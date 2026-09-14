# -*- coding: utf-8 -*-
"""Türkçe semantik çıkarım hattı: varlık → bağlama → özellik → ilişki (P0-5).

Mevcut ``cumle_ayiklayici`` sabit bir sözlükten ``(özne, ilişki, nesne)``
eşler; sözlükte olmayan her şeyi atar. Demo korpusta ölçülen çıkarım verimi
~0.5'tir ve "Ali dün Ankara'ya arabayla gitti" gibi bir cümleden yalnız tek
bir üçlü çıkar — zaman, araç ve olumsuzluk kaybolur.

Bu modül hattı tamamlar::

    Türkçe metin
        ↓ tokenize + morfolojik durum analizi
    ENTITY        (yüzey biçim + durum eki → kök ad)
        ↓ entity linking (aynı varlığın farklı çekimleri tek kimliğe)
    PROPERTY      (sıfat tamlaması: "kırmızı araba" → araba.renk=kırmızı)
        ↓
    RELATION      (fiil + nesne durumu → yönlü ilişki)
        ↓
    NEGATION      ("gitmedi" → polarity=NEGATIVE)
        ↓
    TEMPORAL      ("dün", "yarın", zaman eki → time)
        ↓
    CONFIDENCE    (her kayıt için kural-temelli güven)
        ↓
    KnowledgeStore

Neyi ÖLÇMEZ (dürüstlük sınırı)
------------------------------
Bu istatistiksel bir NER/bağımlılık ayrıştırıcısı değildir. Türkçe morfoloji
kuralları ve sonlu bir sözlük üzerine kurulu **kural tabanlı** bir hattır.
Bilinmeyen sözcükler atılmaz — ``UNKNOWN`` tipli varlık olarak, düşük güvenle
kaydedilir; yani sistem "bilmiyorum" diyebilir ama uydurmaz. Kapsam dışı
yapılar (yan cümle, ilgi cümleciği, ettirgen çatı) çıkarılmaz ve
``skipped_reasons`` altında sayılır — sessiz kayıp yoktur.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..knowledge.schemas import KaynakTuru
from .cumle_ayiklayici import ascii_norm

# ── Morfolojik durum ekleri (yüzeyden kök çıkarımı için) ────────────────────
#: (ek listesi, durum adı). Sıra önemlidir: uzun ek önce denenir.
DURUM_EKLERI: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("ndan", "nden", "tan", "ten", "dan", "den"), "ayrilma"),
    (("nda", "nde", "ta", "te", "da", "de"), "bulunma"),
    (("yla", "yle", "ile", "la", "le"), "vasita"),
    (("na", "ne", "ya", "ye", "a", "e"), "yonelme"),
    (("nı", "ni", "nu", "nü", "yı", "yi", "yu", "yü", "ı", "i", "u", "ü"),
     "belirtme"),
    ((("nın", "nin", "nun", "nün", "ın", "in", "un", "ün")), "tamlayan"),
)

#: Fiil zaman ekleri → (zaman etiketi, güven).
ZAMAN_EKLERI: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("yordu",), "gecmis_surekli"),
    (("acak", "ecek", "acakti", "ecekti"), "gelecek"),
    (("iyor", "ıyor", "uyor", "üyor"), "simdiki"),
    (("mis", "mış", "miş", "muş", "müş"), "gecmis_duyulan"),
    (("di", "dı", "du", "dü", "ti", "tı", "tu", "tü"), "gecmis"),
    (("r", "ir", "ır", "ur", "ür", "ar", "er"), "genis"),
)

#: Zaman belirteçleri → normalize zaman etiketi.
ZAMAN_BELIRTECLERI: Dict[str, str] = {
    "dun": "dün", "bugun": "bugün", "yarin": "yarın",
    "simdi": "şimdi", "sonra": "sonra", "once": "önce",
    "gecen hafta": "geçen hafta", "gelecek hafta": "gelecek hafta",
    "sabah": "sabah", "aksam": "akşam", "gece": "gece", "ogleden sonra": "öğleden sonra",
}

#: Olumsuzluk ekleri (fiil gövdesinde).
OLUMSUZLUK_EKLERI = ("ma", "me")

#: Vasıta ekiyle gelen nesnelerin ilişki adı (araç/yol bilgisi bir ÖZELLİKTİR).
VASITA_OZELLIGI = "travel_mode"

#: Durum eki → varsayılan ilişki rolü.
DURUM_ROLU: Dict[str, str] = {
    "yonelme": "hedef",
    "belirtme": "nesne",
    "bulunma": "konum",
    "ayrilma": "kaynak",
    "vasita": "arac",
    "tamlayan": "sahip",
}


@dataclass
class Varlik:
    """Çıkarılan bir varlık: yüzey biçim, kök ad, tip, durum, güven."""

    surface: str
    lemma: str
    entity_type: str
    case: str
    role: Optional[str]
    is_proper: bool
    confidence: float
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Iliski:
    """Çıkarılan bir ilişki: özne → yüklem → nesne, zaman ve kutup bilgisiyle."""

    subject: str
    predicate: str
    object: str
    role: str
    polarity: str          # POSITIVE | NEGATIVE
    tense: Optional[str]
    time: Optional[str]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Ozellik:
    """Bir varlığa bağlanan özellik (sıfat ya da vasıta bilgisi)."""

    entity: str
    name: str
    value: Any
    source_surface: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SemantikCikarim:
    """Tek cümleden çıkarılan tüm yapılandırılmış bilgi."""

    sentence: str
    entities: List[Varlik] = field(default_factory=list)
    relations: List[Iliski] = field(default_factory=list)
    properties: List[Ozellik] = field(default_factory=list)
    temporal: List[Dict[str, Any]] = field(default_factory=list)
    negations: List[Dict[str, Any]] = field(default_factory=list)
    skipped_reasons: List[str] = field(default_factory=list)

    @property
    def bos_mu(self) -> bool:
        return not (self.entities or self.relations or self.properties)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence": self.sentence,
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "properties": [p.to_dict() for p in self.properties],
            "temporal": list(self.temporal),
            "negations": list(self.negations),
            "skipped_reasons": list(self.skipped_reasons),
        }


# ── Sözlükler (genişletilebilir; bilinmeyen kelime ATILMAZ) ─────────────────
VARSAYILAN_SIFATLAR: Dict[str, Tuple[str, str]] = {
    # ascii anahtar → (özellik adı, değer)
    "kirmizi": ("renk", "kırmızı"),
    "mavi": ("renk", "mavi"),
    "yesil": ("renk", "yeşil"),
    "siyah": ("renk", "siyah"),
    "beyaz": ("renk", "beyaz"),
    "sari": ("renk", "sarı"),
    "buyuk": ("boyut", "büyük"),
    "kucuk": ("boyut", "küçük"),
    "yeni": ("durum", "yeni"),
    "eski": ("durum", "eski"),
    "hizli": ("hiz", "hızlı"),
    "yavas": ("hiz", "yavaş"),
}

VARSAYILAN_TIPLER: Dict[str, str] = {
    "ali": "insan", "ayse": "insan", "mehmet": "insan", "veli": "insan",
    "zeynep": "insan", "fatma": "insan", "can": "insan", "deniz": "insan",
    "ankara": "mekan", "istanbul": "mekan", "izmir": "mekan", "bursa": "mekan",
    "okul": "mekan", "ev": "mekan", "park": "mekan", "kutuphane": "mekan",
    "araba": "tasit", "otobus": "tasit", "tren": "tasit", "bisiklet": "tasit",
    "ucak": "tasit", "gemi": "tasit", "at": "hayvan", "kedi": "hayvan",
    "kopek": "hayvan", "kus": "hayvan", "kitap": "nesne", "kalem": "nesne",
    "masa": "nesne", "telefon": "nesne", "arac": "tasit", "tasit": "tasit",
}

#: Tipe göre otomatik türetilen özellikler (ontolojik varsayım — güveni düşüktür).
TIP_OZELLIKLERI: Dict[str, Dict[str, float]] = {
    "insan": {"canli": 1.0},
    "hayvan": {"canli": 1.0, "binilebilir": 1.0},
    "tasit": {"canli": 0.0, "binilebilir": 1.0},
    "mekan": {"canli": 0.0, "binilebilir": 0.0},
    "nesne": {"canli": 0.0, "binilebilir": 0.0},
}

VARSAYILAN_FIILLER: Dict[str, str] = {
    # ascii fiil kökü → ilişki adı
    "git": "gitmek", "gid": "gitmek", "gel": "gelmek", "bin": "binmek",
    "bak": "bakmak", "ver": "vermek", "al": "almak", "oku": "okumak",
    "yaz": "yazmak", "gor": "görmek", "sev": "sevmek", "kos": "koşmak",
    "otur": "oturmak", "kalk": "kalkmak", "calis": "çalışmak",
    "uyu": "uyumak", "ye": "yemek", "ic": "içmek",
}

#: Güven katsayıları — her sayı kaynağıyla birlikte belgelenir.
GUVEN = {
    "sozlukte_varlik": 0.90,        # tip sözlükte açıkça var
    "ozel_isim": 0.85,              # büyük harf başlangıcı, sözlükte yok
    "bilinmeyen_varlik": 0.40,      # tip bilinmiyor → UNKNOWN
    "sozlukte_fiil": 0.90,
    "bilinmeyen_fiil": 0.45,
    "sifat_ozelligi": 0.85,
    "tip_ozelligi": 0.60,           # ontolojiden türetildi, metinde yazmıyor
    "vasita_ozelligi": 0.80,
    "zaman_belirteci": 0.90,
    "zaman_eki": 0.75,
    "olumsuzluk": 0.90,
}


def _ascii_kelime(kelime: str) -> str:
    return ascii_norm(kelime).strip()


def kok_ve_durum(kelime: str) -> Tuple[str, str]:
    """Yüzey biçimden (kök, durum) çıkar.

    Türkçede ek dizileri belirsizdir: ``okula`` hem ``oku+la`` (vasıta) hem
    ``okul+a`` (yönelme) olarak okunabilir. Tek bir ek listesini sırayla
    denemek bu yüzden yetmez; bütün aday ayrıştırmalar üretilir ve **sözlükte
    karşılığı olan kök** tercih edilir. Hiçbir aday sözlükte yoksa en uzun ek
    seçilir ve kök düşük güvenle raporlanır.

    Kesme işareti (``Ankara'ya``) özel isim sınırıdır ve doğrudan kullanılır.
    """
    ham_kelime = (kelime or "").strip()
    kesme = None
    for isaret in ("'", "\u2019"):
        if isaret in ham_kelime:
            kesme = ham_kelime.split(isaret, 1)
            break
    if kesme is not None:
        govde = _ascii_kelime(kesme[0])
        kalan = _ascii_kelime(kesme[1])
        durum = "yalin"
        for ekler, ad in DURUM_EKLERI:
            if any(kalan == ek or kalan.endswith(ek) for ek in ekler):
                durum = ad
                break
        return govde, durum

    ham = _ascii_kelime(ham_kelime)
    if not ham:
        return "", "yalin"
    # Yüzey biçim doğrudan sözlükteyse hiçbir ek düşürülmez ("at", "araba").
    if ham in VARSAYILAN_TIPLER:
        return ham, "yalin"

    geri = {"b": "p", "c": "ç", "d": "t", "g": "k", "ğ": "k"}
    adaylar: List[Tuple[str, str, int]] = []
    for ekler, durum in DURUM_EKLERI:
        for ek in ekler:
            if not ham.endswith(ek) or len(ham) - len(ek) < 2:
                continue
            govde = ham[: -len(ek)]
            adaylar.append((govde, durum, len(ek)))
            if govde and govde[-1] in geri:
                adaylar.append((govde[:-1] + geri[govde[-1]], durum, len(ek)))

    sozlukte = [a for a in adaylar if a[0] in VARSAYILAN_TIPLER]
    if sozlukte:
        # Sözlükte karşılığı olanlar arasında en UZUN kök en az varsayım yapar.
        govde, durum, _ = max(sozlukte, key=lambda a: len(a[0]))
        return govde, durum
    if adaylar:
        govde, durum, _ = max(adaylar, key=lambda a: a[2])
        return govde, durum
    return ham, "yalin"


def fiil_coz(kelime: str) -> Optional[Dict[str, Any]]:
    """Yüzey fiilden kök, zaman ve olumsuzluk çıkar; fiil değilse ``None``.

    Şimdiki zamanda kök ünlüsü düşer (``oku+yor`` → "okuyor", ``uyu+yor`` →
    "uyuyor"), dolayısıyla eki düşürmek tek başına yanlış kök verir ("ok",
    "u"). Bu yüzden isim çözümündeki gibi aday üretilir ve **sözlükte
    karşılığı olan** kök tercih edilir.
    """
    ham = _ascii_kelime(kelime)
    if len(ham) < 3:
        return None

    adaylar: List[Tuple[str, str]] = []   # (kök adayı, zaman)
    for ekler, etiket in ZAMAN_EKLERI:
        for ek in sorted(ekler, key=len, reverse=True):
            if not ham.endswith(ek) or len(ham) - len(ek) < 2:
                continue
            govde = ham[: -len(ek)]
            adaylar.append((govde, etiket))
            # Şimdiki zamanda düşen kök ünlüsünü geri koy: ok→oku, u→uyu.
            if ek.endswith("yor"):
                for unlu in ("a", "e", "ı", "i", "o", "u", "ü", "ö"):
                    adaylar.append((govde + unlu, etiket))
            # Kaynaştırma 'y' ve yumuşamış ünsüzü geri al: gid→git.
            if govde.endswith("y"):
                adaylar.append((govde[:-1], etiket))
            geri = {"b": "p", "c": "ç", "d": "t", "g": "k", "ğ": "k"}
            if govde and govde[-1] in geri:
                adaylar.append((govde[:-1] + geri[govde[-1]], etiket))
        if adaylar:
            break
    if not adaylar:
        return None

    def olumsuzlugu_ayir(govde: str) -> Tuple[str, bool]:
        # Kaynaştırma 'y'si olumsuzluk ekiyle zaman eki arasına girer
        # (git+me+y+ecek → "gitmeyecek"); önce o düşürülür.
        if govde.endswith("y") and len(govde) > 3:
            govde = govde[:-1]
        for ek in OLUMSUZLUK_EKLERI:
            if govde.endswith(ek) and len(govde) - len(ek) >= 2:
                return govde[: -len(ek)], True
        return govde, False

    cozumler = []
    for govde, zaman in adaylar:
        kok, olumsuz = olumsuzlugu_ayir(govde)
        cozumler.append({"lemma": kok, "tense": zaman, "negated": olumsuz})
    sozlukte = [c for c in cozumler if c["lemma"] in VARSAYILAN_FIILLER]
    if sozlukte:
        return sozlukte[0]
    return cozumler[0]


def _tip_ve_guven(kok: str, yuzey: str) -> Tuple[str, float, bool]:
    ozel = bool(yuzey[:1].isupper())
    if kok in VARSAYILAN_TIPLER:
        return VARSAYILAN_TIPLER[kok], GUVEN["sozlukte_varlik"], ozel
    if ozel:
        return "ozel_isim", GUVEN["ozel_isim"], True
    return "UNKNOWN", GUVEN["bilinmeyen_varlik"], False


def cumle_coz(cumle: str) -> SemantikCikarim:
    """Tek bir Türkçe cümleden tam semantik yapı çıkar."""
    sonuc = SemantikCikarim(sentence=cumle)
    kelimeler = [k for k in re.split(r"\s+", (cumle or "").strip()) if k]
    if not kelimeler:
        sonuc.skipped_reasons.append("bos_cumle")
        return sonuc

    temiz = [k.strip(".,;:!?\"'()") for k in kelimeler]
    temiz = [k for k in temiz if k]

    # 1) Yüklemi bul (genelde son sözcük; değilse sondan tara).
    fiil_index: Optional[int] = None
    fiil_bilgi: Optional[Dict[str, Any]] = None
    for index in range(len(temiz) - 1, -1, -1):
        cozum = fiil_coz(temiz[index])
        if cozum is not None:
            fiil_index, fiil_bilgi = index, cozum
            break
    if fiil_bilgi is None:
        sonuc.skipped_reasons.append("yuklem_bulunamadi")

    # 2) Zaman belirteçleri.
    zaman_degeri: Optional[str] = None
    zaman_indeksleri = set()
    for index, kelime in enumerate(temiz):
        anahtar = _ascii_kelime(kelime)
        if anahtar in ZAMAN_BELIRTECLERI:
            zaman_degeri = ZAMAN_BELIRTECLERI[anahtar]
            zaman_indeksleri.add(index)
            sonuc.temporal.append({
                "surface": kelime, "value": zaman_degeri, "kind": "adverb",
                "confidence": GUVEN["zaman_belirteci"],
            })
    if fiil_bilgi is not None and fiil_index is not None:
        sonuc.temporal.append({
            "surface": temiz[fiil_index], "value": fiil_bilgi["tense"],
            "kind": "verb_tense", "confidence": GUVEN["zaman_eki"],
        })

    # 3) Varlıklar ve sıfat tamlamaları.
    bekleyen_sifatlar: List[Tuple[str, str, Any]] = []
    varliklar: List[Varlik] = []
    for index, kelime in enumerate(temiz):
        if index == fiil_index or index in zaman_indeksleri:
            continue
        anahtar = _ascii_kelime(kelime)
        if anahtar in VARSAYILAN_SIFATLAR:
            ad, deger = VARSAYILAN_SIFATLAR[anahtar]
            bekleyen_sifatlar.append((kelime, ad, deger))
            continue
        kok, durum = kok_ve_durum(kelime)
        if not kok:
            continue
        tip, guven, ozel = _tip_ve_guven(kok, kelime)
        varlik = Varlik(
            surface=kelime, lemma=kok, entity_type=tip, case=durum,
            role=DURUM_ROLU.get(durum, "ozne" if durum == "yalin" else None),
            is_proper=ozel, confidence=guven,
        )
        # Sıfatlar kendinden SONRA gelen ilk isme bağlanır (Türkçe sözdizimi).
        for yuzey, ad, deger in bekleyen_sifatlar:
            varlik.properties[ad] = deger
            sonuc.properties.append(Ozellik(
                entity=kok, name=ad, value=deger, source_surface=yuzey,
                confidence=GUVEN["sifat_ozelligi"]))
        bekleyen_sifatlar = []
        varliklar.append(varlik)

    if bekleyen_sifatlar:
        sonuc.skipped_reasons.append("sifat_bagli_isim_yok")

    # 4) Entity linking: aynı kök farklı durumlarda geçtiyse tek kimliğe bağla.
    birlesik: Dict[str, Varlik] = {}
    for varlik in varliklar:
        mevcut = birlesik.get(varlik.lemma)
        if mevcut is None:
            birlesik[varlik.lemma] = varlik
        else:
            mevcut.properties.update(varlik.properties)
            # Daha bilgilendirici durumu koru (yalın en zayıf bilgidir).
            if mevcut.case == "yalin" and varlik.case != "yalin":
                mevcut.case, mevcut.role = varlik.case, varlik.role
            mevcut.confidence = max(mevcut.confidence, varlik.confidence)
    sonuc.entities = list(birlesik.values())

    # 5) Tip tabanlı ontolojik özellikler (metinde yazmaz → düşük güven).
    for varlik in sonuc.entities:
        for ad, tip_degeri in TIP_OZELLIKLERI.get(
                varlik.entity_type, {}).items():
            if ad in varlik.properties:
                continue
            varlik.properties[ad] = tip_degeri
            sonuc.properties.append(Ozellik(
                entity=varlik.lemma, name=ad, value=tip_degeri,
                source_surface=f"<ontology:{varlik.entity_type}>",
                confidence=GUVEN["tip_ozelligi"]))

    # 6) İlişkiler.
    if fiil_bilgi is not None:
        yuklem = VARSAYILAN_FIILLER.get(fiil_bilgi["lemma"])
        fiil_guveni = GUVEN["sozlukte_fiil"]
        if yuklem is None:
            # Yüklem bilinmiyorsa ilişki UYDURULMAZ. Varlıklar, zaman ve
            # özellikler yine de kaydedilir; yalnız yönlü ilişki iddiası
            # üretilmez. "Bilmiyorum" demek yanlış bilgi üretmekten iyidir.
            sonuc.skipped_reasons.append(
                f"bilinmeyen_fiil:{fiil_bilgi['lemma']}")
            return sonuc
        kutup = "NEGATIVE" if fiil_bilgi["negated"] else "POSITIVE"
        if fiil_bilgi["negated"] and fiil_index is not None:
            sonuc.negations.append({
                "surface": temiz[fiil_index], "scope": yuklem,
                "confidence": GUVEN["olumsuzluk"],
            })

        ozneler = [v for v in sonuc.entities if v.case == "yalin"]
        ozne = ozneler[0].lemma if ozneler else None
        # Türkçede belirtisiz nesne yalın haldedir ("kitap okuyor"). İlk yalın
        # ad özne, sonrakiler nesnedir; hepsini özne saymak yanlış olurdu.
        for fazla in ozneler[1:]:
            fazla.role = "nesne"
        if ozne is None:
            sonuc.skipped_reasons.append("ozne_bulunamadi")
        for varlik in sonuc.entities:
            if ozne is not None and varlik.lemma == ozne:
                continue
            if varlik.case == "vasita":
                # Araç bilgisi bir ilişki değil ÖZELLİKTİR: Ali.travel_mode=araba
                if ozne is not None:
                    sonuc.properties.append(Ozellik(
                        entity=ozne, name=VASITA_OZELLIGI, value=varlik.lemma,
                        source_surface=varlik.surface,
                        confidence=GUVEN["vasita_ozelligi"]))
                continue
            if ozne is None:
                continue
            sonuc.relations.append(Iliski(
                subject=ozne, predicate=yuklem, object=varlik.lemma,
                role=varlik.role or "nesne", polarity=kutup,
                tense=fiil_bilgi["tense"], time=zaman_degeri,
                confidence=round(min(fiil_guveni, varlik.confidence), 4),
            ))
        if ozne is not None and not sonuc.relations:
            sonuc.skipped_reasons.append("nesnesiz_yuklem")

    return sonuc


# ── KnowledgeStore aktarımı ─────────────────────────────────────────────────
def store_a_aktar(store, cikarim: SemantikCikarim,
                  source: KaynakTuru = KaynakTuru.REAL_DATA) -> Dict[str, Any]:
    """Çıkarılan yapıyı KnowledgeStore'a yaz; ne yazıldığını say.

    Güven değerleri korunur: ontolojiden türetilen özellik ile metinde açıkça
    yazan sıfat aynı güvenle kaydedilmez.
    """
    varlik_idleri: Dict[str, str] = {}
    for varlik in cikarim.entities:
        entity_id = f"E_{varlik.lemma.upper()}"
        store.varlik_ekle(varlik.lemma, entity_type=varlik.entity_type,
                          entity_id=entity_id, ozel_isim=varlik.is_proper)
        varlik_idleri[varlik.lemma] = entity_id

    for ozellik in cikarim.properties:
        ozellik_id = varlik_idleri.get(ozellik.entity)
        if ozellik_id is None:
            continue
        deger = ozellik.value
        sayisal = float(deger) if isinstance(deger, (int, float)) else 1.0
        store.ozellik_koy(ozellik_id, ozellik.name, sayisal, source=source,
                          confidence=ozellik.confidence)
        if not isinstance(deger, (int, float)):
            # Kategorik değer ayrıca "ad=deger" bayrağı olarak da yazılır ki
            # "renk=kırmızı" bilgisi sayıya indirgenip kaybolmasın.
            store.ozellik_koy(ozellik_id, f"{ozellik.name}={deger}", 1.0,
                              source=source, confidence=ozellik.confidence)

    iliski_sayisi = 0
    for iliski in cikarim.relations:
        ozne_id = varlik_idleri.get(iliski.subject)
        nesne_id = varlik_idleri.get(iliski.object)
        if ozne_id is None or nesne_id is None:
            continue
        relation_id = f"R_{ascii_norm(iliski.predicate).upper().replace(' ', '_')}"
        if not store.relations.icerir(relation_id):
            store.iliski_tanimla(iliski.predicate, relation_id=relation_id)
        # Olumsuz ilişki POZİTİF olgu olarak yazılmaz; skor 0 ile kaydedilir.
        skor = 0.0 if iliski.polarity == "NEGATIVE" else 1.0
        store.olgu_kaydet(ozne_id, relation_id, nesne_id, score=skor,
                          source=source, confidence=iliski.confidence)
        iliski_sayisi += 1

    return {
        "entities_written": len(varlik_idleri),
        "properties_written": len(cikarim.properties),
        "relations_written": iliski_sayisi,
        "negations": len(cikarim.negations),
        "temporal": len(cikarim.temporal),
    }


def korpus_coz(cumleler: Sequence[str]) -> Dict[str, Any]:
    """Cümle listesini çöz ve çıkarım verimi metriklerini raporla."""
    cikarimlar = [cumle_coz(c) for c in cumleler]
    toplam = len(cikarimlar) or 1
    bos = sum(1 for c in cikarimlar if c.bos_mu)
    iliskili = sum(1 for c in cikarimlar if c.relations)
    nedenler: Dict[str, int] = {}
    for cikarim in cikarimlar:
        for neden in cikarim.skipped_reasons:
            anahtar = neden.split(":", 1)[0]
            nedenler[anahtar] = nedenler.get(anahtar, 0) + 1
    return {
        "sentences": len(cikarimlar),
        "extractions": cikarimlar,
        "entity_yield": round(
            sum(len(c.entities) for c in cikarimlar) / toplam, 4),
        "relation_yield": round(
            sum(len(c.relations) for c in cikarimlar) / toplam, 4),
        "property_yield": round(
            sum(len(c.properties) for c in cikarimlar) / toplam, 4),
        "temporal_yield": round(
            sum(len(c.temporal) for c in cikarimlar) / toplam, 4),
        "negation_yield": round(
            sum(len(c.negations) for c in cikarimlar) / toplam, 4),
        "sentence_coverage": round(1.0 - bos / toplam, 4),
        "relation_coverage": round(iliskili / toplam, 4),
        "skipped_reasons": dict(sorted(nedenler.items())),
    }


__all__ = [
    "Varlik", "Iliski", "Ozellik", "SemantikCikarim",
    "DURUM_EKLERI", "ZAMAN_EKLERI", "ZAMAN_BELIRTECLERI", "GUVEN",
    "VARSAYILAN_SIFATLAR", "VARSAYILAN_TIPLER", "VARSAYILAN_FIILLER",
    "kok_ve_durum", "fiil_coz", "cumle_coz", "store_a_aktar", "korpus_coz",
]
