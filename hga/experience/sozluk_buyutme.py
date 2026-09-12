# -*- coding: utf-8 -*-
"""
Sözlük Büyütme — gerçek korpustan YENİ VARLIK desenleri çıkarma
================================================================
(v1.0+ — rapor §8 "dış korpus ölçeği"; §12 "gerçek veri → temsil")

`CumleAyiklayici` dar ve lexicon tabanlıdır; bilinmeyen cümleler atlanır ve
sistem ASLA uydurmaz. Bu modül, "sözlük gerçek korpustan çıkarılan desenlerle
büyütülmeli" maddesini aynı dürüstlük ilkesiyle gerçekler: bilinen bir fiil
deseniyle eşleşen basit cümlelerden yalnızca YENİ ÖZNE ve YENİ NESNE varlıkları
sözlüğe eklenir. YENİ İLİŞKİ ASLA otomatik eklenmez (ilişki anlamı tek bir
cümleden güvenle çıkarılamaz).

Büyüme kuralları (dar, belgeli, yalnız yüzey eşlemesi):
    1. Cümlede sözlükteki bir yüklem (fiil yüzey biçimi) bulunmalı.
    2. ÖZNE: ya sözlükte vardır ya da cümlenin İLK kelimesidir ve özel isimdir
       (büyük harfle başlar) → "insan" tipiyle eklenir. Küçük harfli yeni özne
       EKLENMEZ (tipi tek cümleden güvenle çıkarılamaz).
    3. NESNE: ya sözlükte vardır ya da fiilin HEMEN ÖNÜNDEKİ kelimedir (SOV) ve
       ilişkinin beklenen durum ekiyle biter (örn. binmek→yönelme -a/-e/-ya/-ye)
       → kökü çıkarılarak "varlik" tipiyle eklenir. Durum eki uyuşmazsa ATLANIR.
    4. Eşleşmeyen/kuşkulu cümleler sessizce atlanır — sistem asla uydurmaz.

Dürüst sınırlar: bu bir sözdizimsel ayrıştırıcı DEĞİLDİR; "ilk kelime = özne,
fiilin önü = nesne" kısa/yalın Türkçe cümleler için geçerli bir yüzey buluşsalıdır
(belgeli; kapsam dışı cümleler atlanır). Kök çıkarma en-iyi-çabadır (durum eki
soyulur); Türkçe büyük-İ (İzmir→Izmir değil) ayrımı kapsam dışıdır.
"""
import copy
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

from .cumle_ayiklayici import ascii_norm
from .turkce import hece_sayisi

# İlişkinin beklediği nesne durumu → kabul edilen ek sonları (en uzun eşleşme önce).
_DURUM_SONLARI = {
    "yonelme": ["ya", "ye", "a", "e"],
    "belirtme": ["yı", "yi", "yu", "yü", "ı", "i", "u", "ü"],
    "bulunma": ["da", "de", "ta", "te"],
    "ayrilma": ["dan", "den", "tan", "ten"],
}

# Ünsüz yumuşamasının TERSİ (kök çıkarma, en-iyi-çaba; belgeli):
# sesliyle başlayan bir durum eki soyulduğunda kökün sonundaki yumuşamış ünsüz
# (p→b, ç→c, t→d, k→g) sert biçimine çevrilir. Türkçe öz sözcükler sözcük
# sonunda b/c/d/g taşımaz; bu yüzden bu dört geri-çevirme yüksek güvenlidir.
# "ğ" AMBİGU olduğu için DOKUNULMAZ (dağ = asıl kök; bebek→bebeğ = yumuşama).
# Borçlanma sözcüklerin (psikolog, katalog…) sonundaki g/ğ bu yüzden yanlış
# çevrilebilir — dar, yalın Türkçe korpuslar için belgeli bir sınırdır.
_YUMUSAMA_GERI = {"b": "p", "c": "ç", "d": "t", "g": "k"}
# Tek heceli olup yumuşayan (düzensiz) sözcüklerin yumuşamış gövdesi → kök.
_YUMUSAMA_GERI_TEK_HECELI = {
    "kab": "kap", "reng": "renk", "gög": "gök", "cog": "çok", "yurd": "yurt",
}


@dataclass
class SozlukBuyutmeRaporu:
    taranan_cumle: int = 0
    eslesen_cumle: int = 0
    atlanan_cumle: int = 0
    yeni_ozne: int = 0
    yeni_nesne: int = 0
    yeni_iliski: int = 0   # DAİMA 0 — ilişki asla otomatik uydurulmaz

    def to_dict(self) -> Dict:
        return asdict(self)


def _ozgun_kelimeler(cumle: str) -> List[str]:
    """Orijinal (büyük/küçük harfi koruyan) kelimeler."""
    return re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşüâîû]+", cumle or "")


def _ozel_isim_mi(kelime: str) -> bool:
    """Büyük harfle başlayan, 1 harften uzun kelime → özel isim adayı."""
    return bool(kelime) and len(kelime) > 1 and kelime[0].isupper()


def _baslik(kelime: str) -> str:
    """İlk harfi büyüt, gerisini koru (capitalize()'ın İ→I tuzağına düşme)."""
    k = (kelime or "").strip()
    if not k:
        return k
    return k[0].upper() + k[1:]


def _kok_bul(yuzey: str, durum: str) -> str:
    """Yüzey biçiminden durum ekini soyar (en-iyi-çaba kök; özgün harfler korunur).

    Sesliyle başlayan ek soyulduktan sonra kalan kök yumuşamış bir ünsüzle
    bitiyorsa (b/c/d/g) sert biçime çevrilir (`_YUMUSAMA_GERI`); "ğ" ambigua
    olduğu için dokunulmaz. Tek heceli gövdeler yalnızca bilinen yumuşayan
    sözcüklerse çevrilir. Ünlü düşmesi (resim→resmi→"resm") geri alınmaz —
    bu da belgeli bir sınırdır.
    """
    k = (yuzey or "").lower()
    for son in _DURUM_SONLARI.get(durum or "", []):
        if len(k) > len(son) + 1 and k.endswith(son):
            kok = yuzey[:len(yuzey) - len(son)]
            if kok and kok[-1] in _YUMUSAMA_GERI:
                anahtar = kok.lower()
                if hece_sayisi(anahtar) >= 2 or anahtar in _YUMUSAMA_GERI_TEK_HECELI:
                    kok = kok[:-1] + _YUMUSAMA_GERI[kok[-1]]
            return kok
    return yuzey


def _durum_uyuyor(kelime: str, durum: str) -> bool:
    """Kelime, ilişkinin beklediği durum ekiyle bitiyor mu? (durum yoksa: serbest)"""
    if not durum:
        return True
    k = (kelime or "").lower()
    return any(k.endswith(son) for son in _DURUM_SONLARI.get(durum, []))


def _nesne_adayi(norm: List[str], yuklem: str, ozne_ascii: str) -> str:
    """Fiilin hemen önündeki, özneden farklı kelime (SOV nesne adayı)."""
    if yuklem not in norm:
        return ""
    i = norm.index(yuklem)
    if i == 0:
        return ""
    aday = norm[i - 1]
    if aday == ozne_ascii:
        return ""
    return aday


def sozlugu_buyut(sozluk: Dict, cumleler: List[str]):
    """Sözlüğü gerçek cümlelerden büyütür: (yeni_sozluk, rapor, bulunan_ucluler).

    `sozluk` DEĞİŞTİRİLMEZ (derin kopya döner). `bulunan_ucluler`, her eşleşen
    cümle için (özne_token, ilişki_token, nesne_token) üçlüsüdür (yeni varlıklar
    dahil). Yeni ilişki EKLENMEZ.
    """
    sozluk = copy.deepcopy(sozluk or {})
    rapor = SozlukBuyutmeRaporu()
    bulunan: List[Tuple[str, str, str]] = []

    for cumle in cumleler:
        rapor.taranan_cumle += 1
        norm = ascii_norm(cumle).split()
        if not norm:
            rapor.atlanan_cumle += 1
            continue
        ozgun = _ozgun_kelimeler(cumle)
        ozgun_map = {ascii_norm(w): w for w in ozgun}
        eslesti = False

        for _, sablon in sozluk.items():
            yuklem = next((k for k in norm if k in sablon.get("yuklemler", [])),
                          None)
            if yuklem is None:
                continue

            # ── özne ──
            ozne_ascii = next((k for k in norm
                               if k in sablon.get("ozneler", {})), None)
            ozne_yeni = False
            if ozne_ascii is None:
                ilk = ozgun[0] if ozgun else ""
                if _ozel_isim_mi(ilk) and ascii_norm(ilk) in norm:
                    ozne_ascii = ascii_norm(ilk)
                    ozne_yeni = True
            if ozne_ascii is None:
                continue

            # ── nesne ──
            nesne_ascii = next((k for k in norm
                                if k in sablon.get("nesneler", {})), None)
            nesne_yeni = False
            if nesne_ascii is None:
                aday = _nesne_adayi(norm, yuklem, ozne_ascii)
                if aday and _durum_uyuyor(aday, sablon.get("nesne_durumu")):
                    nesne_ascii = aday
                    nesne_yeni = True
            if nesne_ascii is None:
                continue

            eslesti = True

            if ozne_yeni:
                sablon["ozneler"][ozne_ascii] = {
                    "token": ozgun_map.get(ozne_ascii, _baslik(ozne_ascii)),
                    "tip": "insan",
                    "ozellikler": {},
                }
                rapor.yeni_ozne += 1
            if nesne_yeni:
                nesne_ozgun = ozgun_map.get(nesne_ascii, nesne_ascii)
                kok = _kok_bul(nesne_ozgun, sablon.get("nesne_durumu"))
                sablon["nesneler"][nesne_ascii] = {
                    "token": _baslik(kok),
                    "tip": "varlik",
                    "ozellikler": {},
                }
                rapor.yeni_nesne += 1

            bulunan.append((sablon["ozneler"][ozne_ascii]["token"],
                            sablon["iliski"],
                            sablon["nesneler"][nesne_ascii]["token"]))
            break  # her cümle tek desenle eşleşir

        if eslesti:
            rapor.eslesen_cumle += 1
        else:
            rapor.atlanan_cumle += 1

    return sozluk, rapor, bulunan
