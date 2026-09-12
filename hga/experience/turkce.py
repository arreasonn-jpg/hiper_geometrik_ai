# -*- coding: utf-8 -*-
"""
Türkçe Morfoloji — ek uyumu yardımcıları (v0.2++)
==================================================
(rapor §19 v0.2: "Generator'ı ... gerçek metin/olay üretimine genişlet")

Tam bir doğal dil üreteci kurmak yerine, metin üretiminde gereken EN KÜÇÜK
morfoloji parçasını doğru uygular:

    * sesli uyumu      — kalın (a/ı/o/u) / ince (e/i/ö/ü)
    * yönelme  (-a/-e) ve belirtme (-ı/-i/-u/-ü): sesliyle biten gövdede 'y'
    * bulunma  (-da/-de) ve ayrılma (-dan/-den): ötümsüz ünsüzden sonra
      ünsüz benzeşmesi (-ta/-te, -tan/-ten)
    * çoğul    (-lar/-ler)
    * ünsüz yumuşaması: çok heceli gövdelerde p/ç/t/k → b/c/d/ğ (sesliyle
      başlayan ek gelince); tek heceli ve istisna sözcüklerde uygulanmaz.
    * ünlü düşmesi: iki heceli, ikinci hecesinde DAR ünlü (ı/i/u/ü) taşıyan
      sözcüklerde sesliyle başlayan ek gelince o ünlü düşer (burun→burna,
      şehir→şehre); küratörlü listeyle sınırlı.
    * iyelik ekleri (6 kişi) + iyelik sonrası durum zincirleri: 3. tekil
      iyelikten sonra durum ekleri 'n' ara harfi alır (evi→evine, araba→
      arabasına).
    * görülen geçmiş zaman 3. tekil fiil çekimi (-dı/-di/-du/-dü; ötümsüzden
      sonra -tı/-ti/-tu/-tü): bin→bindi, bak→baktı, gör→gördü.
    * özel isimlerde kesme işareti (').

KAPSAM VE DÜRÜST SINIRLAR (bilinçli):
    * Ünsüz yumuşaması ve ünlü düşmesi kural-genelleme DEĞİL, küratörlü
      istisna listeleriyle sınırlı bir yaklaşımdır; Türkçede tek heceli
      sözcükler (top→topa, at→ata, ama kap→kaba, renk→renge) ve alıntılar
      (saat→saate) düzensizdir.
    * İsim tamlamaları (belirtili/belirtisiz), geniş zaman/şimdiki zaman/
      gelecek zaman çekimleri ve kişi ekli fiil çekimleri uygulanmaz.
    * Fonksiyonlar büyük/küçük harfi DEĞİŞTİRMEZ; cümle içinde cins isimler
      çağıran tarafından küçük harfe çevrilerek verilir.
"""
import re

UNLULER = "aeiouöüı"
KALIN = set("aıou")           # kalın ünlüler → -a / -ı / -u / -da / -lar
INCE = set("eiöü")            # ince ünlüler → -e / -i / -ü / -de / -ler
YUVARLAK = set("ouöü")        # yuvarlak ünlüler (belirtme -u/-ü seçiminde)
UNVOICED = set("fstkçşhp")    # ötümsüz ünsüzler (ünsüz benzeşmesi: -ta/-tan)

# Kalın ünlü taşıyıp İNCE uyumlu ek alan alıntı sözcükler (saat→saate, harf→harfe).
# "vakt", "vakit"in ünlü düşmeli gövdesidir (vakti/vakte de ince uyumludur).
INCE_UYUMLU_ISTISNALAR = {
    "saat", "harf", "kalp", "gol", "rol", "petrol", "kontrol", "alkol",
    "normal", "sembol", "santral", "vakit", "vakt",
}

# İyelik ekli / istisna kelimeler: genel kaynaştırma kurallarına uymazlar.
# (3. tekil iyelik -ı/-i/-u/-ü/-sı/-si'den sonra durum ekleri 'n' alır.)
YONELME_ISTISNALARI = {"gökyüzü": "gökyüzüne", "yüzü": "yüzüne",
                       "su": "suya", "ne": "neye"}
BELIRTME_ISTISNALARI = {"gökyüzü": "gökyüzünü", "su": "suyu", "ne": "neyi"}
BULUNMA_ISTISNALARI = {"gökyüzü": "gökyüzünde", "su": "suda"}
AYRILMA_ISTISNALARI = {"gökyüzü": "gökyüzünden", "su": "sudan"}

# Ünsüz yumuşaması: sesliyle başlayan ek gelince son ünsüzün dönüşümü.
YUMUSAMA = {"p": "b", "ç": "c", "t": "d", "k": "ğ"}
# Çok heceli olup YUMUŞAMAYAN alıntı/istisna sözcükler.
YUMUSAMAYAN_ISTISNALAR = {
    "saat", "hukuk", "ceket", "sepet", "cumhuriyet", "sanat",
    "merhamet", "ahlak", "seyahat", "isabet", "millet",
}
# Tek heceli olup YUMUŞAYAN sözcükler (düzensiz; küratörlü liste).
YUMUSAYAN_TEK_HECELI = {
    "çok": "çoğ", "gök": "göğ", "yurt": "yurd", "renk": "reng", "kap": "kab",
}

# Ünlü düşmesi: iki heceli, ikinci hecesinde DAR ünlü (ı/i/u/ü) taşıyan
# sözcüklerin sesliyle başlayan ek öncesi gövdesi (burun→burn-, şehir→şehr-).
# Düzensiz olduğu için küratörlü liste (kural-genelleme DEĞİL). Bazı girdiler
# ünlü düşmesi + ünsüz yumuşamasını birlikte taşır (kayıp→kayb-, fesat→fesad-).
UNLU_DUSMESI = {
    "burun": "burn", "ağız": "ağz", "oğul": "oğl", "şehir": "şehr",
    "karın": "karn", "alın": "aln", "beyin": "beyn", "isim": "ism",
    "fikir": "fikr", "nehir": "nehr", "resim": "resm", "vakit": "vakt",
    "gönül": "gönl", "ömür": "ömr", "hüküm": "hükm", "nesil": "nesl",
    "asıl": "asl",
    "kayıp": "kayb", "fesat": "fesad",
}

# İyelik (sahiplik) kişi kodları → ek kuralı açıklaması.
#   1t/2t/3t: benim/senin/onun (tekil)  1c/2c/3c: bizim/sizin/onların (çoğul)
IYELIK_KISILER = ("1t", "2t", "3t", "1c", "2c", "3c")


def son_unlu(kelime: str) -> str:
    """Kelimenin son sesli harfi (küçük harf); yoksa ''."""
    for harf in reversed((kelime or "").lower()):
        if harf in UNLULER:
            return harf
    return ""


def hece_sayisi(kelime: str) -> int:
    """Yaklaşık hece sayısı = sesli gruplarının sayısı."""
    k = (kelime or "").lower()
    n = 0
    onceki_unlu = False
    for harf in k:
        if harf in UNLULER:
            if not onceki_unlu:
                n += 1
            onceki_unlu = True
        else:
            onceki_unlu = False
    return n


def _sesliyle_bitiyor(kelime: str) -> bool:
    return bool(kelime) and kelime[-1].lower() in UNLULER


def _son_harf(kelime: str) -> str:
    return kelime[-1].lower() if kelime else ""


def _belirtme_unlusu(kelime: str, ozel_isim: bool = False) -> str:
    """Belirtme ekinin ünlüsü (4-yönlü uyum + ince-uyumlu istisnalar).

    Kalın+düz (a/ı)→ı, ince+düz (e/i)→i, kalın+yuvarlak (o/u)→u,
    ince+yuvarlak (ö/ü)→ü. 'saat' gibi ince-uyumlu alıntılarda öncelik incedir.
    """
    u = son_unlu(kelime)
    ince = (not ozel_isim and (kelime or "").lower() in INCE_UYUMLU_ISTISNALAR) \
        or (u in INCE)
    if u in YUVARLAK:
        return "ü" if ince else "u"
    return "i" if ince else "ı"


def _kalın_mı(kelime: str, ozel_isim: bool = False) -> bool:
    """2-yönlü ekler (a/e) için kalın mı? İnce-uyumlu istisnalar ince sayılır."""
    if not ozel_isim and (kelime or "").lower() in INCE_UYUMLU_ISTISNALAR:
        return False
    return son_unlu(kelime) in KALIN


def unsuz_yumusat(kelime: str, ozel_isim: bool = False) -> str:
    """Sesliyle başlayan ekten ÖNCE gövdeyi hazırla (ünsüz yumuşaması).

    Özel isimlerde yumuşama yazılmaz (Mehmet'e, Zonguldak'a). Çok heceli
    gövdelerde p/ç/t/k → b/c/d/ğ; tek heceli ve istisna sözcüklerde korunur.
    """
    k = (kelime or "").strip()
    if not k or ozel_isim:
        return k
    anahtar = k.lower()
    if anahtar in YUMUSAYAN_TEK_HECELI:
        return YUMUSAYAN_TEK_HECELI[anahtar]
    son = anahtar[-1]
    if son not in YUMUSAMA:
        return k
    if anahtar in YUMUSAMAYAN_ISTISNALAR:
        return k
    if hece_sayisi(anahtar) < 2:
        return k
    return k[:-1] + YUMUSAMA[son]


def unlu_dusmesi(kelime: str, ozel_isim: bool = False) -> str:
    """Sesliyle başlayan ekten ÖNCE gövdeyi hazırla: ünlü düşmesi + yumuşama.

    İki heceli, ikinci hecesi DAR ünlü taşıyan sözcüklerde o ünlü düşer
    (burun→burn-, şehir→şehr-); ardından ünsüz yumuşaması uygulanır.
    Özel isimlerde ve ünsüzle başlayan eklerde (bulunma/ayrılma/çoğul)
    UYGULANMAZ. Sınırlı, küratörlü bir listedir (kural-genelleme değil).
    """
    k = (kelime or "").strip()
    if not k or ozel_isim:
        return k
    anahtar = k.lower()
    if anahtar in UNLU_DUSMESI:
        return UNLU_DUSMESI[anahtar]
    return unsuz_yumusat(k, ozel_isim)


def kucult(kelime: str) -> str:
    """Cins isim için küçük harfe çevir (İ→i sorununu da çözer)."""
    return (kelime or "").replace("İ", "i").replace("I", "i").lower()


def temizle(metin: str) -> str:
    """Noktalama ve fazla boşlukları temizle (cümle ön-işleme için)."""
    return re.sub(r"\s+", " ", (metin or "")).strip()


def yonelme_eki(kelime: str, ozel_isim: bool = False) -> str:
    """Yönelme (-a/-e) hâlini ekle.

    Kural: son ünlü kalınsa '-a', inceyse '-e'; sesliyle biten gövdede 'y'
    kaynaştırma; özel isimde kesme işareti; ünsüz yumuşaması uygulanır.
    """
    k = (kelime or "").strip()
    if not k:
        return ""
    istisna = YONELME_ISTISNALARI.get(k.lower())
    if istisna is not None and not ozel_isim:
        return istisna
    govde = unlu_dusmesi(k, ozel_isim)
    ek = "a" if _kalın_mı(govde, ozel_isim) else "e"
    if ozel_isim:
        return f"{k}'{('y' if _sesliyle_bitiyor(k) else '')}{ek}"
    return f"{govde}{('y' if _sesliyle_bitiyor(govde) else '')}{ek}"


def belirtme_eki(kelime: str, ozel_isim: bool = False) -> str:
    """Belirtme (-ı/-i/-u/-ü) hâlini ekle."""
    k = (kelime or "").strip()
    if not k:
        return ""
    istisna = BELIRTME_ISTISNALARI.get(k.lower())
    if istisna is not None and not ozel_isim:
        return istisna
    govde = unlu_dusmesi(k, ozel_isim)
    ek = _belirtme_unlusu(govde, ozel_isim)
    if ozel_isim:
        return f"{k}'{('y' if _sesliyle_bitiyor(k) else '')}{ek}"
    return f"{govde}{('y' if _sesliyle_bitiyor(govde) else '')}{ek}"


def bulunma_eki(kelime: str, ozel_isim: bool = False) -> str:
    """Bulunma (-da/-de; ötümsüzden sonra -ta/-te) hâlini ekle."""
    k = (kelime or "").strip()
    if not k:
        return ""
    istisna = BULUNMA_ISTISNALARI.get(k.lower())
    if istisna is not None and not ozel_isim:
        return istisna
    uyum = "a" if _kalın_mı(k, ozel_isim) else "e"
    once = "d"
    if not _sesliyle_bitiyor(k) and _son_harf(k) in UNVOICED:
        once = "t"
    ek = once + uyum
    return f"{k}'{ek}" if ozel_isim else f"{k}{ek}"


def ayrilma_eki(kelime: str, ozel_isim: bool = False) -> str:
    """Ayrılma (-dan/-den; ötümsüzden sonra -tan/-ten) hâlini ekle."""
    k = (kelime or "").strip()
    if not k:
        return ""
    istisna = AYRILMA_ISTISNALARI.get(k.lower())
    if istisna is not None and not ozel_isim:
        return istisna
    uyum = "a" if _kalın_mı(k, ozel_isim) else "e"
    once = "d"
    if not _sesliyle_bitiyor(k) and _son_harf(k) in UNVOICED:
        once = "t"
    ek = once + uyum + "n"
    return f"{k}'{ek}" if ozel_isim else f"{k}{ek}"


def cogul_eki(kelime: str, ozel_isim: bool = False) -> str:
    """Çoğul (-lar/-ler) eki ekle."""
    k = (kelime or "").strip()
    if not k:
        return ""
    ek = "lar" if _kalın_mı(k, ozel_isim) else "ler"
    return f"{k}'{ek}" if ozel_isim else f"{k}{ek}"


# ── İyelik (sahiplik) ekleri ────────────────────────────────────────────────
def iyelik_eki(kelime: str, kisi: str) -> str:
    """Sahiplik eki ekle (kisi: 1t/2t/3t tekil, 1c/2c/3c çoğul).

    Sesliyle başlayan ek olduğu için ünlü düşmesi + ünsüz yumuşaması uygulanır
    (kitap→kitabım, burun→burnum). 3. tekilde sesliyle biten gövde 's' kaynaştırma
    alır (araba→arabası, ev→evi). Özel isim iyelik/tamlama KAPSAM DIŞIDIR.
    """
    k = (kelime or "").strip()
    if not k:
        return ""
    if kisi not in IYELIK_KISILER:
        raise ValueError(f"kisi {kisi!r} tanınmadı; beklenen: {IYELIK_KISILER}")
    govde = unlu_dusmesi(k)
    dar = _belirtme_unlusu(govde)  # ı/i/u/ü (4-yönlü uyum)
    sesli = _sesliyle_bitiyor(govde)

    if kisi == "1t":
        return govde + ("" if sesli else dar) + "m"        # -m / -ım / -um
    if kisi == "2t":
        return govde + ("" if sesli else dar) + "n"        # -n / -ın / -un
    if kisi == "3t":
        return govde + (("s" + dar) if sesli else dar)     # -(s)ı/i/u/ü
    if kisi == "1c":
        return govde + ("" if sesli else dar) + "m" + dar + "z"   # -(ı)mız
    if kisi == "2c":
        return govde + ("" if sesli else dar) + "n" + dar + "z"   # -(ı)nız
    # 3c: -ları/-leri ünsüzle başlar → yumuşama/ünlü düşmesi YOK (kitapları).
    return k + ("ları" if _kalın_mı(k) else "leri")


def iyelik_li_durum(kelime: str, kisi: str, durum: str) -> str:
    """İyelik eki + durum eki zinciri (belirtme/bulunma/ayrılmada da çalışır).

    durum: 'yonelme' | 'belirtme' | 'bulunma' | 'ayrilma'.
    3. tekil iyelikten sonra durum eki 'n' ara harfi alır: ev→evi→evine,
    araba→arabası→arabasına (diğer kişilerde ara harf YOK: evim→evime).
    """
    if kisi not in IYELIK_KISILER:
        raise ValueError(f"kisi {kisi!r} tanınmadı; beklenen: {IYELIK_KISILER}")
    iyelikli = iyelik_eki(kelime, kisi)
    govde = iyelikli + ("n" if kisi == "3t" else "")
    return _durum_ekle(govde, durum)


def _durum_ekle(govde: str, durum: str) -> str:
    """HAZIR gövdeye (yumuşama/ünlü düşmesi yapılmış) durum ekini ekle.

    İstisna sözlüklerine bakmaz; iyelikli zincirlerde kullanılır.
    """
    durum = (durum or "").strip().lower()
    if durum in ("yonelme", "y"):
        ek = "a" if _kalın_mı(govde) else "e"
        return f"{govde}{('y' if _sesliyle_bitiyor(govde) else '')}{ek}"
    if durum in ("belirtme", "b"):
        ek = _belirtme_unlusu(govde)
        return f"{govde}{('y' if _sesliyle_bitiyor(govde) else '')}{ek}"
    if durum in ("bulunma", "bl"):
        uyum = "a" if _kalın_mı(govde) else "e"
        once = "t" if (not _sesliyle_bitiyor(govde) and _son_harf(govde) in UNVOICED) else "d"
        return f"{govde}{once}{uyum}"
    if durum in ("ayrilma", "a"):
        uyum = "a" if _kalın_mı(govde) else "e"
        once = "t" if (not _sesliyle_bitiyor(govde) and _son_harf(govde) in UNVOICED) else "d"
        return f"{govde}{once}{uyum}n"
    raise ValueError(f"durum {durum!r} tanınmadı; beklenen: "
                     "yonelme/belirtme/bulunma/ayrilma")


# ── Fiil çekimi (görülen geçmiş zaman 3. tekil) ─────────────────────────────
def gecmis_zaman_3tekil(koku: str) -> str:
    """Fiil köküne görülen geçmiş zaman 3. tekil eki ekle.

    -dı/-di/-du/-dü (son ünlüye göre 4-yönlü); ötümsüz ünsüzden sonra
    -tı/-ti/-tu/-tü. Kök '-mak/-mek' mastarıyla verilirse o düşürülür.
    bin→bindi, bak→baktı, gör→gördü, dur→durdu, yazmak→yazdı.
    """
    k = (koku or "").strip()
    if not k:
        return ""
    if k.lower().endswith(("mak", "mek")):
        k = k[:-3]
    u = son_unlu(k)
    once = "t" if (not _sesliyle_bitiyor(k) and _son_harf(k) in UNVOICED) else "d"
    if u in KALIN:
        return f"{k}{once}{'u' if u in YUVARLAK else 'ı'}"
    return f"{k}{once}{'ü' if u in YUVARLAK else 'i'}"
