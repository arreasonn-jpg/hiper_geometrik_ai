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
    * özel isimlerde kesme işareti (').

KAPSAM VE DÜRÜST SINIRLAR (bilinçli — tam morfoloji modülü gerektirir):
    * Ünsüz yumuşaması kural-genelleme DEĞİL, küratörlü istisna listeleriyle
      sınırlı bir yaklaşımdır; Türkçede tek heceli sözcükler (top→topa,
      at→ata, ama kap→kaba, renk→renge) ve alıntılar (saat→saate) düzensizdir.
    * Ünlü düşmesi (burun→burna), belirtme/bulunma/ayrılmada iyelik zincirleri,
      çekimli fiil üretimi, isim tamlamaları uygulanmaz.
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
INCE_UYUMLU_ISTISNALAR = {
    "saat", "harf", "kalp", "gol", "rol", "petrol", "kontrol", "alkol",
    "normal", "sembol", "santral",
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
    govde = unsuz_yumusat(k, ozel_isim)
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
    govde = unsuz_yumusat(k, ozel_isim)
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
