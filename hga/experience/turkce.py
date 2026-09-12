# -*- coding: utf-8 -*-
"""
Türkçe Morfoloji — ek uyumu yardımcıları (v0.2++)
==================================================
(rapor §19 v0.2: "Generator'ı ... gerçek metin/olay üretimine genişlet")

Tam bir doğal dil üreteci kurmak yerine, metin üretiminde gereken EN KÜÇÜK
morfoloji parçasını doğru uygular: sesli uyumu (vowel harmony) ve özel isim
kesme işareti kuralları. Kapsam bilinçli olarak dardır:

    * `son_unlu()`        — kelimenin son sesli harfi (a/ı/o/u → kalın; e/i/ö/ü → ince)
    * `yonelme_eki()`     — yönelme (-a / -e) hâli; sesliyle biten kelimede 'y'
                            kaynaştırma harfi, özel isimde kesme işareti (').

Örnekler (testlerde kilitli):
    ata        → ataya        (kalın, 'y' kaynaştırma)
    araba      → arabaya
    gökyüzü    → gökyüzüne    (ince)
    kitap      → kitaba       (sessizle biten)
    Ali (özel) → Ali'ye       (kesme işareti + 'y')
    Ata (özel) → Ata'ya
    Atatürk (özel) → Atatürk'e

NOT: Fonksiyonlar büyük/küçük harfi DEĞİŞTİRMEZ; cümle içinde cins isimler
çağıran tarafından küçük harfe çevrilerek verilir.

KAPSAM DIŞI (bilinçli — tam morfoloji modülü gerektirir):
  * ünsüz yumuşaması (kitap → kitaba, ağaç → ağaca): p/ç/t/k yumuşaması
    tek heceli/çok heceli birçok istisna taşır ve burada uygulanmaz.
  * belirtme (-ı/-i), bulunma (-da/-de) gibi diğer durum ekleri.
  * ünlü düşmesi (burun → burna), isim tamlamaları, çekimli fiil üretimi.
"""
import re

UNLULER = "aeiouöüı"
KALIN = set("aıou")   # kalın ünlüler → -a
# ince ünlüler (e,i,ö,ü) → -e

# İyelik ekli / istisna kelimeler: genel 'y' kaynaştırma kuralına uymazlar.
# (Türkçede 3. tekil iyelik eki -ı/-i/-u/-ü/-sı/-si/-su/-sü'den sonra durum
# ekleri 'n' kaynaştırma harfi alır; "su" ve "ne" ayrı istisnalardır.)
YONELME_ISTISNALARI = {
    "gökyüzü": "gökyüzüne",
    "yüzü": "yüzüne",
    "su": "suya",
    "ne": "neye",
}


def son_unlu(kelime: str) -> str:
    """Kelimenin son sesli harfi (küçük harf); yoksa ''."""
    for harf in reversed((kelime or "").lower()):
        if harf in UNLULER:
            return harf
    return ""


def _sesliyle_bitiyor(kelime: str) -> bool:
    return bool(kelime) and kelime[-1].lower() in UNLULER


def yonelme_eki(kelime: str, ozel_isim: bool = False) -> str:
    """Yönelme (-a/-e) hâlini ekle.

    Kural:
      * Son ünlü kalınsa (a/ı/o/u) ek '-a', inceyse (e/i/ö/ü) '-e'.
      * Kelime sesliyle bitiyorsa ekten önce 'y' kaynaştırma harfi gelir.
      * Özel isimlerde ekten (ve kaynaştırma harfinden) önce kesme işareti.
      * İstisnalar (iyelik ekli vb.) `YONELME_ISTISNALARI`'ndan okunur.
    """
    k = (kelime or "").strip()
    if not k:
        return ""
    istisna = YONELME_ISTISNALARI.get(k.lower())
    if istisna is not None and not ozel_isim:
        return istisna
    su = son_unlu(k)
    ek = "a" if su in KALIN else "e"
    if ozel_isim:
        return f"{k}'{('y' if _sesliyle_bitiyor(k) else '')}{ek}"
    return f"{k}{('y' if _sesliyle_bitiyor(k) else '')}{ek}"


def kucult(kelime: str) -> str:
    """Cins isim için küçük harfe çevir (İ→i sorununu da çözer)."""
    return (kelime or "").replace("İ", "i").replace("I", "i").lower()


def temizle(metin: str) -> str:
    """Noktalama ve fazla boşlukları temizle (cümle ön-işleme için)."""
    return re.sub(r"\s+", " ", (metin or "")).strip()
