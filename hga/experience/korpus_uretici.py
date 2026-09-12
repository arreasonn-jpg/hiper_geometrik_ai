# -*- coding: utf-8 -*-
"""
Korpus Üretici — çevrimdışı, belirleyici Türkçe cümle korpusu
===============================================================
(v1.0+ — rapor §8 "dış korpus ölçeği"nin çevrimdışı provası)

`egitim/veri_toplayici.py` canlı korpusu (Wikipedia/HF) AĞ üzerinden çeker; bu
modül ise ağ GEREKTİRMEYEN bir araçtır: Türkçe morfoloji yardımcılarını
(`turkce.py`) kullanarak BELİRLEYİCİ (tohumlu) ve dilbilgisel olarak doğru SOV
cümleleri üretir. Amacı, korpus borusunu (`korpus_boru.py`) ve sözlük büyütmeyi
(`sozluk_buyutme.py`) gerçekçi bir ölçekte, tekrarlanabilir biçimde sınamaktır.

    * `sentetik_korpus_uret(ozne_sayisi, nesne_sayisi, gurultu_orani, tohum)`
      → cümle listesi (SOV + kasıtlı gürültü)
    * Üretilen her cümle ya bilinen bir fiil desenine birebir uyar (→ REAL_DATA)
      ya da KASITLI olarak uymaz (→ dürüst atlama sınaması).

DÜRÜSTLÜK NOTU: üretilen metin GERÇEK dünya korpusu DEĞİLDİR — bilinen fiil
desenleriyle sınırlı, şablon tabanlı SENTETİK bir stres testidir. "Milyon kelime"
hedefinin canlı gerçeklenmesi `veri_toplayici.py`'nin işidir (ağ gerektirir).

BİLİNEN SINIR (belgeli): `sozluk_buyutme.py`'nin "fiilin hemen önü = nesne"
yüzey buluşsalı, SESLİYLE BİTEN isimlerde belirsizdir — çekimsiz "manzara"
ile yönelme hâlindeki "manzar+a" yüzeyden ayırt edilemez. Bu yüzden nesne
listeleri yalnızca bu belirsizliği tetiklemeyen köklerden seçilir ve gürültü
cümleleri de KASITLI olarak net (ünsüzle biten) biçimde kurulur.
"""
import random
from typing import Dict, List

from .turkce import yonelme_eki

# ── Küratörlü varlık listeleri ────────────────────────────────────────────
# Yaygın Türkçe özel adlar (büyük harfle başlar → "insan" tipiyle özne olur).
TURKCE_ADLAR = [
    "Mehmet", "Ahmet", "Ayşe", "Fatma", "Zeynep", "Ali", "Veli", "Hasan",
    "Hüseyin", "Mustafa", "Emine", "Hatice", "Kemal", "Elif",
    "Murat", "Selin", "Burak", "Ceren", "Emre", "Gamze", "İrem", "Kerem",
    "Leyla", "Mert", "Nazlı", "Onur", "Ömer", "Pelin", "Rıza", "Seda",
    "Tolga", "Umut", "Volkan", "Yasemin", "Zeki", "Arda", "Belgin",
    "Canan", "Derya", "Ece", "Furkan", "Gül", "İpek", "Jale", "Kaan",
    "Lale", "Meltem", "Nuri", "Okan", "Pınar", "Şule", "Çetin", "Burcu",
]
# NOT: "Deniz" bilerek listede yok — "deniz" (nesne) ile aynı token'a düşer
# (kişi adı ↔ deniz sesteşliği) ve varlık indeksi token-bazlı olduğu için
# ikisini birbirine karıştırırdı. Sesteş çakışması belgeli bir sınırdır.

# Binilebilir nesneler (yönelme; yumuşama/ünlü düşmesi içermeyen güvenli kökler —
# böylece sözlük büyütmenin "en-iyi-çaba" kök soyması temiz sonuç verir).
BINILECEK_NESNELER = [
    "kamyon", "vapur", "otobüs", "tren", "taksi", "metro",
    "minibüs", "gemi", "sandal", "otomobil", "troleybüs", "helikopter",
]

# Bakılabilir nesneler (yönelme; aynı güvenli-kök ilkesiyle).
BAKILACAK_NESNELER = [
    "deniz", "ay", "manzara", "tablo", "ayna", "fotoğraf", "harita",
    "panorama", "tabela", "vitrin", "sahne", "duvar", "pencere",
    "merdiven", "cadde",
]

# ── İlişki kalıpları (bilinen fiil desenleri; ilişki İCAT EDİLMEZ) ────────
# yuklemler: `VARSAYILAN_SOZLUK`'un o ilişki için tanıdığı fiil yüzey biçimleri.
_KALIPLAR: List[Dict] = [
    {"iliski": "Binmek", "durum": "yonelme", "nesneler": BINILECEK_NESNELER,
     "yuklemler": ["bindi", "biniyor", "binecek", "biner"]},
    {"iliski": "Bakmak", "durum": "yonelme", "nesneler": BAKILACAK_NESNELER,
     "yuklemler": ["bakti", "bakiyor", "bakar"]},
]

# Kasıtlı gürültü: eşleşmemesi GEREKEN kalıplar (dürüst atlama sınaması).
# DİKKAT: "eksik ek" gürültüsünde nesne ÜNSÜZLE BİTEN bir kök olmalıdır; aksi
# hâlde sesliyle biten isimlerin çekimsiz biçimi yönelme hâliyle karışır.
_EKSIK_EKLI_NESNE = "{ad} kamyon bindi."     # nesne çekimsiz (ünsüzle biter) → atlanır
_KUCUK_HARFLI_OZNE = "kamyon denize bakti."   # özne küçük harfli → tipi bilinmez
_SOZLUK_DISI = "kuantum bilgisayar nedir?"    # hiçbir kalıba uymaz


def _durumla(nesne: str, durum: str) -> str:
    """Nesneyi ilişkinin beklediği durum ekine göre çekimle (şimdilik yönelme)."""
    if durum == "yonelme":
        return yonelme_eki(nesne)
    raise ValueError(f"desteklenmeyen durum: {durum!r}")


def sentetik_korpus_uret(ozne_sayisi: int = 50, nesne_sayisi: int = 10,
                         gurultu_orani: float = 0.05,
                         tohum: int = 0) -> List[str]:
    """Belirleyici SOV cümle korpusu üretir (bilinen fiiller + kasıtlı gürültü).

    * `ozne_sayisi` : kaç farklı özel ad kullanılacağı (≤ len(TURKCE_ADLAR)).
    * `nesne_sayisi`: kalıp başına kaç farklı nesne kullanılacağı.
    * `gurultu_orani`: gerçek cümlelere oranla eklenen eşleşmeyen cümle oranı.
    * `tohum`        : `random.Random` tohumu — aynı tohum → aynı korpus.

    Dönen cümleler karıştırılmıştır; tekrarlanabilirdir.
    """
    rng = random.Random(tohum)
    ozneler = rng.sample(TURKCE_ADLAR, min(int(ozne_sayisi), len(TURKCE_ADLAR)))

    cumleler: List[str] = []
    for kalip in _KALIPLAR:
        secilen = rng.sample(kalip["nesneler"],
                             min(int(nesne_sayisi), len(kalip["nesneler"])))
        for ad in ozneler:
            for nesne in secilen:
                # zaman ekseni: kalıbın yüklemlerini dönüşümlü kullan
                yuklem = kalip["yuklemler"][(len(cumleler)) % len(kalip["yuklemler"])]
                cumleler.append(f"{ad} {_durumla(nesne, kalip['durum'])} "
                                f"{yuklem}.")

    # ── Kasıtlı gürültü: eşleşmemesi gereken cümleler ────────────────────
    n_gurultu = int(len(cumleler) * gurultu_orani)
    if n_gurultu > 0:
        gurultu: List[str] = []
        for _ in range(n_gurultu):
            ad = rng.choice(ozneler)
            gurultu.append(_EKSIK_EKLI_NESNE.format(ad=ad))
            gurultu.append(_KUCUK_HARFLI_OZNE)
        gurultu.append(_SOZLUK_DISI)
        cumleler.extend(gurultu)

    rng.shuffle(cumleler)
    return cumleler
