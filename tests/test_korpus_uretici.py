# -*- coding: utf-8 -*-
"""
Korpus üretici testleri — çevrimdışı sentetik Türkçe korpus (torch'suz)
========================================================================
Çalıştırma:
    python tests/test_korpus_uretici.py
    pytest tests/test_korpus_uretici.py -q
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.experience import (sentetik_korpus_uret, korpus_borusu,  # noqa: E402
                            sozlugu_buyut, VARSAYILAN_SOZLUK,
                            yonelme_eki)
from hga.experience.korpus_uretici import (BINILECEK_NESNELER,  # noqa: E402
                                           BAKILACAK_NESNELER)
from hga.experience.sozluk_buyutme import _kok_bul  # noqa: E402
from hga.knowledge import KnowledgeStore  # noqa: E402


def test_determinizm():
    a = sentetik_korpus_uret(ozne_sayisi=10, nesne_sayisi=4, tohum=0)
    b = sentetik_korpus_uret(ozne_sayisi=10, nesne_sayisi=4, tohum=0)
    c = sentetik_korpus_uret(ozne_sayisi=10, nesne_sayisi=4, tohum=1)
    assert a == b                       # aynı tohum → aynı korpus
    assert a != c                       # farklı tohum → farklı korpus


def test_gercek_cumle_sayisi():
    # gürültüsüz: özne_sayısı × (binmek + bakmak nesneleri)
    c = sentetik_korpus_uret(ozne_sayisi=7, nesne_sayisi=5,
                             gurultu_orani=0.0, tohum=0)
    assert len(c) == 7 * (5 + 5)
    # her cümle noktayla biter ve bir fiil yüzey biçimi taşır
    fiiller = {"bindi", "biniyor", "binecek", "biner",
               "bakti", "bakiyor", "bakar"}
    for cumle in c:
        assert cumle.endswith(".")
        assert cumle.split()[-1][:-1] in fiiller


def test_nesneler_guvenli_kok():
    # üreticinin nesne listeleri yönelme hâline geçince TEMİZ kök verir
    # (yumuşama/ünlü düşmesi/sesli-belirsizliği içermez).
    for n in BINILECEK_NESNELER + BAKILACAK_NESNELER:
        cekim = yonelme_eki(n)
        kok = _kok_bul(cekim, "yonelme")
        assert kok.lower() == n, f"{n!r} → {cekim!r} → kök {kok!r}"


def test_aktarim_ve_atlama():
    # 6 özne × 3 nesne × 2 ilişki = 36 gerçek cümle; gerisi kasıtlı gürültü.
    c = sentetik_korpus_uret(ozne_sayisi=6, nesne_sayisi=3,
                             gurultu_orani=0.1, tohum=0)
    k = KnowledgeStore()
    rapor = korpus_borusu(k, "\n".join(c))
    assert rapor.aktarilan_uclu == 6 * (3 + 3)   # her gerçek cümle bir üçlü
    assert rapor.cumle_sayisi > rapor.aktarilan_uclu  # gürültü atlandı
    assert rapor.buyutme.yeni_iliski == 0          # ilişki asla uydurulmaz


def test_gurultu_dogrulanabilir():
    # gürültü cümleleri bilinen kalıplara uymaz → sozlugu_buyut atlar.
    c = sentetik_korpus_uret(ozne_sayisi=5, nesne_sayisi=2,
                             gurultu_orani=0.2, tohum=0)
    _, rapor, _ = sozlugu_buyut(VARSAYILAN_SOZLUK, c)
    gurultu = [x for x in c
               if "kuantum" in x or "kamyon denize" in x or x.endswith("kamyon bindi.")]
    assert rapor.atlanan_cumle >= len(gurultu) - 1
    # hiçbir gürültü cümlesi üçlü ÜRETMEMELİ (kuantum hariç hepsi kalıpla çakışmaz)
    assert rapor.eslesen_cumle == rapor.taranan_cumle - rapor.atlanan_cumle


def test_iliski_icat_edilmez_buyuk_olcekte():
    # ölçekte de yeni ilişki asla icat edilmez.
    c = sentetik_korpus_uret(ozne_sayisi=20, nesne_sayisi=5,
                             gurultu_orani=0.05, tohum=3)
    _, rapor, _ = sozlugu_buyut(VARSAYILAN_SOZLUK, c)
    assert rapor.yeni_iliski == 0


if __name__ == "__main__":
    testler = [(ad, fn) for ad, fn in sorted(globals().items())
               if ad.startswith("test_") and callable(fn)]
    basarisiz = 0
    for ad, fn in testler:
        try:
            fn()
            print(f"  ✅ {ad}")
        except Exception as e:  # noqa: BLE001
            basarisiz += 1
            print(f"  ❌ {ad}: {type(e).__name__}: {e}")
    print("\nTÜM TESTLER GEÇTİ" if basarisiz == 0 else f"{basarisiz} test başarısız")
    sys.exit(1 if basarisiz else 0)
