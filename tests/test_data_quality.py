# -*- coding: utf-8 -*-
"""Veri kalite filtresi testleri."""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.data import kalite_skoru, temizle_cumleler  # noqa: E402
from hga.knowledge import KnowledgeStore  # noqa: E402
from hga.experience.korpus_boru import korpus_borusu  # noqa: E402


def test_temizle_cumleler_rapor():
    cumleler = [
        "Ali ataya bindi.",
        "Ali ataya bindi.",  # duplicate
        "",                 # boş
        "kısa",             # kısa
        "bozuk � metin",    # encoding
        "12345 !!! ???",    # düşük harf oranı
        "aaaaaaaaaaaaaaaaaaaa Türkçe",  # aşırı tekrar
        "Ayşe arabaya bindi ve güvenli şekilde indi.",
    ]
    temiz, rapor = temizle_cumleler(cumleler)
    assert temiz == ["Ali ataya bindi.", "Ayşe arabaya bindi ve güvenli şekilde indi."]
    assert rapor.toplam == len(cumleler)
    assert rapor.kabul == 2
    assert rapor.duplicate == 1
    assert rapor.bos == 1
    assert rapor.kisa == 1
    assert rapor.bozuk_encoding == 1
    assert rapor.dusuk_harf_orani == 1
    assert rapor.asiri_tekrar == 1
    assert abs(rapor.to_dict()["kabul_orani"] - 0.25) < 1e-9


def test_kalite_skoru_aralik():
    assert 0.0 <= kalite_skoru("Ali ataya bindi.") <= 1.0
    assert kalite_skoru("Ali ataya bindi.") > kalite_skoru("???? 12345 �")


def test_korpus_borusu_kalite_filtresi():
    k = KnowledgeStore()
    rapor = korpus_borusu(k, "Ali ataya bindi. Ali ataya bindi. bozuk � metin.",
                          kalite_filtresi=True)
    assert rapor.kalite is not None
    assert rapor.kalite.duplicate == 1
    assert rapor.kalite.bozuk_encoding == 1
    assert rapor.cumle_sayisi == 1


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
