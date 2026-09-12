# -*- coding: utf-8 -*-
"""Mimari n/K/context taraması testleri."""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.evaluation import nk_taramasi, parametre_tahmini  # noqa: E402


def test_parametre_tahmini_monotonik():
    a = parametre_tahmini(n=64, k=2, baglam=16, vocab=1000, emb=32, seyrek_satir=0)
    b = parametre_tahmini(n=128, k=2, baglam=16, vocab=1000, emb=32, seyrek_satir=0)
    c = parametre_tahmini(n=64, k=4, baglam=16, vocab=1000, emb=32, seyrek_satir=0)
    assert b["tahmini_parametre"] > a["tahmini_parametre"]
    assert c["etkilesim_uzayi_ust_siniri"] > a["etkilesim_uzayi_ust_siniri"]
    assert c["zincir_ops_yaklasik"] > a["zincir_ops_yaklasik"]


def test_nk_taramasi_sayisi():
    rows = nk_taramasi(n_degerleri=[64, 128], k_degerleri=[2, 4],
                       baglam_degerleri=[16], vocab=1000, emb=32, seyrek_satir=0)
    assert len(rows) == 4
    assert {r["n"] for r in rows} == {64, 128}
    assert {r["K"] for r in rows} == {2, 4}


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
