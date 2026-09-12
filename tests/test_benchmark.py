# -*- coding: utf-8 -*-
"""
Benchmark testleri — döngüyü ground-truth'a karşı ölçme
=========================================================
Çalıştırma:
    python tests/test_benchmark.py
    pytest tests/test_benchmark.py -q
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.experience import (aritmetik_etki_alani, kos, ozetle,  # noqa: E402
                            karsilastirma, AritmetikOrtam)


def test_aritmetik_etki_alani():
    k = aritmetik_etki_alani()
    assert len(k.entities) == 6
    assert len(k.relations) == 1


def test_karsilastirma_yanlis_kabul():
    """6 varlık × 5 nesne = 30 üçlü; ground-truth'ta yalnız 6 doğru.

    Kural tabanlı değerlendirme (MODEL_GENERATED) hepsini VALID yapar →
    false acceptance = 24, false rejection = 0. Bu, rapor §18/§21'in
    "kural tabanlı değerlendirme tek başına yetersiz" tezini ölçer.
    """
    sonuc = karsilastirma(adimlar=1)
    o = sonuc["ozet"]
    assert o.toplam_uretilen == 30
    assert o.toplam_valid == 30
    assert o.yanlis_kabul == 24
    assert o.yanlis_ret == 0
    # MODEL_GENERATED asla VERIFIED olmaz
    assert o.toplam_verified == 0


def test_kos_coklu_adim_ozet():
    k = aritmetik_etki_alani()
    raporlar = kos(k, adimlar=3, dogrulayici=AritmetikOrtam().aday_dogrula)
    o = ozetle(raporlar)
    assert o.adim_sayisi == 3
    # yalnız ilk adımda yeni üçlü var (sonraki adımlar aynı kombinasyonlar)
    assert o.toplam_novel == 30
    assert o.toplam_uretilen == 90
    # replay verimliliği [0,1]
    assert 0.0 <= o.son_replay_verimliligi <= 1.0


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
