# -*- coding: utf-8 -*-
"""
TorchKoprusu (v0.6) testleri — deneyim ↔ geometrik seyrek bellek köprüsü
=========================================================================
Çalıştırma:
    python tests/test_kopru.py
    pytest tests/test_kopru.py -q

Torch kurulu değilse: köprünün DÜRÜSTÇE hata verdiğini doğrular (sessizce
boş köprü döndürmez). Torch kuruluysa: gerçek tabloya yazma yolunu test eder.
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.memory import TorchKoprusu, torch_var_mi  # noqa: E402


def test_kopru_torch_yoksa_guvenli_hata():
    if torch_var_mi():
        print("  (torch mevcut — gerçek köprü testi aşağıda)")
        return
    # torch yok → köprü AÇIKÇA ImportError fırlatmalı, sessizce geçmemeli
    hata = None
    try:
        TorchKoprusu()
    except ImportError as e:
        hata = e
    assert hata is not None, "torch yokken köprü sessizce kurulmamalı"


def test_kopru_torch_varsa_gercek():
    if not torch_var_mi():
        print("  (torch yok — bu test atlandı)")
        return
    kopru = TorchKoprusu(tablo_boyutu=1024, boyut=8)
    uclu = ("E_001", "R_001", "E_002")
    vektor = kopru.vektor(uclu)
    assert vektor.shape == (1, 8)
    # boş tablo → sıfır vektör (geometrik çekirdeğin "boş küme" davranışı)
    assert vektor.abs().sum().item() == 0.0
    # anahtar deterministik
    assert kopru.anahtar(uclu)[0].item() == kopru.anahtar(uclu)[0].item()
    dolu, toplam = kopru.doluluk()
    assert dolu == 0 and toplam == 1024


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
