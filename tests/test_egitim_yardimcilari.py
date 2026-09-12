# -*- coding: utf-8 -*-
"""Eğitim yardımcıları: deterministik seed ve hafif değerlendirme importu."""
import os
import random
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from egitim import tohumla  # noqa: E402


def test_tohumla_python_random_deterministik():
    r1 = tohumla(123)
    a = [random.random() for _ in range(5)]
    r2 = tohumla(123)
    b = [random.random() for _ in range(5)]
    assert a == b
    assert r1["seed"] == r2["seed"] == 123
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_perplexity_import_edilebilir():
    from egitim.degerlendirme import perplexity
    assert callable(perplexity)


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
