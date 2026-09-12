# -*- coding: utf-8 -*-
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from egitim.scheduler import warmup_cosine_factor


def test_warmup_cosine_shape():
    degerler = [warmup_cosine_factor(i, total_steps=10, warmup_steps=3, min_factor=0.1)
                for i in range(10)]
    assert degerler[0] < degerler[1] <= degerler[2]
    assert abs(degerler[2] - 1.0) < 1e-9
    assert degerler[3] <= 1.0
    assert degerler[-1] >= 0.1
    assert degerler[-1] < degerler[3]


def test_sinir_degerleri():
    assert warmup_cosine_factor(-5, total_steps=0, warmup_steps=-1, min_factor=0.2) == 1.0
    assert 0.05 <= warmup_cosine_factor(999, total_steps=5, warmup_steps=0, min_factor=0.05) <= 1.0


if __name__ == "__main__":
    test_warmup_cosine_shape()
    test_sinir_degerleri()
    print("OK")
