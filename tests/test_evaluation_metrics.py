# -*- coding: utf-8 -*-
"""Halüsinasyon / factual consistency metrik testleri."""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.evaluation import hallucination_metrics  # noqa: E402
from hga.experience import (  # noqa: E402
    AritmetikOrtam,
    ExperienceEvaluator,
    ExperienceGenerator,
    aritmetik_etki_alani,
)
from hga.knowledge import DeneyimDurumu, ExperienceCandidate, KaynakTuru  # noqa: E402


def test_hallucination_metrics_validator():
    k = aritmetik_etki_alani()
    gen = ExperienceGenerator(tip_filtresi=False)
    ev = ExperienceEvaluator()
    adaylar = gen.uret(k)
    for a in adaylar:
        ev.degerlendir(a, k)
    r = hallucination_metrics(adaylar, store=k, validator=AritmetikOrtam().aday_dogrula)
    assert r.toplam == 30
    assert r.checked == 30
    assert r.dogru == 6
    assert r.yanlis == 24
    assert abs(r.hallucination_rate - 24 / 30) < 1e-9
    assert abs(r.factual_consistency_score - 6 / 30) < 1e-9


def test_hallucination_metrics_proxy():
    adaylar = [
        ExperienceCandidate("X1", "E1", "R1", "E2", state=DeneyimDurumu.VERIFIED,
                            source=KaynakTuru.REAL_DATA),
        ExperienceCandidate("X2", "E1", "R1", "E3", state=DeneyimDurumu.VALID,
                            source=KaynakTuru.MODEL_GENERATED),
        ExperienceCandidate("X3", "E1", "R1", "E4", state=DeneyimDurumu.CONFLICT),
    ]
    r = hallucination_metrics(adaylar)
    assert r.verified == 1
    assert r.valid_model_generated == 1
    assert r.conflict == 1
    assert abs(r.unsupported_rate - 2 / 3) < 1e-9
    assert abs(r.factual_consistency_score - 1 / 3) < 1e-9


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
