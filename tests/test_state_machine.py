# -*- coding: utf-8 -*-
"""Experience durum makinesi testleri."""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.experience import DeneyimDurumMakinesi  # noqa: E402
from hga.knowledge import DeneyimDurumu, ExperienceCandidate, KaynakTuru  # noqa: E402


def test_candidate_yollari():
    sm = DeneyimDurumMakinesi()
    a = ExperienceCandidate("X", "E1", "R1", "E2")
    assert sm.uygula(a, DeneyimDurumu.CONFLICT).ok
    assert a.state == DeneyimDurumu.CONFLICT
    assert sm.uygula(a, DeneyimDurumu.EXPLORE).ok
    assert a.state == DeneyimDurumu.EXPLORE


def test_invalid_reject_terminal():
    sm = DeneyimDurumMakinesi()
    a = ExperienceCandidate("X", "E1", "R1", "E2", state=DeneyimDurumu.INVALID)
    assert sm.uygula(a, DeneyimDurumu.REJECT).ok
    assert a.state == DeneyimDurumu.REJECT
    assert sm.uygula(a, DeneyimDurumu.VALID).ok is False


def test_model_generated_verified_engeli():
    sm = DeneyimDurumMakinesi()
    a = ExperienceCandidate("X", "E1", "R1", "E2", state=DeneyimDurumu.VALID,
                            source=KaynakTuru.MODEL_GENERATED)
    tr = sm.uygula(a, DeneyimDurumu.VERIFIED)
    assert tr.ok is False
    assert "MODEL_GENERATED" in tr.neden
    assert a.state == DeneyimDurumu.VALID


def test_harici_kaynak_bagimsiz_dogrulayici_kimligi_gerektirir():
    sm = DeneyimDurumMakinesi()
    a = ExperienceCandidate("X", "E1", "R1", "E2", state=DeneyimDurumu.VALID,
                            source=KaynakTuru.REAL_DATA)
    assert not sm.uygula(a, DeneyimDurumu.VERIFIED).ok
    a.source = KaynakTuru.EXTERNAL_VERIFIED
    a.verified_by = "test-verifier"
    assert sm.uygula(a, DeneyimDurumu.VERIFIED).ok
    assert a.state == DeneyimDurumu.VERIFIED


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
