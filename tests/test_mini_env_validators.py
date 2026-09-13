# -*- coding: utf-8 -*-
"""
Mantık + tutarlılık doğrulayıcıları.
Çalıştırma:
    python tests/test_mini_env_validators.py
    pytest tests/test_mini_env_validators.py -q
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.experience import MantikOrtam, TutarlilikOrtam  # noqa: E402
from hga.knowledge import ExperienceCandidate, KnowledgeStore  # noqa: E402


def test_mantik_modus_ponens():
    m = MantikOrtam()
    assert m.iddia_dogrula("A -> B, A |- B") is True
    assert m.iddia_dogrula("A → B, A ⊢ C") is False
    assert m.iddia_dogrula("A, B |- C") is None


def test_tutarlilik_aday_dogrula():
    k = KnowledgeStore()
    k.varlik_ekle("Ali", entity_type="insan", properties={"canli": 1}, entity_id="E_001")
    k.varlik_ekle("Ata", entity_type="hayvan", properties={"binilebilir": 1}, entity_id="E_002")
    k.varlik_ekle("Gökyüzü", entity_type="mekan", properties={"binilebilir": 0}, entity_id="E_003")
    k.iliski_tanimla("Binmek", relation_id="R_001",
                     subject_types=["insan"], requires_object_props={"binilebilir": 1.0})
    t = TutarlilikOrtam()
    ok = ExperienceCandidate(experience_id="X1", subject_id="E_001",
                             relation_id="R_001", object_id="E_002")
    bad = ExperienceCandidate(experience_id="X2", subject_id="E_001",
                              relation_id="R_001", object_id="E_003")
    assert t.aday_dogrula(k, ok) is True
    assert t.aday_dogrula(k, bad) is False


def test_ozellik_celiskisi():
    k = KnowledgeStore()
    e = k.varlik_ekle("Araba", entity_type="tasit", properties={"binilebilir": 1}, entity_id="E_001")
    t = TutarlilikOrtam()
    assert t.ozellik_celiskisi(k, e.entity_id, "binilebilir", 0.0) is True
    assert t.ozellik_celiskisi(k, e.entity_id, "binilebilir", 1.0) is False
    assert t.ozellik_celiskisi(k, e.entity_id, "ucabilir", 1.0) is None


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
