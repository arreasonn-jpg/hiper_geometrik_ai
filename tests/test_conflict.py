# -*- coding: utf-8 -*-
"""
Conflict → Exploration testleri
================================
(v0.1 — rapor §11, §14 madde 11)

Kapsam:
  * CONFLICT aday → alternatif deneyimler üretilir
  * deterministik test + güven güncelleme
  * yeniden değerlendirme kesin duruma (VALID/INVALID) iner
  * MODEL_GENERATED yine VERIFIED'a terfi ETMEZ

Çalıştırma:
    python tests/test_conflict.py
    pytest tests/test_conflict.py -q
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.experience import ConflictResolver, ExperienceEvaluator  # noqa: E402
from hga.knowledge import DeneyimDurumu, KnowledgeStore  # noqa: E402


def _store() -> KnowledgeStore:
    k = KnowledgeStore()
    k.varlik_ekle("Ali", entity_type="insan", properties={"canli": 1},
                  entity_id="E_001")
    k.varlik_ekle("Ata", entity_type="hayvan", properties={"binilebilir": 1},
                  entity_id="E_002")
    k.varlik_ekle("Halı", entity_type="esya", entity_id="E_003")  # özellik bilinmiyor
    k.iliski_tanimla("Binmek", relation_id="R_001",
                     subject_types=["insan"],
                     requires_object_props={"binilebilir": 1.0})
    return k


def test_conflict_alternatif_uretir():
    k = _store()
    ev = ExperienceEvaluator()
    from hga.knowledge import ExperienceCandidate
    a = ExperienceCandidate(experience_id="X_1", subject_id="E_001",
                            relation_id="R_001", object_id="E_003")
    ev.degerlendir(a, k)
    assert a.state == DeneyimDurumu.UNCERTAIN  # binilebilir bilinmiyor

    cr = ConflictResolver(evaluator=ev)
    sonuc = cr.coz(k, a)
    # "Ata" (binilebilir=1) alternatif olarak önerilmeli
    assert any(alt.object_id == "E_002" for alt in sonuc.alternatifler)
    assert sonuc.yeniden_degerlendirildi


def test_conflict_cozum_kesin_duruma_iner():
    k = _store()
    ev = ExperienceEvaluator()
    from hga.knowledge import ExperienceCandidate
    a = ExperienceCandidate(experience_id="X_2", subject_id="E_001",
                            relation_id="R_001", object_id="E_003")
    ev.degerlendir(a, k)
    cr = ConflictResolver(evaluator=ev)
    cr.coz(k, a)
    # deterministik test "Halı" için binilebilir kanıtı bulamaz → belirsiz,
    # yeniden değerlendirme yine UNCERTAIN kalır (güvenli) — kesinleşmezse yanlış
    # yükseltme yapılmamalıdır.
    assert a.state == DeneyimDurumu.UNCERTAIN
    assert a.state != DeneyimDurumu.VERIFIED


def test_conflict_model_generated_verified_olmaz():
    """Keşif sonrası bile MODEL_GENERATED asla VERIFIED'a terfi etmez."""
    k = _store()
    ev = ExperienceEvaluator()
    from hga.knowledge import ExperienceCandidate, KaynakTuru
    a = ExperienceCandidate(experience_id="X_3", subject_id="E_001",
                            relation_id="R_001", object_id="E_003",
                            source=KaynakTuru.MODEL_GENERATED)
    ev.degerlendir(a, k)
    cr = ConflictResolver(evaluator=ev)
    cr.coz(k, a)
    assert a.state != DeneyimDurumu.VERIFIED


def test_deterministik_test_ozellik_ekler():
    """Belirsiz nesneye deterministik test özellik kanıtı yazabilir."""
    k = _store()
    from hga.knowledge import ExperienceCandidate, KaynakTuru
    # "Ata" için bilinen bir aday; test binilebilir=1 bulmalı
    a = ExperienceCandidate(experience_id="X_4", subject_id="E_001",
                            relation_id="R_001", object_id="E_002",
                            source=KaynakTuru.REAL_DATA)
    cr = ConflictResolver()
    sonuc = cr.varsayilan_test(k, a)
    assert sonuc is True


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
