# -*- coding: utf-8 -*-
"""
ExperienceEvaluator + state machine testleri
=============================================
(v0.1 — rapor §8, §9, §15, EK-C)

Kapsam (rapor §15 senaryosu birebir):
  * Ali + Binmek + Ata     → VALID
  * Ali + Binmek + Araba   → VALID
  * Ali + Binmek + Gökyüzü → INVALID (kural ihlali), asla otomatik VERIFIED
  * MODEL_GENERATED deneyimler ASLA otomatik VERIFIED edilmez (§9/§21)
  * yetersiz kanıt → CONFLICT (araştırma kuyruğuna)

Çalıştırma:
    python tests/test_experience_eval.py
    pytest tests/test_experience_eval.py -q
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.knowledge import KnowledgeStore, KaynakTuru, DeneyimDurumu  # noqa: E402
from hga.experience import ExperienceEvaluator, Scoring  # noqa: E402


def demo_store() -> KnowledgeStore:
    """Rapor §15'teki bilgi tabanı."""
    k = KnowledgeStore()
    k.varlik_ekle("Ali", entity_type="insan", properties={"canli": 1},
                  entity_id="E_001")
    k.varlik_ekle("Ata", entity_type="hayvan", properties={"binilebilir": 1},
                  entity_id="E_002")
    k.varlik_ekle("Araba", entity_type="tasit", properties={"binilebilir": 1},
                  entity_id="E_003")
    k.varlik_ekle("Gökyüzü", entity_type="mekan", properties={"binilebilir": 0},
                  entity_id="E_004")
    k.iliski_tanimla("Binmek", relation_id="R_001",
                     subject_types=["insan"],
                     requires_object_props={"binilebilir": 1.0})
    return k


def aday(subject, relation, object_, **kw) -> "ExperienceCandidate":
    from hga.knowledge import ExperienceCandidate
    return ExperienceCandidate(
        experience_id=kw.pop("experience_id", "X_TEST"),
        subject_id=subject, relation_id=relation, object_id=object_, **kw)


def test_ali_ata_binmek_valid():
    k = demo_store()
    ev = ExperienceEvaluator()
    a = aday("E_001", "R_001", "E_002")
    ev.degerlendir(a, k)
    assert a.state == DeneyimDurumu.VALID
    assert a.scores["property_compatibility"] == 1.0


def test_ali_araba_binmek_valid():
    k = demo_store()
    ev = ExperienceEvaluator()
    a = aday("E_001", "R_001", "E_003")
    ev.degerlendir(a, k)
    assert a.state == DeneyimDurumu.VALID


def test_ali_gokyuzu_binmek_invalid():
    """Gökyüzü binilebilir=0 → deterministik kural ihlali → INVALID."""
    k = demo_store()
    ev = ExperienceEvaluator()
    a = aday("E_001", "R_001", "E_004")
    ev.degerlendir(a, k)
    assert a.state == DeneyimDurumu.INVALID
    # hiçbir şekilde otomatik VERIFIED değil (§15)
    assert a.state != DeneyimDurumu.VERIFIED


def test_model_generated_asla_verified_olmaz():
    """Rapor §9/§21'in en kritik kuralı: model üretimi otomatik VERIFIED edilemez."""
    k = demo_store()
    ev = ExperienceEvaluator()
    # kaynak MODEL_GENERATED, puan yüksek olsa bile
    a = aday("E_001", "R_001", "E_002", source=KaynakTuru.MODEL_GENERATED,
             source_confidence=0.9)
    ev.degerlendir(a, k)
    assert a.state == DeneyimDurumu.VALID        # en fazla VALID
    assert a.state != DeneyimDurumu.VERIFIED     # asla VERIFIED


def test_harici_kaynak_yuksek_puan_verified():
    """REAL_DATA + yüksek puan → VERIFIED (kalıcı bilgiye yükseltilebilir)."""
    k = demo_store()
    ev = ExperienceEvaluator()
    a = aday("E_001", "R_001", "E_002", source=KaynakTuru.REAL_DATA,
             source_confidence=1.0)
    ev.degerlendir(a, k)
    assert a.state == DeneyimDurumu.VERIFIED


def test_yetersiz_kanit_conflict():
    """Binilebilir özelliği bilinmeyen nesne → CONFLICT (araştırma kuyruğu)."""
    k = demo_store()
    k.varlik_ekle("Halı", entity_type="esya", entity_id="E_005")  # özellik yok
    ev = ExperienceEvaluator()
    a = aday("E_001", "R_001", "E_005")
    ev.degerlendir(a, k)
    assert a.state == DeneyimDurumu.CONFLICT


def test_bilinmeyen_varlik_invalid():
    k = demo_store()
    ev = ExperienceEvaluator()
    a = aday("E_999", "R_001", "E_002")
    ev.degerlendir(a, k)
    assert a.state == DeneyimDurumu.INVALID


def test_skor_araligi_ve_bilesenleri():
    k = demo_store()
    sc = Scoring()
    a = aday("E_001", "R_001", "E_002")
    subject = k.entities.getir("E_001")
    relation = k.relations.iliski_al("R_001")
    object_ = k.entities.getir("E_002")
    br = sc.skorla(k, a, subject, relation, object_)
    for anahtar in ("property_compatibility", "relation_compatibility",
                    "context_consistency", "memory_support", "novelty",
                    "source_confidence", "contradiction", "weighted"):
        assert 0.0 <= br.to_dict()[anahtar] <= 1.0
    # kontrollü adayda çelişki cezası sıfır
    assert br.contradiction == 0.0


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
