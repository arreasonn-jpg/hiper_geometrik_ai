# -*- coding: utf-8 -*-
"""
v0.1 Milestone — Uçtan Uca Deneyim Döngüsü (rapor §14, §15)
=============================================================
§14'teki 12 maddeyi ve §15'teki test senaryosunu uçtan uca doğrular:

  1..4  EntityIndex / PropertyIndex / RelationIndex / KnowledgeStore kurulur
  5..7  ExperienceCandidate + rule/score tabanlı evaluator + durum makinesi
  8     MODEL_GENERATED otomatik VERIFIED edilmez
  9     Ali + Ata + Bindi     → VALID
  10    Ali + Araba + Bindi   → VALID
  11    Ali + Gökyüzü + Bindi → INVALID/CONFLICT (asla otomatik VERIFIED)
  12    tüm testler doğrulanır + konsolidasyon + seyrek slot köprüsü

Çalıştırma:
    python tests/test_milestone_v01.py
    pytest tests/test_milestone_v01.py -q
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.experience import (  # noqa: E402
    Consolidator,
    DogrulamaHatti,
    ExperienceEvaluator,
    ExperienceGenerator,
)
from hga.knowledge import DeneyimDurumu, KaynakTuru, KnowledgeStore  # noqa: E402
from hga.memory import DeneyimSlotlari  # noqa: E402


def _kur() -> KnowledgeStore:
    """§14 madde 1-4: üç dizin + KnowledgeStore."""
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


def test_milestone_uygulanir():
    k = _kur()
    ev = ExperienceEvaluator()
    gen = ExperienceGenerator()  # kontrollü kombinasyon (§7)

    # §14 madde 5-7: aday üret + değerlendir + durum makinesi
    adaylar = gen.uret(k, relation_ids=["R_001"])
    assert len(adaylar) == 3          # Ali × {Ata, Araba, Gökyüzü}
    assert all(a.state == DeneyimDurumu.CANDIDATE for a in adaylar)
    assert all(a.source == KaynakTuru.MODEL_GENERATED for a in adaylar)

    sonuc = {a.object_id: a for a in adaylar}
    for a in adaylar:
        ev.degerlendir(a, k)

    # §14 madde 9-10: VALID
    assert sonuc["E_002"].state == DeneyimDurumu.VALID    # Ali + Ata
    assert sonuc["E_003"].state == DeneyimDurumu.VALID    # Ali + Araba

    # §14 madde 11: Gökyüzü INVALID/CONFLICT, asla otomatik VERIFIED
    gokyuzu = sonuc["E_004"].state
    assert gokyuzu in (DeneyimDurumu.INVALID, DeneyimDurumu.CONFLICT)
    assert sonuc["E_004"].state != DeneyimDurumu.VERIFIED

    # §14 madde 8: hiçbir MODEL_GENERATED aday VERIFIED değil
    assert all(a.state != DeneyimDurumu.VERIFIED for a in adaylar)


def test_milestone_konsolidasyon_ve_kopru():
    """Konsolidasyon: VALID belleğe aday olur, bilgi büyümesi ölçülür,
    deneyimler seyrek slotlara yazılır (§22 commit 6)."""
    k = _kur()
    ev = ExperienceEvaluator()
    gen = ExperienceGenerator()
    adaylar = gen.uret(k, relation_ids=["R_001"])
    for a in adaylar:
        ev.degerlendir(a, k)

    cons = Consolidator()
    rapor = cons.konsolide_et(k, adaylar)
    # 2 VALID + 1 INVALID
    assert rapor.valid == 2
    assert rapor.invalid == 1
    assert rapor.conflict == 0
    assert rapor.verified == 0
    # kabul oranı (acceptance rate, §20)
    assert abs(rapor.to_dict()["acceptance_rate"] - 2 / 3) < 1e-3

    # seyrek slot köprüsü: VALID deneyimler slotlara yazılır
    slotlar = DeneyimSlotlari(slot_sayisi=64)
    for a in adaylar:
        if a.state == DeneyimDurumu.VALID:
            slotlar.yaz(a.experience_id, a.uclusu)
    dolu, toplam = slotlar.doluluk_orani()
    assert dolu == 2 and toplam == 64


def test_milestone_harici_verified_bilgi_buyurur():
    """VERIFIED bir deneyim kalıcı bilgiye yazılır → bilgi büyümesi ≥ 1."""
    k = _kur()
    ev = ExperienceEvaluator()
    from hga.knowledge import ExperienceCandidate
    a = ExperienceCandidate(experience_id="X_VER", subject_id="E_001",
                            relation_id="R_001", object_id="E_002",
                            source=KaynakTuru.REAL_DATA, source_confidence=1.0)
    ev.degerlendir(a, k)
    assert a.state == DeneyimDurumu.VALID

    rapor = DogrulamaHatti(lambda store, aday: True).isle(k, [a])
    assert a.state == DeneyimDurumu.VERIFIED
    assert rapor.dogrulanan == 1
    assert rapor.bilgi_buyumesi == 1
    # kalıcı kanıt yazıldı
    assert k.relations.olgu_agrega("E_001", "R_001", "E_002")["count"] == 1


def test_milestone_model_generated_kalici_olamaz():
    """Güvenlik: kaynak sahte yollarla VERIFIED işaretlense bile konsolidasyon
    MODEL_GENERATED'ı kalıcı bilgiye yazmaz (§9/§21)."""
    k = _kur()
    from hga.knowledge import ExperienceCandidate
    a = ExperienceCandidate(experience_id="X_HILE", subject_id="E_001",
                            relation_id="R_001", object_id="E_002",
                            source=KaynakTuru.MODEL_GENERATED,
                            state=DeneyimDurumu.VERIFIED)  # elle zorla
    cons = Consolidator()
    rapor = cons.konsolide_et(k, [a])
    assert rapor.verified == 0
    assert rapor.invalid == 1
    # kalıcı bilgi yazılmadı
    assert k.relations.olgu_agrega("E_001", "R_001", "E_002") is None


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
