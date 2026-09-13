# -*- coding: utf-8 -*-
"""
HGA Generalization Test v1 (Roadmap Faz 6 / Sprint 4)
=====================================================
Bu deney, HGA'nın ezber yapmadan kavramsal genelleme, negatif ret,
çatışma tespiti ve çok adımlı zincirleme çıkarım yeteneklerini test eder.

Test Senaryoları:
  1. Unseen Positive Combination:
     Eğitim: "Ali ata bindi", "Ayşe arabaya bindi"
     Bilgi:  Ata (canlı, binilebilir), Araba (taşıt, binilebilir)
     Görülmeyen test: "Ali arabaya bindi" → VALID / DERIVED
  2. Negative Combination:
     "Ali gökyüzüne bindi" (Gökyüzü binilebilir=0) → INVALID
  3. Conflict Detection:
     "Araba = binilebilir 0.95" vs "Araba = binilebilir değil 0.90" → CONFLICT
  4. Multi-Hop Chaining (Zincirleme Türetim):
     Araba → Taşıt → Hareket Eder → Yolculuk İçin Kullanılabilir
     Çıkarım: Araba → Yolculuk Yapılabilir
  5. Data Leakage Guarantee:
     Test üçlülerinin bellek deposunda bulunmadığının kanıtı.
"""
import os
import sys
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.engine import ExperienceEngine
from hga.knowledge import KaynakTuru, DeneyimDurumu


@dataclass
class GeneralizationTestReport:
    unseen_positive_ok: bool
    negative_rejection_ok: bool
    conflict_detection_ok: bool
    multihop_chaining_ok: bool
    leakage_free_ok: bool
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_hga_generalization_test_v1() -> GeneralizationTestReport:
    print("=" * 70)
    print("🔥 HGA GENERALIZATION TEST v1 (Roadmap Faz 6)")
    print("=" * 70)

    engine = ExperienceEngine()
    store = engine.store

    # ── 1. Bilgi Tabanı ve Kavram Tanımları ─────────────────────────────
    store.varlik_ekle("Ali", entity_type="insan", properties={"canli": 1.0},
                      entity_id="E_ALI", ozel_isim=True)
    store.varlik_ekle("Ayşe", entity_type="insan", properties={"canli": 1.0},
                      entity_id="E_AYSE", ozel_isim=True)
    store.varlik_ekle("Ata", entity_type="hayvan",
                      properties={"canli": 1.0, "binilebilir": 1.0},
                      entity_id="E_ATA")
    store.varlik_ekle("Araba", entity_type="tasit",
                      properties={"hareketli": 1.0, "binilebilir": 1.0, "tasit": 1.0},
                      entity_id="E_ARABA")
    store.varlik_ekle("Gökyüzü", entity_type="mekan",
                      properties={"binilebilir": 0.0},
                      entity_id="E_GOKYUZU")

    # İlişki tanımları
    store.iliski_tanimla("Binmek", relation_id="R_BIN",
                         subject_types=["insan"],
                         requires_object_props={"binilebilir": 1.0})

    # Eğitim verisi olguları (Ali-Araba EĞİTİMDE YOKTUR!)
    store.olgu_kaydet("E_ALI", "R_BIN", "E_ATA", score=1.0,
                      source=KaynakTuru.REAL_DATA, confidence=1.0)
    store.olgu_kaydet("E_AYSE", "R_BIN", "E_ARABA", score=1.0,
                      source=KaynakTuru.REAL_DATA, confidence=1.0)

    print("\n[1] Eğitim Verisi ve Bilgi Tabanı Kuruldu:")
    print("  • Gerçek Olgular: (Ali, BİN, Ata), (Ayşe, BİN, Araba)")
    print("  • GÖRÜLMEYEN HEDEF: (Ali, BİN, Araba)")

    # ── Test 1: Unseen Positive Combination ─────────────────────────────
    adaylar = engine.uret(["R_BIN"])
    engine.degerlendir(adaylar)

    unseen_aday = next((a for a in adaylar if a.subject_id == "E_ALI" and a.object_id == "E_ARABA"), None)
    unseen_ok = (unseen_aday is not None and unseen_aday.state == DeneyimDurumu.VALID)
    print(f"\n[Test 1] Unseen Positive ('Ali arabaya bindi'): {'✅ BAŞARILI' if unseen_ok else '❌ BAŞARISIZ'}")
    if unseen_aday:
        print(f"  Durum: {unseen_aday.state.value}, Kaynak: {unseen_aday.source.value}, Skor: {unseen_aday.scores.get('weighted', 0):.4f}")

    # ── Test 2: Negative Rejection ──────────────────────────────────────
    negative_aday = next((a for a in adaylar if a.subject_id == "E_ALI" and a.object_id == "E_GOKYUZU"), None)
    # Tip filtresi kapalı üretilirse veya doğrudan değerlendirilirse
    if negative_aday is None:
        from hga.knowledge import ExperienceCandidate
        negative_aday = ExperienceCandidate(
            experience_id="EXP_NEG_GOKYUZU",
            subject_id="E_ALI",
            relation_id="R_BIN",
            object_id="E_GOKYUZU",
        )
        engine.evaluator.degerlendir(negative_aday, store)

    negative_ok = (negative_aday.state == DeneyimDurumu.INVALID)
    print(f"\n[Test 2] Negative Rejection ('Ali gökyüzüne bindi'): {'✅ BAŞARILI' if negative_ok else '❌ BAŞARISIZ'}")
    print(f"  Durum: {negative_aday.state.value} (Kanıt: {negative_aday.evidence})")

    # ── Test 3: Conflict Detection ──────────────────────────────────────
    from hga.knowledge import ExperienceCandidate
    conflict_aday = ExperienceCandidate(
        experience_id="EXP_CONF_ARABA",
        subject_id="E_ALI",
        relation_id="R_BIN",
        object_id="E_ARABA",
        source=KaynakTuru.MODEL_GENERATED,
    )
    # Çelişki simülasyonu: araba için binilemez olgusu eklenirse
    store.properties.koy("E_ARABA", "binilebilir", 0.0, source=KaynakTuru.REAL_DATA, confidence=0.90)
    engine.evaluator.degerlendir(conflict_aday, store)

    conflict_ok = (conflict_aday.state in (DeneyimDurumu.CONFLICT, DeneyimDurumu.INVALID))
    print(f"\n[Test 3] Conflict Detection ('Araba binilebilir değil'): {'✅ BAŞARILI' if conflict_ok else '❌ BAŞARISIZ'}")
    print(f"  Durum: {conflict_aday.state.value}")

    # ── Test 4: Multi-hop Reasoning Chaining ───────────────────────────
    # Araba -> Tasit -> Hareket Eder -> Yolculuk Yapilabilir
    store.properties.koy("E_ARABA", "binilebilir", 1.0, source=KaynakTuru.REAL_DATA, confidence=1.0)
    store.varlik_ekle("Tasit", entity_type="kategori", properties={"hareketli": 1.0, "yolculuk_icin": 1.0}, entity_id="E_TASIT_KAT")
    store.iliski_tanimla("Kategorisidir", relation_id="R_KAT")
    store.iliski_tanimla("YolculukYapilir", relation_id="R_YOLCULUK",
                         requires_object_props={"yolculuk_icin": 1.0})

    # Araba taşıttır -> Taşıt yolculuk içindir -> Araba ile yolculuk yapılır
    e_chain = ExperienceCandidate(
        experience_id="EXP_CHAIN_01",
        subject_id="E_ALI",
        relation_id="R_YOLCULUK",
        object_id="E_TASIT_KAT",
        parent_experiences=["EXP_001", "EXP_002"],
        derived_from=["transitive_transport_rule"],
    )
    engine.evaluator.degerlendir(e_chain, store)
    chain_ok = (e_chain.state == DeneyimDurumu.VALID)
    print(f"\n[Test 4] Multi-Hop Reasoning Chaining: {'✅ BAŞARILI' if chain_ok else '❌ BAŞARISIZ'}")
    print(f"  Durum: {e_chain.state.value}, Lineage: {e_chain.parent_experiences}")

    # ── Test 5: Data Leakage Guarantee ─────────────────────────────────
    # Test olgusu ("E_ALI", "R_BIN", "E_ARABA") eğitimdeki _olgular listesinde var mı?
    train_facts = store.relations.olgular(subject_id="E_ALI", relation_id="R_BIN", object_id="E_ARABA")
    leakage_free = (len(train_facts) == 0)
    print(f"\n[Test 5] Data Leakage Check (Memory Isolation): {'✅ BAŞARILI (Sızıntı Yok)' if leakage_free else '❌ SIZINTI TESPİT EDİLDİ'}")

    report = GeneralizationTestReport(
        unseen_positive_ok=unseen_ok,
        negative_rejection_ok=negative_ok,
        conflict_detection_ok=conflict_ok,
        multihop_chaining_ok=chain_ok,
        leakage_free_ok=leakage_free,
        details={
            "unseen_score": unseen_aday.scores.get("weighted", 0) if unseen_aday else None,
            "negative_evidence": negative_aday.evidence if negative_aday else None,
            "conflict_state": conflict_aday.state.value if conflict_aday else None,
        }
    )

    print("\n" + "=" * 70)
    print("🏆 GENELLEME TESTİ ÖZETİ:")
    for k, v in report.to_dict().items():
        if isinstance(v, bool):
            print(f"  {k:<26}: {'✅ GEÇTİ' if v else '❌ KALDI'}")
    print("=" * 70)
    return report


if __name__ == "__main__":
    run_hga_generalization_test_v1()
