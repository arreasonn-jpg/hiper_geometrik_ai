# -*- coding: utf-8 -*-
"""
Experience Graph & Exploration Engine Testleri
================================================
(Roadmap Faz 23, Faz 24, Faz 25)
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.experience.exploration import ExplorationEngine
from hga.experience.graph import ExperienceGraph
from hga.knowledge import DeneyimDurumu, ExperienceCandidate, KaynakTuru, KnowledgeStore


def test_experience_graph_ekleme_ve_lineage():
    store = KnowledgeStore()
    store.varlik_ekle("Ali", entity_type="insan", entity_id="E_001")
    store.varlik_ekle("Ata", entity_type="hayvan", entity_id="E_002")
    store.varlik_ekle("Araba", entity_type="tasit", entity_id="E_003")
    store.iliski_tanimla("Binmek", relation_id="R_001")

    graph = ExperienceGraph()

    e1 = ExperienceCandidate(
        experience_id="EXP_001",
        subject_id="E_001",
        relation_id="R_001",
        object_id="E_002",
        source=KaynakTuru.REAL_DATA,
        state=DeneyimDurumu.VERIFIED,
        verified_by="manual_fact",
    )
    graph.deneyim_kaydet(e1)

    e2 = ExperienceCandidate(
        experience_id="EXP_002",
        subject_id="E_001",
        relation_id="R_001",
        object_id="E_003",
        source=KaynakTuru.MODEL_GENERATED,
        state=DeneyimDurumu.VALID,
        parent_experiences=["EXP_001"],
        derived_from=["rule_rideable_transport"],
    )
    graph.deneyim_kaydet(e2)

    # Lineage testi (EXP_002 -> EXP_001)
    lin = graph.lineage("EXP_002")
    assert len(lin) >= 2
    dugum_idleri = [x["dugum"]["id"] for x in lin]
    assert "EXP_002" in dugum_idleri
    assert "EXP_001" in dugum_idleri

    # Açıklama testi
    aciklama_metni = graph.aciklama("EXP_002", store=store)
    assert "Ali" in aciklama_metni
    assert "Araba" in aciklama_metni
    assert "VALID" in aciklama_metni
    assert "EXP_001" in aciklama_metni

    ozet = graph.ozet()
    assert ozet["deneyim_sayisi"] == 2


def test_exploration_engine_uzay_haritasi_ve_aktif_ogrenme():
    store = KnowledgeStore()
    store.varlik_ekle("Ali", entity_type="insan", properties={"canli": 1.0}, entity_id="E_001")
    store.varlik_ekle("Ata", entity_type="hayvan", properties={"binilebilir": 1.0}, entity_id="E_002")
    store.varlik_ekle("Araba", entity_type="tasit", properties={"binilebilir": 1.0}, entity_id="E_003")
    store.iliski_tanimla("Binmek", relation_id="R_001", requires_object_props={"binilebilir": 1.0})

    # Bilinen bir olgu ekle (high confidence)
    store.olgu_kaydet("E_001", "R_001", "E_002", score=1.0, source=KaynakTuru.REAL_DATA, confidence=1.0)

    e_known = ExperienceCandidate(
        experience_id="EXP_KNOWN",
        subject_id="E_001",
        relation_id="R_001",
        object_id="E_002",
        state=DeneyimDurumu.VALID,
    )

    e_unknown = ExperienceCandidate(
        experience_id="EXP_UNKNOWN",
        subject_id="E_001",
        relation_id="R_001",
        object_id="E_003",
        state=DeneyimDurumu.VALID,
    )

    e_conflicted = ExperienceCandidate(
        experience_id="EXP_CONFLICT",
        subject_id="E_001",
        relation_id="R_001",
        object_id="E_002",
        state=DeneyimDurumu.CONFLICT,
        contradicts=["EXP_KNOWN"],
    )

    explorer = ExplorationEngine()
    harita = explorer.uzay_haritasi(store, [e_known, e_unknown, e_conflicted])

    assert len(harita.known) == 1
    assert len(harita.unknown) == 1
    assert len(harita.conflicted) == 1

    # Aktif öğrenme seçimi (Bilinmeyen ve bilgi kazancı yüksek olan üstte olmalı)
    secilenler = explorer.aktif_ogrenme_sec(store, [e_known, e_unknown], k=2)
    assert len(secilenler) == 2
    # Bilinmeyen deneyim daha yüksek bilgi kazancı ve yenilik taşır
    assert secilenler[0][0].experience_id == "EXP_UNKNOWN"
