# -*- coding: utf-8 -*-
"""
HGA Experience Engine — Knowledge / Experience / Memory katmanı
================================================================
(v0.1 — "Hiper Geometrik AI: Experience Engine / Self-Expanding Knowledge
Architecture" yol haritasının ilk fiziksel karşılığı)

Bu paket, mevcut geometrik çekirdeğin (`mimari/`) ÜZERİNE eklenen, tamamen
saf Python ile çalışan (torch bağımlılığı OLMAYAN) modüler katmandır:

    hga/knowledge/   — EntityIndex, PropertyIndex, RelationIndex, KnowledgeStore
    hga/experience/  — Generator, Evaluator, Scoring, Conflict, Consolidation
    hga/memory/      — Seyrek deneyim slotları + Experience Replay (köprü)
    hga/evaluation/  — Halüsinasyon/factual consistency metrikleri
    hga/observability/ — Attention/geometri/bellek/deneyim akışı raporları
    hga/data/        — Veri kalite filtresi
    hga/config/      — Ağırlık/eşik/model yapılandırmaları

Temel döngü (rapor §2):
    Gerçek veri → Temsil → Yeni deneyim adayı → Değerlendirme
    → VALID / UNCERTAIN / CONFLICT / INVALID → Hafıza / EXPLORE / REJECT
    → bağımsız doğrulama → Bilgi güncellemesi

Kullanım:
    from hga.knowledge import KnowledgeStore
    from hga.experience import ExperienceGenerator, ExperienceEvaluator, Consolidator
"""
from hga.knowledge.schemas import (
    KAYNAK_GUVENIRLIGI,
    DeneyimDurumu,
    Entity,
    ExperienceCandidate,
    KaynakTuru,
    PropertyValue,
    Relation,
    RelationFact,
)

__all__ = [
    "KaynakTuru",
    "DeneyimDurumu",
    "KAYNAK_GUVENIRLIGI",
    "Entity",
    "PropertyValue",
    "Relation",
    "RelationFact",
    "ExperienceCandidate",
]
