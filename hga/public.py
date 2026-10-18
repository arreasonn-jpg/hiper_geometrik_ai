# -*- coding: utf-8 -*-
"""HGA Public API — İngilizce alias'lar (uluslararası katkı için).

Türkçe isimler korunur; bu modül onların İngilizce eşlerini sunar.
Hiçbir davranış değişmez, yalnızca aynı nesnelere İngilizce ad verir.

Kullanım:
    from hga.public import KnowledgeBase, ExperienceEngine, Relation
    kb = KnowledgeBase()
    engine = ExperienceEngine(store=kb)
"""
from __future__ import annotations

# ── Experience engine ──
from hga.engine import ExperienceEngine
from hga.knowledge import (
    KAYNAK_GUVENIRLIGI as SOURCE_RELIABILITY,
)

# ── Knowledge schemas ──
from hga.knowledge import (
    DeneyimDurumu as ExperienceStatus,
)
from hga.knowledge import (
    Entity,
    ExperienceCandidate,
    PropertyValue,
    Relation,
    RelationFact,
)
from hga.knowledge import (
    KaynakTuru as SourceKind,
)
from hga.knowledge import (
    KnowledgeStore as KnowledgeBase,
)

# ── Memory ──
from hga.memory import BellekEntegrasyonu as MemoryIntegration

__all__ = [
    # Knowledge
    "KnowledgeBase",
    "SourceKind",
    "ExperienceStatus",
    "SOURCE_RELIABILITY",
    "Entity",
    "PropertyValue",
    "Relation",
    "RelationFact",
    "ExperienceCandidate",
    # Experience
    "ExperienceEngine",
    # Memory
    "MemoryIntegration",
]
