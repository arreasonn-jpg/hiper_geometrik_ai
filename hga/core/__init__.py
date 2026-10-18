"""HGA Core — Model, bellek, bilgi ve deneyim çekirdeği.

Bu facade, çekirdek modülleri tek isim alanı altında toplar:

    from hga.core import KnowledgeBase, ExperienceEngine, MemoryIntegration

Fiziksel taşıma YAPILMAZ; mevcut modüller (`hga.knowledge`,
`hga.memory`, `hga.experience`, `hga.engine`) buradan re-export edilir.
"""
from __future__ import annotations

# Experience engine
from hga.engine import ExperienceEngine  # noqa: F401

# Knowledge
from hga.knowledge import (  # noqa: F401
    Entity,
    ExperienceCandidate,
    KaynakTuru,
    KnowledgeStore,
    PropertyValue,
    Relation,
    RelationFact,
)

# Memory
from hga.memory import BellekEntegrasyonu  # noqa: F401

__all__ = [
    "KnowledgeStore",
    "KaynakTuru",
    "Entity",
    "PropertyValue",
    "Relation",
    "RelationFact",
    "ExperienceCandidate",
    "ExperienceEngine",
    "BellekEntegrasyonu",
]
