# -*- coding: utf-8 -*-
"""
Consolidation — Değerlendirilen Deneyimleri Bilgiye Dönüştürme
===============================================================
(v0.1 — rapor §12, EK-C)

Değerlendirilen deneyimlerin duruma göre AKIBETİ (EK-C):

    CANDIDATE  → Evaluator'a gönder (buraya gelmemeli)
    VALID      → belleğe ADAY olarak ekle (kalıcı bilgi DEĞİL)
    CONFLICT   → araştırma kuyruğuna gönder
    INVALID    → reddet (çelişki günlüğüne not düş)
    VERIFIED   → kalıcı bilgiye yükselt (KnowledgeStore.dogrula)

Bu modül, "doğrulanmış bilgiyi konsolide et" adımını uygular (rapor §12) ve
deney metrikleri için bir özet rapor üretir (rapor §20: acceptance rate,
conflict rate, false acceptance, bilgi büyümesi).
"""
from dataclasses import dataclass, field
from typing import Dict, List

from ..knowledge.schemas import (ExperienceCandidate, DeneyimDurumu,
                                 KaynakTuru)


@dataclass
class ConsolidationReport:
    toplam: int = 0
    valid: int = 0
    conflict: int = 0
    invalid: int = 0
    verified: int = 0
    arastirma_kuyrugu: List[str] = field(default_factory=list)
    reddedilenler: List[str] = field(default_factory=list)
    dogrulananlar: List[str] = field(default_factory=list)
    bilgi_buyumesi: int = 0

    def to_dict(self) -> Dict:
        return {
            "toplam": self.toplam,
            "valid": self.valid,
            "conflict": self.conflict,
            "invalid": self.invalid,
            "verified": self.verified,
            "acceptance_rate": round(self.valid / self.toplam, 4) if self.toplam else 0.0,
            "conflict_rate": round(self.conflict / self.toplam, 4) if self.toplam else 0.0,
            "bilgi_buyumesi": self.bilgi_buyumesi,
            "arastirma_kuyrugu": self.arastirma_kuyrugu,
        }


class Consolidator:
    """Duruma göre deneyimleri konsolide eder; güvenliği burada da korur."""

    def __init__(self):
        self.arastirma_kuyrugu: List[ExperienceCandidate] = []

    def konsolide_et(self, store, adaylar: List[ExperienceCandidate],
                     ) -> ConsolidationReport:
        rapor = ConsolidationReport(toplam=len(adaylar))
        onceki_versiyon = store.versiyon

        for a in adaylar:
            if a.state == DeneyimDurumu.CANDIDATE:
                rapor.invalid += 1  # değerlendirilmeden gelirse reddet (güvenli varsayılan)
                rapor.reddedilenler.append(a.experience_id)
                continue
            if a.state == DeneyimDurumu.VALID:
                rapor.valid += 1
                # belleğe ADAY olarak yaz (kalıcı bilgi değil — kaynak korunur)
                store.olgu_kaydet(a.subject_id, a.relation_id, a.object_id,
                                  a.scores.get("weighted", 0.5),
                                  source=a.source, confidence=a.source_confidence)
            elif a.state == DeneyimDurumu.CONFLICT:
                rapor.conflict += 1
                rapor.arastirma_kuyrugu.append(a.experience_id)
                self.arastirma_kuyrugu.append(a)
            elif a.state == DeneyimDurumu.INVALID:
                rapor.invalid += 1
                rapor.reddedilenler.append(a.experience_id)
            elif a.state == DeneyimDurumu.VERIFIED:
                rapor.verified += 1
                rapor.dogrulananlar.append(a.experience_id)
                # Güvenlik: MODEL_GENERATED asla kalıcı bilgiye terfi etmez (§9/§21)
                if a.source == KaynakTuru.MODEL_GENERATED:
                    rapor.invalid += 1
                    rapor.verified -= 1
                    rapor.reddedilenler.append(a.experience_id)
                    store.celiski_logla(a, "model-generated-verified-ihlali")
                    continue
                store.dogrula(a, a.scores.get("weighted", 1.0))

        rapor.bilgi_buyumesi = store.versiyon - onceki_versiyon
        return rapor

    def arastir(self, store, resolver=None):
        """Araştırma kuyruğundaki CONFLICT adayları çözmeye çalış.

        `resolver` (ConflictResolver) verilmezse, hafif bir varsayılan
        içe aktarılır (döngüsel import'tan kaçınmak için geç import).
        """
        if resolver is None:
            from .conflict import ConflictResolver
            resolver = ConflictResolver()
        sonuclar = []
        for a in list(self.arastirma_kuyrugu):
            sonuclar.append(resolver.coz(store, a))
        self.arastirma_kuyrugu = [a for a in self.arastirma_kuyrugu
                                  if a.state == DeneyimDurumu.CONFLICT]
        return sonuclar
