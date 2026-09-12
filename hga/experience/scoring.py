# -*- coding: utf-8 -*-
"""
Scoring — Deneyim Puanlama (sinyal ayrıştırması)
=================================================
(v0.1 — rapor §8, §16, EK-A.8)

Evaluator birden fazla bağımsız sinyal kullanır (rapor §8): özellik
uyumluluğu, ilişki uyumluluğu, bağlam tutarlılığı, bellek desteği, yenilik,
kaynak güveni ve çelişki. Puan bunların ağırlıklı birleşimidir (EK-A.8):

    Score(e) = w1*property_compatibility
             + w2*relation_compatibility
             + w3*context_consistency
             + w4*memory_support
             + w5*novelty
             + w6*source_confidence
             - w7*contradiction

DİKKAT (rapor §16): toplama sonucu TEK BAŞINA doğruluk kanıtı DEĞİLDİR.
Kısıt ihlalleri (INVALID) ve çelişkiler (CONFLICT) puan düşse de düşmese de
ayrıca karar mekanizmasında ele alınır — bu modül yalnızca KANIT toplar.
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from ..knowledge.schemas import (Entity, Relation, ExperienceCandidate,
                                 KaynakTuru, KAYNAK_GUVENIRLIGI)


@dataclass
class ScoreBreakdown:
    property_compatibility: float
    relation_compatibility: float
    context_consistency: float
    memory_support: float
    novelty: float
    source_confidence: float
    contradiction: float
    weighted: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# Varsayılan ağırlıklar (experience_config.yaml ile örtüşür)
VARSAYILAN_AGIRLIKLAR = {
    "w1_property": 0.25,
    "w2_relation": 0.20,
    "w3_context": 0.15,
    "w4_memory": 0.15,
    "w5_novelty": 0.10,
    "w6_source": 0.15,
    "w7_contradiction": 0.30,   # ceza olarak çıkarılır
}


def _uyum(deger: Optional[float], hedef: float) -> float:
    """Özellik değerinin hedefe uyumu [0,1].

    Bilinmeyen özellik → 0.5 (nötr: "kanıt yok" ne destekler ne reddeder).
    Bilinen değer → 1 - |deger - hedef|.
    """
    if deger is None:
        return 0.5
    return max(0.0, min(1.0, 1.0 - abs(deger - hedef)))


class Scoring:
    """Bağımsız kanıt sinyallerini hesaplayıp ağırlıklı puana çevirir."""

    def __init__(self, agirliklar: Optional[Dict[str, float]] = None):
        self.agirliklar = dict(VARSAYILAN_AGIRLIKLAR)
        if agirliklar:
            self.agirliklar.update(agirliklar)

    # ── Tek sinyaller ────────────────────────────────────────────────────
    def property_uyumluluk(self, store, subject: Entity, relation: Relation,
                           object_: Entity) -> float:
        """İlişkinin özne/nesne özellik gereksinimlerine uyum (0..1)."""
        parcalar: List[float] = []
        for ad, hedef in relation.requires_object_props.items():
            pv = store.properties.al(object_.entity_id, ad)
            parcalar.append(_uyum(None if pv is None else pv.deger, hedef))
        for ad, hedef in relation.requires_subject_props.items():
            pv = store.properties.al(subject.entity_id, ad)
            parcalar.append(_uyum(None if pv is None else pv.deger, hedef))
        if not parcalar:
            return 1.0  # kısıt yok → özellik açısından çelişki yok
        return sum(parcalar) / len(parcalar)

    def iliski_uyumluluk(self, store, subject_id: str, relation_id: str,
                         object_id: str) -> float:
        """Aynı üçlüye dair kayıtlı kanıt uyumu; yoksa nötr 0.5."""
        agrega = store.relations.olgu_agrega(subject_id, relation_id, object_id)
        if agrega is None:
            return 0.5
        return agrega["score"]

    def baglam_tutarliligi(self, subject: Entity, relation: Relation,
                           object_: Entity) -> float:
        """Özne/nesne tipinin ilişkinin izin verdiği tiplerle uyumu (0/1)."""
        uyum = 1.0
        if relation.subject_types and subject.entity_type not in relation.subject_types:
            uyum = 0.0
        if relation.object_types and object_.entity_type not in relation.object_types:
            uyum = 0.0
        return uyum

    def bellek_destegi(self, store, subject_id: str, relation_id: str,
                       object_id: str) -> float:
        return store.bellek_destegi(subject_id, relation_id, object_id)

    def yenilik(self, store, subject_id: str, relation_id: str,
                object_id: str) -> float:
        """Bu üçlü ne kadar yeni? 1.0 = hiç görülmedi; tekrar gördükçe azalır."""
        n = len(store.relations.olgular(subject_id, relation_id, object_id))
        return 1.0 / (1.0 + n)

    def kaynak_guveni(self, aday: ExperienceCandidate) -> float:
        """Adayın kaynak güveni = source_confidence × kaynak güvenilirliği (§10)."""
        guvenilirlik = KAYNAK_GUVENIRLIGI.get(aday.source, 0.5)
        return round(aday.source_confidence * guvenilirlik, 4)

    # ── Toplam ───────────────────────────────────────────────────────────
    def skorla(self, store, aday: ExperienceCandidate, subject: Entity,
               relation: Relation, object_: Entity,
               celiski: bool = False) -> ScoreBreakdown:
        """Tüm sinyalleri hesapla ve ağırlıklı puanı döndür.

        `celiski` bayrağı Evaluator'ın çelişki tespitinden gelir; ceza olarak
        puanın içine `-w7` ile girer.
        """
        pc = self.property_uyumluluk(store, subject, relation, object_)
        rc = self.iliski_uyumluluk(store, subject.entity_id,
                                   relation.relation_id, object_.entity_id)
        cc = self.baglam_tutarliligi(subject, relation, object_)
        ms = self.bellek_destegi(store, subject.entity_id,
                                 relation.relation_id, object_.entity_id)
        nv = self.yenilik(store, subject.entity_id, relation.relation_id,
                          object_.entity_id)
        sc = self.kaynak_guveni(aday)
        ct = 1.0 if celiski else 0.0

        w = self.agirliklar
        weighted = (
            w["w1_property"] * pc
            + w["w2_relation"] * rc
            + w["w3_context"] * cc
            + w["w4_memory"] * ms
            + w["w5_novelty"] * nv
            + w["w6_source"] * sc
            - w["w7_contradiction"] * ct
        )
        return ScoreBreakdown(
            property_compatibility=round(pc, 4),
            relation_compatibility=round(rc, 4),
            context_consistency=round(cc, 4),
            memory_support=round(ms, 4),
            novelty=round(nv, 4),
            source_confidence=sc,
            contradiction=ct,
            weighted=round(max(0.0, min(1.0, weighted)), 4),
        )
