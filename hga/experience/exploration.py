# -*- coding: utf-8 -*-
r"""
Exploration Engine & Active Learning Selector
==============================================
(Roadmap Faz 24 & Faz 25 — P2-036)

Sistem "Ne biliyorum?" sorusundan sonra "Ne bilmiyorum?" ve "Bana en çok
bilgi kazandıracak deneyim hangisi?" sorularını sorar.

Uzay Ayrımı (Faz 24):
  * ``KNOWN``      : Yüksek güvenilirlikte (≥ 0.85) doğrulanmış/kayıtlı bilgi
  * ``UNKNOWN``    : Henüz denenmemiş veya kütüphanede yer almayan kombinasyonlar
  * ``CONFLICTED`` : Çelişki kuyruğundaki veya zıt kanıt barındıran üçlüler
  * ``UNCERTAIN``  : Yetersiz kanıt veya düşük güven taşıyan alanlar

Aktif Öğrenme (Faz 25):
  $$E^* = \arg\max_{E} \left[ w_1 \cdot \text{InfoGain}(E) + w_2 \cdot \text{Novelty}(E) + w_3 \cdot \text{Uncertainty}(E) - w_4 \cdot \text{ConflictPenalty}(E) \right]$$
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from ..knowledge.schemas import ExperienceCandidate, DeneyimDurumu, KaynakTuru
from .scoring import Scoring


class EpistemicSpace(str, Enum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    CONFLICTED = "CONFLICTED"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class ExplorationMap:
    known: List[Tuple[str, str, str]] = field(default_factory=list)
    unknown: List[Tuple[str, str, str]] = field(default_factory=list)
    conflicted: List[Tuple[str, str, str]] = field(default_factory=list)
    uncertain: List[Tuple[str, str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "known_count": len(self.known),
            "unknown_count": len(self.unknown),
            "conflicted_count": len(self.conflicted),
            "uncertain_count": len(self.uncertain),
            "toplam_uzay": len(self.known) + len(self.unknown) + len(self.conflicted) + len(self.uncertain),
        }


class ExplorationEngine:
    """Bilinmeyen kavram uzayını haritalandırır ve en bilgilendirici deneyimleri seçer."""

    def __init__(self, scoring: Optional[Scoring] = None,
                 w_gain: float = 0.40,
                 w_novelty: float = 0.35,
                 w_uncertainty: float = 0.25,
                 w_conflict_penalty: float = 0.20):
        self.scoring = scoring or Scoring()
        self.w_gain = float(w_gain)
        self.w_novelty = float(w_novelty)
        self.w_uncertainty = float(w_uncertainty)
        self.w_conflict_penalty = float(w_conflict_penalty)

    def uzay_haritasi(self, store, adaylar: List[ExperienceCandidate]) -> ExplorationMap:
        """Deneyim adaylarını epistemik uzay kategorilerine (KNOWN/UNKNOWN/CONFLICTED/UNCERTAIN) ayırır."""
        harita = ExplorationMap()
        for aday in adaylar:
            uclu = aday.uclusu
            if aday.state == DeneyimDurumu.CONFLICT or len(aday.contradicts) > 0:
                harita.conflicted.append(uclu)
                continue

            agrega = store.relations.olgu_agrega(aday.subject_id, aday.relation_id, aday.object_id)
            if agrega is not None and agrega["confidence"] >= 0.85:
                harita.known.append(uclu)
            elif agrega is not None and agrega["confidence"] < 0.50:
                harita.uncertain.append(uclu)
            else:
                harita.unknown.append(uclu)

        return harita

    def bilgi_kazanci_skoru(self, store, aday: ExperienceCandidate) -> float:
        """Deneyimin bilgi kazancı optimizasyon skorunu (Faz 25) hesaplar."""
        try:
            subject = store.entities.getir(aday.subject_id)
            relation = store.relations.iliski_al(aday.relation_id)
            object_ = store.entities.getir(aday.object_id)
            score_breakdown = self.scoring.skorla(
                store, aday, subject, relation, object_,
                celiski=(aday.state == DeneyimDurumu.CONFLICT)
            )
            gain = score_breakdown.information_gain
            novelty = score_breakdown.novelty
        except Exception:
            gain = 0.5
            novelty = 0.5

        # Belirsizlik puanı (bilinen kanıtın azlığı)
        agrega = store.relations.olgu_agrega(aday.subject_id, aday.relation_id, aday.object_id)
        if agrega is None:
            uncertainty = 1.0
        else:
            uncertainty = max(0.0, 1.0 - agrega["confidence"])

        conflict_penalty = 1.0 if aday.state == DeneyimDurumu.CONFLICT else 0.0

        skor = (
            self.w_gain * gain +
            self.w_novelty * novelty +
            self.w_uncertainty * uncertainty -
            self.w_conflict_penalty * conflict_penalty
        )
        return max(0.0, min(1.0, round(skor, 4)))

    def aktif_ogrenme_sec(self, store, adaylar: List[ExperienceCandidate],
                          k: int = 10) -> List[Tuple[ExperienceCandidate, float]]:
        """Adaylar arasından en fazla bilgi kazandıracak en iyi K tanesini seçer (argmax InfoGain)."""
        puanli = []
        for a in adaylar:
            puan = self.bilgi_kazanci_skoru(store, a)
            puanli.append((a, puan))

        puanli.sort(key=lambda item: item[1], reverse=True)
        return puanli[:max(1, int(k))]


__all__ = ["EpistemicSpace", "ExplorationMap", "ExplorationEngine"]
