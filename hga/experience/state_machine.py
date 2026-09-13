# -*- coding: utf-8 -*-
"""Deneyim durum makinesi (Roadmap P0-012).

Durum Akışı:
    CANDIDATE → EVALUATING → (VALID | UNCERTAIN | CONFLICT | INVALID)
    VALID → VERIFYING → (VERIFIED | UNCERTAIN | INVALID)
    UNCERTAIN → (VERIFYING | EXPLORE)
    CONFLICT → EXPLORE → (VALID | UNCERTAIN | INVALID | CONFLICT)
    INVALID → REJECT

Altın kural: ``MODEL_GENERATED`` kaynaklı aday deterministik/harici kanıt
olmadan asla ``VERIFIED`` olamaz.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from ..knowledge.schemas import DeneyimDurumu, ExperienceCandidate, KaynakTuru

GECISLER: Dict[DeneyimDurumu, Tuple[DeneyimDurumu, ...]] = {
    DeneyimDurumu.CANDIDATE: (
        DeneyimDurumu.EVALUATING,
        DeneyimDurumu.VALID,
        DeneyimDurumu.UNCERTAIN,
        DeneyimDurumu.CONFLICT,
        DeneyimDurumu.INVALID,
    ),
    DeneyimDurumu.EVALUATING: (
        DeneyimDurumu.VALID,
        DeneyimDurumu.UNCERTAIN,
        DeneyimDurumu.CONFLICT,
        DeneyimDurumu.INVALID,
    ),
    DeneyimDurumu.VALID: (
        DeneyimDurumu.VERIFYING,
        DeneyimDurumu.VERIFIED,
        DeneyimDurumu.CONFLICT,
    ),
    DeneyimDurumu.VERIFYING: (
        DeneyimDurumu.VERIFIED,
        DeneyimDurumu.UNCERTAIN,
        DeneyimDurumu.INVALID,
        DeneyimDurumu.CONFLICT,
    ),
    DeneyimDurumu.UNCERTAIN: (
        DeneyimDurumu.VERIFYING,
        DeneyimDurumu.EXPLORE,
        DeneyimDurumu.VALID,
        DeneyimDurumu.CONFLICT,
        DeneyimDurumu.INVALID,
    ),
    DeneyimDurumu.CONFLICT: (
        DeneyimDurumu.EXPLORE,
        DeneyimDurumu.VALID,
        DeneyimDurumu.INVALID,
    ),
    DeneyimDurumu.EXPLORE: (
        DeneyimDurumu.VALID,
        DeneyimDurumu.UNCERTAIN,
        DeneyimDurumu.INVALID,
        DeneyimDurumu.CONFLICT,
    ),
    DeneyimDurumu.INVALID: (
        DeneyimDurumu.REJECT,
    ),
    DeneyimDurumu.REJECT: (),
    DeneyimDurumu.VERIFIED: (),
}


@dataclass
class StateTransition:
    onceki: DeneyimDurumu
    yeni: DeneyimDurumu
    ok: bool
    neden: str = ""

    def to_dict(self):
        return {
            "onceki": self.onceki.value,
            "yeni": self.yeni.value,
            "ok": self.ok,
            "neden": self.neden,
        }


class DeneyimDurumMakinesi:
    """Durum geçişlerini doğrular ve adaya uygular."""

    def gecis_mumkun_mu(self, aday: ExperienceCandidate,
                        yeni: DeneyimDurumu) -> StateTransition:
        onceki = aday.state
        if yeni not in GECISLER.get(onceki, ()):
            return StateTransition(onceki, yeni, False,
                                   f"{onceki.value} → {yeni.value} geçişi tanımsız")
        if (aday.source == KaynakTuru.MODEL_GENERATED and
                yeni == DeneyimDurumu.VERIFIED):
            return StateTransition(onceki, yeni, False,
                                   "MODEL_GENERATED otomatik VERIFIED olamaz")
        if yeni == DeneyimDurumu.VERIFIED and not aday.verified_by:
            return StateTransition(
                onceki, yeni, False,
                "VERIFIED geçişi bağımsız doğrulayıcı kimliği (verified_by) gerektirir",
            )
        return StateTransition(onceki, yeni, True, "ok")

    def uygula(self, aday: ExperienceCandidate, yeni: DeneyimDurumu,
               neden: str = "") -> StateTransition:
        tr = self.gecis_mumkun_mu(aday, yeni)
        if tr.ok:
            aday.state = yeni
            aday.evidence.append(neden or f"state: {tr.onceki.value} → {tr.yeni.value}")
        return tr


__all__ = ["GECISLER", "StateTransition", "DeneyimDurumMakinesi"]
