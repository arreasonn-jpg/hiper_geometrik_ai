# -*- coding: utf-8 -*-
"""Somut halüsinasyon / factual consistency metrikleri."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, Optional

from ..knowledge.schemas import DeneyimDurumu, KaynakTuru


@dataclass
class HallucinationReport:
    toplam: int = 0
    checked: int = 0
    dogru: int = 0
    yanlis: int = 0
    belirsiz: int = 0
    verified: int = 0
    valid_model_generated: int = 0
    conflict: int = 0
    invalid: int = 0
    hallucination_rate: float = 0.0
    factual_consistency_score: float = 0.0
    unsupported_rate: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


def hallucination_metrics(adaylar: Iterable, store=None,
                          validator: Optional[Callable] = None) -> HallucinationReport:
    """Aday deneyimler için factual consistency raporu üret.

    ``validator`` verilirse imzası ``validator(store, aday) -> True/False/None``
    olmalıdır. Bu durumda:
      * False = ölçülmüş halüsinasyon/yanlış kabul
      * True  = doğrulanmış tutarlılık
      * None  = belirsiz

    Validator yoksa güvenli proxy metrik kullanılır: VERIFIED tutarlı kabul
    edilir; MODEL_GENERATED+VALID ve CONFLICT desteklenmemiş risk olarak sayılır.
    """
    r = HallucinationReport()
    for a in list(adaylar):
        r.toplam += 1
        state = getattr(a, "state", None)
        source = getattr(a, "source", None)
        if state == DeneyimDurumu.VERIFIED:
            r.verified += 1
        if state == DeneyimDurumu.VALID and source == KaynakTuru.MODEL_GENERATED:
            r.valid_model_generated += 1
        if state == DeneyimDurumu.CONFLICT:
            r.conflict += 1
        if state == DeneyimDurumu.INVALID:
            r.invalid += 1

        if validator is not None:
            sonuc = validator(store, a)
            if sonuc is True:
                r.dogru += 1
                r.checked += 1
            elif sonuc is False:
                r.yanlis += 1
                r.checked += 1
            else:
                r.belirsiz += 1

    if validator is not None:
        r.hallucination_rate = r.yanlis / r.checked if r.checked else 0.0
        r.factual_consistency_score = r.dogru / r.checked if r.checked else 0.0
    else:
        # Proxy: doğrulanmamış kabul + conflict = desteklenmemiş bilgi riski.
        risk = r.valid_model_generated + r.conflict
        r.unsupported_rate = risk / r.toplam if r.toplam else 0.0
        r.hallucination_rate = r.unsupported_rate
        r.factual_consistency_score = r.verified / r.toplam if r.toplam else 0.0
    if validator is not None:
        risk = r.valid_model_generated + r.conflict
        r.unsupported_rate = risk / r.toplam if r.toplam else 0.0
    return r


__all__ = ["HallucinationReport", "hallucination_metrics"]
