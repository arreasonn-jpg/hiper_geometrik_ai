# -*- coding: utf-8 -*-
"""Deneyim akış metrikleri."""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable


def _val(x):
    return getattr(x, "value", str(x))


def deneyim_akisi(adaylar: Iterable) -> Dict:
    """Üretilen/kabul edilen/reddedilen deneyimlerin durum serisi özeti.

    ``adaylar`` ExperienceCandidate veya aynı alanlara sahip nesneler olabilir.
    Dönüş JSON-serileştirilebilir dict'tir.
    """
    adaylar = list(adaylar)
    durum = Counter(_val(getattr(a, "state", "UNKNOWN")) for a in adaylar)
    kaynak = Counter(_val(getattr(a, "source", "UNKNOWN")) for a in adaylar)
    accepted = durum.get("VALID", 0) + durum.get("VERIFIED", 0)
    rejected = durum.get("INVALID", 0) + durum.get("REJECT", 0)
    conflict = durum.get("CONFLICT", 0) + durum.get("EXPLORE", 0)
    toplam = len(adaylar)
    return {
        "toplam": toplam,
        "durum": dict(sorted(durum.items())),
        "kaynak": dict(sorted(kaynak.items())),
        "accepted": accepted,
        "rejected": rejected,
        "conflict": conflict,
        "acceptance_rate": accepted / toplam if toplam else 0.0,
        "rejection_rate": rejected / toplam if toplam else 0.0,
        "conflict_rate": conflict / toplam if toplam else 0.0,
    }


__all__ = ["deneyim_akisi"]
