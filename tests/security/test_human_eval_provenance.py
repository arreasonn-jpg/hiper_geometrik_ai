"""P0-4 regresyon testi: sentetik/simülasyon insan verisi skor üretemez.

Kural: `human_evaluation` bölümü yalnızca üç koşul birlikte sağlandığında
skorlanır:

    1. checks.human_ratings_collected == True
    2. checks.rater_attestation_present == True
    3. source_kind == "real"

Bu testler, capability_vector.build_scorecard'ın bu üç koşulu doğru
uyguladığını doğrular.
"""
from __future__ import annotations

import pytest

from hga.experiments.capability_vector import build_scorecard


def _minimal_scorecard(**kwargs):
    """scorecard üret; human_evaluation dışındaki bölümler None kabul eder."""
    return build_scorecard(**kwargs)


def _insan_bolumu(card):
    """Scorecard.sections Dict[str, Dict] — key ile eriş."""
    b = card.sections.get("human_evaluation")
    assert b is not None, "human_evaluation bölümü bulunamadı"
    # Görsel tutarlılık için dict döndürüyoruz; .score erişimi için
    # testler b["score"] kullanmalı. Eski .score attribute erişimini
    # desteklemek için basit bir wrapper:
    class _B:
        def __init__(self, d):
            self._d = d
            self.score = d.get("score")
            self.inputs = d.get("inputs", {})
            self.section = d.get("section", "human_evaluation")
            self.evidence = d.get("evidence", [])
            self.rationale = d.get("rationale", "")
    return _B(b)


# --- 1) Simülasyon verisi: ratings_collected=True, attestation=False ---
def test_simulasyon_verisi_skor_uretmez():
    he = {
        "checks": {
            "human_ratings_collected": True,
            "rater_attestation_present": False,  # ← kritik
            "reliability_meets_threshold": True,
        },
        "design": {"prompts": 50, "raters": 10},
        "source_kind": "unattested",
        "results": {"_summary": {"all_acceptable": True}},
        "arm_results": {"hga": {"means": {"dogruluk": 4.8}}},
    }
    card = _minimal_scorecard(human_evaluation=he)
    bolum = _insan_bolumu(card)
    assert bolum.score is None, (
        "Attestation yokken insan bölümü n/a olmalı; "
        f"ama skor={bolum.score} çıktı."
    )


# --- 2) Eksik source_kind: eski davranış geri gelmesin ---
def test_source_kind_yoksa_skor_uretmez():
    he = {
        "checks": {
            "human_ratings_collected": True,
            "rater_attestation_present": True,   # attestation var
            # source_kind YOK
        },
        "design": {"prompts": 50, "raters": 10},
        "results": {"_summary": {"all_acceptable": True}},
    }
    card = _minimal_scorecard(human_evaluation=he)
    bolum = _insan_bolumu(card)
    assert bolum.score is None, "source_kind yokken skor üretilmemeli"


# --- 3) source_kind="simulation" — kesin reddedilmeli ---
@pytest.mark.parametrize("kaynak", ["simulation", "unattested", "protocol_only", ""])
def test_gercek_olmayan_kaynaklar_reddedilir(kaynak):
    he = {
        "checks": {
            "human_ratings_collected": True,
            "rater_attestation_present": True,
            "reliability_meets_threshold": True,
        },
        "source_kind": kaynak,
        "design": {"prompts": 50, "raters": 10},
        "results": {"_summary": {"all_acceptable": True}},
    }
    card = _minimal_scorecard(human_evaluation=he)
    bolum = _insan_bolumu(card)
    assert bolum.score is None, (
        f"source_kind={kaynak!r} ile skor üretildi — kabul edilemez."
    )


# --- 4) Protokol-only (ratings_collected=False) — zaten n/a ---
def test_protokol_cikitisi_skor_uretmez():
    he = {
        "checks": {
            "human_ratings_collected": False,
            "rater_attestation_present": False,
            "reliability_meets_threshold": False,
        },
        "design": {"prompts": 50, "raters": 10},
        "source_kind": "protocol_only",
    }
    card = _minimal_scorecard(human_evaluation=he)
    bolum = _insan_bolumu(card)
    assert bolum.score is None


# --- 5) Gerçek veri — üç koşul birlikte sağlanınca skor üretilir ---
def test_gercek_beyanli_veri_skor_uretir():
    he = {
        "checks": {
            "human_ratings_collected": True,
            "rater_attestation_present": True,
            "reliability_meets_threshold": True,
        },
        "source_kind": "real",
        "design": {"prompts": 50, "raters": 10},
        "results": {"_summary": {"all_acceptable": True}},
    }
    card = _minimal_scorecard(human_evaluation=he)
    bolum = _insan_bolumu(card)
    # Üç koşul sağlandığı için skor None olmamalı; ayrıca 0..10 arasında olmalı.
    assert bolum.score is not None, (
        "Gerçek, beyanlı insan verisi skor üretmeli."
    )
    assert 0.0 <= bolum.score <= 10.0
    # Girdilerde source_kind doğru damgalanmış olmalı.
    assert bolum.inputs.get("source_kind") == "real"
    assert bolum.inputs.get("rater_attestation_present") is True
    assert bolum.inputs.get("real_human_ratings") is True
