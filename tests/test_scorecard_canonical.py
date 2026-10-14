"""Kanonik skor kartı sözleşmesi testi (docs/SCORECARD_SPEC.md).

Bu testler, depoda commit edilmiş `docs/scorecard.json`'un kanonik
sözleşmeye uyduğunu doğrular. Kart bozulursa CI kırmızı olur.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

KOK = Path(__file__).resolve().parents[1]
KART = Path(os.environ.get("SCORECARD_PATH", str(KOK / "docs" / "scorecard.json")))

KANONIK_BOLUMLER = {
    "architecture", "memory", "verification", "generalization",
    "reasoning", "self_learning", "statistical_rigor",
    "human_evaluation", "turkish_nlp", "language_modeling",
    "reproducibility", "scientific_evidence", "engineering",
}


@pytest.fixture(scope="module")
def kart() -> dict:
    assert KART.exists(), f"Kanonik kart yok: {KART}"
    return json.loads(KART.read_text(encoding="utf-8"))


def test_zorunlu_alanlar(kart):
    for alan in ("generated_at_utc", "overall", "scored_sections",
                 "unscored_sections", "sections"):
        assert alan in kart, f"Zorunlu alan eksik: {alan}"


def test_generated_at_iso(kart):
    s = kart["generated_at_utc"]
    assert "T" in s and ("+00:00" in s or s.endswith("Z")), (
        f"generated_at_utc ISO 8601 değil: {s!r}"
    )


def test_kanonik_bolumler(kart):
    assert set(kart["sections"].keys()) == KANONIK_BOLUMLER, (
        f"Bölüm kümesi uyuşmuyor: {set(kart['sections'].keys()) ^ KANONIK_BOLUMLER}"
    )


def test_skor_araligi(kart):
    for ad, b in kart["sections"].items():
        s = b.get("score")
        assert s is None or (isinstance(s, (int, float)) and 0.0 <= s <= 10.0), (
            f"{ad}: skor aralık dışı: {s!r}"
        )


def test_overall_ortalamaya_yakin(kart):
    skorlar = [b["score"] for b in kart["sections"].values()
               if b.get("score") is not None]
    if not skorlar:
        assert kart["overall"] is None
        return
    beklenen = round(sum(skorlar) / len(skorlar), 2)
    assert abs(kart["overall"] - beklenen) < 0.05, (
        f"overall={kart['overall']} beklenen≈{beklenen}"
    )


def test_unscored_tutarliligi(kart):
    gercek = sorted(ad for ad, b in kart["sections"].items()
                    if b.get("score") is None)
    ilan = sorted(kart["unscored_sections"])
    assert ilan == gercek, f"unscored listesi uyuşmuyor: {ilan} vs {gercek}"
    assert kart["scored_sections"] == 13 - len(gercek)


def test_human_evaluation_kurali(kart):
    """SEC-2026-10-13-02: sentetik veri skor üretemez."""
    b = kart["sections"]["human_evaluation"]
    inputs = b.get("inputs", {})
    gercek = (
        inputs.get("ratings_collected") is True
        and inputs.get("rater_attestation_present") is True
        and inputs.get("source_kind") == "real"
    )
    if not gercek:
        assert b.get("score") is None, (
            "human_evaluation: sentetik/eksik veriye karşın skor üretilmiş"
        )
