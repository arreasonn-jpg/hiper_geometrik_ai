# -*- coding: utf-8 -*-
"""HGA English EWT parameter sweep guard rails.

The important assertion is negative: a narrow three-size sweep must *not* be
silently promoted to a scaling law.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hga.evaluation.english_ewt import (
    HGA_SCALE_CONFIGS,
    english_hga_scaling_markdown,
    run_english_hga_scaling_probe,
)


def test_hga_scale_probe_gercek_cekirdegi_arttirilan_parametreyle_olcer():
    pytest.importorskip("torch")
    report = run_english_hga_scaling_probe(seeds=(1, 2), profile="smoke")

    assert report["status"] == "EXPLORATORY_NOT_A_SCALING_LAW"
    assert list(HGA_SCALE_CONFIGS) == ["small", "base", "large"]
    assert [row["scale"] for row in report["rows"]] == ["small", "base", "large"]
    parameters = [row["physical_parameters"] for row in report["rows"]]
    assert parameters == sorted(parameters)
    assert report["parameter_range_ratio"] < 100.0
    assert all(report["checks"].values())
    assert "not a scaling law" in english_hga_scaling_markdown(report)


def test_kuraturlenmis_bes_seed_probe_scaling_law_iddiasini_engeller():
    path = Path(__file__).resolve().parents[2] / "docs" / "hga_english_scaling_probe.json"
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["seeds"] == [1, 2, 3, 4, 5]
    assert report["status"] == "EXPLORATORY_NOT_A_SCALING_LAW"
    assert report["checks"]["scaling_law_claim_blocked_by_narrow_range"] is True
    assert len(report["limitations"]) >= 3
