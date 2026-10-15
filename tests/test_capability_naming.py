"""P1-4 regresyon testi: capability naming + kapsam alanları."""
from __future__ import annotations

import pytest

from hga.evaluation.capability_vector import (
    TERMINOLOGY,
    CapabilityEntry,
    build_capability_vector,
)


def test_yeni_isimler_daraltildi():
    assert TERMINOLOGY["C_R"]["name"] == "Verified Relational Chain Depth"
    assert TERMINOLOGY["C_RD"]["name"] == "Distractor-Resistant Chain Depth"
    assert TERMINOLOGY["C_G"]["name"] == "Compositional Fixture v2 Score"
    assert TERMINOLOGY["C_MR"]["name"] == "Exact Record Lookup Rate"
    assert TERMINOLOGY["C_H"]["name"] == "Signature Benchmark Resistance"


@pytest.mark.parametrize("sembol", ["C_R", "C_RD", "C_G", "C_MR", "C_H"])
def test_kapsam_alanlari_dolu(sembol):
    meta = TERMINOLOGY[sembol]
    assert meta.get("protocol"), f"{sembol}: protocol boş"
    assert meta.get("scope"), f"{sembol}: scope boş"
    assert meta.get("limitations"), f"{sembol}: limitations boş"


def test_capability_entry_yeni_alanlari_tasiyor():
    entries = build_capability_vector(
        capacity={"physical_parameters": 40_524_865},
    )
    by_symbol = {e.symbol: e for e in entries}
    if "C_H" in by_symbol:
        e = by_symbol["C_H"]
        assert e.protocol == "hga_signature_benchmark_v1"
        assert e.scope
        assert e.limitations
        assert "Açık dünya" in e.limitations


def test_eski_isimler_kaldirildi():
    genel_isimler = {
        "Measured Generalization Capacity",
        "Measured Reliable Reasoning Depth",
        "Measured Memory Recall",
        "Hallucination Resistance",
    }
    mevcut = {meta["name"] for meta in TERMINOLOGY.values()}
    kalan = genel_isimler & mevcut
    assert not kalan, f"Genel isimler hâlâ var: {kalan}"


def test_capability_entry_default_degerler():
    e = CapabilityEntry(
        symbol="X", name="Test", kind="measured", value=1.0,
        unit="adet", evidence=None,
    )
    assert e.protocol == ""
    assert e.scope == ""
    assert e.limitations == ""
