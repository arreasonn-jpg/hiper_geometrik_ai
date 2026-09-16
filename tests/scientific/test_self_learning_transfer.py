# -*- coding: utf-8 -*-
"""P1 cross-domain self-learning transfer benchmark testleri."""
from __future__ import annotations

import pytest

from hga.evaluation.multi_environment import ENVIRONMENTS
from hga.evaluation.self_learning_transfer import (
    PROTOCOL,
    run_self_learning_transfer_benchmark,
    self_learning_transfer_markdown,
)


@pytest.fixture(scope="module")
def rapor():
    return run_self_learning_transfer_benchmark(
        seeds=(1, 2),
        environments=("physics", "causal", "language"),
        source_examples=30,
        target_support_examples=10,
        target_unseen_examples=10,
    )


def test_protokol_ve_pair_sayisi(rapor):
    assert rapor.protocol == PROTOCOL
    assert rapor.schema_version == 1
    assert rapor.aggregate["pairs"] == 2 * 3 * 2


def test_source_only_hedefte_cekimser(rapor):
    assert rapor.aggregate["source_only_target_decisions"] == 0
    assert rapor.checks["source_only_abstains_on_target"] is True
    for row in rapor.pair_results:
        assert row["source_only"]["coverage"] == 0.0


def test_hedef_destek_adaptasyonu_calısıyor_ama_pozitif_transfer_iddiasi_yok(rapor):
    assert rapor.aggregate["scratch_coverage"]["mean"] > 0.0
    assert rapor.aggregate["transfer_coverage"]["mean"] > 0.0
    assert rapor.aggregate["coverage_delta_transfer_minus_scratch"]["mean"] == pytest.approx(0.0)
    assert rapor.aggregate["positive_transfer_status"] == "NOT_DEMONSTRATED"
    assert rapor.checks["positive_transfer_not_claimed_when_absent"] is True


def test_tum_kapilar_geciyor(rapor):
    assert [k for k, v in rapor.checks.items() if not v] == []


def test_gecersiz_girdiler():
    with pytest.raises(ValueError):
        run_self_learning_transfer_benchmark(seeds=())
    with pytest.raises(ValueError):
        run_self_learning_transfer_benchmark(environments=("physics",))
    with pytest.raises(ValueError):
        run_self_learning_transfer_benchmark(environments=("yok", "physics"))
    with pytest.raises(ValueError):
        run_self_learning_transfer_benchmark(source_examples=0)


def test_markdown_durust_sinir_iceriyor(rapor):
    md = self_learning_transfer_markdown(rapor)
    assert "Pozitif transfer durumu" in md
    assert "NOT_DEMONSTRATED" in md
    assert "pozitif transfer iddiası değildir" in md


def test_varsayilan_ortamlar_besli():
    r = run_self_learning_transfer_benchmark(
        seeds=(1,), source_examples=5,
        target_support_examples=5,
        target_unseen_examples=5)
    assert set(r.environments) == set(ENVIRONMENTS)
