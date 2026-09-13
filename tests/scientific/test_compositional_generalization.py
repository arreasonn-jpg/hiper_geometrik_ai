"""Türkçe compositional generalization bilimsel sözleşme testleri."""
from __future__ import annotations

import copy

import pytest

from hga.evaluation.compositional import (
    DIMENSIONS,
    CompositionalDataset,
    run_compositional_benchmark,
)


def test_dataset_elle_sabit_split_ve_hash_sozlesmesi():
    dataset = CompositionalDataset()
    assert dataset.document["curation"]["kind"] == "manually_authored_repository_fixture"
    assert len(dataset.dataset_hash()) == 64
    assert {row["split"] for row in dataset.entities} == {"train", "test"}
    assert {row["split"] for row in dataset.relations} == {"train", "test"}
    assert dataset.leakage_audit().clean
    assert dataset.surface_leakage_clean()


def test_ali_araba_gorulmeden_uretilir_ve_dogrulanir():
    report = run_compositional_benchmark()
    target = next(
        row for row in report.predictions if row["experience_id"] == "TE-COMP-001"
    )
    assert target["candidate_generated"] is True
    assert target["predicted"] == "VALID"
    assert target["generated_sentence"] == "Ali arabaya bindi."
    assert target["generation_exact"] is True
    assert target["parsing_correct"] is True


def test_tum_boyutlar_ve_cg_acikca_olculur():
    report = run_compositional_benchmark()
    assert set(report.dimensions) == set(DIMENSIONS)
    assert report.metrics.accuracy == 1.0
    assert report.metrics.generation_exact_rate == 1.0
    assert report.metrics.parsing_accuracy == 1.0
    assert report.dimensions["unseen_entity"].total > 0
    assert report.dimensions["unseen_relation"].total > 0
    assert report.dimensions["unseen_combination"].total > 0
    assert report.dimensions["unseen_wording"].total > 0
    assert report.dimensions["ood"].uncertain_total > 0
    assert report.generalization_capacity.symbol == "C_G"
    assert report.generalization_capacity.is_theoretical_capacity is False
    assert report.generalization_capacity.eligible_cases == 5
    assert report.generalization_capacity.score == 1.0


def test_semantik_sizinti_fixture_olusturulurken_reddedilir():
    document = copy.deepcopy(CompositionalDataset().document)
    contaminated = next(row for row in document["test"] if row["semantic_novel"])
    contaminated.update({
        "subject_id": "E_ALI",
        "relation_id": "R_BINMEK",
        "object_id": "E_AT",
    })
    with pytest.raises(ValueError, match="semantic_novel"):
        CompositionalDataset(document)
