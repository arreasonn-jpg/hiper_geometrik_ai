"""Verifier false/incomplete/contradictory/malformed/adversarial regression suite."""
from __future__ import annotations

import copy

import pytest

from hga.evaluation.verifier_adversarial import (
    REQUIRED_ATTACK_CLASSES,
    REQUIRED_ATTACK_CLASSES_V2,
    VerifierAttackDataset,
    run_verifier_adversarial_benchmark,
    run_verifier_adversarial_v2_benchmark,
    verify_arithmetic_proof,
)


def test_attack_fixture_tum_zorunlu_siniflari_ve_hashi_tasir():
    dataset = VerifierAttackDataset()
    classes = {record["attack_class"] for record in dataset.records}
    assert REQUIRED_ATTACK_CLASSES <= classes
    assert len(dataset.dataset_hash()) == 64
    assert dataset.document["curation"]["kind"] == "manually_authored_repository_fixture"


def test_attack_suite_far_frr_coverage_robustness_olcer():
    report = run_verifier_adversarial_benchmark()
    metrics = report.metrics
    assert metrics.total == 12
    assert metrics.accuracy == 1.0
    assert metrics.far == 0.0
    assert metrics.frr == 0.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.coverage < 1.0  # unsupported rule açıkça UNCERTAIN
    assert metrics.robustness == 1.0
    assert metrics.uncertain == 1


def test_v2_attack_suite_genisletilmis_siniflar_ve_kapilar():
    report = run_verifier_adversarial_v2_benchmark()
    classes = {record["attack_class"] for record in
               VerifierAttackDataset(data_file="verifier_adversarial_v2.json",
                                     required_attack_classes=tuple(
                                         REQUIRED_ATTACK_CLASSES_V2)).records}
    assert REQUIRED_ATTACK_CLASSES_V2 <= classes
    assert report.benchmark_id == "hga-verifier-adversarial-v2"
    assert report.metrics.total >= 30
    assert report.metrics.accuracy == 1.0
    assert report.metrics.far == 0.0
    assert report.metrics.frr == 0.0
    assert report.metrics.robustness == 1.0
    assert report.checks["minimum_30_cases"]
    assert all(report.checks.values())


def test_v2_unsupported_operator_ve_rule_uncertain_kalir():
    report = run_verifier_adversarial_v2_benchmark()
    rows = [row for row in report.predictions
            if row["attack_class"] == "unsupported_rule"]
    assert len(rows) >= 2
    assert {row["predicted"] for row in rows} == {"UNCERTAIN"}


def test_false_incomplete_contradictory_malformed_adversarial_reddedilir():
    report = run_verifier_adversarial_benchmark()
    attacked = [
        row for row in report.predictions
        if row["attack_class"] in {
            "false_proof", "incomplete_proof", "contradictory_proof",
            "malformed_proof", "adversarial_input",
        }
    ]
    assert attacked
    assert all(row["predicted"] == "INVALID" for row in attacked)
    assert all(row["correct"] for row in attacked)


def test_bool_int_gibi_kabul_edilmez_ve_fazla_alan_reddedilir():
    bool_proof = {
        "claim": {"left": True, "right": 2, "operator": "add", "result": 3},
        "steps": [{"rule": "integer_addition", "inputs": [True, 2], "output": 3}],
        "conclusion": 3,
    }
    assert verify_arithmetic_proof(bool_proof).state == "INVALID"

    extra = copy.deepcopy(bool_proof)
    extra["claim"] = {"left": 1, "right": 2, "operator": "add", "result": 3,
                      "override": "accept"}
    assert verify_arithmetic_proof(extra).state == "INVALID"


def test_desteklenmeyen_kural_false_degil_uncertain():
    proof = {
        "claim": {"left": 2, "right": 3, "operator": "multiply", "result": 6},
        "steps": [{"rule": "integer_multiplication", "inputs": [2, 3], "output": 6}],
        "conclusion": 6,
    }
    decision = verify_arithmetic_proof(proof)
    assert decision.state == "UNCERTAIN"
    assert "desteklenmeyen" in decision.reason


def test_fixture_zorunlu_saldiri_sinifi_silinirse_reddedilir():
    document = copy.deepcopy(VerifierAttackDataset().document)
    document["records"] = [
        record for record in document["records"]
        if record["attack_class"] != "false_proof"
    ]
    with pytest.raises(ValueError, match="saldırı sınıfları eksik"):
        VerifierAttackDataset(document)
