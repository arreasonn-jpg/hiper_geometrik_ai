"""Çoklu verifier ensemble regresyon testleri."""
from __future__ import annotations

import pytest

from hga.evaluation.verifier_adversarial import ProofDecision
from hga.evaluation.verifier_ensemble import (
    ENSEMBLE_PROTOCOL,
    VerifierMember,
    run_verifier_ensemble_benchmark,
    verifier_ensemble_markdown,
    verify_normal_form_addition,
    verify_trace_replay_addition,
)


def test_ensemble_uc_bagimsiz_uyeyle_sifir_false_accept():
    report = run_verifier_ensemble_benchmark()
    assert report.protocol == ENSEMBLE_PROTOCOL
    assert report.members == [
        "strict_schema_oracle", "normal_form_oracle", "trace_replay_oracle",
    ]
    assert report.metrics.accuracy == 1.0
    assert report.metrics.far == 0.0
    assert report.metrics.frr == 0.0
    assert report.metrics.robustness == 1.0
    assert report.checks["member_count_at_least_3"]
    assert all(report.checks.values())


def test_unsupported_rule_verified_olmaya_zorlanmaz():
    report = run_verifier_ensemble_benchmark()
    unsupported = next(row for row in report.predictions
                       if row["attack_class"] == "unsupported_rule")
    assert unsupported["predicted"] == "UNCERTAIN"
    assert {vote["state"] for vote in unsupported["votes"].values()} == {"UNCERTAIN"}


def test_normal_form_ve_trace_false_proofu_reddeder():
    proof = {
        "claim": {"left": 2, "right": 3, "operator": "add", "result": 6},
        "steps": [{"rule": "integer_addition", "inputs": [2, 3], "output": 6}],
        "conclusion": 6,
    }
    assert verify_normal_form_addition(proof).state == "INVALID"
    assert verify_trace_replay_addition(proof).state == "INVALID"


def test_her_case_tum_oylari_tasir_ve_markdown_kapilari_yazar():
    report = run_verifier_ensemble_benchmark()
    assert all(len(row["votes"]) == 3 for row in report.predictions)
    md = verifier_ensemble_markdown(report)
    assert "Kabul kapıları" in md
    assert "strict_schema_oracle" in md
    assert "zero_false_acceptance" in md


def test_ensemble_tek_uyeyle_kosulmaz():
    with pytest.raises(ValueError, match="en az üç"):
        run_verifier_ensemble_benchmark(
            members=[VerifierMember("only", verify_normal_form_addition)])


def test_bir_uye_reddederse_gecerli_ispat_bile_verified_olmaz():
    def rejector(proof, minimum_operand, maximum_operand, maximum_steps):
        return ProofDecision("INVALID", "sentetik rejector")

    report = run_verifier_ensemble_benchmark(
        members=[
            VerifierMember("normal", verify_normal_form_addition),
            VerifierMember("trace", verify_trace_replay_addition),
            VerifierMember("rejector", rejector),
        ],
    )
    valid_rows = [row for row in report.predictions if row["expected"] == "VERIFIED"]
    assert valid_rows and all(row["predicted"] == "INVALID" for row in valid_rows)
    assert report.metrics.false_rejection == report.metrics.valid_proofs
    assert report.checks["zero_false_rejection"] is False
