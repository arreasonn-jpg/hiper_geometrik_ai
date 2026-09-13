"""Scientific tests for leakage-safe uncertainty calibration."""
import pytest

from hga.evaluation.calibration import (
    COVERAGE_TARGETS,
    calibration_metrics,
    fit_temperature,
    temperature_scaling_report,
)


def _overconfident_fixture():
    # Correct signs are mixed with errors; oversized margins are deliberately
    # overconfident so dev-only temperature scaling has a measurable job.
    logits = [
        [-4.0, 4.0], [4.0, -4.0], [-5.0, 5.0], [5.0, -5.0],
        [-4.0, 4.0], [4.0, -4.0], [-3.0, 3.0], [3.0, -3.0],
        [-4.0, 4.0], [4.0, -4.0],
    ]
    labels = [True, False, True, False, True, False, False, True, False, True]
    return logits, labels


def test_temperature_yalniz_dev_nll_ile_fit_edilir_ve_argmaxi_degistirmez():
    dev_logits, dev_labels = _overconfident_fixture()
    fit = fit_temperature(dev_logits, dev_labels)
    assert fit["temperature"] > 1.0
    assert fit["calibrated_dev_nll"] < fit["uncalibrated_dev_nll"]

    test_logits = [[-2.0, 2.0], [2.0, -2.0], [-1.0, 1.0], [1.0, -1.0]]
    test_labels = [True, False, False, True]
    report = temperature_scaling_report(
        dev_logits,
        dev_labels,
        test_logits,
        test_labels,
        [("all", "entity_disjoint")] * 2 + [("all", "relation_disjoint")] * 2,
    )
    assert all(report["checks"].values())
    assert report["fit_split"] == "dev"
    assert report["evaluation_split"] == "test"
    assert set(report["slices"]) == {"all", "entity_disjoint", "relation_disjoint"}
    assert report["slices"]["all"]["before"]["accuracy"] == (
        report["slices"]["all"]["after"]["accuracy"]
    )


def test_ece_brier_nll_adaptive_ece_ve_selective_risk_raporlanir():
    logits, labels = _overconfident_fixture()
    metrics = calibration_metrics(logits, labels, temperature=2.0, bins=5)
    for name in ("ece", "adaptive_ece", "brier", "nll", "accuracy", "aurc"):
        assert 0.0 <= metrics[name]
    assert len(metrics["risk_coverage_points"]) == len(COVERAGE_TARGETS)
    assert [point["target_coverage"] for point in metrics["risk_coverage_points"]] == list(
        COVERAGE_TARGETS
    )
    assert metrics["risk_coverage_points"][-1]["achieved_coverage"] == 1.0


def test_calibration_gecersiz_girdi_ve_test_dimension_mismatchini_reddeder():
    with pytest.raises(ValueError):
        calibration_metrics([], [])
    with pytest.raises(ValueError):
        calibration_metrics([[0.0, 1.0]], [True], temperature=0.0)
    with pytest.raises(ValueError):
        temperature_scaling_report(
            [[0.0, 1.0]], [True], [[0.0, 1.0]], [True], []
        )
