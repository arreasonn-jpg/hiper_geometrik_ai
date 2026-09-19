import pytest

from hga.evaluation.scale_protocol import (
    ONE_MILLION_TOKENS,
    LongContextObservation,
    ScalingObservation,
    assess_long_context_stability,
    fit_power_law,
)

HASH = "a" * 64


def _point(parameters: int, loss: float) -> ScalingObservation:
    return ScalingObservation(
        run_id=f"run-{parameters}",
        parameters=parameters,
        training_flops=float(parameters * 10),
        validation_loss=loss,
        train_tokens=1_000_000,
        dataset_revision="dataset-revision",
        code_revision="code-revision",
        checkpoint_sha256=HASH,
    )


def test_power_law_fit_reports_log_space_parameters_and_predeclared_gate():
    report = fit_power_law([_point(100, 1.0), _point(400, 0.5), _point(1600, 0.25)], "parameters")
    assert report["passes_r_squared_target"] is True
    assert report["r_squared_log_loss"] == pytest.approx(1.0)
    assert report["exponent"] == pytest.approx(-0.5)
    assert report["fixed_conditions"]["train_tokens"] == 1_000_000


def test_power_law_rejects_mixed_training_conditions():
    other = _point(400, 0.5)
    other = ScalingObservation(**{**other.__dict__, "train_tokens": 2_000_000})
    with pytest.raises(ValueError, match="share train_tokens"):
        fit_power_law([_point(100, 1.0), other, _point(1600, 0.25)], "parameters")


def test_one_million_context_gate_never_treats_empty_trace_as_stable():
    empty = assess_long_context_stability([])
    assert empty["target_context_tokens"] == ONE_MILLION_TOKENS
    assert empty["stable"] is False

    trace = [
        LongContextObservation("scale-1b", ONE_MILLION_TOKENS, step, loss, 1000, 0.8)
        for step, loss in ((100, 1.00), (200, 1.02), (300, 0.99))
    ]
    stable = assess_long_context_stability(trace)
    assert stable["finite_loss"] is True
    assert stable["stable"] is True


def test_context_gate_refuses_sub_million_target():
    with pytest.raises(ValueError, match="at least 1,000,000"):
        assess_long_context_stability([], target_context_tokens=64_000)
