"""Leakage-safe uncertainty calibration and selective prediction metrics.

Temperature is selected only on the fixed development split.  Test labels are
used exclusively for final measurement; they never influence the temperature
or an abstention threshold.  The routines are dependency-free so the metric
contract can be unit-tested independently from the neural models.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

CALIBRATION_PROTOCOL = "dev-temperature-scaling-selective-risk-v1"
TEMPERATURE_GRID = tuple(
    round(math.exp(math.log(0.1) + index * math.log(100.0) / 240), 12)
    for index in range(241)
)
COVERAGE_TARGETS = (0.10, 0.25, 0.50, 0.75, 1.0)


def _round(value: float) -> float:
    return round(float(value), 8)


def _probability(logits: Sequence[float], temperature: float) -> float:
    margin = (float(logits[1]) - float(logits[0])) / temperature
    if margin >= 0:
        return 1.0 / (1.0 + math.exp(-min(margin, 709.0)))
    exponential = math.exp(max(margin, -709.0))
    return exponential / (1.0 + exponential)


def _nll(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    epsilon = 1e-12
    losses = [
        -math.log(max(epsilon, min(1.0 - epsilon, p if label else 1.0 - p)))
        for p, label in zip(probabilities, labels)
    ]
    return sum(losses) / len(losses)


def _ece(confidences: Sequence[float], correctness: Sequence[bool], bins: int) -> float:
    total = len(confidences)
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [
            position for position, confidence in enumerate(confidences)
            if lower <= confidence < upper or (index == bins - 1 and confidence == 1.0)
        ]
        if selected:
            accuracy = sum(correctness[position] for position in selected) / len(selected)
            mean_confidence = sum(confidences[position] for position in selected) / len(selected)
            error += len(selected) / total * abs(accuracy - mean_confidence)
    return error


def _adaptive_ece(
    confidences: Sequence[float], correctness: Sequence[bool], bins: int
) -> float:
    order = sorted(range(len(confidences)), key=lambda index: (confidences[index], index))
    total = len(order)
    error = 0.0
    for bin_index in range(bins):
        start = bin_index * total // bins
        end = (bin_index + 1) * total // bins
        selected = order[start:end]
        if selected:
            accuracy = sum(correctness[position] for position in selected) / len(selected)
            mean_confidence = sum(confidences[position] for position in selected) / len(selected)
            error += len(selected) / total * abs(accuracy - mean_confidence)
    return error


def _selective_metrics(
    confidences: Sequence[float], correctness: Sequence[bool]
) -> Dict[str, Any]:
    # Stable index tie-break makes the policy byte-reproducible.
    order = sorted(
        range(len(confidences)), key=lambda index: (-confidences[index], index)
    )
    cumulative_errors = 0
    risks = []
    for rank, index in enumerate(order, start=1):
        cumulative_errors += int(not correctness[index])
        risks.append(cumulative_errors / rank)
    points = []
    for target in COVERAGE_TARGETS:
        retained = max(1, math.ceil(target * len(order)))
        selected = order[:retained]
        points.append({
            "target_coverage": target,
            "achieved_coverage": _round(retained / len(order)),
            "retained": retained,
            "confidence_threshold": _round(min(confidences[index] for index in selected)),
            "selective_accuracy": _round(1.0 - risks[retained - 1]),
            "selective_risk": _round(risks[retained - 1]),
        })
    return {
        "aurc": _round(sum(risks) / len(risks)),
        "risk_coverage_points": points,
    }


def calibration_metrics(
    logits: Sequence[Sequence[float]],
    labels: Sequence[bool],
    temperature: float = 1.0,
    bins: int = 10,
) -> Dict[str, Any]:
    """Measure probability and confidence calibration for a binary classifier."""
    if len(logits) != len(labels) or not labels:
        raise ValueError("Logit ve label sayıları eşit ve sıfırdan büyük olmalı")
    if temperature <= 0 or bins <= 1:
        raise ValueError("Temperature pozitif, bins en az 2 olmalı")
    probabilities = [_probability(row, temperature) for row in logits]
    predictions = [probability >= 0.5 for probability in probabilities]
    correctness = [prediction == label for prediction, label in zip(predictions, labels)]
    confidences = [max(probability, 1.0 - probability) for probability in probabilities]
    brier = sum(
        (probability - float(label)) ** 2
        for probability, label in zip(probabilities, labels)
    ) / len(labels)
    result = {
        "count": len(labels),
        "temperature": _round(temperature),
        "accuracy": _round(sum(correctness) / len(correctness)),
        "mean_confidence": _round(sum(confidences) / len(confidences)),
        "nll": _round(_nll(probabilities, labels)),
        "brier": _round(brier),
        "ece": _round(_ece(confidences, correctness, bins)),
        "adaptive_ece": _round(_adaptive_ece(confidences, correctness, bins)),
        "bin_count": bins,
    }
    result.update(_selective_metrics(confidences, correctness))
    return result


def fit_temperature(
    dev_logits: Sequence[Sequence[float]], dev_labels: Sequence[bool]
) -> Dict[str, Any]:
    """Select a scalar temperature by deterministic dev-only NLL minimization."""
    if len(dev_logits) != len(dev_labels) or not dev_labels:
        raise ValueError("Dev logit ve label sayıları eşit ve sıfırdan büyük olmalı")
    losses = []
    for temperature in TEMPERATURE_GRID:
        probabilities = [_probability(row, temperature) for row in dev_logits]
        losses.append(_nll(probabilities, dev_labels))
    best_index = min(range(len(losses)), key=lambda index: (losses[index], index))
    return {
        "temperature": TEMPERATURE_GRID[best_index],
        "objective": "development binary negative log likelihood",
        "grid_min": TEMPERATURE_GRID[0],
        "grid_max": TEMPERATURE_GRID[-1],
        "grid_size": len(TEMPERATURE_GRID),
        "uncalibrated_dev_nll": _round(losses[TEMPERATURE_GRID.index(1.0)]),
        "calibrated_dev_nll": _round(losses[best_index]),
        "optimum_on_grid_boundary": best_index in (0, len(TEMPERATURE_GRID) - 1),
    }


def temperature_scaling_report(
    dev_logits: Sequence[Sequence[float]],
    dev_labels: Sequence[bool],
    test_logits: Sequence[Sequence[float]],
    test_labels: Sequence[bool],
    test_dimensions: Sequence[Sequence[str]],
) -> Dict[str, Any]:
    """Fit on development data and evaluate overall and subgroup test calibration."""
    if len(test_logits) != len(test_dimensions):
        raise ValueError("Test logit ve dimension sayıları eşit olmalı")
    fit = fit_temperature(dev_logits, dev_labels)
    temperature = float(fit["temperature"])
    dimension_names = sorted({name for names in test_dimensions for name in names})
    slices: Dict[str, Mapping[str, Any]] = {}
    for name in dimension_names:
        selected = [index for index, names in enumerate(test_dimensions) if name in names]
        slice_logits = [test_logits[index] for index in selected]
        slice_labels = [test_labels[index] for index in selected]
        slices[name] = {
            "before": calibration_metrics(slice_logits, slice_labels),
            "after": calibration_metrics(slice_logits, slice_labels, temperature),
        }
    before_predictions = [row[1] >= row[0] for row in test_logits]
    calibrated_predictions = [
        _probability(row, temperature) >= 0.5 for row in test_logits
    ]
    scalar_metric_names = (
        "accuracy", "mean_confidence", "nll", "brier", "ece",
        "adaptive_ece", "aurc",
    )
    checks = {
        "fit_uses_development_only": True,
        "test_labels_not_used_for_temperature_or_threshold": True,
        "positive_temperature": temperature > 0.0,
        "dev_nll_nonworsening": fit["calibrated_dev_nll"] <= fit["uncalibrated_dev_nll"],
        "temperature_preserves_argmax_predictions": before_predictions == calibrated_predictions,
        "all_declared_test_dimensions_measured": bool(slices) and "all" in slices,
        "all_probability_metrics_finite": all(
            math.isfinite(float(metrics[metric_name]))
            for result in slices.values()
            for metrics in (result["before"], result["after"])
            for metric_name in scalar_metric_names
        ),
    }
    return {
        "protocol": CALIBRATION_PROTOCOL,
        "schema_version": 1,
        "fit_split": "dev",
        "evaluation_split": "test",
        "abstention_policy": (
            "rank by calibrated max-class confidence; fixed coverage targets; "
            "no test-fitted threshold"
        ),
        "fit": fit,
        "slices": slices,
        "checks": checks,
        "limitations": [
            "Temperature scaling calibrates probabilities but cannot improve argmax accuracy.",
            "ECE depends on binning; fixed-bin and adaptive-bin ECE are both reported.",
            "Selective risk is diagnostic at predeclared coverage targets, not a deployed threshold.",
        ],
    }


__all__ = [
    "CALIBRATION_PROTOCOL",
    "COVERAGE_TARGETS",
    "calibration_metrics",
    "fit_temperature",
    "temperature_scaling_report",
]
