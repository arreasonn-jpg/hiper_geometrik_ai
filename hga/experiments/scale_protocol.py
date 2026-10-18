# -*- coding: utf-8 -*-
"""CKPT-003 için ölçümden bağımsız scaling ve 1M-context kanıt protokolü.

Bu modül eğitim yapmaz ve sentetik gözlem üretmez. Görevi, dış GPU koşularından
alınan değişmez sonuç kartlarını doğrulamak, açık bir power-law fit'i yapmak ve
önceden ilan edilmiş uzun-bağlam kararlılık kapısını değerlendirmektir.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence

ONE_MILLION_TOKENS = 1_000_000
SCALING_R2_TARGET = 0.95


@dataclass(frozen=True)
class ScalingObservation:
    """Bir tamamlanmış eğitim koşusundan gelen, karşılaştırılabilir tek nokta."""

    run_id: str
    parameters: int
    training_flops: float
    validation_loss: float
    train_tokens: int
    dataset_revision: str
    code_revision: str
    checkpoint_sha256: str

    def validate(self) -> None:
        if not self.run_id or not self.dataset_revision or not self.code_revision:
            raise ValueError("run_id, dataset_revision and code_revision are required")
        if not self.checkpoint_sha256 or len(self.checkpoint_sha256) != 64:
            raise ValueError("checkpoint_sha256 must be a 64-character SHA-256")
        if self.parameters <= 0 or self.training_flops <= 0 or self.train_tokens <= 0:
            raise ValueError("parameters, training_flops and train_tokens must be positive")
        if not math.isfinite(self.validation_loss) or self.validation_loss <= 0:
            raise ValueError("validation_loss must be finite and positive")


@dataclass(frozen=True)
class LongContextObservation:
    """Bir doğrulama aralığındaki uzun-bağlam loss telemetrisi."""

    run_id: str
    context_tokens: int
    evaluation_step: int
    validation_loss: float
    peak_memory_bytes: int
    retrieval_metric: float

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")
        if self.context_tokens <= 0 or self.evaluation_step < 0:
            raise ValueError("context_tokens must be positive and evaluation_step non-negative")
        if self.peak_memory_bytes <= 0:
            raise ValueError("peak_memory_bytes must be positive")
        if not math.isfinite(self.validation_loss) or self.validation_loss <= 0:
            raise ValueError("validation_loss must be finite and positive")
        if not math.isfinite(self.retrieval_metric):
            raise ValueError("retrieval_metric must be finite")


def fit_power_law(
    observations: Sequence[ScalingObservation],
    axis: str,
    r_squared_target: float = SCALING_R2_TARGET,
) -> Dict[str, Any]:
    """Fit ``loss = coefficient * axis**exponent`` in log space.

    ``r_squared`` is explicitly the log-loss OLS R². It must not be compared
    across a different loss transform. All points must share training token
    count, dataset revision and code revision; this prevents a nominal model
    scaling plot from silently mixing data or implementation changes.
    """
    if axis not in {"parameters", "training_flops"}:
        raise ValueError("axis must be parameters or training_flops")
    if len(observations) < 3:
        raise ValueError("at least three completed observations are required")
    if not 0.0 < r_squared_target <= 1.0:
        raise ValueError("r_squared_target must be in (0, 1]")
    for observation in observations:
        observation.validate()
    tokens = {item.train_tokens for item in observations}
    datasets = {item.dataset_revision for item in observations}
    code = {item.code_revision for item in observations}
    if len(tokens) != 1 or len(datasets) != 1 or len(code) != 1:
        raise ValueError("scaling points must share train_tokens, dataset_revision and code_revision")

    xs = [math.log(float(getattr(item, axis))) for item in observations]
    ys = [math.log(item.validation_loss) for item in observations]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered = sum((x - mean_x) ** 2 for x in xs)
    if centered == 0.0:
        raise ValueError(f"{axis} values must not all be equal")
    exponent = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / centered
    intercept = mean_y - exponent * mean_x
    predicted = [intercept + exponent * x for x in xs]
    total = sum((y - mean_y) ** 2 for y in ys)
    residual = sum((y - estimate) ** 2 for y, estimate in zip(ys, predicted))
    r_squared = 1.0 if total == 0.0 and residual == 0.0 else 1.0 - residual / total
    coefficient = math.exp(intercept)
    return {
        "schema_version": 1,
        "fit": "loss = coefficient * axis ** exponent",
        "axis": axis,
        "coefficient": coefficient,
        "exponent": exponent,
        "r_squared_log_loss": r_squared,
        "r_squared_target": r_squared_target,
        "passes_r_squared_target": r_squared >= r_squared_target,
        "fixed_conditions": {
            "train_tokens": next(iter(tokens)),
            "dataset_revision": next(iter(datasets)),
            "code_revision": next(iter(code)),
        },
        "observations": [asdict(item) for item in observations],
        "limitations": [
            "This is an OLS descriptive fit in log-loss space, not a causal proof of a scaling law.",
            "A high R-squared does not validate extrapolation beyond the observed range.",
            "No result exists until supplied observations correspond to completed, hashed runs.",
        ],
    }


def assess_long_context_stability(
    observations: Sequence[LongContextObservation],
    target_context_tokens: int = ONE_MILLION_TOKENS,
    maximum_relative_loss_span: float = 0.05,
    minimum_observations: int = 3,
) -> Dict[str, Any]:
    """Apply a predeclared finite-loss/drift gate at one target context length.

    The gate is intentionally conservative: it requires at least three
    telemetry points from exactly one run/context, no non-finite loss, and
    ``(max(loss)-min(loss))/mean(loss) <= maximum_relative_loss_span``.
    Passing is operational stability evidence, not competitive quality.
    """
    if target_context_tokens < ONE_MILLION_TOKENS:
        raise ValueError("CKPT-003 target_context_tokens must be at least 1,000,000")
    if not 0.0 <= maximum_relative_loss_span < 1.0 or minimum_observations < 2:
        raise ValueError("invalid stability thresholds")
    selected = [item for item in observations if item.context_tokens == target_context_tokens]
    for item in selected:
        item.validate()
    runs = {item.run_id for item in selected}
    if len(runs) > 1:
        raise ValueError("assess one run at a time; aggregate runs separately")
    losses = [item.validation_loss for item in selected]
    enough = len(selected) >= minimum_observations
    mean_loss = sum(losses) / len(losses) if losses else None
    span = ((max(losses) - min(losses)) / mean_loss) if mean_loss else None
    stable = bool(enough and span is not None and span <= maximum_relative_loss_span)
    return {
        "schema_version": 1,
        "target_context_tokens": target_context_tokens,
        "minimum_observations": minimum_observations,
        "observed_count": len(selected),
        "maximum_relative_loss_span": maximum_relative_loss_span,
        "mean_validation_loss": mean_loss,
        "relative_loss_span": span,
        "finite_loss": all(math.isfinite(loss) for loss in losses),
        "stable": stable,
        "observations": [asdict(item) for item in selected],
        "limitations": [
            "An empty or undersampled trace is not stable by default.",
            "This gate does not establish retrieval quality, task accuracy, or fairness against RAG.",
            "Hardware OOM/crash receipts must be retained separately, including distributed topology.",
        ],
    }


__all__ = [
    "ONE_MILLION_TOKENS",
    "SCALING_R2_TARGET",
    "LongContextObservation",
    "ScalingObservation",
    "assess_long_context_stability",
    "fit_power_law",
]
