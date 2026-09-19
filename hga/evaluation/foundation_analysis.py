"""CKPT-001 analytic contracts and small deterministic diagnostic grids.

This module keeps three categories separate:

* exact algebraic facts about ``X -> A X B``;
* ideal-uniform hash models, explicitly labelled as models rather than facts
  about the implementation hash; and
* finite, seeded numerical diagnostics.

It is not a substitute for a general VC/pseudo-dimension theorem, a proof of
uniform hashing, or an optimization convergence theorem for the full HGA model.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .experiment import seed_everything
from .kronecker_rank import measure_chain_collapse, measure_single_layer_rank

HASH_KEY_BITS = 31


def _probability_no_collision(item_count: int, bucket_count: int) -> float:
    """Birthday-model probability of no collision, evaluated stably in log space."""
    if item_count < 0 or bucket_count < 1:
        raise ValueError("item_count >= 0 and bucket_count >= 1 are required")
    if item_count > bucket_count:
        return 0.0
    log_probability = math.fsum(
        math.log1p(-index / bucket_count) for index in range(item_count)
    )
    return math.exp(log_probability)


def sparse_memory_theory(
    *, vocabulary_size: int = 8_000, window_size: int = 16,
    slot_count: int = 1_048_576, table_count: int = 1,
    embedding_dimension: int = 32, queried_unique_contexts: int = 2_171,
    scalar_storage_bits: int = 32,
) -> Dict[str, Any]:
    """Return an auditable ideal hash/occupancy model for sparse memory.

    Args:
        vocabulary_size: Number of possible token IDs in a context position.
        window_size: Number of token IDs in one full context key.
        slot_count: Physical rows in each embedding table.
        table_count: Number of address tables used in the signature.
        embedding_dimension: Float components stored per physical row.
        queried_unique_contexts: Unique contexts considered by the occupancy
            model. Repeated contexts are intentionally not counted twice.
        scalar_storage_bits: Storage width used for a *storage* bit budget, not
            a Shannon-capacity theorem.

    Returns:
        Exact namespace/storage bounds plus collision formula values under a
        clearly named independent-uniform-address model.

    Notes:
        ``HashlenmisKureselTablo`` first produces a 31-bit key. Therefore a
        token namespace of ``vocabulary_size ** window_size`` is an input
        domain, not a collision-free address space. Its physical address
        functions share that key, so independent-table formulas are a useful
        reference model only and are not asserted for the implementation.
    """
    values = {
        "vocabulary_size": vocabulary_size,
        "window_size": window_size,
        "slot_count": slot_count,
        "table_count": table_count,
        "embedding_dimension": embedding_dimension,
        "queried_unique_contexts": queried_unique_contexts,
        "scalar_storage_bits": scalar_storage_bits,
    }
    if any(int(value) < 1 for value in values.values()):
        raise ValueError("All sparse-memory theory inputs must be positive")
    vocabulary_size = int(vocabulary_size)
    window_size = int(window_size)
    slot_count = int(slot_count)
    table_count = int(table_count)
    embedding_dimension = int(embedding_dimension)
    queried_unique_contexts = int(queried_unique_contexts)
    scalar_storage_bits = int(scalar_storage_bits)

    input_domain_log2 = window_size * math.log2(vocabulary_size)
    input_namespace_count = vocabulary_size ** window_size
    prehash_classes = 2 ** HASH_KEY_BITS
    raw_signature_space = slot_count ** table_count
    implementation_signature_upper_bound = min(
        input_namespace_count, prehash_classes, raw_signature_space
    )
    model_bucket_count = raw_signature_space
    occupied = model_bucket_count * (-math.expm1(
        queried_unique_contexts * math.log1p(-1.0 / model_bucket_count)
    ))
    expected_collisions = queried_unique_contexts - occupied
    no_collision = _probability_no_collision(queried_unique_contexts, model_bucket_count)
    isolation = math.exp(
        (queried_unique_contexts - 1) * math.log1p(-1.0 / model_bucket_count)
    )
    physical_rows = slot_count * table_count
    storage_bits = physical_rows * embedding_dimension * scalar_storage_bits

    return {
        "model": "independent-uniform-address-reference-v1",
        "assumptions": [
            "The selected unique contexts are independently and uniformly mapped to signature buckets.",
            "For table_count > 1, table addresses are treated as independent only in this reference model.",
            "The implementation's shared 31-bit prehash key is not proven uniform or independent.",
        ],
        "input_namespace": {
            "context_count_symbolic": "V^w",
            "context_count_upper_bound": input_namespace_count,
            "log2_context_count": round(input_domain_log2, 8),
            "context_count_decimal_order": round(input_domain_log2 / math.log2(10), 8),
        },
        "hash_and_signature_bounds": {
            "prehash_key_bits": HASH_KEY_BITS,
            "prehash_key_classes_upper_bound": prehash_classes,
            "raw_physical_signature_count_upper_bound": raw_signature_space,
            "implementation_signature_count_upper_bound": implementation_signature_upper_bound,
            "explanation": (
                "This is an upper bound on distinguishable address signatures, not on learned facts. "
                "It is bounded by the input namespace, the 31-bit key, and physical signatures."
            ),
        },
        "physical_storage": {
            "physical_rows": physical_rows,
            "embedding_components": physical_rows * embedding_dimension,
            "storage_bits_at_declared_scalar_width": storage_bits,
            "storage_bytes_at_declared_scalar_width": storage_bits // 8,
            "meaning": "Allocated floating-point storage, not an information-theoretic semantic capacity.",
        },
        "ideal_uniform_occupancy": {
            "queried_unique_contexts": queried_unique_contexts,
            "model_bucket_count": model_bucket_count,
            "expected_occupied_signatures": round(occupied, 8),
            "expected_colliding_contexts": round(expected_collisions, 8),
            "expected_occupancy_fraction": round(occupied / model_bucket_count, 12),
            "probability_of_zero_collision": round(no_collision, 12),
            "per_context_isolation_probability": round(isolation, 12),
            "recall_interpretation": (
                "Isolation is the probability that a queried context has no other context in its "
                "idealized address signature. It is not end-to-end retrieval accuracy."
            ),
        },
        "limitations": [
            "Observed implementation collision rates must be measured with carpisma_istatistigi; this model cannot certify them.",
            "An address signature does not guarantee recall: embeddings, optimization, interference, and query distribution also matter.",
            "V^w describes possible input strings. It does not bypass the implementation's 31-bit prehash bottleneck.",
        ],
    }


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional runtime
        raise ImportError("Foundation gradient diagnostics require PyTorch.") from error
    return torch


@dataclass(frozen=True)
class GradientDiagnostic:
    """One seeded local gradient-flow observation; it is not a landscape proof."""

    n: int
    K: int
    activation: Optional[str]
    loss: float
    input_gradient_norm: float
    mean_factor_gradient_norm: float
    max_factor_gradient_norm: float
    output_to_input_norm_ratio: float
    linear_spectral_norm_product: float
    all_gradients_finite: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def measure_gradient_flow(
    *, n: int, k: int, seed: int = 1,
    activation: Optional[str] = "silu", batch_size: int = 8,
) -> GradientDiagnostic:
    """Measure local gradient magnitudes for a seeded bilinear chain.

    For an activation-free chain, the reported spectral product is the exact
    induced Frobenius-norm operator norm. With SiLU it is only the product of
    linear components; nonlinear local derivatives and residuals are outside
    that number.
    """
    if n < 2 or k < 1 or batch_size < 1:
        raise ValueError("n >= 2, k >= 1 and batch_size >= 1 are required")
    if activation not in (None, "silu"):
        raise ValueError("activation must be None or 'silu'")
    torch = _torch()
    seed_everything(seed)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 91_337)
    inputs = torch.randn(batch_size, n, n, generator=generator, requires_grad=True)
    target = torch.randn(batch_size, n, n, generator=generator)
    factors: List[Tuple[Any, Any]] = []
    value = inputs
    spectral_product = 1.0
    for _ in range(k):
        left = torch.nn.Parameter(torch.empty(n, n))
        right = torch.nn.Parameter(torch.empty(n, n))
        torch.nn.init.xavier_uniform_(left, generator=generator)
        torch.nn.init.xavier_uniform_(right, generator=generator)
        factors.append((left, right))
        spectral_product *= float(torch.linalg.matrix_norm(left.detach(), ord=2).item())
        spectral_product *= float(torch.linalg.matrix_norm(right.detach(), ord=2).item())
        value = torch.matmul(left, torch.matmul(value, right))
        if activation == "silu":
            value = torch.nn.functional.silu(value)
    loss = torch.mean((value - target).square())
    loss.backward()
    gradients = [factor.grad for pair in factors for factor in pair]
    norms = [float(gradient.detach().norm().item()) for gradient in gradients if gradient is not None]
    finite = bool(torch.isfinite(inputs.grad).all()) if inputs.grad is not None else False
    finite = finite and all(bool(torch.isfinite(gradient).all()) for gradient in gradients if gradient is not None)
    input_norm = float(inputs.detach().norm().item())
    output_norm = float(value.detach().norm().item())
    return GradientDiagnostic(
        n=int(n), K=int(k), activation=activation, loss=float(loss.detach().item()),
        input_gradient_norm=float(inputs.grad.detach().norm().item()) if inputs.grad is not None else float("nan"),
        mean_factor_gradient_norm=math.fsum(norms) / len(norms) if norms else float("nan"),
        max_factor_gradient_norm=max(norms) if norms else float("nan"),
        output_to_input_norm_ratio=output_norm / input_norm if input_norm else float("nan"),
        linear_spectral_norm_product=float(spectral_product), all_gradients_finite=finite,
    )


@dataclass
class FoundationAnalysisReport:
    """Machine-readable CKPT-001 theory/diagnostic report."""

    protocol: str
    seed: int
    n_values: List[int]
    k_values: List[int]
    rank_grid: List[Dict[str, Any]]
    gradient_grid: List[GradientDiagnostic]
    sparse_memory: Dict[str, Any]
    checks: Dict[str, bool]
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol,
            "seed": self.seed,
            "n_values": self.n_values,
            "k_values": self.k_values,
            "rank_grid": self.rank_grid,
            "gradient_grid": [row.to_dict() for row in self.gradient_grid],
            "sparse_memory": self.sparse_memory,
            "checks": self.checks,
            "limitations": self.limitations,
        }


def run_foundation_analysis(
    *, n_values: Sequence[int] = (4, 8, 16), k_values: Sequence[int] = (1, 2, 4),
    seed: int = 1, samples: int = 512,
) -> FoundationAnalysisReport:
    """Run the deterministic CKPT-001 diagnostic grid and analytic memory model."""
    normalized_n = [int(value) for value in n_values]
    normalized_k = [int(value) for value in k_values]
    if not normalized_n or not normalized_k:
        raise ValueError("n_values and k_values must not be empty")
    # A linear least-squares proxy needs at least n² independent input rows;
    # otherwise it can interpolate a nonlinear sample and falsely suggest collapse.
    if int(samples) < max(value * value for value in normalized_n):
        raise ValueError("samples must be at least max(n_values)^2 for collapse diagnostics")
    rank_grid: List[Dict[str, Any]] = []
    gradients: List[GradientDiagnostic] = []
    for n in normalized_n:
        single = measure_single_layer_rank(n=n, seed=seed, trained=False)
        for k in normalized_k:
            linear = measure_chain_collapse(
                n=n, k=k, seed=seed, activation=None, samples=samples
            )
            nonlinear = measure_chain_collapse(
                n=n, k=k, seed=seed, activation="silu", samples=samples
            )
            rank_grid.append({
                "n": n,
                "K": k,
                "single_layer_spectrum": single.to_dict(),
                "linear_chain": linear.to_dict(),
                "silu_chain": nonlinear.to_dict(),
            })
            gradients.append(measure_gradient_flow(n=n, k=k, seed=seed, activation=None))
            gradients.append(measure_gradient_flow(n=n, k=k, seed=seed, activation="silu"))
    memory = sparse_memory_theory()
    checks = {
        "all_linear_chains_collapsed": all(
            bool(row["linear_chain"]["collapsed"]) for row in rank_grid
        ),
        "all_gradient_observations_finite": all(row.all_gradients_finite for row in gradients),
        "n16_entropy_effective_dimension_reproduced_when_in_grid": (
            True if 16 not in normalized_n else any(
                row["n"] == 16 and abs(
                    float(row["single_layer_spectrum"]["measured"]["entropy_effective_dimension"])
                    - 98.379852
                ) < 1e-5 for row in rank_grid
            )
        ),
        "memory_signature_bound_respects_prehash": (
            int(memory["hash_and_signature_bounds"]["implementation_signature_count_upper_bound"])
            <= 2 ** HASH_KEY_BITS
        ),
    }
    return FoundationAnalysisReport(
        protocol="hga-ckpt-001-foundation-analysis-v1", seed=int(seed),
        n_values=normalized_n, k_values=normalized_k, rank_grid=rank_grid,
        gradient_grid=gradients, sparse_memory=memory, checks=checks,
        limitations=[
            "The rank and gradient values are local seeded diagnostics, not a proof of optimization convergence or generalization.",
            "The nonlinear chain is fitted by a linear least-squares proxy only to test linear collapse; its spectrum is not a nonlinear function-class dimension.",
            "The hash formulas assume independent uniform signatures and do not prove properties of the implementation hash.",
            "No closed-form VC/pseudo-dimension theorem for the full trainable HGA chain is supplied by this report.",
        ],
    )


def foundation_analysis_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise human-readable companion for :func:`run_foundation_analysis`."""
    lines = [
        "# CKPT-001 foundation analysis", "",
        f"- Protocol: `{report['protocol']}` · seed: `{report['seed']}`",
        "- This is a deterministic diagnostic grid plus an explicitly idealized hash model.", "",
        "## Rank / collapse grid", "",
        "| n | K | single-layer entropy dimension | linear collapse residual | SiLU linear-proxy residual |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in report["rank_grid"]:
        spectrum = row["single_layer_spectrum"]["measured"]
        lines.append(
            f"| {row['n']} | {row['K']} | {spectrum['entropy_effective_dimension']:.4f} | "
            f"{row['linear_chain']['collapse_residual']:.3e} | "
            f"{row['silu_chain']['collapse_residual']:.3e} |"
        )
    lines.extend(["", "## Gradient diagnostics", "",
                  "| n | K | activation | input grad norm | mean factor grad norm | finite |",
                  "|---:|---:|---|---:|---:|:--:|"])
    for row in report["gradient_grid"]:
        lines.append(
            f"| {row['n']} | {row['K']} | {row['activation'] or 'identity'} | "
            f"{row['input_gradient_norm']:.6g} | {row['mean_factor_gradient_norm']:.6g} | "
            f"{'PASS' if row['all_gradients_finite'] else 'FAIL'} |"
        )
    memory = report["sparse_memory"]
    occupancy = memory["ideal_uniform_occupancy"]
    bounds = memory["hash_and_signature_bounds"]
    lines.extend([
        "", "## Sparse-memory model", "",
        f"- Prehash-key upper bound: `{bounds['prehash_key_classes_upper_bound']:,}` classes ({HASH_KEY_BITS} bits).",
        f"- Physical-signature upper bound after prehash: `{bounds['implementation_signature_count_upper_bound']:,}`.",
        f"- Ideal-model expected occupancy for `{occupancy['queried_unique_contexts']:,}` unique contexts: "
        f"`{occupancy['expected_occupied_signatures']:.4f}`; zero-collision probability: "
        f"`{occupancy['probability_of_zero_collision']:.6g}`.",
        "", "## Checks", "",
    ])
    lines.extend(
        f"- {'PASS' if value else 'FAIL'} — `{key}`" for key, value in report["checks"].items()
    )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


__all__ = [
    "HASH_KEY_BITS", "GradientDiagnostic", "FoundationAnalysisReport",
    "foundation_analysis_markdown", "measure_gradient_flow", "run_foundation_analysis",
    "sparse_memory_theory",
]
