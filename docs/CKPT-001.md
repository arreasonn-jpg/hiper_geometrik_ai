# CKPT-001 — v0.2.0-foundation draft

**Goal:** a one-command reproducible, theory-framed, CI-protected foundation.
**Machine-readable status:** [`checkpoints/CKPT-001.json`](checkpoints/CKPT-001.json).
`v0.2.0-foundation` is a research-checkpoint label, not a retroactive change
to the package version declared in `pyproject.toml`.

This checkpoint follows CKPT-000 without rewriting its claim boundaries. In
particular, a structured Kronecker operator, a physical parameter count, a
hash address namespace, a benchmark score, and a VC/pseudo-dimension are still
different quantities.

## Repository-complete foundation

| Deliverable | Evidence |
|---|---|
| One-command CPU reproduction bundle | [`Makefile`](../Makefile), [`REPRODUCE_ALL.md`](REPRODUCE_ALL.md), `python -m hga reproduce-all` |
| Atomic manifests and deterministic controls | [`experiment.py`](../hga/evaluation/experiment.py) |
| CUDA-ready plus CPU-fallback Docker recipes | [`Dockerfile.cuda`](../Dockerfile.cuda), [`Dockerfile`](../Dockerfile) |
| Pinned Conda entry point | [`environment.yml`](../environment.yml), [`requirements-lock.txt`](../requirements-lock.txt) |
| Formal/differential/hash analysis draft | [`paper/technical_report.tex`](../paper/technical_report.tex) |
| Executable n×K rank/gradient/hash diagnostics | [`foundation_analysis.py`](../hga/evaluation/foundation_analysis.py), curated [`FOUNDATION_ANALYSIS.md`](FOUNDATION_ANALYSIS.md) |
| MkDocs API/protocol site | [`../mkdocs.yml`](../mkdocs.yml) |
| Community/review interfaces | [`../CONTRIBUTING.md`](../CONTRIBUTING.md), [`../CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md), `.github/` templates |

## Determinism contract

Every seed runner calls `seed_everything(seed)`, which seeds Python, NumPy and
PyTorch, seeds all CUDA devices if available, disables cuDNN benchmarking, and
enables strict deterministic Torch algorithms. The launch interfaces also set
`PYTHONHASHSEED=0` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` **before** Python/Torch
initializes. The manifest records effective controls and runtime inventory.

No false cross-hardware promise is made: unsupported deterministic CUDA kernels
fail, while legal CPU/GPU implementations can still differ in floating-point
rounding. The bundle comparison target is the documented value fingerprint and
its seed-level values, not timestamps or elapsed time.

## Theory boundaries

The technical report proves the exact linear function class

\[
\mathcal{K}_n=\{X\mapsto AXB:A,B\in\mathbb{R}^{n\times n}\}
\]

and its vectorized Kronecker form. It also proves linear-chain collapse,
dimension/rank distinctions, and local derivative identities. It does **not**
claim a closed-form full-HGA VC/pseudo-dimension result, uniformity of the
implementation hash, a global non-convex optimization theorem, or an
empirical scaling law. The executable grid reports the prior n=16 entropy
spectral effective-dimension value near `98.379852` under its named seeded
initialization, not as a universal architectural constant.

For sparse memory, `V^w` is an input namespace. The actual implementation
reduces it to a 31-bit prehash key before physical table addressing; any claim
that all `V^w` contexts are collision-free is prohibited.

## External validation gates — intentionally not fabricated

The following conditions require a person/service outside this repository and
must not be marked completed by source code:

- a separate machine successfully running the Docker bundle;
- a green GitHub Actions run on the pushed commit;
- public GHCR package visibility at
  `ghcr.io/arreasonn-jpg/hiper_geometrik_ai:latest`;
- independent mathematical reading and author approval of the LaTeX draft.

The repository supplies a GHCR workflow and a reviewable report, but at this
moment its truthful checkpoint status remains
`artifact-complete-external-validation-pending`. After authorized humans
validate these four items, they may update a new checkpoint/release record;
they should not rewrite this one retroactively.
