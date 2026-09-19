# CKPT-001 foundation reproduction

`make reproduce` is the canonical one-command **foundation bundle**. It does
not silently mean every historical exploratory command in the repository. The
bundle runs, under one seed list and artifact receipt:

1. the manifest-backed Research Benchmark Suite (including TWT and EWT);
2. English EWT dense/Transformer/BERT-style/GPT-style controls;
3. the real-HGA English parameter probe, retaining its non-scaling-law guard;
4. the rank/collapse/gradient diagnostic grid and sparse-memory reference model.

## Local CPU run

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps .
make reproduce
```

The default is `SEEDS=1,2,3,4,5`, `PROFILE=smoke`, and writes a new atomic
`artifacts/reproduce-all/EXP-NNNN/` directory. Its top-level
`reproduce_summary.json`, `manifest.json`, source/config hashes, per-component
reports, Markdown companions, and `REPRODUCE.md` receipt must be kept together.

For a non-default output location:

```bash
make reproduce ARTIFACT_ROOT=/absolute/path/to/artifacts PROFILE=full
```

At least two seeds are required by the scaling probe. Fewer than five makes the
bundle explicitly `COMPLETED_WITH_LIMITATIONS`, not a five-seed result.

## Docker CPU fallback

```bash
docker build -t hga:ckpt-001-cpu -f Dockerfile .
mkdir -p artifacts
docker run --rm \
  -v "$PWD/artifacts:/opt/hga/artifacts" \
  hga:ckpt-001-cpu make reproduce
```

The CPU image is the portable reference path. PyTorch's pinned package can
contain CUDA support libraries, but no GPU is assumed or required.

## CUDA reference image

```bash
docker build -t hga:ckpt-001-cuda -f Dockerfile.cuda .
docker run --rm --gpus all \
  -v "$PWD/artifacts:/opt/hga/artifacts" \
  hga:ckpt-001-cuda make reproduce
```

`Dockerfile.cuda` pins its CUDA/cuDNN base image and the same Python lock.
A compatible NVIDIA driver/container runtime is an external host requirement.
If a CUDA operation has no deterministic PyTorch implementation, strict
Torch determinism turns it into a visible failure rather than silently
producing a non-reproducible result.

## Conda

```bash
conda env create -f environment.yml
conda activate hga-ckpt-001
make reproduce
```

`environment.yml` pins CPython/pip and delegates Python packages to the same
`requirements-lock.txt` used by Docker. The Docker CPU route is the supported
fallback when Conda or GPU driver resolution differs by host.

## What “same” means

Compare `deterministic_value_fingerprint`, dataset/config hashes, seed lists,
and component checks. Timestamps, absolute paths, `EXP-NNNN` identifiers,
hardware inventory, durations, and potentially cross-device floating-point
last bits are intentionally excluded from that fingerprint. Identical wall
clock logs or byte-identical GPU values are not promised.

The bundle cannot by itself prove that it was executed on a second machine,
that GitHub Actions is currently green, that a GHCR image is public, or that a
mathematician approved the technical report. Those are documented external
validation gates in [`CKPT-001.md`](CKPT-001.md).
