# Contributing and independent reproduction

HGA is a research prototype. Contributions are welcome, but no contributor is
implicitly added as an author and no benchmark result may be described as a
publication result without the project owner's approval. Participation is
subject to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Before proposing a change

1. Read [`docs/CKPT-000.md`](docs/CKPT-000.md) and preserve the distinction
   between physical parameters, interaction upper bounds, and memory-address
   upper bounds.
2. Do not replace a limitation with an unmeasured claim. New benchmarks need a
   task definition, split/leakage audit, provenance/licensing, seeds, and a
   machine-readable report.
3. Do not add downloaded model weights, credentials, private data, or generated
   experiment directories to Git.
4. Keep upstream revisions and SHA-256 source hashes pinned for redistributed
   data.

## Reproduction minimum

```bash
pip install -r requirements-lock.txt
pytest
python test_mimari.py
python -m hga research-benchmark
python -m hga english-ewt --seeds 1,2,3,4,5
```

A Docker reference environment is documented in `docs/CKPT-000.md`. Reports
must be written outside the image to a mounted artifact directory.

## Code and documentation standard

New or materially changed public Python functions should use complete type hints
and Google-style docstrings (`Args`, `Returns`, `Raises`, `Notes` where useful).
Keep new source formatted for the configured Ruff import/order rules. Build the
public documentation locally with `make docs`; the MkDocs configuration is a
site/navigation contract, not a substitute for API tests.

## Research-change checklist

- Add/extend tests that fail before the implementation change.
- Record every stochastic run with a seed and `EXP-NNNN` manifest.
- Report mean ± population standard deviation; do not promote one seed to a
  comparative conclusion.
- State whether a BERT/GPT arm is pretrained or merely architecture-style.
- State whether an observed parameter sweep is a descriptive probe or a
  validated scaling-law fit.
- Request an independent reviewer for theoretical or broad-performance claims.

This guide lowers a single-maintainer bottleneck through reproducible review
interfaces; it does not pretend that the project already has a multi-person
research team.
