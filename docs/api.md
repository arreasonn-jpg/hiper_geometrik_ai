# API contract

The project is in active research development. The public functions below are
versioned protocol entry points; internal model details can change only with
corresponding manifest/test updates.

## Reproduction

```python
from hga.evaluation import run_reproduce_all

receipt = run_reproduce_all(
    root="artifacts/reproduce-all",
    seeds=(1, 2, 3, 4, 5),
    profile="smoke",
)
```

`receipt` names an atomic `EXP-NNNN` directory and a deterministic value
fingerprint. It excludes timestamps, wall-clock durations and hardware
inventory by design. Command-line equivalent:

```bash
python -m hga reproduce-all --artifact-root artifacts/reproduce-all
```

## Foundation theory diagnostics

```python
from hga.evaluation import run_foundation_analysis, sparse_memory_theory

analysis = run_foundation_analysis(n_values=(4, 8, 16), k_values=(1, 2, 4))
memory_model = sparse_memory_theory()
```

`run_foundation_analysis` reports seeded finite rank/collapse and gradient
observations. `sparse_memory_theory` returns an ideal-uniform reference model;
it is not a proof that the production hash is uniform or that retrieval is
perfect.

CLI equivalent:

```bash
python -m hga foundation-analysis --n-values 4,8,16 --k-values 1,2,4
```

## Core experiment protocol

```python
from hga.evaluation.experiment import run_seed_sweep, seed_everything
```

`run_seed_sweep` creates one atomic manifest directory per seed. `seed_everything`
seeds Python/NumPy/PyTorch and turns on strict deterministic PyTorch controls.
For Python hash seeding and CUDA cuBLAS determinism, use the Make/Docker launch
paths so process-start environment variables are present.

## English EWT controls

```python
from hga.evaluation import run_english_ewt_baselines, run_english_hga_scaling_probe
```

The baseline functions use pinned UD English EWT sources and report small
scratch-trained architecture controls. BERT-style/GPT-style are not pretrained
checkpoint comparisons. The scaling probe deliberately returns
`EXPLORATORY_NOT_A_SCALING_LAW`.
