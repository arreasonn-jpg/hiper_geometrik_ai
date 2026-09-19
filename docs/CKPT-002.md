# CKPT-002 — comparative evaluation (experimental)

**Release label:** `v0.3.0-experimental`  
**Status:** planning and measurement infrastructure only. No comparative score,
claim of superiority, or completion claim is made in this checkpoint.

## Evidence gate

A result row is eligible for release only when it contains immutable dataset and
code revisions, local SHA-256 artifact hashes, seeds, training budget, parameter
count, FLOP estimate, metric direction, mean, sample standard deviation, 95% CI,
effect sizes, raw p-value, corrected p-value, correction family, and the
baseline identity. Missing fields are a blocked result, not a zero or a pass.

The repository default of five seeds is useful for descriptive pilot work but
cannot produce a two-sided exact sign-flip p-value below 0.05: its smallest
possible p-value is 0.0625. Confirmatory comparisons therefore require a
pre-registered seed count and a power plan. `paired_power_plan` is only a
normal-approximation planning aid; its result must be sensitivity checked
before a run begins.

## Regimes and fairness

Every comparison is reported separately under **equal parameter count** and
**equal estimated training FLOPs**. A matching table records tokenizer,
sequence length, data split/revision, update count, hardware, precision,
optimizer schedule, wall clock, peak memory, trainable parameters, and FLOPs.
No row may be copied between regimes.

The baseline registry reserves adapters for parameter-matched BERT-base and
GPT-2 small; Switch Transformer, Product Key Memory, Compressive Transformer;
and RESCAL and DistMult where the task is relational. An adapter is not a
completed comparison: upstream revision, license, preprocessing and matching
record must be present before use.

## Dataset decision register

`docs/checkpoints/CKPT-002.json` is the canonical planning register. Dataset
records stay `external-input-required` until an authoritative release,
license, revision, split and checksum are recorded. This deliberately includes
Penn Treebank and the requested Turkish TurkBench/TR-MMLU sources: neither is
vendored or assumed open here. A Hugging Face dataset card or a secondary
redistribution is not by itself a project-wide license decision.

Planned coverage is GLUE/SuperGLUE, WikiText-103/Penn Treebank, XNLI/MLQA/TyDi
QA, TurkBench/TR-MMLU, and long-context 4K/16K/64K. The existing implementation
only has a 1024-token long-context pilot; it must not be represented as a 4K,
16K, or 64K test.

## Ablations

The planned full factorial is declared in the register: `K={1,2,4,8,16}`;
`n={64,128,256,512}`; sparse table/hash/double-hash/LRU/decay; SiLU/GELU/ReLU/
tanh; and LayerNorm/RMSNorm/Pre-LN/Post-LN. This is a release target, not a
claim that all cells have run. The ablation manifest supports splitting the
matrix into preregistered blocks, but each absent cell remains visible.

## Statistical policy

Use paired seed-level comparisons where pairing is real, bootstrap 95% CIs,
paired Cohen's `d_z`/Hedges `g`, Cliff's delta with its non-paired caveat, raw
p-values and either Holm/Bonferroni family-wise correction or BH-FDR. Report
losses symmetrically: HGA loses whenever the defined direction-adjusted
baseline comparison favours the baseline, irrespective of statistical
significance. Do not equate non-significance with equality.

## Exit criteria

The experimental label can only be reconsidered after results on at least
three English benchmarks, at least two baselines, statistical tests, and a
completed ablation matrix. Those are eventual gates, currently unmet.
