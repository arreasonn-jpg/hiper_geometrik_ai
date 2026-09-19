# HGA English EWT parameter-scaling probe

- Status: **EXPLORATORY_NOT_A_SCALING_LAW**
- Seeds: `[1, 2, 3, 4, 5]`
- Parameter range: `3.28×`
- Descriptive log(params) → log(cross-entropy) slope: `-1.38898153`

| Size | Parameters | Test F1 (mean ± std) | Test cross-entropy (mean ± std) |
|---|---:|---:|---:|
| small | 47,487 | 0.9134 ± 0.0129 | 0.3097 ± 0.0180 |
| base | 99,483 | 0.9623 ± 0.0222 | 0.1236 ± 0.0590 |
| large | 155,559 | 0.9750 ± 0.0215 | 0.0585 ± 0.0378 |

## Guard rails

- PASS — `same_pinned_english_task_all_sizes`
- PASS — `same_seed_set_all_sizes`
- PASS — `at_least_three_sizes`
- PASS — `at_least_two_seeds`
- PASS — `parameter_count_strictly_increases`
- PASS — `scaling_law_claim_blocked_by_narrow_range`

## Why this is not a scaling law

- Three small configurations over less than two orders of magnitude cannot identify a scaling law.
- Training compute, token count and model size are not independently swept; the fitted log slope is descriptive only.
- This controlled dependency-arc task is not next-token language-model scaling and cannot predict foundation-model behaviour.
- A scaling-law claim requires a much wider compute/data/model grid, repeated convergence runs and held-out extrapolation.
