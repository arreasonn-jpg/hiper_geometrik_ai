# CKPT-001 foundation analysis

- Protocol: `hga-ckpt-001-foundation-analysis-v1` · seed: `1`
- This is a deterministic diagnostic grid plus an explicitly idealized hash model.

## Rank / collapse grid

| n | K | single-layer entropy dimension | linear collapse residual | SiLU linear-proxy residual |
|---:|---:|---:|---:|---:|
| 4 | 1 | 4.4542 | 2.482e-07 | 4.939e-01 |
| 4 | 2 | 4.4542 | 1.780e-07 | 5.300e-01 |
| 4 | 4 | 4.4542 | 2.546e-07 | 7.237e-01 |
| 8 | 1 | 19.4904 | 2.549e-07 | 5.149e-01 |
| 8 | 2 | 19.4904 | 2.685e-07 | 6.602e-01 |
| 8 | 4 | 19.4904 | 2.974e-07 | 6.963e-01 |
| 16 | 1 | 98.3799 | 4.975e-07 | 3.776e-01 |
| 16 | 2 | 98.3799 | 5.078e-07 | 4.338e-01 |
| 16 | 4 | 98.3799 | 5.158e-07 | 4.985e-01 |

## Gradient diagnostics

| n | K | activation | input grad norm | mean factor grad norm | finite |
|---:|---:|---|---:|---:|:--:|
| 4 | 1 | identity | 0.46174 | 1.18469 | PASS |
| 4 | 1 | silu | 0.189804 | 0.462262 | PASS |
| 4 | 2 | identity | 0.397326 | 1.52143 | PASS |
| 4 | 2 | silu | 0.0630382 | 0.14861 | PASS |
| 4 | 4 | identity | 1.47376 | 4.33318 | PASS |
| 4 | 4 | silu | 0.0259021 | 0.0449303 | PASS |
| 8 | 1 | identity | 0.264767 | 1.02183 | PASS |
| 8 | 1 | silu | 0.147223 | 0.514972 | PASS |
| 8 | 2 | identity | 0.368319 | 1.21587 | PASS |
| 8 | 2 | silu | 0.0664863 | 0.194256 | PASS |
| 8 | 4 | identity | 0.427749 | 1.69406 | PASS |
| 8 | 4 | silu | 0.0156326 | 0.0320189 | PASS |
| 16 | 1 | identity | 0.0989957 | 0.588714 | PASS |
| 16 | 1 | silu | 0.0495645 | 0.263575 | PASS |
| 16 | 2 | identity | 0.192051 | 1.13145 | PASS |
| 16 | 2 | silu | 0.0327487 | 0.154797 | PASS |
| 16 | 4 | identity | 0.368679 | 2.1743 | PASS |
| 16 | 4 | silu | 0.00683317 | 0.0288022 | PASS |

## Sparse-memory model

- Prehash-key upper bound: `2,147,483,648` classes (31 bits).
- Physical-signature upper bound after prehash: `1,048,576`.
- Ideal-model expected occupancy for `2,171` unique contexts: `2168.7551`; zero-collision probability: `0.105614`.

## Checks

- PASS — `all_linear_chains_collapsed`
- PASS — `all_gradient_observations_finite`
- PASS — `n16_entropy_effective_dimension_reproduced_when_in_grid`
- PASS — `memory_signature_bound_respects_prehash`

## Limitations

- The rank and gradient values are local seeded diagnostics, not a proof of optimization convergence or generalization.
- The nonlinear chain is fitted by a linear least-squares proxy only to test linear collapse; its spectrum is not a nonlinear function-class dimension.
- The hash formulas assume independent uniform signatures and do not prove properties of the implementation hash.
- No closed-form VC/pseudo-dimension theorem for the full trainable HGA chain is supplied by this report.
