# CKPT-001 theory scope and review map

The full technical draft is [`paper/technical_report.tex`](../paper/technical_report.tex).
This short map identifies which statements are proofs, which are executable
measurements, and which remain open.

| Topic | CKPT-001 status | Evidence / assumptions |
|---|---|---|
| `vec(A X B) = (B^T ⊗ A) vec(X)` | proved | finite-dimensional matrix identity |
| One-layer rank identity | proved | `rank(B^T ⊗ A)=rank(A)rank(B)` |
| One-layer generic Kronecker manifold dimension | framed | nonzero generic stratum, reciprocal scaling quotient; singular strata excluded |
| Activation-free chain collapse | proved | ordinary associative matrix multiplication, no norm/residual/activation between layers |
| SiLU/LayerNorm/residual HGA collapse | not applicable | nonlinear full model does not satisfy the linear-collapse hypothesis |
| Full HGA VC/pseudo-dimension | `NOT_ESTABLISHED` | no parameter-count substitution is allowed |
| n=16 effective spectral dimension `98.379852` | seeded measurement | fixed random initialization, float precision, spectrum convention |
| Gradient health | seeded local diagnostic | finite n×K grid; not global convergence/Hessian proof |
| Hash occupancy/collision probability | ideal reference model | independent-uniform signatures; implementation hash uniformity not proved |
| Sparse-memory observed collisions | measurable | run `carpisma_istatistigi` on actual unique context windows |
| Information capacity / recall | separated accounting | input namespace, prehash/signature bound, physical storage and end-to-end recall differ |

## Review questions for an independent mathematician

1. Are vectorization conventions and factor order explicit and consistent?
2. Is the `2n²−1` generic-dimension claim restricted to the nonzero quotient
   stratum as written?
3. Are all uses of rank distinguished from algebraic dimension and from a
   statistical capacity measure?
4. Does the gradient section avoid treating local norm identities as a global
   non-convex optimization theorem?
5. Does the memory section avoid assuming independent uniform hashes for the
   actual 31-bit prehash implementation?

A reader may approve, correct, or reject this draft only through an external
review record. This checklist does not claim that such a review has occurred.
