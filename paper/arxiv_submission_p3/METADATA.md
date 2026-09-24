# arXiv Submission Metadata — Preprint 3

## Title
Implicit Regularization in Kronecker Architectures: When Structural Constraints Help

## Authors
Erdem Esa (Independent Researcher)

## Abstract
The value of Kronecker architectures is conventionally explained by
structural alignment: parameter efficiency when the task's functional
dependencies match tensor factorisation. We show that this explanation
is wrong. Across six tasks, autocorrelation and Kronecker win rate
correlate negatively (r = -0.38). We propose instead that the Kronecker
parameter constraint (2n^2 parameters for an n^2 x n^2 operator) acts
as an implicit regulariser. Across five noise levels and 50 seeds per
point, noise predicts Kronecker's advantage with r = +0.956, p = 0.011.
Grid structure is necessary but not sufficient: on a graph task the
advantage disappears entirely, and on an exact bilinear task
(Y = AXB --- the very form Kronecker represents natively) Flat
Kronecker wins only 1/50 seeds. The bottleneck is optimisation, not
expressivity. Our findings suggest an architectural design principle:
match the parameter budget to the regularisation needs of the task.

## Primary Category
cs.LG (Machine Learning)

## Cross-list Categories
- cs.NE (Neural and Evolutionary Computing)
- cs.AI (Artificial Intelligence)
- stat.ML (Machine Learning - Statistics)

## Comments
9 pages, 4 tables, 8 references. Third in a series.
Preprint 1: arXiv:2026.XXXXX (hybrid gain formula).
Preprint 2: arXiv:2026.XXXXX (unified theory).

## Keywords
implicit regularization, Kronecker factorization, architectural
constraints, overfitting, task-appropriateness

## Submission Files
- main.tex (English, primary)
- main.pdf (compiled)
- main_tr_supplement.tex (Turkish, supplemental)
- main_tr_supplement.pdf (compiled)

## Repository
https://github.com/arreasonn-jpg/hiper_geometrik_ai
Tag: preprint3-latex-v2

## License
CC-BY-4.0
