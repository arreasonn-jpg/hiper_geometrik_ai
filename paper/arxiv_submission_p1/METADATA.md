# arXiv Submission Metadata — Preprint 1

## Title
Hybrid Symbolic-Neural Paradigm: A Universal Formula for Predicting Gain over Neural Baselines

## Authors
Erdem Esa (Independent Researcher)

## Abstract
We present a hybrid symbolic-neural paradigm whose gain over neural
baselines can be predicted by a simple formula:
gain = cov_sym * (acc_sym - acc_neu). The formula was validated on
12 independent datasets across 8 language families, 4 task types
(binary, multiclass, multi-label, NER), and 2 metrics (accuracy,
bit-accuracy). Mean prediction error is 0.002. We derive the formula
exactly and show that its error is bounded by
(1 - cov) * |acc_neu^abst - acc_neu^all|. Under symbolic noise
(0--50%) the prediction error remains invariant at 0.0066, indicating
that the linear identity captures a structural property of the hybrid
ensemble.

## Primary Category
cs.CL (Computation and Language)

## Cross-list Categories
- cs.LG (Machine Learning)
- cs.AI (Artificial Intelligence)

## Comments
6 pages, 4 tables, 8 references. First in a series of three.
Companion papers:
- Preprint 2: Task-Appropriate Hybrid Architectures
- Preprint 3: Implicit Regularization in Kronecker Architectures

## Keywords
symbolic-neural hybrid, gain formula, cross-lingual validation,
macro F1, task generality

## Submission Files
- main.tex (English, primary)
- main.pdf (compiled)
- main_tr_supplement.tex (Turkish, supplemental)
- main_tr_supplement.pdf (compiled)

## Repository
https://github.com/arreasonn-jpg/hiper_geometrik_ai
Tag: supervisor-package-v10

## License
CC-BY-4.0
