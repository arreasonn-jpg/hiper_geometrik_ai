# Uncertainty Calibration Protocol

## Scope

`dev-temperature-scaling-selective-risk-v1` measures whether the four
parameter-matched TWT neural classifiers' confidence values correspond to
observed correctness. It does **not** reinterpret Experience Engine's
`ScoreBreakdown.weighted`, source confidence, or rule confidence as a
probability. Those fields remain evidence/priority signals unless separately
calibrated against labels.

This protocol extends the existing `twt-parameter-matched-architectures-v1`
harness; it does not add a CLI command or alter the repository tokenizer.

## Split isolation

1. Models are trained on the fixed, leakage-audited TWT **train** candidates.
2. A single positive scalar temperature is selected on fixed **dev** logits by
   minimum binary negative log likelihood (NLL).
3. The temperature search is deterministic: 241 log-spaced values in
   `[0.1, 10.0]`, with the lower temperature winning an exact loss tie.
4. Fixed **test** labels are used only to measure the frozen temperature.
   Neither temperature nor an abstention threshold is fitted on test.
5. Positive scalar temperature scaling preserves class-logit ordering and thus
   every argmax prediction. A scientific check enforces this invariant.

The report names `fit_split=dev` and `evaluation_split=test`, stores the normal
TWT dataset/config/split/candidate hashes and is enclosed by the Research Suite
seed manifest (`git_commit`, config/dataset hashes, Python/Torch/CUDA,
hardware, parameter count and duration).

## Metrics

Before (`T=1`) and after (`T=T_dev`) values are reported on all test examples
and each existing entity-, relation-, composition-, wording-, sentence- and
seen-composition slice:

- **NLL**: binary negative log likelihood;
- **Brier**: mean squared error of the positive-class probability;
- **ECE**: 10 equal-width confidence bins;
- **adaptive ECE**: 10 deterministic approximately equal-count bins;
- mean max-class confidence and unchanged accuracy;
- **AURC**: area under the empirical selective risk-coverage curve;
- selective accuracy/risk and frozen confidence cutoffs at predeclared
  coverage targets `10%, 25%, 50%, 75%, 100%`.

The selective policy ranks test examples by calibrated max-class confidence.
Coverage targets are declared in code and no target is selected after looking
at test performance. These are diagnostic operating points, not a deployed
abstention threshold.

## Acceptance gates

For every architecture and seed:

- calibration fit uses dev only and test labels do not tune temperature or a
  threshold;
- temperature is positive and dev NLL cannot worsen relative to `T=1`;
- calibrated and uncalibrated argmax predictions are exactly identical;
- every declared test disjoint slice has calibration metrics;
- temperature, ECE/adaptive ECE, Brier, NLL, AURC and risk-coverage points are
  finite and report-visible.

Test ECE/NLL/Brier are **measurements**, not gates requiring improvement.
Selecting only improving test outcomes would itself leak test information.
Likewise, temperature scaling cannot repair task accuracy or bad ranking.

## Scientific limits

TWT is real Turkish text with human dependency annotations, but the task is a
structured binary dependency-arc verification benchmark rather than general
language generation. Calibration on its candidates does not establish
calibration under arbitrary domain shift, open-world facts, generated prose,
or deployed user traffic. ECE is bin-dependent, so fixed and adaptive variants
are both retained and NLL/Brier are reported alongside them.
