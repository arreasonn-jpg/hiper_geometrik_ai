# English UD EWT Architecture Baselines

- Protocol: `hga-english-ud-ewt-architecture-baselines-v1` · profile: `smoke` · seeds: `[1, 2, 3, 4, 5]`
- Dataset hash: `2f852e91da70301e5a572f717b3e9d6998d14ca6bf38daa8bb77f4baa5b3ddd3` · config hash: `b11ef05c7711182c0e3b8fa21921972a4ff965290ef4525591b1e6cad746b969`
- Official-source selected sentences: train/dev/test = `512/128/128`

| Model | Params | Test accuracy (mean ± std) | Test F1 (mean ± std) | FAR | FRR |
|---|---:|---:|---:|---:|---:|
| dense | 102,116 | 0.9572 ± 0.0144 | 0.9575 ± 0.0143 | 0.0501 | 0.0355 |
| transformer | 96,938 | 0.9212 ± 0.0042 | 0.9209 ± 0.0045 | 0.0750 | 0.0826 |
| bert_style | 97,586 | 0.9370 ± 0.0190 | 0.9357 ± 0.0202 | 0.0489 | 0.0772 |
| gpt_style | 96,938 | 0.9448 ± 0.0182 | 0.9456 ± 0.0189 | 0.0773 | 0.0332 |

## Integrity checks

- PASS — `official_english_ud_sources_hash_verified`
- PASS — `all_four_baseline_families_present`
- PASS — `all_models_trained_from_scratch_and_reported`
- PASS — `parameter_budget_within_twelve_percent`
- PASS — `five_or_more_seeds`
- PASS — `train_dev_test_candidate_hashes_distinct`

## Limitations

- This is English-only; together with the separate Turkish TWT protocol it is bilingual evidence, not a multilingual foundation-model evaluation.
- The task is basic UD dependency-arc verification, not end-to-end parsing, language modelling, semantic relation extraction or general reasoning.
- BERT-style and GPT-style arms are small random-initialized architectural baselines trained from scratch, not pretrained BERT/GPT checkpoints.
- The deterministic source-verified sentence cap controls CPU cost; it is not the full EWT corpus or a SOTA evaluation.
