# Çoklu Verifier Ensemble Benchmarkı

- Protokol: `multi_verifier_ensemble_v1`
- Dataset: `hga-verifier-adversarial-v2` · hash: `ebf5bb4c5f09`
- Ensemble hash: `1c759b597ab7`
- Politika: `unanimous_accept_any_invalid_reject_uncertain_preserved`
- Üyeler: strict_schema_oracle, normal_form_oracle, trace_replay_oracle
- Accuracy: `1.000`
- FAR: `0.000` · FRR: `0.000`
- Coverage: `0.903` · Robustness: `1.000`
- Üye anlaşmazlık oranı: `0.129`

## Üye metrikleri

| üye | accuracy | FAR | FRR | coverage | robustness |
|---|---:|---:|---:|---:|---:|
| strict_schema_oracle | 1.000 | 0.000 | 0.000 | 0.903 | 1.000 |
| normal_form_oracle | 0.871 | 0.154 | 0.000 | 0.903 | 0.833 |
| trace_replay_oracle | 1.000 | 0.000 | 0.000 | 0.903 | 1.000 |

## Saldırı sınıfları

| sınıf | N | doğru | accuracy | disagreement |
|---|---:|---:|---:|---:|
| adversarial_input | 6 | 6 | 1.000 | 0.000 |
| boundary_case | 4 | 4 | 1.000 | 0.000 |
| contradictory_proof | 2 | 2 | 1.000 | 0.500 |
| false_proof | 5 | 5 | 1.000 | 0.400 |
| incomplete_proof | 2 | 2 | 1.000 | 0.000 |
| malformed_proof | 7 | 7 | 1.000 | 0.143 |
| unsupported_rule | 2 | 2 | 1.000 | 0.000 |
| valid_control | 3 | 3 | 1.000 | 0.000 |

## Kabul kapıları

| kapı | sonuç |
|---|---|
| member_count_at_least_3 | GEÇTİ |
| all_cases_have_all_votes | GEÇTİ |
| zero_false_acceptance | GEÇTİ |
| zero_false_rejection | GEÇTİ |
| robustness_at_least_1_0 | GEÇTİ |
| ensemble_accuracy_at_least_best_member | GEÇTİ |
| unsupported_rule_not_forced_verified | GEÇTİ |

## Sınırlar

- Bu suite yalnız katı şemalı tam sayı toplama ispatlarını doğrular; genel theorem prover değildir.
- Unsupported operator/rule UNCERTAIN döner ve coverage hesabında açıkça görünür.
- v2 saldırıları elle sabitlenmiş regresyon vakalarıdır; fuzzing veya formal verification yerine geçmez.
- Toplama dışındaki matematiksel doğruluk bilinçli olarak başarı sayılmaz; alan dışı kalır.
- Ensemble yalnız aynı kapalı aritmetik proof sözleşmesini üç ayrı denetimle ölçer; genel theorem prover değildir.
- Oybirliği politikası false accept riskini azaltır ama desteklenmeyen doğru ispatları UNCERTAIN bırakabilir.
