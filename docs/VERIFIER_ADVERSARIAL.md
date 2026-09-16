# HGA Verifier Adversarial Suite

- Benchmark: `hga-verifier-adversarial-v2`
- Dataset hash: `ebf5bb4c5f0953655397db629a047a6c1bd607eb89aabe61870ab19477716b9c`
- Accuracy: `1.000`
- FAR: `0.000`
- FRR: `0.000`
- Coverage: `0.903`
- Robustness: `1.000`

| Saldırı sınıfı | N | Doğru | Accuracy |
|---|---:|---:|---:|
| adversarial_input | 6 | 6 | 1.000 |
| boundary_case | 4 | 4 | 1.000 |
| contradictory_proof | 2 | 2 | 1.000 |
| false_proof | 5 | 5 | 1.000 |
| incomplete_proof | 2 | 2 | 1.000 |
| malformed_proof | 7 | 7 | 1.000 |
| unsupported_rule | 2 | 2 | 1.000 |
| valid_control | 3 | 3 | 1.000 |

## Kabul kapıları

| kapı | sonuç |
|---|---|
| required_attack_classes_present | GEÇTİ |
| minimum_30_cases | GEÇTİ |
| zero_false_acceptance | GEÇTİ |
| zero_false_rejection | GEÇTİ |
| robustness_at_least_1_0 | GEÇTİ |
| unsupported_rules_not_accepted | GEÇTİ |

## Sınırlar

- Bu suite yalnız katı şemalı tam sayı toplama ispatlarını doğrular; genel theorem prover değildir.
- Unsupported operator/rule UNCERTAIN döner ve coverage hesabında açıkça görünür.
- v2 saldırıları elle sabitlenmiş regresyon vakalarıdır; fuzzing veya formal verification yerine geçmez.
- Toplama dışındaki matematiksel doğruluk bilinçli olarak başarı sayılmaz; alan dışı kalır.
