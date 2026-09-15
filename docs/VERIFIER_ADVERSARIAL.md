# HGA Verifier Adversarial Suite v1

- Dataset hash: `77cf912ee0dd3d53b938b41c9e10be89b6d78618ae7b99bafea39a56e82a0db7`
- Accuracy: `1.000`
- FAR: `0.000`
- FRR: `0.000`
- Coverage: `0.917`
- Robustness: `1.000`

| Saldırı sınıfı | N | Doğru | Accuracy |
|---|---:|---:|---:|
| adversarial_input | 3 | 3 | 1.000 |
| boundary_case | 2 | 2 | 1.000 |
| contradictory_proof | 1 | 1 | 1.000 |
| false_proof | 1 | 1 | 1.000 |
| incomplete_proof | 1 | 1 | 1.000 |
| malformed_proof | 2 | 2 | 1.000 |
| unsupported_rule | 1 | 1 | 1.000 |
| valid_control | 1 | 1 | 1.000 |

## Sınırlar

- Bu suite yalnız katı şemalı tam sayı toplama ispatlarını doğrular; genel theorem prover değildir.
- Unsupported rule UNCERTAIN döner ve coverage hesabında açıkça görünür.
- Saldırılar elle sabitlenmiş regresyon vakalarıdır; fuzzing veya formal verification yerine geçmez.
