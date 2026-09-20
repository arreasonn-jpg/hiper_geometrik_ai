# TWT Karşılaştırma — Flat vs Hierarchical HGA

- Seeds: [1]
- Profile: smoke
- Train/Test: 127602/16036
- Vocab: 17582

## Özet

| Model | Params | Test Acc | Test F1 |
|---|---:|---:|---:|
| flat_hga | 291,827 | 0.8038 ± 0.0000 | 0.7922 ± 0.0000 |
| hiyerarsik_eq | 289,415 | 0.7962 ± 0.0000 | 0.7952 ± 0.0000 |
| hiyerarsik_rich | 294,583 | 0.8363 ± 0.0000 | 0.8348 ± 0.0000 |

## Yorum

❌ Flat kazanıyor: 0.8038 vs 0.7962

✅ Zengin hiyerarşik de kazanıyor: 0.8363