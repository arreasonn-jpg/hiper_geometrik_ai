# Türkçe LM 20-Seed Sonuçları (kanonik)

Protokol: `turkish_lm_v1` v2 · korpus `tr_corpus_v1` · profil `full`
Tohumlar: **20 seed** (1–20) · RNG: deterministic

> P1-2'nin kanonik referansı. `docs/TURKISH_LM.md` tarihsel 2-seed
> versiyonudur; sonuçlar burada 20 seed ile güçlendirildi.

## Test sonuçları (20 seed)

| Kol | PPL ↓ (ort ± std) | min | max | top-1 ↑ |
|---|---:|---:|---:|---:|
| `transformer` | **208.54 ±4.22** | 202.59 | 222.59 | **0.1753** |
| `dense` | 230.84 ±4.04 | 223.01 | 239.30 | 0.1751 |
| `hga` | 621.21 ±25.95 | 579.26 | 673.29 | 0.1181 |
| `bigram` (interp.) | 397.96 | — | — | — |
| `unigram` (add-α) | 1307.57 | — | — | — |

## Kabul kapıları

- PASS — `best_neural_beats_interpolated_bigram_ppl`
- PASS — `best_neural_beats_majority_top1`
- PASS — `best_neural_beats_unigram_ppl`
- PASS — `corpus_at_least_1m_words`
- PASS — `document_disjoint_splits`
- PASS — `finite_metrics_all_arms`
- PASS — `multi_seed_reported` (n=20)
- PASS — `parameter_budget_max_min_ratio_leq_1_05`
- PASS — `real_human_corpus_hash_verified`
- PASS — `tokenizer_fit_train_only`

## 2-seed → 20-seed karşılaştırması

| Kol | 2-seed PPL | 20-seed PPL | Δ |
|---|---:|---:|---:|
| transformer | 209.85 ±0.86 | 208.54 ±4.22 | −1.31 |
| dense | 229.80 ±0.54 | 230.84 ±4.04 | +1.04 |
| hga | 632.28 ±22.07 | 621.21 ±25.95 | −11.07 |

**Yorum:** Ortalama değerler seed artışıyla **stabil**. 2-seed
std'leri yapay dardı (±0.5-0.9); 20-seed std'leri gerçekçi CI verir
(±4.0-4.2).

## Kritik gözlem (dürüst)

`transformer` **eş-parametreli** HGA'yı PPL'de **3× geçiyor**
(208 vs 621). HGA'nın seed varyansı (25.95) transformer'ın **6 katı**
(4.22) → HGA daha az kararlı.

Bu sonuç HGA mimarisinin Türkçe dil modellemede zayıf olduğunu
gösteriyor; hangi bileşenin zarar verdiği bir sonraki deneyin
konusudur (bkz. P1-5 ablation önerileri).

## Üretim yolu

Batch komutları `scripts/turkish_lm_batch.sh` üzerinden:

- `turkish_lm_batch.sh 1 5 batch_01_05.json`
- `turkish_lm_batch.sh 6 10 batch_06_10.json`
- `turkish_lm_batch.sh 11 15 batch_11_15.json`
- `turkish_lm_batch.sh 16 20 batch_16_20.json`

Sonra `scripts/turkish_lm_merge.py` ile 4 batch tek JSON'a birleştirilir:
`docs/turkish_lm_20seed.json`.

## Referans

- P1-2 (birleşik audit raporu)
- SECURITY.md: SEC-2026-10-13-01, SEC-2026-10-13-02
- Tarihsel: `docs/TURKISH_LM.md` (2 seed)
