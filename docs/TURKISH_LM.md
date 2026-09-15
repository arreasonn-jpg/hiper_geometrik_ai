# Gerçek Türkçe Dil Modelleme Benchmarkı (tr_corpus_v1 — 1.11M kelime)

Protokol: `turkish_lm_v1` v1 · korpus `tr_corpus_v1` · profil `full` · veri imzası `d278278dd310` · tohumlar [1, 2]

Görev: belge-ayrık held-out Türkçe metinde next-token tahmini. Tokenizer YALNIZ train metninde eğitildi.

## Korpus

| Split | Belge | Cümle | Kelime | BPE token |
|---|---:|---:|---:|---:|
| train | 3,514 | 85,804 | 912,839 | 1,891,700 |
| dev | 399 | 7,608 | 75,687 | 150,943 |
| test | 452 | 10,818 | 121,983 | 257,896 |

Tokenizer: BPE, sözlük 5,332 (byte fallback: True)

## Parametre bütçesi

| Kol | Parametre |
|---|---:|
| `dense` | 474,905 |
| `transformer` | 477,311 |
| `hga` | 477,333 |
| **max/min oranı** | **1.0051** |

## Test sonuçları

| Kol | PPL ↓ (ort ± std) | top-1 ↑ | top-5 ↑ |
|---|---:|---:|---:|
| `dense` | 229.80 ±0.54 | 0.1749 | 0.3025 |
| `transformer` | 209.85 ±0.86 | 0.1723 | 0.3045 |
| `hga` | 632.28 ±22.07 | 0.1202 | 0.2044 |
| `unigram` (add-α) | 1307.57 | — | — |
| `bigram` (interp.) | 397.96 | — | — |

Dejenere çoğunluk kolu top-1: `0.0470` — en iyi neural kol: `transformer`

## Kabul kapıları

- PASS — `best_neural_beats_interpolated_bigram_ppl`
- PASS — `best_neural_beats_majority_top1`
- PASS — `best_neural_beats_unigram_ppl`
- PASS — `corpus_at_least_1m_words`
- PASS — `document_disjoint_splits`
- PASS — `finite_metrics_all_arms`
- PASS — `multi_seed_reported`
- PASS — `parameter_budget_max_min_ratio_leq_1_05`
- PASS — `real_human_corpus_hash_verified`
- PASS — `tokenizer_fit_train_only`

## Sınırlar

- Korpus tr_corpus_v1'dir (UD r2.14 ×8 + Bible CC0 + TWT; ≥1.1M kelime, tamamı gerçek insan metni, hash doğrulamalı). Tür dağılımı dengeli değildir: İncil çevirisi korpusun ~%40'ıdır ve dil/biçem olarak moderndir ama tematik olarak dardır. TWT hem burada hem twt_real_results_v1 arc benchmarkındadır; görevler farklıdır ve bu çapraz kullanım bilinçlidir.
- Ölçülen şey held-out perplexity ve next-token doğruluğudur; üretim kalitesi, talimat takibi ve sohbet kalitesi bu protokolün dışındadır.
- Bağlam pencereleri belge içidir; belgeler arası uzun-bağlam etkisi bu sürümde ölçülmez.
- 2 tohum dağılım raporlar; kesin anlamlılık iddiası için 20-tohum çekirdek protokolü gerekir.

