# Uzun Bağlam Dil Modelleme (P1)

Protokol: `long_context_lm_v1` v1 · profil `full` · veri imzası `c414db2ee70b`
Tohumlar: [1, 2] · korpus token: {'train': 1891700, 'dev': 150943, 'test': 257896} · sözlük 5332
Pencereler: train 40,000 / eval 4,000 (deterministik örnekleme, imza `17252fdd8b52`)

## Test PPL (ortalama ± std)

| Bağlam | dense | transformer | hga | unigram | bigram |
|---:|---:|---:|---:|---:|---:|
| 24 | 2003.9 ±108.1 | 1536.5 ±4.8 | 1772.8 ±8.3 | 1309.8 | 392.2 |
| 64 | 2364.9 ±192.1 | 1573.3 ±30.8 | 1790.5 ±30.5 | 1309.8 | 392.2 |
| 128 | 2480.3 ±130.0 | 1599.9 ±36.2 | 1823.6 ±23.1 | 1309.8 | 392.2 |
| 256 | 2825.3 ±161.5 | 1586.9 ±1.4 | 1791.8 ±11.5 | 1309.8 | 392.2 |

## Bağlam etkisi (24→256 token)

- `dense`: 2003.9 → 2825.3 (+41.0%, degrades)
- `transformer`: 1536.5 → 1586.9 (+3.3%, degrades)
- `hga`: 1772.8 → 1791.8 (+1.1%, degrades)

## Kapılar

- GEÇTİ — `all_cells_completed`
- GEÇTİ — `finite_metrics_all_cells`
- GEÇTİ — `parameter_budget_leq_1_05_every_context`
- GEÇTİ — `positions_shared_across_contexts_and_arms`
- GEÇTİ — `ngram_floor_reported_every_context`
- GEÇTİ — `context_effect_direction_reported`
- GEÇTİ — `multi_seed_reported`

## Bulgular

- `dense`: bağlam 24→256 tokenda test PPL 2003.9→2825.3 (+41.0%, degrades).
- `transformer`: bağlam 24→256 tokenda test PPL 1536.5→1586.9 (+3.3%, degrades).
- `hga`: bağlam 24→256 tokenda test PPL 1772.8→1791.8 (+1.1%, degrades).
- Zemin @ bağlam 256: unigram PPL 1309.8, bigram 392.2; en iyi neural 1586.9.
- AÇIK SINIR: bu kısa eşit-bütçe taramasında (300 adım) hiçbir neural kol bigram zeminini geçmedi. Mutlak LM kalitesi iddiası bu protokolün konusu değildir ve `turkish_lm` full (4000 adım) protokolünde ölçülür; oradaki n-gram kapıları gerçek veriyle PASS. Buradaki sayılar yalnız bağlam ETKİSİNİN yönünü taşır.

## Sınırlar

- PPL, deterministik örneklenmiş pencereler üzerinde hesaplanır; korpus CE'sinin yansız tahminidir, tam sayımı değildir.
- Adım sayısı sabittir ve yakınsama iddiası yoktur; bağlamlar arası karşılaştırma eşit-eğitim-bütçesi karşılaştırmasıdır.
- Girdi düzleştirmeli mimarilerde (dense, HGA encoder) bağlam uzadıkça parametre bütçesi aynı kalsın diye genişlikler düşer; bu, mimari ailelerin uzun bağlamdaki YAPISAL maliyet farkıdır ve gizlenmez.
- Tek korpus (tr_corpus_v1) ve tek dil (Türkçe); genelleme iddiası taşımaz.
