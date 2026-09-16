# Uzun Bağlam Dil Modelleme (P1)

Protokol: `long_context_lm_v1` v1 · profil `smoke_1024` · veri imzası `4f6da1cd2ead`
Tohumlar: [1, 2] · korpus token: {'train': 30472, 'dev': 5761, 'test': 6780} · sözlük 512
Pencereler: train 256 / eval 128 (deterministik örnekleme, imza `c4e3349cfb36`)

## Test PPL (ortalama ± std)

| Bağlam | dense | transformer | hga | unigram | bigram |
|---:|---:|---:|---:|---:|---:|
| 512 | 500.1 ±16.9 | 604.4 ±112.2 | 524.6 ±17.3 | 34.0 | 26.6 |
| 1024 | 514.3 ±0.1 | 446.9 ±67.5 | 496.2 ±17.6 | 34.0 | 26.6 |

## Bağlam etkisi (512→1024 token)

- `dense`: 500.1 → 514.3 (+2.8%, degrades)
- `transformer`: 604.4 → 446.9 (-26.1%, improves)
- `hga`: 524.6 → 496.2 (-5.4%, improves)

## Kapılar

- GEÇTİ — `all_cells_completed`
- GEÇTİ — `finite_metrics_all_cells`
- GEÇTİ — `parameter_budget_leq_1_05_every_context`
- GEÇTİ — `positions_shared_across_contexts_and_arms`
- GEÇTİ — `ngram_floor_reported_every_context`
- GEÇTİ — `context_effect_direction_reported`
- GEÇTİ — `multi_seed_reported`
- GEÇTİ — `long_context_targets_configured`
- GEÇTİ — `max_context_at_least_1024_when_configured`

## Bulgular

- `dense`: bağlam 512→1024 tokenda test PPL 500.1→514.3 (+2.8%, degrades).
- `transformer`: bağlam 512→1024 tokenda test PPL 604.4→446.9 (-26.1%, improves).
- `hga`: bağlam 512→1024 tokenda test PPL 524.6→496.2 (-5.4%, improves).
- Zemin @ bağlam 1024: unigram PPL 34.0, bigram 26.6; en iyi neural 446.9.
- 512/1024 token uzun bağlam kod yolu bu profilde koşuldu; bu bir kalite iddiası değil şekil/bütçe/pozisyon eşleşmesi kapısıdır.
- AÇIK SINIR: bu kısa eşit-bütçe taramasında (3 adım) hiçbir neural kol bigram zeminini geçmedi. Mutlak LM kalitesi iddiası bu protokolün konusu değildir ve `turkish_lm` full (4000 adım) protokolünde ölçülür; oradaki n-gram kapıları gerçek veriyle PASS. Buradaki sayılar yalnız bağlam ETKİSİNİN yönünü taşır.

## Sınırlar

- PPL, deterministik örneklenmiş pencereler üzerinde hesaplanır; korpus CE'sinin yansız tahminidir, tam sayımı değildir.
- Adım sayısı sabittir ve yakınsama iddiası yoktur; bağlamlar arası karşılaştırma eşit-eğitim-bütçesi karşılaştırmasıdır.
- Girdi düzleştirmeli mimarilerde (dense, HGA encoder) bağlam uzadıkça parametre bütçesi aynı kalsın diye genişlikler düşer; bu, mimari ailelerin uzun bağlamdaki YAPISAL maliyet farkıdır ve gizlenmez.
- Tek korpus (tr_corpus_v1) ve tek dil (Türkçe); genelleme iddiası taşımaz.
- `smoke_1024` profili yalnız 512/1024 kod yolunu duman testinden geçirir; tam kalite raporu için `full` profili 24→1024 taramasını çalıştırmalıdır.
