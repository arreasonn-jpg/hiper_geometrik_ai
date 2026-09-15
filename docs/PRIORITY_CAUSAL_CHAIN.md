# Priority(E) Nedensel Zincir Ablasyonu (P0-1)

Protokol: `priority_causal_chain_ablation_v1` · veri imzası: `9bb0550d98d7` · tohumlar: [1, 2, 3, 4, 5] · K=10

## Zincir halkaları

| Kapatılan terim | Skor Δ (ort) | Kendall τ | Spearman ρ | Rank kayması | Top-K örtüşme | Downstream Δ verim | Zincir tam mı |
|---|---:|---:|---:|---:|---:|---:|:--:|
| `w_gain` | 0.1787 | 0.6593 | 0.8113 | 15.458 | 0.680 | -0.2400 | EVET |
| `w_novelty` | 0.1399 | 0.6957 | 0.8399 | 13.687 | 0.300 | +0.1200 | EVET |
| `w_uncertainty` | 0.1351 | 0.6968 | 0.8568 | 14.170 | 0.760 | -0.1600 | EVET |
| `w_conflict_penalty` | 0.0300 | 0.8859 | 0.9392 | 6.788 | 0.640 | -0.3400 | EVET |

## Downstream (seçilen K aday, dış doğrulayıcı)

| Kol | Doğrulama verimi | Yeni bilgi verimi | Yanlış seçim oranı | Özne çeşitliliği |
|---|---:|---:|---:|---:|
| baseline | 0.8800 | 0.6400 | 0.1200 | 0.9800 |
| w_gain=0 | 0.6400 | 0.6400 | 0.3600 | 0.9800 |
| w_novelty=0 | 1.0000 | 0.0600 | 0.0000 | 0.8600 |
| w_uncertainty=0 | 0.7200 | 0.4400 | 0.2800 | 1.0000 |
| w_conflict_penalty=0 | 0.5400 | 0.5200 | 0.4600 | 1.0000 |

## Kabul kapıları

| kapı | sonuç |
|---|---|
| at_least_one_full_causal_chain | GEÇTİ |
| all_terms_move_scores | GEÇTİ |
| all_terms_move_ranking | GEÇTİ |
| all_terms_move_selection | GEÇTİ |
| all_terms_move_downstream | GEÇTİ |
| no_term_harms_downstream_yield | GEÇTİ |

## Bulgular

- 5 tohum × 120 aday; ilk-10 seçimi ölçüldü.
- Baseline downstream doğrulama verimi 0.8800, yeni bilgi verimi 0.6400.
- Zinciri uçtan uca (skor→sıralama→seçim→downstream) taşıyan terimler: ['w_gain', 'w_novelty', 'w_uncertainty', 'w_conflict_penalty'].
- Kendi hedef metriğinde pozitif katkı taşıyan terimler: [('w_novelty', 'novel_knowledge_yield'), ('w_uncertainty', 'verification_yield'), ('w_conflict_penalty', 'false_selection_rate')] — terimi kapatmak hedef metriğini kötüleştiriyor.
- NOT: ['w_novelty'] kapatılınca doğrulama verimi yükseliyor; bu keşif/sömürü takasıdır (yeni bilgi verimi aynı anda düşer), tek metrikli 'zarar' kanıtı değildir.

## Sınırlar

- Domain aritmetik mini-environment'tır; doğal dil aday havuzlarına genellenemez.
- Downstream ölçütü seçilen K adayın dış doğrulayıcı karşısındaki verimidir; model eğitim kazancı DEĞİLDİR.
- Priority skorları kırpılmadan (raw_priority) karşılaştırılır; CLI'nin gösterdiği kırpılmış skor bazı etkileri gizleyebilir.
- Beraberlikler experience_id'ye göre deterministik kırılır; farklı bir kırma kuralı ilk-K örtüşmesini değiştirebilir.

