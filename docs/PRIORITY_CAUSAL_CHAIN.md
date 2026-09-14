# Priority(E) Nedensel Zincir Ablasyonu (P0-1)

Protokol: `priority_causal_chain_ablation_v1` · veri imzası: `5b8111feed57` · tohumlar: [1, 2, 3, 4, 5] · K=10

## Zincir halkaları

| Kapatılan terim | Skor Δ (ort) | Kendall τ | Spearman ρ | Rank kayması | Top-K örtüşme | Downstream Δ verim | Zincir tam mı |
|---|---:|---:|---:|---:|---:|---:|:--:|
| `w_gain` | 0.1673 | 0.5147 | 0.5530 | 25.945 | 0.200 | +0.2400 | EVET |
| `w_novelty` | 0.3135 | 0.9469 | 0.9840 | 2.487 | 1.000 | +0.0000 | hayır |
| `w_uncertainty` | 0.2099 | 0.8952 | 0.9658 | 5.243 | 0.900 | +0.0000 | hayır |
| `w_conflict_penalty` | 0.0217 | 0.9409 | 0.9739 | 4.252 | 0.900 | +0.0600 | hayır |

## Downstream (seçilen K aday, dış doğrulayıcı)

| Kol | Doğrulama verimi | Yeni bilgi verimi | Yanlış seçim oranı | Özne çeşitliliği |
|---|---:|---:|---:|---:|
| baseline | 0.3800 | 0.3800 | 0.1000 | 0.8600 |
| w_gain=0 | 0.6200 | 0.6200 | 0.3200 | 0.9800 |
| w_novelty=0 | 0.3800 | 0.3800 | 0.1000 | 0.8600 |
| w_uncertainty=0 | 0.3800 | 0.3800 | 0.1000 | 0.8800 |
| w_conflict_penalty=0 | 0.4400 | 0.4400 | 0.1000 | 0.8800 |

## Kabul kapıları

| kapı | sonuç |
|---|---|
| at_least_one_full_causal_chain | GEÇTİ |
| all_terms_move_scores | GEÇTİ |
| all_terms_move_ranking | GEÇTİ |
| all_terms_move_selection | KALDI |
| all_terms_move_downstream | KALDI |
| no_term_harms_downstream_yield | KALDI |

## Bulgular

- 5 tohum × 120 aday; ilk-10 seçimi ölçüldü.
- Baseline downstream doğrulama verimi 0.3800, yeni bilgi verimi 0.3800.
- Zinciri uçtan uca (skor→sıralama→seçim→downstream) taşıyan terimler: ['w_gain'].
- `w_novelty` zinciri şu halkada kopuyor: ['selection_changed', 'downstream_changed']. Bu bir hata değil bulgudur: terim bu havuzda o halkanın ötesine geçmiyor.
- `w_uncertainty` zinciri şu halkada kopuyor: ['selection_changed', 'downstream_changed']. Bu bir hata değil bulgudur: terim bu havuzda o halkanın ötesine geçmiyor.
- `w_conflict_penalty` zinciri şu halkada kopuyor: ['selection_changed', 'downstream_changed']. Bu bir hata değil bulgudur: terim bu havuzda o halkanın ötesine geçmiyor.
- YÖN UYARISI: ['w_gain', 'w_conflict_penalty'] terimlerini KAPATMAK doğrulama verimini ARTIRIYOR — bu havuzda katkıları negatif. 'Etkili olmak' 'faydalı olmak' değildir; ağırlıklar bu bulgu incelenmeden savunulamaz.

## Sınırlar

- Domain aritmetik mini-environment'tır; doğal dil aday havuzlarına genellenemez.
- Downstream ölçütü seçilen K adayın dış doğrulayıcı karşısındaki verimidir; model eğitim kazancı DEĞİLDİR.
- Priority skorları kırpılmadan (raw_priority) karşılaştırılır; CLI'nin gösterdiği kırpılmış skor bazı etkileri gizleyebilir.
- Beraberlikler experience_id'ye göre deterministik kırılır; farklı bir kırma kuralı ilk-K örtüşmesini değiştirebilir.
