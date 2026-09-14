# Priority(E) Ağırlık Optimizasyonu + Held-Out Doğrulama

- Protokol: `priority_weight_optimization_v1` v1 (imza `3ca4c32063c1`)
- Profil: `standard` · hedef: `verification_yield`
- Izgara: `[0.0, 0.2, 0.4, 0.6]` · değerlendirilen kombinasyon: `255`
- Eğitim tohumları: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- **Held-out tohumları: `[101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]`** (aramada HİÇ görülmedi)

## Ağırlıklar

| Terim | Varsayılan (elle) | Aramanın bulduğu |
|---|---:|---:|
| `w_gain` | 0.40 | **0.00** |
| `w_novelty` | 0.35 | **0.00** |
| `w_uncertainty` | 0.25 | **0.00** |
| `w_conflict_penalty` | 0.20 | **0.20** |

## Eğitim vs held-out

| Küme | Varsayılan | Optimize | Kazanç |
|---|---:|---:|---:|
| Eğitim | 0.3300 | 0.6400 | +0.3100 |
| **Held-out** | 0.3300 | 0.6700 | **+0.3400** |

- Genelleşme oranı: `1.096774`
- Held-out'ta kazanç korundu mu: **EVET**

> Eğitim kazancının held-out'ta korunan kısmı. Oran 1.0'a yakınsa kazanç gerçek; 0'a yakınsa arama ezberlemiş, negatifse ağırlıklar held-out'ta ZARAR veriyor.

## Held-out istatistiksel karşılaştırma (eşleşmiş)

- Fark %95 CI: `[0.2650, 0.4200]`
- Etki büyüklüğü: `+1.813` (large)
- Permütasyon p: `0.00005`
- Hüküm: AYRIŞMA: p=0.00005, g=1.7408 (large), CI [0.265, 0.42] sıfırı içermiyor.

## Terim duyarlılığı (held-out)

| Terim | En iyi değer | Held-out aralığı | Fark yaratıyor mu |
|---|---:|---:|---|
| `w_gain` | 0.00 | 0.2950 | evet |
| `w_novelty` | 0.20 | 0.0050 | evet |
| `w_uncertainty` | 0.20 | 0.0100 | evet |
| `w_conflict_penalty` | 0.20 | 0.0000 | evet |

## Kabul kapıları

| Kapı | Sonuç |
|---|---|
| train_test_seed_isolation | GEÇTİ |
| search_space_includes_zero | GEÇTİ |
| holdout_evaluated | GEÇTİ |
| optimized_at_least_matches_default_on_holdout | GEÇTİ |
| gain_survives_holdout | GEÇTİ |
| gain_statistically_distinguishable | GEÇTİ |
| overfitting_measured | GEÇTİ |
| sensitivity_measured | GEÇTİ |
| paired_design | GEÇTİ |

## Bulgular

- 255 ağırlık kombinasyonu 10 eğitim tohumunda tarandı; en iyisi 20 HELD-OUT tohumda test edildi.
- Varsayılan (elle seçilmiş) ağırlıklar: {'w_gain': 0.4, 'w_novelty': 0.35, 'w_uncertainty': 0.25, 'w_conflict_penalty': 0.2}.
- Aramanın bulduğu ağırlıklar: {'w_gain': 0.0, 'w_novelty': 0.0, 'w_uncertainty': 0.0, 'w_conflict_penalty': 0.2}.
- Arama şu terimleri TAMAMEN KAPATTI: ['w_gain', 'w_novelty', 'w_uncertainty']. Bu, ablasyonun 'bu terimleri kapatmak verimi artırıyor' bulgusuyla tutarlıdır — elle seçilmiş ağırlıklar yalnız etkisiz değil, bazı yerlerde zararlıydı.
- Eğitim kazancı +0.3100, held-out kazancı +0.3400.
- Kazancın %110'ı held-out'ta korundu; kalanı aşırı uydurmadır.
- Held-out hükmü: AYRIŞMA: p=0.00005, g=1.7408 (large), CI [0.265, 0.42] sıfırı içermiyor.

## Sınırlar

- Arama ızgarası kabadır; sürekli optimizasyon daha iyi bir nokta bulabilir. Bulunan ağırlıklar 'ızgaranın en iyisi'dir, küresel optimum değildir.
- Hedef tek metriktir (doğrulama verimi). Çeşitlilik ya da kapsama gibi başka hedefler farklı ağırlıklar seçtirirdi.
- Havuz aritmetiktir ve doğrulayıcı kapalı formdur; gerçek alanlarda optimal ağırlıklar farklı olabilir.
- Held-out tohumlar aynı havuz üreticisinden gelir; bu bir tohum genellemesidir, ALAN genellemesi değildir.
