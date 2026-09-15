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
| `w_uncertainty` | 0.25 | **0.20** |
| `w_conflict_penalty` | 0.20 | **0.20** |

## Eğitim vs held-out

| Küme | Varsayılan | Optimize | Kazanç |
|---|---:|---:|---:|
| Eğitim | 0.8600 | 1.0000 | +0.1400 |
| **Held-out** | 0.9100 | 1.0000 | **+0.0900** |

- Genelleşme oranı: `0.642857`
- Held-out'ta kazanç korundu mu: **EVET**

> Eğitim kazancının held-out'ta korunan kısmı. Oran 1.0'a yakınsa kazanç gerçek; 0'a yakınsa arama ezberlemiş, negatifse ağırlıklar held-out'ta ZARAR veriyor.

## Held-out istatistiksel karşılaştırma (eşleşmiş)

- Fark %95 CI: `[0.0400, 0.1450]`
- Etki büyüklüğü: `+0.719` (medium)
- Permütasyon p: `0.00495`
- Hüküm: AYRIŞMA: p=0.00495, g=0.6899 (medium), CI [0.04, 0.145] sıfırı içermiyor.

## Terim duyarlılığı (held-out)

| Terim | En iyi değer | Held-out aralığı | Fark yaratıyor mu |
|---|---:|---:|---|
| `w_gain` | 0.00 | 0.1250 | evet |
| `w_novelty` | 0.00 | 0.4450 | evet |
| `w_uncertainty` | 0.20 | 0.4950 | evet |
| `w_conflict_penalty` | 0.20 | 0.5250 | evet |

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
- Aramanın bulduğu ağırlıklar: {'w_gain': 0.0, 'w_novelty': 0.0, 'w_uncertainty': 0.2, 'w_conflict_penalty': 0.2}.
- Arama şu terimleri TAMAMEN KAPATTI: ['w_gain', 'w_novelty']. Bu, ablasyonun 'bu terimleri kapatmak verimi artırıyor' bulgusuyla tutarlıdır — elle seçilmiş ağırlıklar yalnız etkisiz değil, bazı yerlerde zararlıydı.
- Eğitim kazancı +0.1400, held-out kazancı +0.0900.
- Kazancın %64'ı held-out'ta korundu; kalanı aşırı uydurmadır.
- Held-out hükmü: AYRIŞMA: p=0.00495, g=0.6899 (medium), CI [0.04, 0.145] sıfırı içermiyor.

## Sınırlar

- Arama ızgarası kabadır; sürekli optimizasyon daha iyi bir nokta bulabilir. Bulunan ağırlıklar 'ızgaranın en iyisi'dir, küresel optimum değildir.
- Hedef tek metriktir (doğrulama verimi). Çeşitlilik ya da kapsama gibi başka hedefler farklı ağırlıklar seçtirirdi.
- Havuz aritmetiktir ve doğrulayıcı kapalı formdur; gerçek alanlarda optimal ağırlıklar farklı olabilir.
- Held-out tohumlar aynı havuz üreticisinden gelir; bu bir tohum genellemesidir, ALAN genellemesi değildir.

