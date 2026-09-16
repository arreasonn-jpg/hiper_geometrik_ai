# Cross-domain Self-learning Transfer Benchmark

- Protokol: `cross_domain_self_learning_transfer_v1` v1 (imza `306c584f0e26`)
- Ortamlar: physics, causal, spatial, temporal, language
- Tohumlar: `[1, 2, 3]`

> Bu rapor pozitif transfer iddiası değildir. Source-only davranışın hedefte çekimser kalıp kalmadığını ve target-support adaptasyonunun scratch ile farkını ölçer.

## Özet

| Metrik | Değer |
|---|---:|
| Pair satırı | 60 |
| Source-only hedef karar | 0 |
| Source-only hedef konuşma oranı | 0.00000000 |
| Scratch coverage | 0.891667 ± 0.014296 |
| Transfer coverage | 0.891667 ± 0.014296 |
| Transfer − scratch coverage | 0.000000 ± 0.000000 |
| Pozitif transfer durumu | NOT_DEMONSTRATED |

## Örnek pair satırları

| Kaynak→Hedef | Seed | Source-only karar | Scratch cov | Transfer cov | Δcov |
|---|---:|---:|---:|---:|---:|
| physics→causal | 1 | 0 | 0.825000 | 0.825000 | 0.000000 |
| physics→spatial | 1 | 0 | 0.937500 | 0.937500 | 0.000000 |
| physics→temporal | 1 | 0 | 0.887500 | 0.887500 | 0.000000 |
| physics→language | 1 | 0 | 0.975000 | 0.975000 | 0.000000 |
| causal→physics | 1 | 0 | 0.862500 | 0.862500 | 0.000000 |
| causal→spatial | 1 | 0 | 0.937500 | 0.937500 | 0.000000 |
| causal→temporal | 1 | 0 | 0.887500 | 0.887500 | 0.000000 |
| causal→language | 1 | 0 | 0.975000 | 0.975000 | 0.000000 |
| spatial→physics | 1 | 0 | 0.862500 | 0.862500 | 0.000000 |
| spatial→causal | 1 | 0 | 0.825000 | 0.825000 | 0.000000 |
| spatial→temporal | 1 | 0 | 0.887500 | 0.887500 | 0.000000 |
| spatial→language | 1 | 0 | 0.975000 | 0.975000 | 0.000000 |

## Kabul kapıları

| Kapı | Sonuç |
|---|---|
| all_ordered_pairs_evaluated | GEÇTİ |
| source_only_abstains_on_target | GEÇTİ |
| target_support_adaptation_has_coverage | GEÇTİ |
| target_support_accuracy_perfect_on_decided | GEÇTİ |
| transfer_delta_reported | GEÇTİ |
| positive_transfer_not_claimed_when_absent | GEÇTİ |
| multi_seed | GEÇTİ |

## Bulgular

- 60 kaynak→hedef×tohum satırı koştu; kaynak-only hedef karar sayısı 0.
- Kaynak ortam deneyimleri hedef ortamda karar üretmedi; bu smoke koşuda yanlış pozitif transfer/kontaminasyon gözlenmedi.
- Hedef destek örnekleri eklendiğinde scratch ve kaynak-ön-yüklü öğrenen aynı coverage'ı verdi (delta 0.000000).
- Pozitif transfer durumu: NOT_DEMONSTRATED. Bu rapor pozitif genelleme iddiası olarak okunmamalıdır.

## Sınırlar

- Öğrenen exact-triple bellek kullanır; unseen hedef genellemesi beklenmez.
- Ortamlar sentetiktir ve ayrık sembol uzayları kullanır; gerçek domain transferi değildir.
- Bu artifact transfer protokolünü ve negatif kontrolü kapatır; pozitif transfer iddiası için daha güçlü, ortak soyut özellikli görevler gerekir.
