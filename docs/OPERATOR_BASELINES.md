# Operatör Baseline Ailesi — Kronecker vs Gerçek Dense (P0-2)

Protokol: `operator_baseline_family_v1` · veri imzası: `351fe80d38f0` · n=8 · tohumlar: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] · adım: 300

## Bütçe muhasebesi

| Kol | Parametre | FLOP | P oranı (Kron=1) | F oranı | Eşit-P | Eşit-F |
|---|---:|---:|---:|---:|:--:|:--:|
| `kronecker` | 128 | 1,024 | 1.000 | 1.000 | ✓ | ✓ |
| `rank1` | 128 | 128 | 1.000 | 0.125 | ✓ | — |
| `low_rank_param_matched` | 256 | 256 | 2.000 | 0.250 | — | — |
| `low_rank_flop_matched` | 1,024 | 1,024 | 8.000 | 1.000 | — | ✓ |
| `kron_sum_2` | 256 | 2,048 | 2.000 | 2.000 | — | — |
| `full_dense` | 4,096 | 4,096 | 32.000 | 4.000 | — | — |

## Öğretmen: `kronecker_teacher`

| Kol | test nMSE (ort) | 95% CI | ± std | R² | eğitim sn | P |
|---|---:|---:|---:|---:|---:|---:|
| `kronecker` | 0.000000 | [0.000000, 0.000000] | 0.000000 | 1.0000 | 0.097 | 128 |
| `full_dense` | 0.000001 | [0.000000, 0.000002] | 0.000002 | 1.0000 | 0.076 | 4,096 |
| `kron_sum_2` | 0.000275 | [0.000205, 0.000346] | 0.000162 | 0.9997 | 0.119 | 256 |
| `low_rank_flop_matched` | 0.382258 | [0.356985, 0.407531] | 0.057665 | 0.6177 | 0.091 | 1,024 |
| `low_rank_param_matched` | 0.749540 | [0.726682, 0.772398] | 0.052156 | 0.2504 | 0.090 | 256 |
| `rank1` | 0.859933 | [0.842177, 0.877689] | 0.040513 | 0.1400 | 0.087 | 128 |

## Öğretmen: `rank1_teacher`

| Kol | test nMSE (ort) | 95% CI | ± std | R² | eğitim sn | P |
|---|---:|---:|---:|---:|---:|---:|
| `low_rank_param_matched` | 0.000000 | [0.000000, 0.000000] | 0.000000 | 1.0000 | 0.091 | 256 |
| `rank1` | 0.000001 | [0.000000, 0.000003] | 0.000004 | 1.0000 | 0.087 | 128 |
| `full_dense` | 0.000003 | [0.000000, 0.000006] | 0.000007 | 1.0000 | 0.076 | 4,096 |
| `low_rank_flop_matched` | 0.000023 | [0.000019, 0.000027] | 0.000009 | 1.0000 | 0.092 | 1,024 |
| `kron_sum_2` | 0.762561 | [0.740704, 0.784419] | 0.049872 | 0.2374 | 0.120 | 256 |
| `kronecker` | 0.864611 | [0.847142, 0.882079] | 0.039858 | 0.1354 | 0.099 | 128 |

## Öğretmen: `low_rank_teacher`

| Kol | test nMSE (ort) | 95% CI | ± std | R² | eğitim sn | P |
|---|---:|---:|---:|---:|---:|---:|
| `full_dense` | 0.000000 | [0.000000, 0.000000] | 0.000000 | 1.0000 | 0.081 | 4,096 |
| `low_rank_flop_matched` | 0.000003 | [0.000001, 0.000005] | 0.000004 | 1.0000 | 0.095 | 1,024 |
| `low_rank_param_matched` | 0.368479 | [0.348885, 0.388074] | 0.044709 | 0.6315 | 0.094 | 256 |
| `rank1` | 0.645062 | [0.621839, 0.668284] | 0.052987 | 0.3549 | 0.092 | 128 |
| `kron_sum_2` | 0.855472 | [0.846388, 0.864557] | 0.020727 | 0.1445 | 0.124 | 256 |
| `kronecker` | 0.919669 | [0.913692, 0.925647] | 0.013639 | 0.0803 | 0.103 | 128 |

## Öğretmen: `full_dense_teacher`

| Kol | test nMSE (ort) | 95% CI | ± std | R² | eğitim sn | P |
|---|---:|---:|---:|---:|---:|---:|
| `full_dense` | 0.000000 | [0.000000, 0.000000] | 0.000000 | 1.0000 | 0.079 | 4,096 |
| `low_rank_flop_matched` | 0.638899 | [0.634102, 0.643696] | 0.010945 | 0.3611 | 0.094 | 1,024 |
| `low_rank_param_matched` | 0.894320 | [0.891730, 0.896910] | 0.005910 | 0.1056 | 0.094 | 256 |
| `kron_sum_2` | 0.907845 | [0.904623, 0.911067] | 0.007352 | 0.0921 | 0.123 | 256 |
| `rank1` | 0.947528 | [0.944538, 0.950517] | 0.006821 | 0.0524 | 0.089 | 128 |
| `kronecker` | 0.953684 | [0.951184, 0.956184] | 0.005704 | 0.0463 | 0.101 | 128 |

## Kabul kapıları

| kapı | sonuç |
|---|---|
| parameter_accounting_exact | GEÇTİ |
| all_arms_optimization_stable | GEÇTİ |
| full_dense_is_over_budget | GEÇTİ |
| all_teacher_families_present | GEÇTİ |
| all_baseline_arms_present | GEÇTİ |
| official_20_seed_rule_met | GEÇTİ |
| kronecker_beats_rank1_on_structured | GEÇTİ |
| kronecker_loses_to_dense_on_unstructured | GEÇTİ |
| kronecker_beats_flop_matched_dense_on_structured | GEÇTİ |

## Bulgular

- n=8, d=n²=64; 20 tohum × 4 öğretmen × 6 kol, her kol 300 adım.
- `kronecker_teacher`: en iyi kol `kronecker` (nMSE 0.000000); eşit-parametre kolları arasında en iyi `kronecker`.
- `rank1_teacher`: en iyi kol `low_rank_param_matched` (nMSE 0.000000); eşit-parametre kolları arasında en iyi `rank1`.
- `low_rank_teacher`: en iyi kol `full_dense` (nMSE 0.000000); eşit-parametre kolları arasında en iyi `rank1`.
- `full_dense_teacher`: en iyi kol `full_dense` (nMSE 0.000000); eşit-parametre kolları arasında en iyi `rank1`.
- Kronecker'ın kısıtsız dense tavanına göre nMSE farkı: kronecker_teacher: -0.000001, rank1_teacher: +0.864608, low_rank_teacher: +0.919669, full_dense_teacher: +0.953684. Pozitif değer Kronecker'ın tavanın gerisinde olduğunu gösterir.

## Sınırlar

- Görev tek katmanlı operatör regresyonudur; derin ağ, dil modeli veya sınıflandırma sonucu DEĞİLDİR.
- full_dense kolu parametre bütçesini kasıtlı aşar; rakip değil tavandır. Onu yenmek beklenmez, ona olan mesafe raporlanır.
- FLOP sayıları analitik çarpma-toplama sayımıdır; gerçek donanım verimi (BLAS, cache, paralellik) ölçülmez — wall-clock ayrıca verilir.
- Öğretmenler sentetiktir; 'HGA şu problem sınıfında iyidir' sonucu yalnız bu sentetik sınıflar için geçerlidir.
- Tohum sayısı 20; 20 tohum kuralı karşılandı ve rapor çekirdek istatistiksel iddia için kullanılabilir.
