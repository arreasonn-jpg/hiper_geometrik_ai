# Operatör Baseline Ailesi — Kronecker vs Gerçek Dense (P0-2)

Protokol: `operator_baseline_family_v1` · veri imzası: `f633b9ecb28b` · n=8 · tohumlar: [1, 2, 3] · adım: 300

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

| Kol | test nMSE (ort) | ± std | R² | eğitim sn | P |
|---|---:|---:|---:|---:|---:|
| `kronecker` | 0.000000 | 0.000000 | 1.0000 | 0.171 | 128 |
| `full_dense` | 0.000000 | 0.000000 | 1.0000 | 0.132 | 4,096 |
| `kron_sum_2` | 0.000170 | 0.000159 | 0.9998 | 0.197 | 256 |
| `low_rank_flop_matched` | 0.381058 | 0.081016 | 0.6189 | 0.159 | 1,024 |
| `low_rank_param_matched` | 0.767845 | 0.048406 | 0.2321 | 0.160 | 256 |
| `rank1` | 0.874755 | 0.024815 | 0.1252 | 0.144 | 128 |

## Öğretmen: `rank1_teacher`

| Kol | test nMSE (ort) | ± std | R² | eğitim sn | P |
|---|---:|---:|---:|---:|---:|
| `low_rank_param_matched` | 0.000000 | 0.000000 | 1.0000 | 0.154 | 256 |
| `rank1` | 0.000000 | 0.000000 | 1.0000 | 0.146 | 128 |
| `full_dense` | 0.000000 | 0.000000 | 1.0000 | 0.134 | 4,096 |
| `low_rank_flop_matched` | 0.000026 | 0.000004 | 1.0000 | 0.158 | 1,024 |
| `kron_sum_2` | 0.772372 | 0.049778 | 0.2276 | 0.197 | 256 |
| `kronecker` | 0.868704 | 0.024279 | 0.1313 | 0.169 | 128 |

## Öğretmen: `low_rank_teacher`

| Kol | test nMSE (ort) | ± std | R² | eğitim sn | P |
|---|---:|---:|---:|---:|---:|
| `full_dense` | 0.000000 | 0.000000 | 1.0000 | 0.145 | 4,096 |
| `low_rank_flop_matched` | 0.000005 | 0.000007 | 1.0000 | 0.168 | 1,024 |
| `low_rank_param_matched` | 0.376780 | 0.010882 | 0.6232 | 0.162 | 256 |
| `rank1` | 0.643730 | 0.055157 | 0.3562 | 0.147 | 128 |
| `kron_sum_2` | 0.852736 | 0.023163 | 0.1472 | 0.203 | 256 |
| `kronecker` | 0.928002 | 0.007115 | 0.0719 | 0.173 | 128 |

## Öğretmen: `full_dense_teacher`

| Kol | test nMSE (ort) | ± std | R² | eğitim sn | P |
|---|---:|---:|---:|---:|---:|
| `full_dense` | 0.000000 | 0.000000 | 1.0000 | 0.139 | 4,096 |
| `low_rank_flop_matched` | 0.646735 | 0.012055 | 0.3532 | 0.168 | 1,024 |
| `low_rank_param_matched` | 0.895073 | 0.003361 | 0.1049 | 0.159 | 256 |
| `kron_sum_2` | 0.910523 | 0.001910 | 0.0894 | 0.208 | 256 |
| `rank1` | 0.949636 | 0.002897 | 0.0503 | 0.149 | 128 |
| `kronecker` | 0.951678 | 0.000345 | 0.0483 | 0.178 | 128 |

## Kabul kapıları

| kapı | sonuç |
|---|---|
| parameter_accounting_exact | GEÇTİ |
| all_arms_optimization_stable | GEÇTİ |
| full_dense_is_over_budget | GEÇTİ |
| kronecker_beats_rank1_on_structured | GEÇTİ |
| kronecker_loses_to_dense_on_unstructured | GEÇTİ |
| kronecker_beats_flop_matched_dense_on_structured | GEÇTİ |

## Bulgular

- n=8, d=n²=64; 3 tohum × 4 öğretmen × 6 kol, her kol 300 adım.
- `kronecker_teacher`: en iyi kol `kronecker` (nMSE 0.000000); eşit-parametre kolları arasında en iyi `kronecker`.
- `rank1_teacher`: en iyi kol `low_rank_param_matched` (nMSE 0.000000); eşit-parametre kolları arasında en iyi `rank1`.
- `low_rank_teacher`: en iyi kol `full_dense` (nMSE 0.000000); eşit-parametre kolları arasında en iyi `rank1`.
- `full_dense_teacher`: en iyi kol `full_dense` (nMSE 0.000000); eşit-parametre kolları arasında en iyi `rank1`.
- Kronecker'ın kısıtsız dense tavanına göre nMSE farkı: kronecker_teacher: -0.000000, rank1_teacher: +0.868704, low_rank_teacher: +0.928002, full_dense_teacher: +0.951678. Pozitif değer Kronecker'ın tavanın gerisinde olduğunu gösterir.

## Sınırlar

- Görev tek katmanlı operatör regresyonudur; derin ağ, dil modeli veya sınıflandırma sonucu DEĞİLDİR.
- full_dense kolu parametre bütçesini kasıtlı aşar; rakip değil tavandır. Onu yenmek beklenmez, ona olan mesafe raporlanır.
- FLOP sayıları analitik çarpma-toplama sayımıdır; gerçek donanım verimi (BLAS, cache, paralellik) ölçülmez — wall-clock ayrıca verilir.
- Öğretmenler sentetiktir; 'HGA şu problem sınıfında iyidir' sonucu yalnız bu sentetik sınıflar için geçerlidir.
- Tohum sayısı 3; çekirdek bilimsel iddia için 20 tohum hedefi ayrıca koşulmalıdır.
