# HGA Signature Benchmark v1 (P0-4)

Protokol: `hga_signature_benchmark_v1` · profil: `standard` · veri imzası: `d699088469e4` · tohumlar: [1]

## Parametre bütçesi

| Kol | Toplam parametre | Gövde parametresi |
|---|---:|---:|
| `dense` | 83,655 | 81,975 |
| `hga` | 85,938 | 82,482 |
| `hga_no_attention` | 84,881 | 81,425 |
| `hga_no_kronecker` | 83,914 | 80,458 |
| `hga_no_memory` | 85,938 | 82,482 |
| `transformer` | 85,926 | 82,470 |
| **max/min oranı** | **1.0273** | **1.0252** |

## Doğruluk ızgarası (ortalama)

| Görev | `symbolic` | `dense` | `transformer` | `hga` | `hga_no_memory` | `hga_no_kronecker` | `hga_no_attention` | `hybrid` | çoğunluk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_long_chain | 1.0000 | 0.4785 | 0.5508 | 0.4766 | 0.4766 | 0.4746 | 0.5059 | 1.0000 | 0.4785 |
| B_unseen_entity | 1.0000 | 0.4844 | 0.5176 | 0.4766 | 0.4766 | 0.4785 | 0.4863 | 1.0000 | 0.5020 |
| C_unseen_relation | 1.0000 | 0.5176 | 0.5137 | 0.5098 | 0.5098 | 0.5195 | 0.4961 | 1.0000 | 0.5156 |
| D_unseen_both | 1.0000 | 0.5039 | 0.5312 | 0.4980 | 0.4980 | 0.5000 | 0.4922 | 1.0000 | 0.5312 |
| E_conflict | 1.0000 | 0.4941 | 0.4961 | 0.4941 | 0.4941 | 0.5234 | 0.5098 | 0.7617 | 0.5000 |
| F_memory_dependent | 1.0000 | 0.8984 | 0.9863 | 1.0000 | 0.4883 | 1.0000 | 1.0000 | 1.0000 | 0.4844 |
| G_epistemic | 1.0000 | 0.3418 | 0.3848 | 0.3223 | 0.3223 | 0.3105 | 0.3535 | 0.7695 | 0.3438 |
| H_distractor | 1.0000 | 0.5117 | 0.4785 | 0.5156 | 0.5156 | 0.4941 | 0.5234 | 1.0000 | 0.5215 |

## Halüsinasyon oranı (gerçekte çıkarılamayana 'YES')

| Görev | `symbolic` | `dense` | `transformer` | `hga` | `hga_no_memory` | `hga_no_kronecker` | `hga_no_attention` | `hybrid` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_long_chain | 0.0000 | 0.1875 | 0.1934 | 0.1621 | 0.1621 | 0.1914 | 0.1797 | 0.0000 |
| B_unseen_entity | 0.0000 | 0.3242 | 0.4160 | 0.3086 | 0.3086 | 0.3379 | 0.2852 | 0.0000 |
| C_unseen_relation | 0.0000 | 0.2363 | 0.3867 | 0.2578 | 0.2578 | 0.2578 | 0.2031 | 0.0000 |
| D_unseen_both | 0.0000 | 0.2637 | 0.4062 | 0.2832 | 0.2832 | 0.2930 | 0.2539 | 0.0000 |
| E_conflict | 0.0000 | 0.2461 | 0.1504 | 0.2383 | 0.2383 | 0.2051 | 0.2402 | 0.2383 |
| F_memory_dependent | 0.0000 | 0.0449 | 0.0137 | 0.0000 | 0.2031 | 0.0000 | 0.0000 | 0.0000 |
| G_epistemic | 0.0000 | 0.2422 | 0.2656 | 0.2422 | 0.2422 | 0.2148 | 0.2363 | 0.1211 |
| H_distractor | 0.0000 | 0.2402 | 0.0840 | 0.2383 | 0.2383 | 0.2188 | 0.2539 | 0.0000 |

## İmza analizi: HGA nerede, neden?

Rakip = **öğrenen** kollar (dense/transformer; aynı bilgi, aynı bütçe). `symbolic` rakip değil KÂHİN tavandır: gizli olguları ham okur, deterministik dünyada yapısal 1.0 alır; farkı `kâhin açığı` sütununda ayrıca raporlanır.

| Görev | HGA | En iyi rakip | Fark | Kâhin açığı | Bellek katkısı | Kronecker katkısı | Attention katkısı |
|---|---:|---|---:|---:|---:|---:|---:|
| A_long_chain | 0.4766 | transformer (0.5508) | -0.0742 | +0.5234 | +0.0000 | +0.0020 | -0.0293 |
| B_unseen_entity | 0.4766 | transformer (0.5176) | -0.0410 | +0.5234 | +0.0000 | -0.0020 | -0.0098 |
| C_unseen_relation | 0.5098 | dense (0.5176) | -0.0078 | +0.4902 | +0.0000 | -0.0098 | +0.0137 |
| D_unseen_both | 0.4980 | transformer (0.5312) | -0.0332 | +0.5020 | +0.0000 | -0.0020 | +0.0059 |
| E_conflict | 0.4941 | transformer (0.4961) | -0.0020 | +0.5059 | +0.0000 | -0.0293 | -0.0156 |
| F_memory_dependent | 1.0000 | transformer (0.9863) | +0.0137 | +0.0000 | +0.5117 | +0.0000 | +0.0000 |
| G_epistemic | 0.3223 | transformer (0.3848) | -0.0625 | +0.6777 | +0.0000 | +0.0117 | -0.0312 |
| H_distractor | 0.5156 | dense (0.5117) | +0.0039 | +0.4844 | +0.0000 | +0.0215 | -0.0078 |

## Sızıntı denetimi

| Görev | temiz | test held-out entity | test held-out relation |
|---|:--:|:--:|:--:|
| A_long_chain | ✓ | ✓ | ✓ |
| B_unseen_entity | ✓ | ✓ | ✓ |
| C_unseen_relation | ✓ | ✓ | ✓ |
| D_unseen_both | ✓ | ✓ | ✓ |
| E_conflict | ✓ | ✓ | ✓ |
| F_memory_dependent | ✓ | ✓ | ✓ |
| G_epistemic | ✓ | ✓ | ✓ |
| H_distractor | ✓ | ✓ | ✓ |

## Kabul kapıları

| kapı | sonuç |
|---|---|
| parameter_budget_within_gate | GEÇTİ |
| body_parameter_budget_within_gate | GEÇTİ |
| no_held_out_leak | GEÇTİ |
| hga_beats_majority_everywhere | KALDI |
| hga_has_signature_task | GEÇTİ |
| neural_arms_learn_above_chance | GEÇTİ |
| memory_gain_is_task_specific | GEÇTİ |

## Bulgular

- Profil `standard`: 8 görev × 8 kol × 1 tohum.
- HGA'nın rakiplerini geçtiği görevler: F_memory_dependent (+0.0137 vs transformer), H_distractor (+0.0039 vs dense).
- Bellek kanalının katkısı: A_long_chain: +0.0000, B_unseen_entity: +0.0000, C_unseen_relation: +0.0000, D_unseen_both: +0.0000, E_conflict: +0.0000, F_memory_dependent: +0.5117, G_epistemic: +0.0000, H_distractor: +0.0000. Katkı yalnız F'de belirgin değilse kanal bilgi değil kapasite taşıyor demektir.
- Şu görevlerde HİÇBİR parameter-matched nöral kol çoğunluk tabanını +0.10'dan fazla geçemedi: ['A_long_chain', 'B_unseen_entity', 'C_unseen_relation', 'D_unseen_both', 'E_conflict', 'G_epistemic', 'H_distractor']. Bu, mimari farkından önce gelen bir sonuçtur: bu bütçe ve adım sayısında görev nöral kollar için öğrenilemiyor; sembolik kolla karşılaştırma paradigma farkı olarak okunmalıdır.

## Sınırlar

- Görevler sentetik ilişkisel dünyalardır; doğal dil veya gerçek bilgi grafiği sonucu DEĞİLDİR.
- Sembolik kol öğrenmez; doğruluğu görev tanımının kendisinden gelir ve nöral kollarla aynı anlamda 'model performansı' değildir.
- Bellek kanalı iki skaler özetle temsil edilir; tam bir bellek geri çağırma mimarisi değildir. Ölçtüğü şey bilginin ERİŞİLEBİLİRLİĞİdir.
- Parametre bütçesi oranı 1.027 (gövde 1.025); gömme katmanı sözlük boyutuna bağlı olduğu için tam eşitlik değil, kapı (1.1) hedeflenir.
- Tohum sayısı 1; çekirdek iddia için 20 tohum ayrıca koşulmalıdır.
