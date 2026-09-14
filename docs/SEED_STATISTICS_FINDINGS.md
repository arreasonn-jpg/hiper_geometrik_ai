# 20 Tohum + İstatistiksel Çıkarım — Bulgular (P1)

Protokol: `core_seed_statistics_v1` · Modül: `hga/evaluation/seed_statistics.py`
CLI: `python -m hga tohum-istatistik --seeds 1,...,20`
Ham çıktı: `docs/SEED_STATISTICS.md`, `docs/seed_statistics.json`

Çekirdek protokoller (`priority_ablation`, `signature`, `operator_baselines`)
artık 3–5 tohum yerine **20 tohum** ile, **aynı tohum kümesinde eşleşmiş**
olarak koşuyor. 76 eşleşmiş karşılaştırma üretildi; her biri %95 bootstrap
CI + eşleşmiş Cohen's d + permütasyon testi + Wilcoxon testi taşıyor.

n=20'de ulaşılabilir en küçük iki yönlü p = `1.9e-06`, yani tasarım artık
p<0.01 saptayabiliyor. (n=3'te bu sınır 0.25'ti — eski koşumlar hiçbir
anlamlılık iddiasını destekleyemezdi ve rapor bunu açıkça söylüyordu.)

## 1. HGA bileşen ablasyonları — büyük ölçüde dürüst negatif

24 ablasyon karşılaştırmasının yalnız **2'sinde** fark istatistiksel olarak
ayrışıyor:

| Görev | Çıkarılan bileşen | Cohen's d | p |
|---|---|---:|---:|
| F_memory_dependent | bellek | **+2.780** | 0.00005 |
| C_unseen_relation | attention | +0.523 | 0.031 |

Yorum: **bellek gerçek ve büyük bir katkı sağlıyor** — ama yalnız bellek
gerektiren görevde, ki beklenen budur. Attention'ın katkısı yalnız bir
görevde ve orta büyüklükte.

Geri kalan **22 karşılaştırmada bileşeni çıkarmak ölçülebilir bir fark
yaratmıyor.** Özellikle `hga_no_kronecker` hiçbir görevde ayrışmıyor:
Signature görevlerinde Kronecker yapısının katkısı **gösterilememiştir**.
`hga_no_memory` ise F dışındaki tüm görevlerde tam olarak 0.0000 fark
veriyor — yani bellek o görevlerde hiç devreye girmiyor.

Bu, mimarinin bir kısmının bu görev ailesinde **ölü ağırlık** olduğunu
gösterir. Sonuç gizlenmedi.

## 2. HGA vs Dense / Transformer

8 görev × 2 rakip = 16 karşılaştırmanın 4'ünde ayrışma var:

| Görev | Rakip | Δ (HGA lehine) | Sonuç |
|---|---|---:|---|
| F_memory_dependent | dense | +0.3418 | HGA üstün |
| F_memory_dependent | transformer | +0.1867 | HGA üstün |
| B_unseen_entity | dense | +0.0238 | HGA üstün |
| H_distractor | transformer | **−0.0352** | **HGA geride** |

HGA'nın net üstünlüğü **bellek gerektiren göreve** yoğunlaşıyor. Diğer
12 karşılaştırmada kollar ayrıştırılamıyor. Dolgu-yoğun görevde (H)
transformer HGA'yı geçiyor.

`symbolic` ve `hybrid` kolları tüm görevlerde HGA'yı büyük farkla
(d ≈ −12) yeniyor; bu görev ailesinin sembolik olarak çözülebilir
olduğunu, yani nöral kolların asıl rakibinin sembolik çözücü olduğunu
gösterir.

## 3. Kronecker vs Dense — avantaj yapıya koşullu (P0-2 teyit edildi)

20 karşılaştırmanın 19'u ayrışıyor. Kritik olan **yön**:

| Öğretmen (veri yapısı) | Kronecker sonucu |
|---|---|
| `kronecker_teacher` | **4/4 kazanıyor** (rank1, low-rank ×2, kron_sum_2'ye karşı) |
| `rank1_teacher` | 5/5 **kaybediyor** |
| `low_rank_teacher` | 5/5 **kaybediyor** |
| `full_dense_teacher` | 5/5 **kaybediyor** |

Kronecker yalnızca **veri gerçekten Kronecker yapılıysa** kazanıyor. Veri
başka bir yapıdaysa Full Dense ve low-rank kolları onu p<0.001 ile
yeniyor. Bu, 20 tohumla ve eşit parametre/FLOP bütçesiyle doğrulanmış
haliyle P0-2'nin sonucudur: **Kronecker genel bir üstünlük değildir,
yapısal bir eşleşme avantajıdır.**

## 4. Priority(E) ağırlıkları — hangisi gerçekten nedensel?

Downstream `verification_yield` farkının %95 CI'si:

| Ağırlık | Δ ortalama | %95 CI | Sıfırı dışlıyor mu? |
|---|---:|---|---|
| `w_gain` | **+0.2800** | [+0.1750, +0.3800] | **EVET** |
| `w_uncertainty` | −0.0200 | [−0.0400, −0.0050] | **EVET** |
| `w_conflict_penalty` | +0.0200 | [−0.0050, +0.0450] | hayır |
| `w_novelty` | 0.0000 | [0.0000, 0.0000] | hayır |

`w_gain` tek başına baskın nedensel terim (P0-1'deki 0.38→0.62 bulgusunu
20 tohumla teyit eder). `w_uncertainty`'nin etkisi küçük ama gerçek ve
**negatif** — sıfırlamak downstream verimi artırıyor, yani mevcut
ağırlığı zararlı. `w_novelty` downstream'e hiç etki etmiyor.

## Kabul kapıları (8/8 GEÇTİ)

`all_core_protocols_use_same_seeds`, `all_core_protocols_meet_20_seeds`,
`comparisons_are_paired`, `every_comparison_reports_ci`,
`every_comparison_reports_effect_size`,
`every_comparison_reports_two_tests`, `power_limit_documented`,
`design_can_reach_p_0_05`.

## Sınırlar

- **Çoklu karşılaştırma düzeltmesi (Bonferroni/FDR) uygulanmadı.** 76
  kıyasta tek tek p-değerleri iyimserdir; p≈0.03 olan tek sonuç
  (`C_unseen_relation` / attention) düzeltme altında ayakta kalmayabilir.
  d≈12 ve p=0.00005 olan sonuçlar bundan etkilenmez.
- Tohum artırmak varyans tahminini düzeltir, **sistematik yanlılığı
  düzeltmez** (görev tasarımı, veri üretimi).
- Bootstrap CI küçük n'de asimptotik değildir; n=20'de kapsama gerçek
  değerin biraz altında olabilir.
- Eşleşmiş testler aynı tohumun aynı veri havuzunu ürettiğini varsayar;
  protokoller bunu garanti eder.
