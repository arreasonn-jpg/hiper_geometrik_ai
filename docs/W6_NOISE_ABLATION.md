# W6 — Noise Ablasyonu (Preprint 3'ün n=13 genişletmesi)

**Tarih:** 2026-10-14
**Script:** `artifacts/w6_noise_ablation.py`
**Sonuç:** `artifacts/w6_runs/w6_full_50seed.json`
**Protokol:** `w6_noise_ablation_v1`

## Amaç
Preprint 3'ün "n=5" sınırlılığını gidermek. Bootstrap CI,
permütasyon testi ve Holm-Bonferroni düzeltmesi eklemek.

## Yöntem
- 13 noise seviyesi: 0.00, 0.05, ..., 0.60
- 50 seed × 100 epoch × n_train=2000
- Deterministik RNG: seed=20261014
- Bootstrap 10.000, permütasyon 10.000

## Sonuçlar

| Noise | Dense | Flat | Flat kazandı |
|---:|---:|---:|---:|
| 0.00 | 0.9805 | 0.9428 | 0/50 |
| 0.05 | 0.9662 | 0.9325 | 0/50 |
| 0.10 | 0.9412 | 0.9144 | 2/50 |
| 0.15 | 0.9136 | 0.8922 | 2/50 |
| 0.20 | 0.8824 | 0.8687 | 7/50 |
| **0.25** | **0.8502** | **0.8457** | **25/50** |
| 0.30 | 0.8196 | 0.8226 | 29/50 |
| 0.35 | 0.7884 | 0.8005 | 39/50 |
| 0.40 | 0.7603 | 0.7796 | 43/50 |
| 0.45 | 0.7333 | 0.7584 | 45/50 |
| 0.50 | 0.7097 | 0.7408 | 47/50 |
| 0.55 | 0.6870 | 0.7224 | 48/50 |
| 0.60 | 0.6642 | 0.7041 | 47/50 |

## İstatistiksel Analiz

- **Pearson r:** +0.956 (n=13)
- **Bootstrap %95 CI:** [+0.919, +0.983] — sıfırı dışlıyor
- **Permütasyon p:** 0.0001 (10.000 permütasyon)
- **Holm-Bonferroni:** p_ham=0.0001 → p_düzeltilmiş=0.0002, karar **REDDET**

## Yorum

Geçiş noktası **noise ≈ 0.25**'te; bu, W5'in 5 seviyeli koşusunda
"~0.30" olarak tahmin edilen değerle uyumlu.

W5'in "50 seeds, n=5" sınırlılığı kapandı: korelasyon n=13'te
aynı kaldı, ancak artık bootstrap CI, permütasyon testi ve
çoklu karşılaştırma düzeltmesi ile destekleniyor.

## Referans
- Preprint 3: `paper/arxiv_submission_p3/main.tex`
- W5 orijinal: `artifacts/w5_noise_ablation.py`
- SECURITY.md: SEC-2026-10-13-01, SEC-2026-10-13-02
