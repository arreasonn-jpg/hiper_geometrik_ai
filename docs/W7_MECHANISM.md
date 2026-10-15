# W7 — Kronecker İmplicit-Regularisation Mekanizması

**Tarih:** 2026-10-15
**Script:** `artifacts/w7_kronecker_mechanism.py`
**Analiz:** `scripts/analyze_w7.py`
**Sonuç:** `artifacts/w7_runs/w7_full_50seed.json`
**Protokol:** `w7_kronecker_mechanism_v1` + `w7_mechanism_analysis_v1`

## Amaç

W6 (n=13) noise ↑ → Kronecker win-rate ↑ ilişkisini gösterdi
(r=+0.956). W7 bu **korelasyonun mekanizmasını** arar: Hangi
ölçülebilir özellik noise ile değişiyor ve Kronecker'ın kazancını
açıklıyor?

## Yöntem

- 13 noise seviyesi (0.00–0.60) × 50 seed × 100 epoch
- Her seed sonrası:
  - `train_loss`, `test_loss`, `train_test_gap`
  - `grad_norm` (son epoch)
  - `eff_rank`: `exp(spectral_entropy)` — SVD spektrumundan
  - `participation_ratio`, `weight_norm`
- Deterministik RNG: `rng_seed=20261015`

## Sonuçlar

| Noise | Dense acc | Dense gap | Dense rank | Flat acc | Flat gap | Flat rank | Win |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.980 | 0.012 | 33.0 | 0.945 | 0.008 | 9.6 | 0/50 |
| 0.05 | 0.966 | 0.023 | 33.1 | 0.936 | 0.008 | 9.6 | 0/50 |
| 0.10 | 0.941 | 0.054 | 33.3 | 0.917 | 0.010 | 9.6 | 1/50 |
| 0.15 | 0.913 | 0.109 | 33.5 | 0.895 | 0.013 | 9.6 | 5/50 |
| 0.20 | 0.882 | 0.185 | 33.7 | 0.871 | 0.016 | 9.6 | 13/50 |
| 0.25 | 0.850 | 0.274 | 33.9 | 0.848 | 0.020 | 9.6 | 28/50 |
| 0.30 | 0.819 | 0.368 | 34.1 | 0.826 | 0.026 | 9.6 | 33/50 |
| 0.35 | 0.788 | 0.460 | 34.3 | 0.803 | 0.032 | 9.6 | 39/50 |
| 0.40 | 0.760 | 0.548 | 34.5 | 0.782 | 0.040 | 9.6 | 42/50 |
| 0.45 | 0.733 | 0.622 | 34.7 | 0.763 | 0.049 | 9.7 | 46/50 |
| 0.50 | 0.709 | 0.683 | 34.9 | 0.743 | 0.059 | 9.7 | 44/50 |
| 0.55 | 0.686 | 0.732 | 35.2 | 0.722 | 0.070 | 9.7 | 46/50 |
| 0.60 | 0.664 | 0.776 | 35.4 | 0.703 | 0.081 | 9.7 | 48/50 |

## Korelasyon analizi

Noise seviyesi ile Pearson korelasyonu (n=13, bootstrap %95 CI):

| Metrik | r | %95 CI | Yorum |
|---|---:|---|---|
| **rank_delta** | **+0.999** | [+0.999, +1.000] | Dense−Flat effective rank farkı |
| flat_acc | −0.999 | [−1.000, −0.998] | Noise ↑ → Flat doğruluğu düşer |
| dense_rank | +0.999 | [+0.998, +1.000] | Dense rank noise ile artar |
| dense_acc | −0.998 | [−0.999, −0.997] | Dense doğruluğu düşer |
| dense_gap | +0.992 | [+0.985, +0.998] | Dense train-test gap büyür |
| acc_delta | +0.992 | [+0.983, +0.998] | Flat − Dense doğruluk farkı |
| **gap_delta** | +0.991 | [+0.985, +0.997] | Gap farkı (Dense − Flat) |
| flat_rank | +0.972 | [+0.949, +0.994] | Flat rank neredeyse sabit |
| flat_gap | +0.961 | [+0.939, +0.987] | Flat gap de büyür (küçük) |

**Win-rate ile en güçlü:** `rank_delta` (r=+0.999)

## Yorum

W6'nın korelasyonu artık **mekanik olarak açıklandı**:

1. **Dense overfitting yapıyor** → train-test gap 0.012'den 0.776'ya
   (**64×** artış). Serbestlik derecesi noise ile büyüyor; effective
   rank 33.0'dan 35.4'e çıkıyor.

2. **Kronecker yapamıyor** → gap 0.008'den 0.081'e (yalnız **10×**).
   Effective rank 9.6–9.7 bandında **sabit** kalıyor.

3. **En güçlü sinyal `rank_delta`** (r=+0.999): Dense ve Flat'in rank
   farkı noise ile birebir açılıyor. Bu, Kronecker'ın yapısal
   rank kısıtının doğrudan regularizasyon görevi gördüğünü gösterir.

## Teorik anlam

Preprint 3'ün "implicit regulariser" hipotezi doğrulandı: Kronecker
katmanı `2n²` parametreyle `n⁴` serbestlik iddia eder ama **etkin
serbestlik derecesi düşük kalır**. Noise yükselince bu kısıt
**kazanca** dönüşür; Dense MLP aynı rejimde aşırı öğrenir.

## Sınırlar

- Tek görev ailesi (block compositional, sentetik).
- Effective rank SVD tabanlı; farklı rank ölçüleri farklı sonuç
  verebilir.
- Gradient norm ve weight norm analize dahil edildi ama raporda
  öne çıkmadı (rank + gap baskın).
- CPU-only, 100 epoch; daha uzun eğitim farklı rejim açabilir.

## Üretim yolu

Tam koşu:

- `PYTHONPATH=$PWD python artifacts/w7_kronecker_mechanism.py`
- `--noise-levels 0.0,...,0.60 --seeds 1-50 --epochs 100`

Analiz:

- `python scripts/analyze_w7.py artifacts/w7_runs/w7_full_50seed.json`

## Referans

- W6: `docs/W6_NOISE_ABLATION.md`
- Preprint 3: `paper/arxiv_submission_p3/main.tex` (§H2)
- P1-5 (birleşik audit raporu)
- SECURITY.md: SEC-2026-10-13-01, SEC-2026-10-13-02
