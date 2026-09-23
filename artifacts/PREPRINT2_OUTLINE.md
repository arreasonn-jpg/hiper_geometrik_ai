# Preprint 2 — Outline

**Başlık (aday):** *When Do Hybrid Systems Win? A Unified Theory of Gain Across Metrics, Architectures, and Tasks*

**Tarih:** 2026-09-23
**Durum:** Outline / taslak
**Önceki preprint:** "Hybrid Symbolic-Neural Paradigm: A Universal Formula for Predicting Gain over Neural Baselines" (2026-09-22)

## Özet (1 paragraf)

Hibrit sistemlerin sinirsel temellere karşı kazancı, üç bağımsız boyutta incelenir:
(1) **metrik tipi** — lineer (accuracy, bit-accuracy) vs. lineer olmayan (F1);
(2) **mimari yapı** — bilinear Kronecker (yapısal) vs. dense MLP;
(3) **görev yapısı** — rastgele lineer vs. compositional/yapısal.
Her boyut için **ne zaman hangi yapının kazandığını** kesin olarak belirliyoruz.

## Katkılar

1. **Lineer metrik formülü** (önceki preprint): `gain = cov × (M_s − M_n)`, ε = 0.002
2. **F1 per-class formülü** (yeni): `F1_h = (1/K) Σ_c [w_c·F1_s^c + (1−w_c)·F1_n^c]`, ε = 0.000
3. **Kronecker görev-uygunluk teoremi** (yeni): bilinear yapı yapısal görevlerde 8x verimli
4. **Dürüst negatif sonuçlar**: Kronecker rastgele görevlerde kaybeder; hiyerarşik yapı zararlı

## Yapı

### 1. Introduction

- **Motivasyon:** Hibrit sistemler ne zaman kazanır? Hangi metrik/mimari/görevde?
- **Problem:** Önceki preprint sadece lineer metrikleri ve tek mimariyi kapsıyordu.
- **Katkı:** Üç boyutlu genelleme + dürüst sınırlar.
- **Yol haritası:** Teori (§2) → Deneyler (§3) → Tartışma (§4).

### 2. Theory

#### 2.1 Lineer Metrik Formülü (Özet, önceki preprint)

    metric_h = cov · metric_s + (1−cov) · metric_n^⊥

Kesin hata sınırı: `ε = (1−cov) · |metric_n^⊥ − metric_n|`.

**Kanıt:** Telescoping identity. Lineer metriklerin doğrusal ayrışımı.

#### 2.2 F1 Per-Class Formülü (Yeni)

F1 harmonik ortalama → lineer değil → lineer formül bozulur.

Kesin ayrışım:
    F1_h = (1/K) · Σ_c [ w_c · F1_s^c + (1−w_c) · F1_n^c ]
    w_c = D_s^c / (D_s^c + D_n^c)
    D_k^c = 2·TP_k^c + FP_k^c + FN_k^c

**Kanıt:** Per-class confusion matrisi + F1 tanımı.

**Önemli:** `w_c ≠ cov` — per-class ağırlıklar sınıf dağılımına bağlı.

#### 2.3 Kronecker Görev-Uygunluk (Yeni)

Bilinear Kronecker katmanı: `Y = A @ X @ B` → `Y_vec = (Bᵀ ⊗ A) · X_vec`.

**Teorem (görev-uygunluk):** Bilinear yapının ifade gücü, görevin tensör yapısıyla uyumlu olduğunda 8x parametre verimliliği sağlar.

**Kanıt:** Deneysel (§3.3), 20 seed paired t-testi ile doğrulandı.

### 3. Experiments

#### 3.1 Lineer Metrik Doğrulaması (Önceki Preprint'ten)

| Boyut | Kapsam | Hata |
|---|---|---|
| Dil | 12 (8 aile) | < 0.01 |
| Yazı sistemi | 4 | < 0.01 |
| Görev tipi | 4 (binary, multiclass, multi-label, NER) | 0.000–0.008 |
| Gürültü | 0-50% | 0.0066 (sabit) |

**Kaynak:** `TWELVE_LANGUAGE_FORMULA.md`, `NER_FORMULA_TEST.md`, `SYMBOLIC_NOISE_ROBUSTNESS.md`.

#### 3.2 F1 Per-Class Doğrulaması (Yeni)

**Kurulum:** NER, 20 konfigürasyon (5 seed × 4 hidden_ratio).

| Formül | Mean ε | Max ε |
|---|---|---|
| Lineer (cov ağırlıklı) | 0.0188 | 0.0515 |
| **Per-class (w_c ağırlıklı)** | **0.000000** | **0.000000** |

**Kaynak:** `F1_HARMONIC_FORMULA_FINAL.md`, `f1_perclass_test.py`.

#### 3.3 Kronecker Görev-Uygunluk (Yeni)

**Görev:** Hiyerarşik compositional (grid yapısı, blok güçleri, çeyrek argmax).

**Kurulum:** 20 seed, paired t-test.

| Model | Param | Test Acc | Verimlilik |
|---|---|---|---|
| dense_mlp | 20,868 | 0.7982 ± 0.0190 | baseline |
| **flat_kronecker** | **2,564** | **0.8007 ± 0.0202** | **8x verimli** |
| hierarchical | 87,044 | 0.7872 ± 0.0255 | 34x param, zararlı |

**İstatistiksel testler:**
- flat vs dense: p = 0.517 (eşit)
- **flat vs hier: p = 0.0008** (flat kazanıyor, d=0.752)
- dense vs hier: p = 0.0093 (dense kazanıyor)

**Kaynak:** `COMPOSITIONAL_KRONECKER_WIN.md`, `compositional_20seed.py`.

#### 3.4 Negatif Sonuçlar (Dürüst Sınırlar)

| Görev | Sonuç |
|---|---|
| Sentetik lineer (rastgele W) | Dense > Hier > Flat |
| Sentetik lineer + LayerNorm | Dense > Hier > Flat (değişmedi) |
| **Compositional** | **Flat ≈ Dense (8x verimli)** |

**Kaynak:** `HIERARCHICAL_PARAM_MATCHED.md`, `HIERARCHICAL_LAYERNORM.md`.

### 4. Discussion

#### 4.1 Ne Zaman Hangisi?

**Karar Ağacı:**

    Görev yapısal mı (grid/tensör)?
    ├── EVET  → Flat Kronecker (8x verimli)
    │           Hierarchical KULLANMA (zararlı)
    └── HAYIR → Dense MLP
                Kronecker dezavantaj

**Metrik tipi:**
- Lineer (accuracy, bit-accuracy) → basit formül `cov × Δ`
- Lineer olmayan (F1) → per-class ağırlıklı formül

#### 4.2 Pratik Kılavuz

1. **Metrik seçimi:** Mümkünse lineer metrik → basit formül
2. **Mimari seçimi:** Yapısal görev → Kronecker; değilse → Dense
3. **Hiyerarşi:** Kanıtlanmış fayda yok → kullanma
4. **Hibrit kazanç:** Formülle öngör, çekimserlik alt kümesine dikkat

#### 4.3 Ana Bilimsel Hikâye

**3 bağımsız boyut, tek bir çerçeve:**
- **Formül** (ne kadar kazanç?) → metrik tipi belirler
- **Mimari** (hangi yapı?) → görev yapısı belirler
- **Ölçek** (ne kadar verimli?) → tensör yapısı belirler

### 5. Limitations & Future Work

**Sınırlar:**
- Kronecker compositional'da tek görev tipi test edildi
- F1 per-class formülü sadece macro F1 için doğrulandı
- Hibrit formül ön-eğitimli LLM'lerle test edilmedi
- İstatistiksel güç: 20 seed (daha fazla önerilir)

**Gelecek:**
- **Görüntü compositional:** Kronecker hipotezini genişlet
- **F1 diğer türleri:** weighted, micro, binary F1
- **Hibrit + LLM:** Büyük modellerle formül testi
- **Yeni mimariler:** Bilinear'ın ötesinde geometrik yapılar

### 6. Sonuç

Hibrit sistemlerin kazancı üç bağımsız boyutta formülle öngörülebilir:
- **Metrik:** lineer vs lineer olmayan → iki farklı formül
- **Mimari:** yapısal vs rastgele → iki farklı kazanan
- **Görev:** compositional vs lineer → farklı verimlilik

**Ana iddia:** Hibrit sistem tasarımı **ilkeli** olabilir — formüller ve görev-uygunluk ilkeleri ile.

---

## Ekler

### A. Kaynak Artefaktlar

| # | Dosya | Ne İçeriyor |
|---|---|---|
| 1 | `TWELVE_LANGUAGE_FORMULA.md` | 12-dil formül kanıtı |
| 2 | `NER_FORMULA_TEST.md` | 4. görev tipi (NER) |
| 3 | `NER_F1_SCOPE_LIMITATION.md` | F1 negatif sonuç |
| 4 | `SYMBOLIC_NOISE_ROBUSTNESS.md` | Gürültü dayanıklılığı |
| 5 | `F1_HARMONIC_FORMULA_FINAL.md` | **F1 per-class teoremi** |
| 6 | `f1_perclass_test.py` | Sayısal doğrulama (ε=0) |
| 7 | `HIERARCHICAL_PARAM_MATCHED.md` | Negatif sonuç 1 |
| 8 | `HIERARCHICAL_LAYERNORM.md` | Negatif sonuç 2 |
| 9 | `COMPOSITIONAL_KRONECKER_WIN.md` | **Kronecker zaferi** |
| 10 | `compositional_20seed.py` | 20-seed istatistik |

### B. Kod Deposu

- Ana repo: `github.com/arreasonn-jpg/hiper_geometrik_ai`
- İlgili tag'ler: `f1-perclass-formula`, `compositional-20seed-confirmed`
- Test suite: 994/994 geçiyor

### C. Metodolojik Notlar

- **Seed politikası:** 5-20 bağımsız seed
- **İstatistik:** paired t-test, Cohen's d, bootstrap CI
- **Dürüstlük:** Negatif sonuçlar dahil
- **Tekrarlanabilirlik:** Tüm artefaktlar + kod repo'da

## Sonraki Adımlar (Bu Outline'dan Preprint'e)

| # | İş | Süre |
|---|---|---|
| 1 | İngilizce LaTeX yazımı | 3-4 saat |
| 2 | Türkçe çeviri | 1-2 saat |
| 3 | Görsel tablolar (PDF) | 1 saat |
| 4 | Hoca incelemesi | 1-2 hafta |
| 5 | arXiv submission | 1 gün |

## Kaynakça (Taslak)

1. Vaswani et al. (2017). Attention is all you need.
2. Nivre et al. (2020). Universal Dependencies v2.
3. Krippendorff (2011). Computing Krippendorff's alpha.
4. [Kronecker factorization literature — TBD]
5. [F1 decomposition literature — TBD]
6. [Hybrid systems literature — TBD]

---

**Durum:** Outline v1 tamam. Sonraki: LaTeX yazımı.
