# Metrik Kapsamı — F1 Lineer Değildir

**Tarih:** 2026-09-23
**Bulgu:** Formül lineer metrikler için geçerli; F1 için geçerli değil.

## Gözlem

| Metrik | Ortalama ε | Maksimum ε |
|---|---|---|
| Token accuracy | 0.0083 | 0.0175 |
| **F1-macro** | **0.0188** | **0.0515** |

## Neden?

Kesin formül şu özdeşliğe dayanır:

    metric_hybrid = cov × metric_s + (1−cov) × metric_n^⊥

Bu **sadece lineer** metrikler için doğrudur (accuracy, bit-accuracy).
F1 harmonik ortalama içerdiği için özdeşlik **bozulur**:

    F1_hybrid ≠ cov × F1_s + (1−cov) × F1_n^⊥

Sayısal sonuç: 20 konfigürasyonda ratio (obs/pred) ortalama **16.4**
(0.05–234). Bu, kesin formülün F1 için geçersiz olduğunu gösterir.

## Bilimsel Değer

Bu negatif sonuç, formülün **kesin kapsamını** tanımlar:

| Metrik Tipi | Formül |
|---|---|
| Lineer (accuracy, bit-accuracy) | ✅ Geçerli |
| Lineer olmayan (F1, precision, recall) | ❌ Geçerli değil |

## Preprint İçin Öneri

> "The formula holds exactly for linear metrics (accuracy,
> bit-accuracy). For non-linear metrics such as F1, the hybrid's
> metric is not a convex combination of the arms' metrics, and
> the formula's error is no longer bounded by Equation 6."

## Gelecek Çalışma

F1 için harmonik formül gerekir — açık problem.
