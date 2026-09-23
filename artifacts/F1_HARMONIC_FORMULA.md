# F1 Harmonik Formülü — Lineer Olmayan Metrikler İçin Genelleme

**Tarih:** 2026-09-23
**Bağlam:** `NER_F1_SCOPE_LIMITATION.md`'de belgelenen F1 negatif sonucunun matematiksel çözümü.

## Problem

Lineer metrikler (accuracy, bit-accuracy, token-accuracy) için kesin özdeşlik:

    metric_h = cov × metric_s + (1−cov) × metric_n^⊥

F1 için bu **bozulur** — harmonik ortalama lineer değildir.
20 NER konfigürasyonunda gözlenen/öngörülen hata oranı ortalama **16.4**.

## Türetim

Test kümesinde N öğe. Kapsam cov → N_ans = cov·N, N_abs = (1−cov)·N.

**Cevap verilen alt kümede (sembolik):**
- TP_s, FP_s, FN_s, TN_s (N_ans üzerinden)

**Çekimser kalınan alt kümede (sinirsel):**
- TP_n, FP_n, FN_n, TN_n (N_abs üzerinden)

**Hibrit:**
- TP_h = TP_s + TP_n
- FP_h = FP_s + FP_n
- FN_h = FN_s + FN_n

**F1 tanımı:**
    F1_h = 2·TP_h / (2·TP_h + FP_h + FN_h)

**Alt küme paydalarını tanımla:**
    D_s = 2·TP_s + FP_s + FN_s    (cevap verilen)
    D_n = 2·TP_n + FP_n + FN_n    (çekimser kalınan)

Bu durumda:
    F1_s = 2·TP_s / D_s
    F1_n^⊥ = 2·TP_n / D_n

## Ana Sonuç

Hibrit F1:
    F1_h = 2·(TP_s + TP_n) / (D_s + D_n)
         = (D_s · F1_s + D_n · F1_n^⊥) / (D_s + D_n)

**Ağırlık tanımı:** W = D_s / (D_s + D_n)

Böylece:
    F1_h = W · F1_s + (1−W) · F1_n^⊥

Kazanç:
    gain_F1 = W · (F1_s − F1_n) + (1−W) · (F1_n^⊥ − F1_n)

Hata sınırı:
    ε_F1 = (1−W) · |F1_n^⊥ − F1_n|

## Lineer Metrikle İlişki

Lineer metrikte ağırlık **W = cov** (her öğe eşit katkı).
F1'de **W = D_s / (D_s + D_n)** — pozitif içeriğin oranı.

D, "pozitif kütlesi"dir:
    D_s = (predicted_positives_s) + (actual_positives_s)
    D_n = (predicted_positives_n) + (actual_positives_n)

**Kritik gözlem:** Sınıf dengesi alt kümeler arasında homojense,
D_s ∝ cov ve D_n ∝ (1−cov), yani W → cov. Homojen değilse W ≠ cov.

Bu, lineer formülün F1'de neden bozulduğunu **kesin olarak açıklar**.

## Sayısal Doğrulama (Dürüst Sonuç)

20 NER konfigürasyonunda test edildi. **Sürpriz bulgu:**

| Formül | Mean ε | Max ε |
|---|---|---|
| Lineer: `cov × (F1_s − F1_n)` | 0.0188 | 0.0515 |
| **Harmonik: `W × (F1_s − F1_n)`** | **0.0190** | **0.0480** |

**İyileştirme: 1x (yok).** W her konfigürasyonda **tam olarak cov'a eşit** çıktı.

### Neden?

Tek-etiketli sınıflandırmada şu **teorem** geçerlidir:

    D = Σ_c (2·TP_c + FP_c + FN_c) = 2N    (her zaman)

İspat: Σ_c TP_c = doğru, Σ_c FP_c = N−doğru, Σ_c FN_c = N−doğru
      D = 2·doğru + (N−doğru) + (N−doğru) = 2N

Dolayısıyla: W = D_s/(D_s+D_n) = N_ans/N = cov.

Bu, **türetimimin micro F1 için doğru olduğunu** gösterir.
Ama micro F1 = accuracy (lineer zaten). Kod ise **macro F1** hesaplıyor.

### Doğru Macro F1 Formülü

    F1_h_macro = (1/K) · Σ_c [ w_c · F1_s^c + (1−w_c) · F1_n^c ]

    w_c = D_s^c / (D_s^c + D_n^c)    ← per-class!

**Per-class ağırlıklar cov'a eşit değildir.** Macro F1 için gerçek genelleme budur.

## Gelecek Çalışma

Per-class ağırlıklı formülü test etmek için:
1. Her sınıf için ayrı ağırlık hesapla
2. Ağırlıkların dağılımını incele
3. Predictor için ne gerektiğini belirle (cov + sınıf dağılımı?)

## Bilimsel Değer

**Negatif sonuç = pozitif katkı.** İki şey öğrendik:
1. Lineer metrikler için W=cov **matematiksel teorem** (ispatlı)
2. Macro F1 için per-class ağırlık **gereklidir** — doğal sonraki adım
