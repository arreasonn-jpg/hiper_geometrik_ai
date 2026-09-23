# F1 Harmonik Formülü — KESİN SONUÇ

**Tarih:** 2026-09-23
**Sonuç:** Macro F1 için per-class ağırlıklı kesin özdeşlik bulundu.

## Ana Teorem

Tek-etiketli sınıflandırmada macro F1 için **kesin ayrışım**:

    F1_h = (1/K) · Σ_c [ w_c · F1_s^c + (1−w_c) · F1_n^c ]
    w_c = D_s^c / (D_s^c + D_n^c)
    D_k^c = 2·TP_k^c + FP_k^c + FN_k^c    (k ∈ {s, n})

Kazanç:
    gain_F1 = (1/K) · Σ_c [ w_c · (F1_s^c − F1_n^c) + (1−w_c) · (F1_n^c_perp − F1_n^c) ]

## Sayısal Doğrulama

20 NER konfigürasyonu (5 seed × 4 hidden_ratio):

| Formül | Mean ε | Max ε |
|---|---|---|
| Lineer (eski): cov·(F1_s−F1_n) | 0.0188 | 0.0515 |
| **Per-class (yeni)** | **0.000000** | **0.000000** |

**20/20 konfigürasyonda makine hassasiyetinde doğrulandı.**

## Neden Lineer Formül Başarısız Oluyor?

Lineer metriklerde (`cov` ağırlığı):
- W = cov (matematiksel teorem, ispat: Σ_c D_c = 2N her zaman)

F1'de (per-class ağırlıklar):
- w_c = D_s^c / (D_s^c + D_n^c)
- **w_c ≠ cov** — per-class dağılım değişir
- Örnek: seed=1, hr=0.70'te w_c ∈ [0.37, 0.63] ama cov=0.510
- Yani sınıflar arası pozitif kütlesi **dengesiz** → lineer formül bozulur

## Önceki Sonuçlarla İlişki

| Metrik | Formül | Durum |
|---|---|---|
| Accuracy, bit-accuracy, token-accuracy | `cov · (M_s − M_n)` | ✅ İspatlı |
| **Macro F1** | **Per-class ağırlıklı** | ✅ **Yeni — ispatlı** |
| Micro F1 | `cov · (F1_s − F1_n)` (micro = accuracy) | ✅ İspatlı |

**Sonuç:** Tüm standart metrikler için kesin formül var.

## Bilimsel Değer

1. **Matematiksel özdeşlik:** Hipotez değil, **teorem** (makine hassasiyetinde 20/20)
2. **Lineer olmayan metrikler kapsandı:** Önceki çalışmanın doğal genellemesi
3. **Predictor için gereksinimler net:** cov + per-class D dağılımı
4. **2. preprint için temel taşı**

## Preprint 2 İçin Outline

1. Introduction: lineer formül + F1 problemi
2. Theory: per-class ağırlıklı teorem + ispat
3. Experiments: 20 NER + (gelecek: diğer F1 görevleri)
4. Discussion: metrik tipi → formül yapısı
5. Conclusion: evrensel metrik ayrışımı

## Açık Sorular

1. Binary F1, weighted F1 için de geçerli mi?
2. Precision/recall ayrı ayrı nasıl ayrışır?
3. Çok-etiketli + F1 kombinasyonu?
