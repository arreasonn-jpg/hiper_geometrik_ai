# Weighted ve Micro F1 için Kesin Formüller

**Tarih:** 2026-09-24
**Bağlam:** `F1_HARMONIC_FORMULA_FINAL.md` (macro F1) çalışmasının doğal genişletmesi.

## Özet

Önceki çalışmada macro F1 için sınıf-başına ağırlıklı kesin özdeşlik
türetildi:

    F1_h^macro = (1/K) · Σ_c [ w_c · F1_s^c + (1-w_c) · F1_n^c ]
    w_c = D_s^c / (D_s^c + D_n^c)

Bu çalışmada aynı çerçeveyi **weighted F1**, **micro F1** ve
**precision/recall** metriklerine genişletiyoruz.

## 1. Weighted F1

**Tanım (standart):**

    F1_h^weighted = Σ_c (n_c / N) · F1_h^c

burada $n_c$ sınıf $c$'nin gerçek örneklem sayısı, $N$ toplam örnek.

**Ayrışım:** Her $F1_h^c$ için önceki macro formül geçerli:

    F1_h^c = w_c · F1_s^c + (1-w_c) · F1_n^c

Yerine koy:

    F1_h^weighted = Σ_c (n_c / N) · [w_c · F1_s^c + (1-w_c) · F1_n^c]

**Sonuç:** Weighted F1, macro F1'in **sınıf-frekansı ağırlıklı** versiyonudur.
Ağırlık artık iki katmanlı: sınıf frekansı $n_c / N$ × hibrit ağırlık $w_c$.

**Kritik gözlem:** $n_c$ terimi hem cevaplanan hem çekimser kalınan
alt kümeleri kapsar (tüm test setinde sınıf dağılımı). Bu nedenle
$a_c = n_c / N$ **doğrudan gözlenebilir** — predictor için ek bilgi gerekmez.

**Predictor formülü:**

    F1_h^weighted = Σ_c a_c · [w_c · F1_s^c + (1-w_c) · F1_n^c]

burada $a_c, w_c$ test setinden hesaplanabilir.

## 2. Micro F1

**Tanım:** Tüm sınıflar birleşik tek bir ikili sınıflandırma gibi işlenir.

    P_h^micro = Σ_c TP_h^c / (Σ_c TP_h^c + Σ_c FP_h^c)
    R_h^micro = Σ_c TP_h^c / (Σ_c TP_h^c + Σ_c FN_h^c)
    F1_h^micro = 2 · P_h^micro · R_h^micro / (P_h^micro + R_h^micro)

**Basitleştirme:** Tek-etiketli sınıflandırmada (her örnek tek sınıfa ait):

    Σ_c TP_h^c = doğru sayısı (accuracy · N)
    Σ_c FP_h^c = yanlış sayısı
    Σ_c FN_h^c = yanlış sayısı

Bu nedenle:

    P_h^micro = R_h^micro = F1_h^micro = accuracy_h

**Sonuç:** Micro F1 = accuracy = **lineer metrik**.

Dolayısıyla önceki lineer formül doğrudan geçerli:

    F1_h^micro = cov · F1_s^micro + (1-cov) · F1_n^micro

**Predictor:** $w = cov$ yeterli, per-class ağırlık gerekmez.

## 3. Precision ve Recall (Ayrı Ayrı)

**Tanımlar (makro):**

    P_h = (1/K) · Σ_c P_h^c = (1/K) · Σ_c TP_h^c / (TP_h^c + FP_h^c)
    R_h = (1/K) · Σ_c R_h^c = (1/K) · Σ_c TP_h^c / (TP_h^c + FN_h^c)

**Ayrışım:** Her sınıf için TP_h^c = TP_s^c + TP_n^c.
Ama **payda** farklı:
- Precision paydası: TP_h^c + FP_h^c (tahmin sayısı)
- Recall paydası: TP_h^c + FN_h^c (gerçek sayısı)

**Bu, F1'den farklı bir zorluk yaratır.** Precision ve recall için
payda **lineer ayrışmaz** — çünkü $TP_s^c + FP_s^c$ ile
$TP_n^c + FP_n^c$ ayrı ayrı toplanır, ama oranları lineer değildir.

**Sonuç (Precision için):**

    P_h^c = [D_s^{P,c} · P_s^c + D_n^{P,c} · P_n^c] / (D_s^{P,c} + D_n^{P,c})

burada $D_k^{P,c} = TP_k^c + FP_k^c$ (tahmin kütlesi).
Yani **per-class ağırlıklı bir özdeşlik** de geçerlidir:

    P_h^c = u_c · P_s^c + (1-u_c) · P_n^c
    u_c = D_s^{P,c} / (D_s^{P,c} + D_n^{P,c})

**Aynı yapı Recall için de geçerli:**

    R_h^c = v_c · R_s^c + (1-v_c) · R_n^c
    v_c = D_s^{R,c} / (D_s^{R,c} + D_n^{R,c})
    D_k^{R,c} = TP_k^c + FN_k^c

## 4. Genel Çerçeve: Tüm Metrikler İçin Ayrışım

| Metrik | Ağırlık | Lineer mi? |
|---|---|---|
| Accuracy | $w = \text{cov}$ | ✅ (tek ağırlık) |
| Micro F1 | $w = \text{cov}$ | ✅ (accuracy'ye eşit) |
| Macro F1 | $w_c = D_s^c / (D_s^c + D_n^c)$ | ❌ (per-class) |
| Weighted F1 | $a_c \cdot w_c$ | ❌ (iki katmanlı) |
| Precision | $u_c = D_s^{P,c} / (D_s^{P,c} + D_n^{P,c})$ | ❌ (per-class) |
| Recall | $v_c = D_s^{R,c} / (D_s^{R,c} + D_n^{R,c})$ | ❌ (per-class) |

**Ana gözlem:** Tüm **lineer metrikler** tek ağırlıkla ifade edilir
($w = \text{cov}$). Tüm **lineer olmayan metrikler** per-class ağırlık
gerektirir, ve ağırlık tipi metrik tipine bağlıdır:

- **F1**: $D = 2TP + FP + FN$ (harmonik yapı)
- **Precision**: $D = TP + FP$ (tahmin kütlesi)
- **Recall**: $D = TP + FN$ (gerçek kütlesi)

**Bu, tek bir ilkeye indirgenebilir:** Her metrik için, "pozitif kütlesi"
$D$ tanımlanır ve hibrit ağırlık $w = D_s / (D_s + D_n)$ olur.

## 5. Sayısal Doğrulama

**Kurulum:** 20 NER konfigürasyonu (5 seed × 4 hidden_ratio).

| Metrik | Mean ε | Max ε |
|---|---|---|
| Macro F1 | 0.000000 | 0.000000 |
| **Weighted F1** | **0.000000** | **0.000000** |

**Her iki formül de makine hassasiyetinde doğrulandı.**

## 6. Dürüst Çerçeve

**Bu, yeni bir teorem değildir.** Weighted F1 formülü macro F1
formülünün doğrudan bir sonucudur:

    F1_h^weighted = Σ_c a_c · F1_h^c
                  = Σ_c a_c · [w_c · F1_s^c + (1-w_c) · F1_n^c]

Yani matematiksel olarak **trivial bir genişletme**.

**Pratik değeri:**
1. Weighted F1, dengesiz veri setlerinde **yaygın olarak kullanılır**
2. Bu formülün doğrulanması, hibrit sistemlerde weighted F1 için
   predictor gereksinimlerini netleştirir
3. Tek bir ilke altında (per-class ağırlık) **tüm F1 varyantlarını**
   birleştirir

## 7. Ana Katkı: Genel Çerçeve

| Metrik | Ağırlık | Predictor Gereksinimi |
|---|---|---|
| Accuracy | $w = \text{cov}$ | cov |
| Micro F1 | $w = \text{cov}$ | cov |
| Macro F1 | $w_c = D_s^c/(D_s^c+D_n^c)$ | cov + per-class D |
| Weighted F1 | $a_c \cdot w_c$ | cov + per-class D + class freq |
| Precision | $u_c = D_s^{P,c}/(D_s^{P,c}+D_n^{P,c})$ | cov + per-class prediction mass |
| Recall | $v_c = D_s^{R,c}/(D_s^{R,c}+D_n^{R,c})$ | cov + per-class ground truth mass |

**İlke:** Her metrik için "pozitif kütlesi" $D$ tanımlanır,
hibrit ağırlık $w = D_s / (D_s + D_n)$ olur.

**Bu, preprint 2'nin F1 bölümünü güçlendirir** — sadece macro F1 değil,
tüm standart metrikler için kesin formüller var.
