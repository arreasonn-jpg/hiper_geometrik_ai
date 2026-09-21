# Preprint Outline — Hibrit Symbolic-Neural Paradigm

**Hedef Dergi:** arXiv (cs.LG) -> NeurIPS 2026 / ICLR 2027
**Tarih:** 2026-09-21
**Durum:** Taslak (hoca onayi bekleniyor)

## Title (Oneri)

Hybrid Symbolic-Neural Reasoning: +10.6 Points Accuracy and +24.2 Points
Cold-Start Generalization over Neural Baselines

## Abstract (150 kelime)

We present a hybrid symbolic-neural paradigm that combines rule-based
reasoning with neural learning. On a procedurally generated reasoning
task with 120 entities and 8 relations, the hybrid model achieves
0.896 +/- 0.021 accuracy versus 0.790 +/- 0.033 for neural-only and
0.490 +/- 0.036 for symbolic-only (10 seeds). On cold-start generalization
(unseen entities), the hybrid reaches 0.772 versus 0.530 for neural-only.
The mechanism is a loss-free veto: symbolic predictions are trusted when
confidence is high, and neural predictions fill in abstentions. The gain
is statistically significant (Cohen d = 5.07, p < 0.001) and increases
with task size (2.3x entity scaling, 2.4x relation scaling). We also
report five negative results from related experiments (hierarchical
Kronecker, depth scaling, low-rank sparsity, hash refinement, memory
bound) to guide future research.

## 1. Introduction

### Motivasyon
- Klasik neural aglar cold-startta coker
- Sembolik sistemler kesin konusur ama cekimser kalir
- Ikisini birlestirmek dogal cozum

### Katkilar
1. Veto + fallback mekanizmasi (basit, kayipsiz)
2. 10-seed istatistiksel kanit (p<0.001)
3. Mekanizma ablasyonu (monoton azalan kazanc)
4. Robustness (gorevle olcekleniyor)
5. Durust negatif sonuclar (5 hipotez)

## 2. Related Work

### Neuro-Symbolic AI
- Garcez ve Lamb (2023): Neuro-symbolic survey
- Manhaeve ve ark. (2018): DeepProbLog
- dAvila Garcez ve ark. (2019): Logic Tensor Networks

### Hybrid Reasoning
- Marcus (2020): The next decade in AI
- Bengio ve ark. (2021): Hybrid AI roadmap

### Cold-Start Generalization
- Wang ve ark. (2020): Few-shot learning survey
- Bansal ve ark. (2020): Zero-shot learning

## 3. Method

### Task
Sentetik ama prosedurel: (ozne, iliski, nesne) uclusunun dogruluk kontrolu.
- 120 varlik, 8 iliski, 1200 egitim, 400 test
- %30 gizli ozellik (bilgi tabaninda yok)
- %20 gorulmemis varlik (cold-start)

### Kollar
1. Symbolic: Kural bilgi tabani. Ozellik varsa kesin, yoksa cekimser.
2. Neural: Embedding + MLP. Her zaman konusur.
3. Hybrid: Sembolik kesin konusursa o kazanir (veto), yoksa neural doldurur.

### Formul
    hybrid(x) = symbolic(x) if symbolic_confidence(x) > 0 else neural(x)

### Metrikler
- Accuracy (ana metrik)
- Coverage (cekimsizlik orani)
- Accuracy_unseen (cold-start)
- FAR, FRR, F1 (ikili siniflandirma)

## 4. Experiments

### 4.1 Paradigm Comparison (10 seeds)

| Model | Accuracy | Coverage | Unseen |
|---|---|---|---|
| Symbolic | 0.490 +/- 0.036 | 0.490 | 0.501 |
| Neural | 0.790 +/- 0.033 | 1.000 | 0.530 |
| Hybrid | 0.896 +/- 0.021 | 1.000 | 0.772 |

Paired t-test: hybrid-neural d=5.07, p<0.001

### 4.2 Mechanism (Hidden Feature Ablation)

| Hidden Ratio | Symbolic | Neural | Hybrid | Gain |
|---|---|---|---|---|
| 0.00 | 1.0000 | 0.7425 | 1.0000 | +0.2575 |
| 0.30 | 0.4908 | 0.7617 | 0.8850 | +0.1233 |
| 0.70 | 0.0917 | 0.7575 | 0.7733 | +0.0158 |

Kazanc monoton azalir. Formul dogrulanir.

### 4.3 Robustness (Entity + Relation Scaling)

Entity 60->240: gain +0.062 -> +0.142 (2.3x)
Relation 4->16: gain +0.063 -> +0.149 (2.4x)

### 4.4 Negative Results (Kisa)

1. Hierarchical Kronecker: null (30 seeds, p=0.21)
2. Kronecker depth K: etkisiz
3. Low-rank sparse: verimsiz (11.5x kayip)
4. Hash refinement: marjinal (%0.15)
5. Memory bound: ~50K baglam sinir

## 5. Discussion

### Neden Hibrit Kazaniyor?
- Symbolic guvenilir -> hibrit miras alir
- Symbolic cekimser -> neural doldurur
- Kayipsiz veto -> hibrit >= neural her durumda

### Neden Olcekleniyor?
- Neural kapasitesiz kalir gorev buyudukce
- Symbolic iskelet dayaniklilik saglar
- Kazanc gorev buyukluguyle artar

### Sinirlar
- Sentetik gorev (henuz gercek veri yok)
- 120 varlik, 8 iliski (orta olcek)
- CPU-only (GPU'da numerik fark olabilir)

## 6. Conclusion

Hibrit symbolic-neural paradigm, klasik neural aglari +10.6 puan accuracy
ve +24.2 puan cold-start genellemede geciyor. Basit bir veto+fallback
mekanizmasi ile, kayipsiz. Kazanc gorev buyukluguyle artar. Bes negatif
sonuc gelecek arastirmacilara yol gosterir.
