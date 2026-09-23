# Parametre-Eşleşmeli Hiyerarşik Kronecker Ablation — Dürüst Sonuç

**Tarih:** 2026-09-23
**Amaç:** Aynı parametre bütçesinde "hiper geometrik" mimarinin standart MLP'ye karşı avantajı var mı?

## Deney Tasarımı

**Görev:** Sabit doğrusal öğretmen (class = argmax(W·flatten(x)))
**Girdi:** (n_base, n_base) = (8, 8) matris → 64 boyut
**Sınıf:** 4
**Seed:** 5 bağımsız
**Epoch:** 300

**İki koşul:**
- **A) Depth-eşleşmiş:** Tüm modeller K=3 katman
- **B) Param-eşleşmiş:** Hedef ~21,764 param (hier K=3 exp=2 ile eşleşen)

## Sonuçlar (5 seed ortalama)

### A) Depth-eşleşmiş (K=3)

| Model | Parametre | Test Acc |
|---|---|---|
| Flat Kronecker | 644 | 0.558 ± 0.019 |
| Hierarchical Kronecker | 21,764 | 0.831 ± 0.008 |
| **Dense MLP** | 25,348 | **0.908 ± 0.009** |

### B) Param-eşleşmiş (~21,764)

| Model | Parametre | Test Acc |
|---|---|---|
| Flat Kronecker (K=168) | 21,764 | **0.257 ± 0.030** |
| Hierarchical Kronecker | 21,764 | 0.816 ± 0.017 |
| **Dense MLP** | 13,179 | **0.900 ± 0.015** |

## Bulgular

### 1. Flat Kronecker Derinlikte Çöküyor
168 katmanlı flat zincir **0.257** test doğruluğu (rastgele 0.25'ten düşük).
Tekrarlanan bilinear + SiLU normalizasyon olmadan bilgi yok ediyor.
**Sonuç:** "K katman n^(2K) etkileşim" iddiası pratikte **boş**.

### 2. Hierarchical > Flat (İç Karşılaştırma)
Aynı K=3'te: hier (0.831) vs flat (0.558). Hiyerarşik **mimari içi iyileştirme** sağlıyor.

### 3. Dense MLP Her Koşulda Kazanıyor
- A koşulu: Dense 0.908 vs Hier 0.831
- B koşulu: Dense 0.900 (13K param) vs Hier 0.816 (21K param)
- Dense **%66 daha az param, %10 daha iyi sonuç**

## Dürüst Bilimsel Yorum

**Bu negatif bir sonuçtur.** Hiper geometrik Kronecker mimarisi,
parametre-eşleşmeli koşulda standart MLP'yi **geçemiyor**.

**Pozitif çıkarımlar:**
- Hierarchical zincir, flat zincire göre **kesinlikle iyi** (mimari içi)
- Flat Kronecker'ın derinlikte çöküşü **yapısal bir keşif**
- Kronecker yapısının ifade gücü **aktivasyona bağlı** (önceki veriyle tutarlı)

**Öneri:**
- "Hiper geometrik" iddiasını **yeniden çerçevelemek** gerekli
- Kronecker zinciri bir **temsil aracı** olarak kalabilir, ama performans iddiası zayıf
- Ana bilimsel katkı **hibrit formül** (A+B+C+D) — o ayrı ve sağlam

## Sonraki Adımlar

1. **Daha büyük görev** (n=16, daha fazla sınıf) — belki ölçekleme farkı var
2. **Farklı görev tipi** (compositional, NER) — belki Kronecker yapısal göreve uygun
3. **Normalizasyon ekle** (LayerNorm, residual) — çöküşü önleyebilir
4. **Dürüst rapor** — negatif sonuç yayınlanabilir

## Bilimsel Değer

Negatif sonuç **pozitif katkıdır:**
- Kendi mimarimizi dürüstçe test ettik
- Zayıf noktayı belgeledik
- Gelecek çalışma için temel attık
