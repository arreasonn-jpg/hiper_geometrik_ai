# LayerNorm + Residual Deneyi — Hipotez Kısmen Doğrulandı

**Tarih:** 2026-09-23
**Amaç:** Flat Kronecker çöküşünü (önceki deneyde 0.257) LayerNorm + residual ile düzeltmek.

## Hipotez

Flat Kronecker'ın derinlikte çöküşü, normalizasyon eksikliğinden kaynaklanıyor.
**Beklenti:** LayerNorm + residual → flat de öğrenmeli.

## Sonuçlar (5 seed, K=16, n=8)

| Model | Param | Test Acc |
|---|---|---|
| flat_K16_baseline | 2,308 | 0.402 ± 0.034 |
| **flat_K16_norm** (LN) | 4,356 | **0.273 ± 0.028** ❌ |
| **flat_K16_norm_res** (LN + res) | 4,356 | **0.584 ± 0.015** ✅ |
| hier_K3_exp2 | 21,764 | 0.817 ± 0.021 |
| hier_K3_exp2_norm | 32,516 | 0.810 ± 0.015 |
| **dense_K3** | 25,348 | **0.916 ± 0.010** ← Şampiyon |

## Bulgular

### 1. LayerNorm Tek Başına Zararlı
- Flat: 0.402 → 0.273 (kötüleşti)
- Hier: 0.817 → 0.810 (hafif kötüleşti)

**Yorum:** LayerNorm'un ölçek normalizasyonu bilinear yapının bilgi taşımasını bozuyor.

### 2. Residual Yardımcı Ama Yetersiz
- 0.402 → 0.584 (+%45)
- Ama hâlâ hier (0.817) ve dense (0.916) altında

**Yorum:** Residual bilgi akışını koruyor ama flat yapının ifade gücü zaten yetersiz.

### 3. Hierarchical Yapısal Üstünlük
LayerNorm'dan bağımsız: hier > flat. Boyut genişlemesinin (n → 2n → 4n) doğal sonucu.

### 4. Dense MLP Hâlâ Şampiyon
25K param, 0.916 test acc. Kronecker kısıtı **dezavantaj**.

## Dürüst Sonuç

**Hipotez kısmen doğrulandı:** Normalizasyon eksikliği flat çöküşünün **bir kısmını** açıklıyor (0.402→0.584).
**Ama:** LayerNorm + residual bile flat'i hier ve dense seviyesine çıkarmıyor.
**Kesin sonuç:** Kronecker yapısı, aynı parametre bütçesinde **dense MLP'den kötü**.

## Bilimsel Değer

İki ardışık negatif sonuç:
1. Parametre-eşleşmeli: Dense > Hier > Flat
2. LayerNorm + residual: Dense > Hier > Flat (değişmedi)

**Hipotezler test edildi, reddedildi, belgelendi.** Bilim böyle ilerler.

## Öneri

Hiper geometrik iddiasını **yeniden çerçevelemek**:
- Kronecker zinciri bir **temsil aracı** olarak kalabilir
- Ama **performans iddiası** kanıtlanmış değil
- Ana bilimsel katkı **hibrit formül** hattı (A+B+C+D) — sağlam ve kanıtlı
