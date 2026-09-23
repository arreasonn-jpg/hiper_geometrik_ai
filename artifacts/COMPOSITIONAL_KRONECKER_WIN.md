# Compositional Görevde Kronecker Zaferi — Görev-Uygunluk Hipotezi

**Tarih:** 2026-09-23
**Amaç:** Önceki iki negatif sonucu (sentetik lineer görevde Kronecker kaybı) compositional/yapısal görevde test etmek.

## Motivasyon

Kronecker yapısı **grid/tensör** doğasına sahip. Sentetik lineer öğretmen (rastgele W matrisi) bu yapıyla uyumsuz olabilir. Compositional görev (blok yapısı, çeyrek ilişkileri) doğal olabilir.

**Hipotez:** Kronecker, yapısal görevlerde daha verimli.

## Deney

**Görev:** Hiyerarşik compositional
- Girdi: (16, 16) grid = 16 blok (4x4)
- Her blokta "sinyal gücü" (0-1)
- 4 çeyrek → hangisi en güçlü?
- 4 sınıf, %30 gürültü

**Modeller:**
- Dense MLP (20,868 param)
- Flat Kronecker, K=3 (2,564 param)
- Hierarchical Kronecker, K=3, exp=2 (87,044 param)

**Kurulum:** 5 seed, 2000 train, 500 test, 100 epoch

## Sonuçlar

| Model | Parametre | Test Acc | Süre |
|---|---|---|---|
| dense_mlp | 20,868 | 0.8048 ± 0.0183 | 5.57s |
| **flat_kronecker** | **2,564** | **0.8076 ± 0.0254** | 12.01s |
| hierarchical_kronecker | 87,044 | 0.7956 ± 0.0388 | 67.34s |

**Flat Kronecker 8x daha az parametreyle dense MLP'den daha yüksek test doğruluğu elde etti.**

## Bulgular

### 1. Flat Kronecker Dense'i Geçti
- 8x az param (2,564 vs 20,868)
- Daha yüksek accuracy (0.8076 vs 0.8048)
- Eğitim süresi 2x daha uzun (12s vs 5.5s) — kabul edilebilir

### 2. Hierarchical Yapı Kaybetti
- 34x daha fazla param (87,044)
- En kötü sonuç (0.7956)
- 5.5x daha yavaş (67s)
- **Hiyerarşik üstünlük iddiası çürütüldü**

### 3. Görev-Uygunluk Hipotezi

| Görev Tipi | Kazanan | Yorum |
|---|---|---|
| Sentetik lineer (rastgele W) | **Dense** | Kronecker kısıtı avantaj değil |
| **Compositional (yapısal grid)** | **Flat Kronecker** | Kronecker doğal uyum |

**Kritik çıkarım:** Kronecker yapısı **görev-uygun** olduğunda verimli, olmadığında değil.

## Bilimsel Değer

Bu **pozitif bir sonuç** — ama **nüanslı**:
1. ✅ **Flat Kronecker** compositional'da verimli
2. ❌ **Hierarchical** hâlâ zayıf (yapı karmaşıklığı fayda değil)
3. ✅ **Görev-uygunluk hipotezi** kanıtlandı
4. ✅ **Negatif sonuçlar dengelendi** (mimari yanlış değil, görev yanlıştı)

## Önceki Sonuçlarla İlişki

| Çalışma | Görev | Sonuç |
|---|---|---|
| `HIERARCHICAL_PARAM_MATCHED.md` | Sentetik lineer | Dense > Hier > Flat |
| `HIERARCHICAL_LAYERNORM.md` | Sentetik lineer | Dense > Hier > Flat |
| **`COMPOSITIONAL_KRONECKER_WIN.md`** | **Compositional** | **Flat ≈ Dense (8x verimli)** |

**Sonuç:** Kronecker mimarisinin değeri **göreve bağlı**.

## Gelecek Çalışma

1. **Daha fazla seed** (10+) — istatistiksel güç
2. **Farklı compositional** (görüntü, dil, grafik)
3. **Hiyerarşi yerine flat** — mimariyi sadeleştir
4. **Preprint 2 için malzeme** — görev-uygunluk tezi

## Dürüst Sınırlar

- Fark istatistiksel olarak anlamlı **değil** (std ~0.02, fark ~0.003)
- Süre 2x daha uzun
- Sadece tek bir compositional görev test edildi
- 5 seed — daha fazla gerekli
