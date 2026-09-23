# Görüntü Compositional'da Hierarchical Kronecker Zaferi

**Tarih:** 2026-09-23
**Amaç:** Kronecker hipotezini görüntü benzeri (grid + uzamsal korelasyon) görevde test.

## Görev Tasarımı

**Girdi:** (16, 16) görüntü
- 4x4 blok ızgarası (her blok 4x4 piksel)
- Her blokta "sinyal gücü" (0-1)
- %30 Gauss gürültü

**Sınıf:** 4 (hangi çeyrek en güçlü?)

**Yapı:** Grid + uzamsal korelasyon → görüntü benzeri

## Sonuçlar (20 seed)

| Model | Parametre | Test Acc |
|---|---|---|
| dense_mlp | 20,868 | 0.8191 ± 0.0207 |
| flat_kronecker | 2,564 | 0.8267 ± 0.0207 |
| **hierarchical_kronecker** | **87,044** | **0.8353 ± 0.0217** |

## İstatistiksel Testler

### Flat vs Dense
- mean_diff = +0.0076
- paired t = +2.049 (p = 0.0405)
- Cohen's d = +0.458

**Sonuç:** Flat **anlamlı olarak** kazanıyor (8x az parametre).

### Flat vs Hierarchical
- mean_diff = −0.0086
- paired t = −2.363 (p = 0.0181)
- Cohen's d = −0.528

**Sonuç:** Hierarchical **anlamlı olarak** kazanıyor.

### Dense vs Hierarchical
- mean_diff = −0.0162
- paired t = −5.342 (p < 0.0001)
- Cohen's d = −1.194

**Sonuç:** Hierarchical **çok anlamlı olarak** kazanıyor (büyük etki).

## Büyük Resim: Görev Tipi Mimarisi Belirliyor

| Görev Tipi | Kazanan | Verimlilik |
|---|---|---|
| Sentetik lineer (rastgele W) | Dense MLP | baseline |
| Compositional (blok güçleri, özet) | Flat Kronecker | **8x verimli** |
| **Görüntü compositional (grid + uzamsal)** | **Hierarchical Kronecker** | 34x param ama en iyi |

## Yorum

**Görüntü yapısı hiyerarşiye doğal uyum sağlıyor:**
- Uzamsal lokalite → hiyerarşik boyut genişlemesi anlamlı
- Blok → çeyrek → sınıf → piramit yapı
- Dense MLP flatten ile uzamsal bilgiyi kaybediyor
- Flat Kronecker orta seviye, hiyerarşik en iyi

**Kritik çıkarım:** "Hiper geometrik" iddiası **görev tipine bağlı:**
- ❌ Rastgele lineer görevler → kaybeder
- ✅ Basit compositional → flat Kronecker 8x verimli
- ✅✅ Görüntü compositional → hierarchical Kronecker en iyi

## Bilimsel Değer

**Üçüncü pozitif sonuç:**
1. F1 per-class formülü
2. Compositional Kronecker zaferi
3. **Görüntü compositional hierarchical zaferi**

**Preprint 2 için güçlü malzeme:**
- Görev tipi → mimari seçimi karar ağacı
- İstatistiksel olarak doğrulanmış
- Üç farklı görev tipi, üç farklı kazanan
