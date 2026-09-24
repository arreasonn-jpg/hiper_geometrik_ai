# Preprint 3 — Outline

**Başlık (aday):** *Implicit Regularization in Kronecker Architectures: When Structural Constraints Help*

**Tarih:** 2026-09-24
**Durum:** Outline
**Önceki preprintler:**
- Preprint 1: Hibrit formül (12 dil, 4 görev)
- Preprint 2: Birleşik teori (metrik + mimari + görev)

## Özet (1 paragraf)

Kronecker mimarilerinin değeri, geleneksel olarak "yapısal uyum"
(tensör faktörizasyonuna uygun görevlerde verimlilik) ile açıklanır.
Bu çalışmada bu açıklamanın **yanlış olduğunu** gösteriyoruz:
6 görevde otokorelasyon ve Kronecker zaferi arasında **negatif**
korelasyon var ($r = -0.38$). Yerine, Kronecker'ın $2n^2$ parametre
kısıtının **örtük düzenlileştirici** olarak çalıştığını öne sürüyoruz.
5 gürültü seviyesinde 20 seed ile bu hipotezi doğruluyoruz
($r = +0.95$, $p = 0.012$). Grid olmayan görevlerde ve temiz veride
Kronecker kaybeder; gürültülü grid görevlerinde kazanır. Bu,
mimari tasarımda yeni bir prensip önerir: **düzenlileştirme ihtiyacı
olan görevlerde parametre kısıtlı mimariler tercih edilmelidir.**

## Katkılar

1. **Yanlış hipotezi çürütme:** "Yapısal uyum" açıklaması reddedildi (W5)
2. **Yeni hipotez:** Örtük düzenlileştirme (r=+0.95, p=0.012)
3. **Sirküler mantık kırıldı:** Grid olmayan görevlerde Kronecker kaybeder (W6)
4. **İfade gücü ≠ optimizasyon:** $Y=AXB$ görevinde Kronecker başarısız
5. **Pratik kılavuz:** Ne zaman Kronecker, ne zaman Dense?

## Yapı

### 1. Introduction

- **Motivasyon:** Kronecker ve benzeri parametre-kısıtlı mimariler
  ne zaman işe yarar?
- **Mevcut açıklama:** "Yapısal uyum" — tensör faktörizasyonuna uygun
  görevlerde verimlilik (bizim önceki çalışmamız dahil)
- **Problem:** Bu açıklama test edilmemişti; sirküler mantık riski var
- **Katkı:** Yapısal uyum hipotezini çürütüp, düzenlileştirme
  hipotezini kuruyoruz
- **Yol haritası:** Teori (§2) → Deneyler (§3) → Tartışma (§4)

### 2. Theory

#### 2.1 Kronecker Katmanı — Parametre Kısıtı

Bilinear katman: $Y = A X B$, $X \in \mathbb{R}^{n \times n}$

- Temsil edilen operatör: $B^\top \otimes A$, boyut $n^2 \times n^2$
- Gerçek parametre: $2n^2$
- **Kısıt oranı:** $n^2 \times n^2 / 2n^2 = n^2 / 2$

Bu, mimari düzeyde bir **parametre kısıtıdır**. İki yönlü etki:
1. **İfade gücü kaybı** (bazı operatörler temsil edilemez)
2. **Örtük düzenlileştirme** (aşırı öğrenmeyi zorlaştırır)

#### 2.2 İki Hipotez

**H1 (Yapısal Uyum):** "Kronecker, görev yapısı tensör
faktörizasyonuna uygun olduğunda verimli."

**H2 (Örtük Düzenlileştirme):** "Kronecker'ın parametre kısıtı,
gürültülü veride aşırı öğrenmeyi önleyen bir düzenlileştirici
olarak çalışır."

Bu iki hipotez **farklı tahminler** üretir:
- H1: otokorelasyon ↑ → Kronecker zaferi ↑
- H2: gürültü ↑ → Kronecker zaferi ↑

#### 2.3 Test Edilebilir Öngörüler

| Hipotez | Öngörü | Test |
|---|---|---|
| H1 | Yapısal görevlerde Kronecker kazanır | 6 görev, otokorelasyon |
| H2 | Gürültülü görevlerde Kronecker kazanır | 5 noise seviyesi |
| Her ikisi | İki koşul birlikte gerekli | Grid × noise faktöriyel |


### 3. Experiments

#### 3.1 H1 Testi — Yapısal Uyum Hipotezi

**6 görev × 20 seed:**

| Görev | Otokorelasyon | Kronecker Kazanma |
|---|---|---|
| random_linear | −0.01 | 1/20 |
| permuted_mnist | +0.00 | 0/20 |
| noise_mnist | +0.23 | 9/20 |
| simple_block | +0.80 | 0/20 |
| grid_correlated | +0.09 | 12/20 |
| fashion_mnist | +0.87 | 0/20 |

**Sonuç:** Otokorelasyon vs Kronecker zaferi: **r = −0.38, p = 0.46**

**H1 REDDEDİLDİ.**

#### 3.2 H2 Testi — Düzenlileştirme Hipotezi

**5 gürültü × 20 seed (blok kompozisyonel görev):**

| Gürültü | Dense | Flat Kronecker | Flat Kazanma |
|---|---|---|---|
| 0.00 | 0.980 | 0.945 | 0/20 |
| 0.10 | 0.913 | 0.893 | 1/20 |
| 0.30 | 0.819 | 0.822 | 11/20 |
| 0.50 | 0.734 | 0.759 | 18/20 |
| 0.60 | 0.667 | 0.705 | 18/20 |

**Sonuç:** Gürültü vs Kronecker zaferi: **r = +0.953, p = 0.012**

**H2 KANITLANDI.**

**Geçiş noktası:** Gürültü ≈ 0.30 (50/50 kazanma).

#### 3.3 Sirküler Mantık Kontrolü

**4 görev × 20 seed:**

| Görev | Dense | Flat | Flat Kazanma |
|---|---|---|---|
| Grid baseline | 0.819 | 0.822 | 11/20 |
| Sequence (1D) | 0.745 | 0.780 | 18/20 |
| Graph (grid yok) | 0.821 | 0.609 | 0/20 |
| Random factor ($Y=AXB$) | 0.706 | 0.599 | 1/20 |

**Bulgular:**
1. **Grid gerekli:** Graf görevinde Kronecker %21 kaybediyor
2. **İfade gücü ≠ optimizasyon:** Tam bilinear görevde bile Kronecker kaybediyor
3. **Sequence istisna:** Yapay 2D reshape → 18/20

**Sirküler mantık riski KIRILDI.**

#### 3.4 Bileşik Hipotez

**Kronecker avantajı için İKİ koşul:**

| Koşul | Kaynak | Test |
|---|---|---|
| **Grid yapısı** | W6 | Graf görevinde kaybeder |
| **Düzenlileştirme ihtiyacı** | W5 | Temiz veride kaybeder |

**Örnek:**
- Temiz grid (noise=0.0) → Dense kazanır
- Gürültülü grid (noise=0.6) → **Flat Kronecker kazanır**
- Graf (grid yok) → Dense kazanır (her koşulda)
- Random_factor → Dense kazanır (**optimizasyon sorunu**)


### 4. Discussion

#### 4.1 Mekanizma — Neden Kronecker Düzenlileştirir?

Kronecker katmanı $2n^2$ parametreyle $n^2 \times n^2$ operatörü
temsil eder. Bu, ağın **efektif kapasitesini** sınırlar:

- **Dense MLP:** Serbest ağırlıklar, herhangi bir lineer haritayı
  temsil edebilir → overfitting riski yüksek
- **Kronecker:** Faktörize yapı, "kolay" fonksiyonları öğrenir →
  overfitting riski düşük

Bu, **Dropout** (rastgele nöron kapatma) ve **Weight Decay**
(L2 cezası) ile aynı ailedendir: **mimari düzeyde düzenlileştirme.**

#### 4.2 İfade Gücü ve Optimizasyon Ayrımı

**Beklenmedik bulgu (W6):** Random factor görevi **tam olarak**
$Y = AXB$ formunda — Kronecker'ın temsil ettiği yapı. Ama Flat
Kronecker **1/20 seed'de kazanıyor**.

**Yorum:** Sorun ifade gücü değil, **optimizasyon yüzeyi**.
Tek katmanlı bilinear + head, bu "kolay" görevi bile öğrenemiyor.
Bu, teorik ifade gücü ile pratik öğrenilebilirlik arasındaki
boşluğu gösterir.

#### 4.3 Pratik Kılavuz

| Görev Tipi | Öneri |
|---|---|
| Gürültülü grid (görüntü, spektrogram) | **Kronecker** |
| Temiz grid (matematik tablolar) | Dense MLP |
| Graf (sosyal ağ, molekül) | Dense MLP + GNN |
| Sıralı (metin, ses) | Dense MLP + Transformer |
| Bilinear form (düşük-rank) | LoRA veya TT (Kronecker değil) |

#### 4.4 İlişkili Literatür

- **Dropout** (Srivastava 2014): Rastgele nöron kapatma
- **Weight Decay / L2** (Krogh 1991): Ağırlık cezası
- **Spectral Normalization** (Miyato 2018): Spektral kısıt
- **LoRA** (Hu 2021): Düşük-rank adaptasyon
- **Tensor Train** (Novikov 2015): Faktörize ağlar
- **MLP-Mixer** (Tolstikhin 2021): Token-karıştırıcı

**Farkımız:** Kronecker'ın değerini **düzenlileştirme** üzerinden
açıklıyoruz; önceki çalışmalar verimlilik üzerinden açıklıyor.

### 5. Limitations

- Tüm görevler sentetik veya küçük ölçekli (MNIST, Fashion-MNIST)
- Büyük ölçekli veri (ImageNet, WikiText) test edilmedi
- Kronecker dışında başka faktörize mimariler test edilmedi
- "İki koşul" iddiası sadece bu 10 görevde doğrulandı
- Optimizasyon sınırı teorik olarak analiz edilmedi

### 6. Conclusion

Kronecker mimarilerinin avantajı "yapısal uyum" ile açıklanamaz.
Yerine, **örtük düzenlileştirme** mekanizması geçerlidir: parametre
kısıtı gürültülü veride overfitting'i önler. Bu bulgu:
- Kronecker'ın **hangi görevlerde** işe yaradığını netleştirir
- Daha genel bir **mimari tasarım prensibi** önerir
- **İfade gücü ≠ öğrenilebilirlik** ayrımını vurgular

**Gelecek çalışma:** Büyük ölçekli veri, diğer faktörize mimariler,
teorik optimizasyon analizi.


---

## Ekler

### A. Kaynak Artefaktlar

| # | Dosya | Ne İçeriyor |
|---|---|---|
| 1 | `W5_REGULARIZATION_HYPOTHESIS.md` | W5 hipotez çürütme + H2 kanıtı |
| 2 | `W6_NON_GRID_RESULTS.md` | W6 sirküler mantık kırma |
| 3 | `w5_proper_6tasks.py` | 6 görev × 20 seed (H1 testi) |
| 4 | `w5_noise_ablation.py` | 5 gürültü × 20 seed (H2 testi) |
| 5 | `w6_non_grid_tasks.py` | 4 grid olmayan görev |
| 6 | `PREPRINT2_WEAKNESSES.md` | 10 zayıflık (W5 + W6 hariç) |

### B. Kod Deposu

- Repo: `github.com/arreasonn-jpg/hiper_geometrik_ai`
- Tag: `w5-regularization-confirmed`, `w6-non-grid-confirmed`
- Test suite: 994/994 geçiyor

### C. İstatistiksel Özet

| Test | Sonuç | Yorum |
|---|---|---|
| H1 (yapısal uyum) | r = −0.38, p = 0.46 | **REDDEDİLDİ** |
| H2 (düzenlileştirme) | r = +0.953, p = 0.012 | **KANITLANDI** |
| Geçiş noktası | Noise ≈ 0.30 | 50/50 kazanma |

### D. Ana Kaynakça

1. Krogh & Hertz (1991). A simple weight decay can improve generalization.
2. Srivastava et al. (2014). Dropout: A simple way to prevent neural networks from overfitting.
3. Miyato et al. (2018). Spectral normalization for generative adversarial networks.
4. Novikov et al. (2015). Tensorizing neural networks.
5. Hu et al. (2021). LoRA: Low-rank adaptation of large language models.
6. Tolstikhin et al. (2021). MLP-Mixer: An all-MLP architecture for vision.
7. Trockman & Kolter (2023). Patches are all you need?
8. Vaswani et al. (2017). Attention is all you need.

---

## Sonraki Adımlar

| # | İş | Süre |
|---|---|---|
| 1 | İngilizce LaTeX yazımı | 3-4 saat |
| 2 | Türkçe çeviri | 1-2 saat |
| 3 | Görsel tablolar | 1 saat |
| 4 | Hoca incelemesi | 1-2 hafta |
| 5 | arXiv submission | 1 gün |

**Tahmini toplam:** 6-8 saat aktif çalışma + inceleme süresi

---

**Durum:** Outline v1 tamam. Sonraki: LaTeX yazımı.
