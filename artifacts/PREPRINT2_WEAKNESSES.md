# Preprint 2 Hakem Provası — 10 Zayıf Nokta

**Tarih:** 2026-09-24
**Amaç:** Preprint 2'yi arXiv'e göndermeden önce kendi kendini eleştirmek.
**Yöntem:** Sert hakem gözüyle okuma, 10 zayıf nokta, her biri için düzeltme önerisi.

---

## W1: F1 Per-Class Formülü — "Teorem" Değil, Cebirsel Özdeşlik

**Hakem eleştirisi:** Denklem (7)'deki ayrışım, $F_1$ tanımının doğrudan
yerine konmasıyla elde edilir. Bu bir teorem değil, **cebirsel bir
özdeşliktir**. "Machine-precision error" iddiası da bu nedenle
tautolojiktir — aynı confusion matrix'i iki farklı yolla hesaplıyorsunuz.

**Kanıt:** $F_1^c_h = 2 \cdot TP_h / D_h$ tanımından başlayıp
$TP_h = TP_s + TP_n$, $D_h = D_s + D_n$ koyunca çıkıyor. 5 dakikalık cebir.

**Düzeltme:**
1. "**Exact algebraic identity**" olarak çerçevele ("theorem" değil)
2. **Katkıyı yeniden tanımla:** "We show that the linear formula fails
   on $F_1$, and that the correct aggregation is necessarily per-class.
   We identify $w_c \neq \text{cov}$ as the source of the failure."
3. Ampirik gözlemi öne çıkar: $w_c$ ve cov arasındaki farkın
   **veriye bağlı** olduğunu göster (0.37–0.66 aralığında değişiyor).
4. Gelecek çalışma olarak **yeni** bir şey ekle: per-class ağırlıkların
   **tahmin edilebilirliği** (predictor için hangi bilgi gerekli?)

---

## W2: Related Work Bölümü Yok — Sadece 5 Atıf

**Hakem eleştirisi:** Bir arXiv preprint'i için **5 referans aşırı az**.
Ayrıca "Related Work" bölümü **tamamen eksik**. Hangi literatüre
konumlandığınız belirsiz.

**Eksik literatür:**
- **MLP-Mixer, gMLP, ConvMixer** (token-mixing vs channel-mixing) — Kronecker'a en yakın mimari
- **Tensorized neural networks**: Tensor Train (TT), CP/Tucker decomposition
- **Low-rank factorization** of neural layers: LoRA, ALBERT
- **Neurosymbolic literature**: DeepProbLog dışında Logic Tensor Networks, Neural Theorem Provers
- **Hybrid systems theory**: Garcez & Lamb dışında daha fazla
- **$F_1$ decomposition literature**: sklearn weighted F1, micro/macro analizleri

**Düzeltme:** En az **15-20 atıf** + 1 sayfa Related Work bölümü ekle.
Kronecker'ı **MLP-Mixer** ile karşılaştır (çok yakın fikir).

---

## W3: Gerçek Dünya Verisi Yok — Hepsi Sentetik

**Hakem eleştirisi:** "Image compositional" görevi aslında **sentetik bir
grid** (4×4 blok + gürültü). Gerçek görüntü değil. Preprint'in "future
work" maddesinde "(1) Test the Kronecker hypothesis on natural images"
yazıyor — ama bu zaten **yapılması gereken** şey, iddianın temeli.

**Düzeltme:**
- **CIFAR-10, MNIST, Fashion-MNIST** üzerinde hızlı test yap (30 dk)
- Kronecker ve dense MLP'yi **aynı param bütçesinde** karşılaştır
- Sonuçları Section 3.3c olarak ekle
- Ya da "future work" maddesini **iddiadan çıkar**, sadece öneri olarak bırak

---

## W4: "8x Verimli" İddiası — Adil Karşılaştırma Değil

**Hakem eleştirisi:** "Flat Kronecker 8x daha az parametreyle eşit
doğruluk" iddiası **iki farklı şeyi** karşılaştırıyor:
- Flat Kronecker: 2,564 param, doğrusal + bilinear kısıt
- Dense MLP: 20,868 param, serbest ağırlıklar

Bu **parametre-eşleşmeli** bir karşılaştırma değil. "8x verimli"
iddiası, farklı **ifade gücü** sınıflarındaki iki modeli karşılaştırıyor.

**Düzeltme:**
1. **Parametre-eşleşmeli versiyon ekle:** Flat Kronecker'ı derinleştir
   ($K=8, 16, 32, \dots$) veya genişlet ($n=24, 32$) → aynı param
   bütçesine getir → o zaman karşılaştır
2. **İfade gücü analizi ekle:** Kronecker kısıtının hangi fonksiyonları
   temsil edemediğini **teorik** olarak göster
3. "8x verimli" yerine "8x daha az parametreyle **eşit** doğruluk,
   **ama** $K=168$'de çöküyor" gibi **dürüst** ifade kullan

---

## W5: Görev-Uygunluk Hipotezi — Formal Test Edilmemiş

**Hakem eleştirisi:** "Task-Appropriateness Hypothesis" bir **hipotez**
olarak sunuluyor ama **istatistiksel bir test** yok. Sadece 3 görev
tipinde 3 farklı kazanan gösteriliyor. Bu **post-hoc** bir açıklama.

**Eksik olan:**
- Hipotezin **falsifiye edilebilir** bir formülasyonu
- Görev yapısını **önsel** olarak ölçen bir metrik (örn: grid
  yapısının tensör rank'ı, uzamsal otokorelasyon)
- Bu metrik ile Kronecker verimliliği arasında **korelasyon** analizi

**Düzeltme:**
1. Görev yapısını ölç: **2D otokorelasyon**, **tensör rankı**, **blok
   yapısı skoru**
2. Bu metrikleri **kazanan mimari** ile ilişkilendir
3. Bir **karar kuralı** çıkar: "Eğer otokorelasyon > 0.3 ise, Kronecker
   kullan"
4. Bu kuralı **held-out** görevlerde test et

---

## W6: Görüntü Compositional Görev — Sirküler Mantık Riski

**Hakem eleştirisi:** "Image compositional" görevi **grid yapılı** olacak
şekilde tasarlanmış. Kronecker mimarisi de **grid/tensör yapısına**
dayanıyor. Bu, "Kronecker grid görevlerinde iyi" iddiasını **sirküler**
yapabilir.

**Kanıt:** Görev tasarımı:
- 4×4 blok ızgarası → **grid**
- Her blokta sinyal → **grid değeri**
- Çeyrek argmax → **grid özeti**

Kronecker katmanı: $Y = A X B$ → **grid dönüşümü**

**Düzeltme:**
1. **Farklı yapıda** bir görev test et:
   - **Graf yapılı** görev (Kronecker'a uygun değil)
   - **Sıralı** görev (Kronecker'a uygun değil)
   - **Rastgele faktörlü** görev (kısmen uygun)
2. "Grid yapısına sahip görevlerde Kronecker iyi" iddiasını
   **falsifiye edilebilir** hale getir
3. **Kontrollü ablation:** Grid yapısını boz (blokları karıştır) →
   Kronecker'ın avantajı kaybolmalı

---

## W7: İstatistiksel Metodoloji — Çoklu Karşılaştırma Düzeltmesi Yok

**Hakem eleştirisi:** 3 karşılaştırma × 2 görev = **6 test** yapılıyor.
Bonferroni düzeltmesi yok. Aile-bazlı hata oranı **şişirilmiş**.

**Ayrıca:**
- $p = 0.0003$ gibi değerler **tek başına** yeterli değil, **etki
  büyüklüğü** de gerekli
- Cohen's $d = +0.508$ (flat vs dense) — **orta etki** ama
  istatistiksel olarak anlamlı → örneklem büyüklüğü yeterli mi?
- **Bootstrap CI** sonuçları raporda yok (sadece $p$ değerleri var)

**Düzeltme:**
1. **Bonferroni veya Holm-Bonferroni** düzeltmesi uygula
2. Her test için **etki büyüklüğü + %95 CI** ver
3. **Güç analizi** ekle: "50 seed ile $d=0.5$ etkisini tespit etmek
   için güç = 0.85"
4. **Bayesian analiz** ekle (alternatif bakış açısı)

---

## W8: Terminoloji Tutarsızlığı — "Theorem" vs "Identity" vs "Formula"

**Hakem eleştirisi:** Metin boyunca üç terim karışık kullanılıyor:
- "formula" (Denklem 1, 5)
- "identity" (Denklem 6, 9)
- "hypothesis" (Task-Appropriateness)

Okuyucu hangi iddianın **kesin** (ispatlı) hangisinin **deneysel**
olduğunu ayırt edemiyor.

**Düzeltme:**
1. **Terim sözlüğü** ekle (Introduction sonuna):
   - **Exact identity** = matematiksel ispat (Denklem 1, 6, 9)
   - **Predictive formula** = yaklaşık, deneysel doğrulanmış (Denklem 5)
   - **Hypothesis** = test edilmiş ama kanıtlanmamış (Task-Appropriateness)
2. Bu terimleri **tutarlı** kullan
3. Abstract'ta hangi iddianın hangi tipte olduğunu belirt

---

## W9: Reproducibility — Kod ve Veri Bağlantısı Zayıf

**Hakem eleştirisi:** Sadece "github.com/arreasonn-jpg/hiper_geometrik_ai"
referansı var. Ama:
- Hangi commit'te bu sonuçlar üretildi?
- Hangi script'ler çalıştırıldı?
- Veri nerede?
- Docker image var mı?
- `requirements.txt` sabit mi?

**Düzeltme:**
1. **DOI/Zenodo** snapshot ekle (arXiv sonrası)
2. Her tablo için **üreten script'in adını** yaz:
   - Table 1: `artifacts/compositional_20seed.py`
   - Table 2: `artifacts/f1_perclass_test.py`
   - Table 3: `artifacts/image_compositional.py`
3. **Seed'leri** açıkça belirt
4. **Donanım** bilgisi ekle (CPU/GPU, süre)
5. Bir **`reproduce.sh`** script'i ekle

---

## W10: Abstract Çok Yoğun — Okunabilirlik Düşük

**Hakem eleştirisi:** Abstract **14 satır** uzunluğunda, 5 farklı sonucu
sıralıyor. Hakem ilk okumada **ana katkıyı** göremiyor.

**Kanıt:** Abstract'ta:
- 3 boyut (metrik, mimari, görev)
- 2 keşif (F1 per-class, Kronecker)
- 4 sayısal iddia ($\varepsilon = 0.000000$, $8\times$, $p<0.0001$, $d=-1.37$)

**Düzeltme:**
1. **İlk cümle:** ana iddia ("Hybrid gain is not universal, but
   predictable along three independent dimensions.")
2. **İkinci cümle:** her boyut için 1 özet
3. **Üçüncü cümle:** pratik sonuç
4. Sayıları **azalt** — en önemli 2 tanesini tut
5. **Uzunluk:** 8 satıra indir

Örnek taslak:
> Hybrid systems do not win unconditionally: their gain is a function
> of metric type, architecture type, and task structure. We provide an
> exact identity for linear metrics and a per-class extension for macro
> $F_1$; we show that Kronecker architectures are $8\times$ more
> parameter-efficient on compositional tasks. These results suggest
> hybrid design can be made principled.

---

## 🎯 Öncelik Tablosu

| # | Zayıflık | Şiddet | Düzeltme Süresi |
|---|---|---|---|
| **W1** | F1 "teorem" değil | 🔴 Yüksek | 30 dk (yeniden çerçeveleme) |
| **W2** | Related Work yok | 🔴 Yüksek | 90 dk (araştırma + yazma) |
| **W3** | Sentetik veri | 🟡 Orta | 30 dk (CIFAR-10 deneyi) |
| **W4** | 8x verimlilik adil değil | 🟡 Orta | 45 dk (param-eşleşmeli) |
| **W5** | Hipotez test edilmemiş | 🔴 Yüksek | 60 dk (ölçüm metrikleri) |
| **W6** | Sirküler mantık | 🟡 Orta | 45 dk (farklı görev) |
| **W7** | Çoklu karşılaştırma | 🟡 Orta | 20 dk (Bonferroni ekle) |
| **W8** | Terminoloji tutarsız | 🟢 Düşük | 15 dk (sözlük ekle) |
| **W9** | Reproducibility zayıf | 🟡 Orta | 30 dk (script referansları) |
| **W10** | Abstract çok yoğun | 🟢 Düşük | 15 dk (yeniden yaz) |

---

## 📋 Düzeltme Stratejisi

**Kritik (yayın öncesi mutlaka):** W1, W2, W5
**Yüksek (güçlü tavsiye):** W3, W4, W7
**İyileştirme (varsa zaman):** W6, W8, W9, W10

**Tahmini toplam süre:** 6-7 saat (hepsi)
**Minimum yayın için:** 2-3 saat (W1, W2, W5, W7, W10)

---

## Sonuç

Preprint 2'nin **bilimsel içeriği sağlam**, ama **sunumu zayıf**.
En büyük sorun: **W2 (Related Work yok) ve W1 (teorem iddiası)**.
Bu ikisi düzeltilirse, hakem kabul olasılığı **yüksek**.
