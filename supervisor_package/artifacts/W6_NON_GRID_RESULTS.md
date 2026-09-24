# W6: Grid Olmayan Görevlerde Kronecker Testi

**Tarih:** 2026-09-24
**Amaç:** Sirküler mantık riskini kırmak — Kronecker gerçekten grid yapısına mı özgü?

## Deney Tasarımı

4 görev × 20 seed × 2 model:

| Görev | Yapı | Beklenti |
|---|---|---|
| sequence | 1D dizi (16×16 reshape) | Belirsiz |
| graph | Graf düğümleri, grid yok | Dense |
| random_factor | Y = A·X·B (Kronecker'ın tam formu) | **Flat** |
| grid_baseline | Grid + %30 gürültü | Berabere |

## Sonuçlar

| Görev | Dense | Flat | Flat Kazanma | Delta |
|---|---|---|---|---|
| grid_baseline | 0.8187 | 0.8217 | 11/20 | +0.003 |
| sequence | 0.7446 | 0.7797 | 18/20 | +0.035 |
| **graph** | **0.8213** | 0.6092 | **0/20** | **−0.212** |
| **random_factor** | **0.7059** | 0.5986 | **1/20** | **−0.107** |

## Bulgular

### 1. Grid Yapısı Gerekli
Graph görevinde (grid yok) Kronecker **%21 kaybediyor**.
Grid yapısı olmayan görevlerde Kronecker işe yaramıyor.

### 2. İfade Gücü ≠ Optimizasyon
`random_factor` görevi **tam olarak `Y = A·X·B`** formunda — Kronecker'ın
tam temsil ettiği yapı. Ama Flat Kronecker **1/20 seed'de kazanıyor**.

**Yorum:** Sorun ifade gücü değil, **optimizasyon**. Flat Kronecker
bu "kolay" görevi öğrenemiyor (tek katmanlı bilinear + head yeterli değil).

### 3. Sequence Şaşırtıcı Sonuç
1D dizi 16×16'ya reshape edilince **yapay 2D yapı** oluşuyor →
Kronecker avantajlı. Bu, **grid yapısının kritik olduğunu** doğruluyor.

## Sirküler Mantık Riski — KIRILDI

**Önceki endişe:** "Kronecker grid görevlerinde iyi" iddiası, görevlerin
grid yapılı tasarlanmasından kaynaklanıyor olabilir.

**Şimdi:** Grid olmayan görevlerde Kronecker:
- **Graph:** %21 kaybediyor
- **Random factor:** %11 kaybediyor
- **Sequence:** Kazanıyor ama sadece reshape grid etkisi sayesinde

**Sonuç:** Kronecker'ın avantajı **gerçekten grid yapısına bağlı**.

## Bileşik Hipotez (W5 + W6)

**Kronecker avantajı için İKİ koşul gerekli:**
1. **Grid/tensör yapısı** (W6: grid olmayan görevlerde kaybeder)
2. **Düzenlileştirme ihtiyacı** (W5: temiz veride kaybeder)

**Örnek:**
- Temiz grid (W5 noise=0) → Dense kazanır
- Gürültülü grid (W5 noise=0.5) → **Flat Kronecker kazanır**
- Graf (grid yok) → Dense kazanır (her koşulda)
- Random_factor (Kronecker'ın tam formu!) → Dense kazanır

## Bilimsel Değer

**Preprint 2 için kritik katkı:**
- "Task-appropriateness" iddiası **iki koşullu** hale geldi
- Sirküler mantık riski belgelendi ve kırıldı
- Random_factor testi: **ifade gücü ≠ optimizasyon** ayrımı

**Dürüst çerçeve:**
> Kronecker architectures are advantageous only on grid-structured,
> noisy tasks. On graph tasks or clean grid tasks, Dense MLP wins.
> Notably, even when the task is exactly a bilinear form
> $Y = A X B$, Flat Kronecker fails to learn it efficiently —
> suggesting an optimisation, not expressivity, limitation.
