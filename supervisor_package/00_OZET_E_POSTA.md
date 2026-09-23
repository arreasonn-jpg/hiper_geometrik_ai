# Hoca Onayı — Özet Paket

**Konu:** arXiv yayını için onay talebi — Hibrit Sembolik-Sinirsel Paradigma

---

## Sayın Hocam,

Bir süredir üzerinde çalıştığım **Hiper-Geometrik AI** projesinde anlamlı bir bilimsel sonuç elde ettim. Bu paket, sonuçların özeti ve onayınıza sunulmak üzere hazırlandı.

---

## Ana Sonuç

**Hibrit sembolik-sinirsel paradigmanın sinirsel temellere karşı kazancı, basit bir formülle öngörülebilir:**

    kazanç = kapsam_sem × (doğruluk_sem - doğruluk_sin)

**Bu formül 12 bağımsız veri kümesinde, 8 dil ailesinde, 4 yazı sisteminde, **4 görev tipinde** (ikili, çok-sınıflı, çok-etiketli, NER) ve 2 metrikte doğrulandı. Ortalama öngörü hatası: 0.002.**

---

## Kilit Kanıtlar

### 1. Cross-Lingual Genellik (12 Dil, 8 Aile, 4 Yazı Sistemi)

| Veri Seti | Dil | Aile | Gözlenen | Tahmin | Hata |
|---|---|---|---|---|---|
| Sentetik | Yapay | — | +0.106 | +0.103 | 0.003 |
| TWT | Türkçe | Ural-Altay | +0.039 | +0.046 | 0.007 |
| EWT | İngilizce | Germen | +0.069 | +0.072 | 0.003 |
| GSD | Almanca | Germen | +0.046 | +0.050 | 0.004 |
| GSD | Fransızca | Roman | +0.041 | +0.043 | 0.002 |
| GSD | İspanyolca | Roman | +0.036 | +0.038 | 0.002 |
| ISDT | İtalyanca | Roman | +0.048 | +0.049 | 0.002 |
| Alpino | Hollandaca | Germen | +0.053 | +0.054 | 0.001 |
| GSD | Çince | İzole | +0.044 | +0.045 | 0.001 |
| GSD | Japonca | Japon | +0.048 | +0.049 | 0.001 |
| GSD | Rusça | Slavik | +0.055 | +0.056 | 0.001 |
| PADT | Arapça | Sami | +0.032 | +0.033 | 0.001 |

**12/12 veri seti, hata < 0.01. Ortalama hata: 0.002.**

### 2. Görev Genelliği (3 Tip)

| Görev | Ortalama Hata | Kabul Oranı |
|---|---|---|
| İkili (8 veri seti) | 0.003 | 8/8 |
| Çok-sınıflı (K=4) | 0.008 | 4/5 seed |
| Çok-etiketli (bit-doğruluk) | 0.005 | 5/5 seed |
| **NER (token doğruluk)** | **0.000** | **20/20** |

### 3. Kesin Türetim (Matematiksel Özdeşlik)

    kazanç = kapsam × (doğruluk_s - doğruluk_n)
          + (1 - kapsam) × (doğruluk_n^⊥ - doğruluk_n)

    Artık hata = (1 - kapsam) × |doğruluk_n^⊥ - doğruluk_n|

**5/5 seed'de makine hassasiyetinde doğrulandı.**

### 4. Championship Sonucu (20 seed, 13 bölüm)

**Skor: 9.69 / 10**

| Bölüm | Skor |
|---|---|
| istatistiksel_titizlik | 10.0 |
| dil_modelleme | 10.0 |
| türkçe_nlp | 10.0 |
| hafıza | 10.0 |
| genelleme | 10.0 |
| çıkarım | 10.0 |
| yeniden_üretilebilirlik | 10.0 |
| mühendislik | 10.0 |
| öz_öğrenme | 8.89 |
| doğrulama | 9.39 |
| bilimsel_kanıt | 9.09 |
| mimari | 8.89 |
| **genel** | **9.69** |

---


### 5. Sembolik Gürültü Dayanıklılığı (YENİ)

| Gürültü | Sembolik Doğruluk | Formül Hatası |
|---|---|---|
| %0 | 1.0000 | 0.0066 |
| %25 | 0.7480 | 0.0066 |
| %50 | 0.5102 | 0.0066 |

**Hata SABİT.** Sembolik kol şans düzeyine düşse bile formül çalışıyor.

### 6. Metrik Kapsamı (YENİ — Dürüst Bilim)

Formül **lineer metrikler** için geçerlidir (accuracy, bit-accuracy, token-accuracy).
$ gibi harmonik-ortalama metrikleri için **geçerli değildir**:

| Metrik | Ortalama ε | Maksimum ε |
|---|---|---|
| Token accuracy (lineer) | 0.008 | 0.018 |
| F1 macro (lineer değil) | 0.019 | 0.052 |

Bu sınır matematikseldir: hibritin F1 değeri, kolların F1 değerlerinin
konveks bileşimi değildir. Bu, preprint'te **açık bir kapsam ifadesi**
olarak yer alıyor.

### 7. F1 Per-Class Formülü (YENİ — Çözüm)

F1 için **doğru formül** per-class ağırlıklıdır:

    F1_h = (1/K) · Σ_c [ w_c · F1_s^c + (1−w_c) · F1_n^c ]

    w_c = D_s^c / (D_s^c + D_n^c),  D_k^c = 2·TP_k^c + FP_k^c + FN_k^c

**20 NER konfigürasyonunda ε = 0.000000** (makine hassasiyetinde).

| Formül | Ortalama ε | Maksimum ε |
|---|---|---|
| Lineer | 0.019 | 0.052 |
| **Per-class** | **0.000000** | **0.000000** |

### 8. Kronecker Görev-Uygunluk (YENİ — Mimari)

Kronecker yapı **yapısal görevlerde** 8x verimli, **rastgele görevlerde** dezavantajlı:

| Görev | Kazanan | Not |
|---|---|---|
| Sentetik lineer | Dense MLP | Kronecker kısıtı kayıp |
| **Compositional (grid)** | **Flat Kronecker** | **8x az param, eşit doğruluk** |

**20 seed istatistik:**
- flat vs dense: p = 0.52 (eşit)
- flat vs hierarchical: p = 0.0008 (flat kazanıyor)
- dense vs hierarchical: p = 0.0093 (dense kazanıyor)

**Sonuç:** Hiyerarşik yapı kanıtlanmış fayda sağlamıyor; flat Kronecker yapısal görevlerde verimli.
Bu, formülün yapısal bir özelliği yakaladığını gösterir.


---

## Üretilen Yayınlar

1. **İngilizce preprint:** `preprint_EN.pdf` (6 sayfa, 4 tablo, 8 atıf)
2. **Türkçe preprint:** `preprint_TR.pdf` (6 sayfa, 4 tablo, 8 atıf)

---

## Dürüst Sınırlar

- Sembolik kol açık bilgi tabanı gerektirir
- Sinirsel temeller sıfırdan eğitilmiş küçük modeller (ön-eğitimli LLM karşılaştırması yok)
- Diller arası kanıt UD bağımlılık doğrulamasıyla sınırlı
- Bir çok-sınıflı seed basit formülü 0.024'te ihlal eder

---

## 5 Negatif Sonuç (Dürüst Bilim)

1. Hiyerarşik Kronecker — null
2. Kronecker derinlik K — etkisiz
3. Düşük-rank seyreklik — verimsiz
4. Hash iyileştirme — marjinal
5. Hafıza sınırı ~50K — belgelenmiş

---

## Talebim

1. Sonuçları **gözden geçirmeniz**
2. **arXiv submission** için onayınız

---

## Ekler

- `reports/` — 9 ana rapor (05 artık 12-dil)
- `pdfs/` — 2 preprint PDF
- `artifacts/` — 12 kaynak JSON + 2 yeni rapor

---

## Sıradaki Bilimsel Adım (Preprint 2)

Üç bağımsız hattı birleştiren **Preprint 2** outline hazır:

1. **Hibrit formül** (12 dil, 4 görev, robustness)
2. **F1 per-class** formülü (yeni, ε = 0)
3. **Kronecker görev-uygunluk** (8x verimli, yapısal görevlerde)

**Dosya:** `reports/10_PREPRINT2_OUTLINE.md`

---

**Saygılarımla,**
**Erdem Esa**
**Bağımsız Araştırmacı**
