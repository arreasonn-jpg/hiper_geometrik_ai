# Uzun-Baglam Analizi

## Sonuclar

### Smoke (16->64 token)
| Model | PPL | Degisim |
|---|---|---|
| dense | 137.9 -> 178.9 | +29.8% (kötü) |
| transformer | 126.7 -> 127.4 | +0.6% |
| hga | 131.4 -> 131.4 | 0.0% (stabil) |

### 1024 (512->1024 token)
| Model | PPL | Degisim |
|---|---|---|
| dense | 519.6 -> 484.0 | -6.9% (iyi) |
| transformer | 539.7 -> 494.0 | -8.5% (iyi) |
| hga | 535.0 -> 505.3 | -5.6% (iyi) |

Tum 9 kapi GECTI.

## Durust Sinir

Repo soyluyor: "Bu bir kalite iddiasi degil, sekil/butce/pozisyon
eslesmesi kapisidir."

Kisa egitim (120 adim) hicbir neural kolu bigram zeminini geciremiyor.
Gercek kalite icin `full` profil (4000 adim) gerekli.

## Yetenek Haritasi (7/7 TAMAM)

| # | Modul | Sonuc |
|---|---|---|
| 1 | Hibrit (sentetik) | +0.106 |
| 2 | TWT (Turkce) | +0.039 |
| 3 | EWT (Ingilizce) | +0.069 |
| 4 | Self-learning | Lineer ama %99 tekrar |
| 5 | Multi-hop | BFS calisiyor |
| 6 | Epistemik | 1.000, halusinasyon yok |
| 7 | Uzun-baglam | Stabil 512-1024 |
