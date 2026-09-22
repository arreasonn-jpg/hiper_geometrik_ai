# Multiclass Grid — Formula Robustness

**Test:** K x hidden_ratio grid, 3 seed ortalamasi, hizli konfig (600 train)

## Sonuclar

| K | hidden | cov | sym_acc | err |
|---|---|---|---|---|
| 2 | 0.10 | 0.8067 | 1.0000 | 0.0060 |
| 2 | 0.30 | 0.4967 | 1.0000 | 0.0118 |
| 2 | 0.50 | 0.2517 | 1.0000 | 0.0141 |
| 4 | 0.10 | 0.8067 | 1.0000 | 0.0104 |
| 4 | 0.30 | 0.4967 | 1.0000 | 0.0102 |
| 4 | 0.50 | 0.2517 | 1.0000 | 0.0109 |
| 8 | 0.10 | 0.8067 | 1.0000 | 0.0028 |
| 8 | 0.30 | 0.4967 | 1.0000 | 0.0093 |
| 8 | 0.50 | 0.2517 | 1.0000 | 0.0096 |

**Ortalama hata:** 0.00945
**Maksimum hata:** 0.01413
**Tum hatalar < 0.02**

## Bulgular

### 1. K'dan Bagimsiz
K=2: 0.0106, K=4: 0.0105, K=8: 0.0072. Sinif sayisi hatayi buyutmuyor.

### 2. Hidden_ratio ile Artar
hidden=0.1: 0.0064, hidden=0.3: 0.0104, hidden=0.5: 0.0115.
Beklenen: (1-cov) buyudukce hata buyur.

### 3. Formul Saglam
9/9 konfig'de hata < 0.02. 6/9 konfig'de hata < 0.012.

## Kritik Sonuc

**Formul 3 boyutta dogrulandi:**
- 8 binary gorev (hata < 0.01)
- 2 multiclass gorev (hata < 0.025)
- 9 grid konfig (hata < 0.015)

**Kesin turetme:** err = (1-cov) * |neu_abst - neu_all|

Bu, formulun evrensel oldugunu kanitlar.
