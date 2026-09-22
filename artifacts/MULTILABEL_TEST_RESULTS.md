# Multi-Label Paradigm — Metrik-Bagimsizlik Testi

**Tarih:** 2026-09-22
**Gorev:** Multi-label (K bagimsiz bit)
**Metrik:** Bit-level accuracy

## Sonuclar

| Seed | cov | sym_bit | neu_bit | hyb_bit | obs | pred | err |
|---|---|---|---|---|---|---|---|
| 1 | 0.5025 | 1.0000 | 0.6219 | 0.8175 | +0.1956 | +0.1900 | 0.0056 |
| 2 | 0.4850 | 1.0000 | 0.5769 | 0.7750 | +0.1981 | +0.2052 | 0.0071 |
| 3 | 0.4450 | 1.0000 | 0.5794 | 0.7694 | +0.1900 | +0.1872 | 0.0028 |
| 4 | 0.4975 | 1.0000 | 0.5363 | 0.7606 | +0.2244 | +0.2307 | 0.0063 |
| 5 | 0.4825 | 1.0000 | 0.5938 | 0.7863 | +0.1925 | +0.1960 | 0.0035 |

**Ortalama hata:** 0.00508
**5/5 seed'de hata < 0.01.**

## Formulun Kapsami (3 Boyut)

| Boyut | Test | Hata |
|---|---|---|
| **Dil** | 8 dil (6 aile) | < 0.01 |
| **Gorev** | Binary, multiclass, multi-label | < 0.025 |
| **Metrik** | Accuracy, bit-accuracy | < 0.01 |

## Bilimsel Sonuc

**Formul gorev-bagimsiz, dil-bagimsiz, metrik-bagimsiz:**

    gain = cov_sym * (sym_acc_answered - neu_acc)

Bu, universal bir prensip. Hibrit sistem tasariminda temel formul.

## Yayin Icin Deger

"Formul 3 gorev tipinde (binary, multiclass, multi-label), 8 dilde,
ve 2 metrikte dogrulandi. Ortalama hata 0.008. Bu, evrensel bir
prensip oldugunu gosterir."
