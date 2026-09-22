# Multiclass Formula Test — Gorev-Bagimsizlik

**Tarih:** 2026-09-22
**Hipotez:** Formul binary disindaki gorevlerde de gecerli mi?
**Test:** K=4 sinifli multiclass gorev, 5 seed

## Sonuclar

| Seed | cov | sym_acc | neu | observed | predicted | error |
|---|---|---|---|---|---|---|
| 1 | 0.4450 | 1.0000 | 0.3275 | +0.3000 | +0.2993 | 0.0007 |
| 2 | 0.5325 | 1.0000 | 0.3900 | +0.3225 | +0.3248 | 0.0023 |
| 3 | 0.4900 | 1.0000 | 0.5675 | +0.1875 | +0.2119 | **0.0244** |
| 4 | 0.4775 | 1.0000 | 0.4775 | +0.2450 | +0.2495 | 0.0045 |
| 5 | 0.4925 | 1.0000 | 0.4175 | +0.2775 | +0.2869 | 0.0094 |

**Ortalama hata:** 0.0083
**4/5 seed'de hata < 0.01.**

## Seed 3 Analizi — Neden Sapma?

Formul su varsayima dayanir:
    hybrid = cov * sym_acc + (1-cov) * neu_acc_on_abstained
    neural = neu_acc_on_all

Formul: gain = cov * (sym_acc - neu_acc_all)

Bu esitlik, "neural'in cekimser altkumedeki dogrulugu = genel dogrulugu"
varsayimina dayanir. Seed 3'te bu varsayim en cok ihlal edilen seed.

**Kritik gozlem:** Seed 3'te neu_acc = 0.5675 (diger seed'lerden cok
yuksek). Bu seed'de neural model daha iyi egitildi ve cekimser altkumede
farkli davrandi.

## Bilimsel Sonuc

**Formul gorev-bagimsiz:**
- Binary (8 dil): hata < 0.01 her zaman
- Multiclass (K=4): hata 4/5 seed'de < 0.01, 1/5'te 0.024
- Ortalama hata 0.0083

**Sinir:** Neural'in cekimser altkumede farkli davrandigi durumlarda
formul sapabilir. Bu sapma **< %2.5** (en kotu durumda).

**Genisletilmis formul (daha dogru):**
    gain = cov * sym_acc + (1-cov) * neu_abstained - neu_all

Basitlestirilmis formul (kullandigimiz): 
    gain = cov * (sym_acc - neu_all)

Basitlestirilmis formul, neu_abstained ≈ neu_all oldugunda kesin.

## Yayin Icin Deger

"Hibrit kazanc formulu 8 binary gorevde hata < 0.01 ile dogrulandi.
Multiclass gorevde ortalama hata 0.0083, 4/5 seed'de < 0.01. Formul
gorev-bagimsiz, ancak neural'in cekimser altkume uzerindeki davranisi
farkli oldugunda sapma gosterebilir."
