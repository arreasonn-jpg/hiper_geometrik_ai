# Kesin Hibrit Formulu ve Coverage Etkisi

## Tam Formul (Kesin)

    hybrid = cov * sym_acc + (1-cov) * neu_abst

Bu bir tanimdir:
- cov olasilikla sembolik cevap kullanilir (dogrulugu sym_acc)
- (1-cov) olasilikla neural cevap kullanilir (cekimser altkumede
  dogrulugu neu_abst)

## Basitlestirilmis Formul (Yaklasik)

    gain = cov * (sym_acc - neu_all)

Bu, "neu_abst ≈ neu_all" varsayimina dayanir.

## Hata Ozdesligi

Tam formul ile basit formul arasindaki fark:

    err = |gain_actual - gain_simple|
        = |(1-cov) * neu_abst - (1-cov) * neu_all|
        = (1-cov) * |neu_abst - neu_all|
        = (1-cov) * fark

**Bu KESIN matematiksel ozdeslik.**

## Multiclass Test Dogrulamasi

| seed | cov | 1-cov | fark | (1-cov)*fark | observed err |
|---|---|---|---|---|---|
| 1 | 0.4450 | 0.5550 | 0.0013 | 0.00072 | 0.0007 |
| 2 | 0.5325 | 0.4675 | 0.0050 | 0.00234 | 0.0023 |
| 3 | 0.4900 | 0.5100 | 0.0479 | 0.02443 | 0.0244 |
| 4 | 0.4775 | 0.5225 | 0.0086 | 0.00449 | 0.0045 |
| 5 | 0.4925 | 0.5075 | 0.0185 | 0.00939 | 0.0094 |

**Hata = (1-cov) * fark, 5/5 seed'de dogrulandi.**

## Binary vs Multiclass

| Boyut | Binary (8 dil) | Multiclass (K=4) |
|---|---|---|
| cov_sym | ~0.97 | ~0.49 |
| 1-cov | ~0.03 | ~0.51 |
| fark | ~0.01-0.05 | ~0.001-0.048 |
| err | < 0.01 | 0.0007 - 0.0244 |

**Sebep:** Multiclass'ta coverage dusuk (0.49). (1-cov) buyuk
oldugundan fark'a olan hassasiyet artar.

## Bilimsel Sonuclar

### 1. Formul gorev-bagimsizdir
Kesin formul (tam) her gorevde gecerlidir. Basit formul, "neu_abst ≈
neu_all" kosulu altinda gecerlidir.

### 2. Coverage onemli
Yuksek coverage → basit formul kesin.
Dusuk coverage → basit formul yaklasik, tam formul gerekli.

### 3. Pratik cikarim
- Hibrit sistem tasarimcisi once coverage'i yukseltmeli
- Coverage yuksekse basit formul yeterli
- Coverage dusukse neural'in cekimser altkume davranisi onemli

## Yayin Icin

"Hibrit kazanc formulu, coverage ve neural'in altkume davranisina
bagli iki parcadan olusur:

    gain = cov * (sym_acc - neu_all) + (1-cov) * (neu_abst - neu_all)

Birinci terim basit formul, ikinci terim coverage-duzeltmeli.
Yuksek coverage rejiminde ikinci terim ihmal edilebilir; dusuk
coverage rejiminde gerekli."

**Bu, formulu daha genel hale getirir ve gorev-bagimsiz kiler.**
