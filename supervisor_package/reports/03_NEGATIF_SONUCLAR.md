# HGA Negatif Sonuclar — Durust Bilim

**Tarih:** 2026-09-21
**Kapsam:** CKPT-000 -> CKPT-002 boyunca test edilen hipotezler

## Yonetici Ozeti

5 hipotez test edildi. 5i reddedildi veya marjinal kaldi. Her basarisizlik
bir sonraki arastirmaci icin degerli bilgidir.

## 1. Hierarchical Kronecker — Null

**Hipotez:** Buyuyen boyutlu Kronecker piramidi daha iyi ogrenir.

**Test:** 30 seed, 256 adim, TWT long-range (min_dist>=3)

| Seed | delta F1 | Cohen d | p-deger |
|---|---|---|---|
| 10 | +0.0033 | 0.45 | 0.19 |
| 20 | +0.0025 | 0.33 | 0.16 |
| 30 | +0.0018 | 0.24 | 0.21 |

**Yorum:** Daha fazla seed -> kucuk etki -> gurultu. p>0.05 her durumda.

## 2. Kronecker Depth K — Etkisiz

**Hipotez:** K katman derinlik ifade gucunu artirir.

**Test:** n=16,32,64 icin K=1,2,4,8

| n | K=1 etkin boyut | K=8 etkin boyut |
|---|---|---|
| 16 | 98.38 | 98.38 |
| 32 | 368.42 | 368.42 |
| 64 | 1491.38 | 1491.38 |

**Yorum:** Etkin boyut Kdan bagimsiz. Aktivasyonsuz zincir tek katmana coker.

## 3. Low-Rank Sparse — Verimsiz

**Hipotez:** Dusuk rank faktorel parametre azaltir, kapasiteyi korur.

**Test:** n_base=32, rank_budget=0,16,32,64

| rank | parametre | etkin boyut | verimlilik |
|---|---|---|---|
| full | 86,016 | 9,905 | 0.115 |
| 16 | 21,504 | 214 | 0.0099 |
| 64 | 69,632 | 2,005 | 0.0288 |

**Yorum:** Parametre 4x azaldi, etkin boyut 46x azaldi. Verimlilik 11.5x dustu.


## 4. Hash Sonlandirici — Marjinal

**Hipotez:** Ek karistirma adimi hash kalitesini artirir.

**Test:** 100K rastgele pencere, mevcut vs ek adimli

| n | Mevcut | Iyi | Fark |
|---|---|---|---|
| 1,000 | 999 | 1,000 | +1 |
| 10,000 | 9,962 | 9,954 | -8 |
| 100,000 | 95,308 | 95,450 | +142 |

**Yorum:** %0.15 fark. Hash zaten iyi. Darbogaz politika.

## 5. Bellek Kapasite Siniri — Ampirik

**Hipotez:** 1M slotlu hash tablosu yeterli.

**Test:** memory-benchmark, 1K->1M baglam

| Baglam | Carpisma | Geri cagirma |
|---|---|---|
| 1,000 | 0.6% | 99.4% |
| 10,000 | 7.1% | 92.9% |
| 100,000 | 48.8% | 51.2% |
| 1,000,000 | 93.4% | 6.6% |

**Yorum:** Pratik sinir ~50K baglam. first-writer-wins politikasi kaybi buyutuyor.


## Bilimsel Deger

Bu 5 negatif sonuc:
1. Gelecek arastirmacilara yol gosterir
2. Kaynak israfini onler
3. Basarili sonuclari cerceveler

## Basarili Sonuclarla Kontrast

| Bulgu | Sonuc |
|---|---|
| Hibrit paradigm (10 seed) | p<0.001, d=5-14 |
| Determinizm | Iki kosu ayni fingerprint |
| Robustness | Gain gorevle artar 2.3-2.4x |

## Sonuc

Modern ML arastirmasinin cogu negatif sonuclardan olusur. HGAnin degeri:
durustce olcmus, belgelemis, tekrarlanabilir kalmis.

## Meta-Ders

- Kucuk orneklemdeki buyuk farklar cogunlukla gurultudur
- Matematiksel guzellik pratik fayda garantisi degildir
- Kapasite, gorev onu gerektirdiginde deger katar
- Durust olcum, basarili sonuclardan daha degerlidir
