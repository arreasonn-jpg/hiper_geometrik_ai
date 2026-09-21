# Bellek Siniri Analizi — 100K Neden Limit?

**Tarih:** 2026-09-21

## Ozet

Seyrek bellek, 1M slot hash tablosu kullanir. Ancak:
- 31-bit hash (2.1 milyar olası anahtar)
- Tablo 1M slot (2^20)
- Tablo, hash uzayinin 1/2048i

Bu yuzden anahtarlar 2048 kat sikisiyor ve 100K baglamda carpisma %48.8e ulasiyor.

## Gozlenen Kapasite Egrisi

| Baglam | Carpisma | Geri Cagirma |
|---|---|---|
| 1,000 | 0.6% | 99.4% |
| 10,000 | 7.1% | 92.9% |
| 100,000 | 48.8% | 51.2% |
| 1,000,000 | 93.4% | 6.6% |

Kritik esik: 50K-100K arasi.


## Matematiksel Analiz

Ideal uniform hash icin:
    P(carpisma) = 1 - exp(-n / tablo_boyutu)

| n | Tablo 1M | Gozlenen | Oran |
|---|---|---|---|
| 10K | 1.0% | 7.1% | 7.1x |
| 100K | 9.5% | 48.8% | 5.1x |
| 1M | 63% | 93.4% | 1.5x |

Gozlenen carpisma, ideal modelden 5-7x yuksek.

### Sebep: Modulo 2^20 + Hash Alt-Bit Korelasyonu

Kod satiri:
    return (anahtar * tuz + (tablo_indeksi + 1) * 0x9E37) % self.tablo_boyutu

tablo_boyutu = 2^20 oldugundan modulo 2^20, hashin alt 20 bitini secer.
Splitmix son XOR islemi ust ve alt 16 biti karistirir. Alt 20 bit beklenen
kalitede degil -> kumeleme (clustering).


## Cozum Onerileri

### A) Tablo Boyutunu Artir (En Kolay)

    tablo_boyutu = 16_777_216   # 16M slot (16x kapasite)

Kazanc: 100Kda carpisma %48.8 -> ~5% (10x iyilesme)
Maliyet: 16x bellek (2 GB) - cihaz icin sorun

### B) Hash Sonlandiriciyi Duzelt

Ek karistirma adimi:
    acc = ((acc ^ (acc >> 16)) * 0x85EBCA6B) & self.MASK31

Kazanc: Alt bit korelasyonunu azaltir -> ~3-5x iyilesme
Maliyet: Sifir

### C) 64-bit Hash

Kazanc: Kalite artar. Maliyet: Ayni bellek.

### D) Iki Farkli Asal ile Cift Hash

Mevcut kod iki tablo kullaniyor ama ayni hash ile. Gercek Bloom filter
icin iki farkli hash gerekir. Kazanc: Carpisma karesel azalir.

## Yayin Icin Deger

### Limitations

HGA seyrek bellegi, sabit boyutlu hash tablosu kullanir. 1M slotluk tablo,
~50K baglama kadar guvenilirdir. 100Kda carpisma %48.8e ulasir.

### Future Work

Bellek kapasitesi uc yolla artirilabilir: (1) tablo boyutunu 16Mye cikarmak,
(2) hash sonlandiriciyi duzeltmek, (3) iki farkli asal ile Bloom hash.
Kombinasyon 100Kda carpismayi %48.8den <%1e dusurebilir.
