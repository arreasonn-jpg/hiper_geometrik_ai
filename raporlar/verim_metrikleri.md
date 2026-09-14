# Deneyim Verimi Ayrıştırması (P1-005)

Tohumlar: [1, 2, 3, 4, 5]

| Metrik | Ortalama | %95 GA |
|---|---:|---:|
| EY  (klasik) | 0.2385 | [0.2327, 0.2444] |
| NY  Yenilik | 0.1904 | [0.1863, 0.1946] |
| UEY Kullanışlı | 0.1606 | [0.1575, 0.1646] |
| GY  Genelleme | 0.6419 | [0.6116, 0.6721] |
| VID Bilgi yoğ. | 0.9434 | [0.9227, 0.9640] |

## Bulgular (tohum 1)

- EY=0.2292 fakat NY=0.1854: üretimin bir kısmı zaten bilinen üçlülerin tekrarı (224 tekrar üretim).
- NY=0.1854 fakat UEY=0.1562: doğrulanan bilginin bir kısmı bellekte kalıcı/kullanılabilir değil (70 çakışma).
- GY=0.6047: holdout doğruluğu öğrenmeyle 0.0000 → 0.6047 arttı (52/86 örnekte karar verilebildi, karar verilenlerde isabet 1.0000). Genelleme, görülmemiş negatifleri fonksiyonel teklikten çıkarmaktır.
- VID=0.9186 bit/deneyim (olgu başına 4.9542 bit, sonuç uzayı R=31).

## Sınırlar

- Sentetik aritmetik domain; sonuçlar genel dil görevlerine taşınmaz.
- GY tek bir çıkarım kuralıyla (fonksiyonel teklik) ölçülür: öğrenilen pozitiften görülmemiş negatifi elemek. Bu gerçek ama DAR bir genellemedir; yeni toplama bağıntısı keşfetmek değildir.
- GY'nin paydası holdout kümesidir; kapsam (coverage) düşükken GY de düşük görünür — bu öğrenme eksikliğidir, metrik kusuru değil.
- UEY bellek geri çağrımına bağlıdır; farklı slot sayısı farklı UEY verir.
- VID, her olgunun sonuç uzayında tekdüze dağıldığını varsayar.
- Güven aralıkları 5 tohumun bootstrap dağılımından hesaplandı; tohum sayısı düşük olduğu için aralıklar geniştir.
