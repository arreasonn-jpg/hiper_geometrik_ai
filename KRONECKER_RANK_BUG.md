# kronecker-rank Bug: n^(2K) Overflow

## Belirti
`--n-values 16,32,64,128,256 --k-values 1,2,4,8,16,32` komutu 1+ saat çalıştı, çökmeden diski doldurdu.

## Kök Neden
n=128, K=16 için n^(2K) = 10^67 elementlik sanal matris üretilmeye çalışıldı.
Bu, hem RAM'i hem diski aştı.

## Güvenli Sınır
- n ≤ 64
- K ≤ 8
- n^(2K) ≤ 10^30

## Önerilen Düzeltme
1. `n^(2K)` değerini **logaritmik** hesapla (string olarak "10^154" yaz)
2. n ve K için üst sınır kontrolü ekle
3. `--force` flag'i olmadan büyük konfigürasyonu reddet

## Kanıt
Başarılı çalıştırma: n=16,32,64 × K=1,2,4,8 → ~1 dk
Başarısız çalıştırma: n=16..256 × K=1..32 → 1+ saat, disk doldu
