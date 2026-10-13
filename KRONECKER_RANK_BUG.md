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

## Düzeltme (UYGULANDI — 2026-10-13)
1. `n^(2K)` değerini **logaritmik** hesapla (string olarak "10^154" yaz)
2. n ve K için üst sınır kontrolü ekle
3. `--force` flag'i olmadan büyük konfigürasyonu reddet

## Kanıt
Başarılı çalıştırma: n=16,32,64 × K=1,2,4,8 → ~1 dk
Başarısız çalıştırma: n=16..256 × K=1..32 → 1+ saat, disk doldu

## Uygulanan Çözüm (P0-8, 2026-10-13)

`hga/evaluation/kronecker_rank.py` içine `guvenli_mi(n, k)` fonksiyonu
eklendi. `run_nk_rank_sweep` hesaplamadan **önce** tüm (n, K) çiftlerini
bu fonksiyonla doğrular.

### Güvenli Sınırlar

| Sınır | Değer | Gerekçe |
|---|---|---|
| `n` üst sınırı | 64 | n⁴ = 16.7M element, güvenli |
| `K` üst sınırı | 8 | Uzun zincirler için yeterli |
| `log10(n^(2K))` | 30 | 10³⁰ element ~ 8×10¹² TB, astronomik |

### Davranış

- **Güvensiz konfigürasyon + `force=False`** → `ValueError`
- **Güvensiz konfigürasyon + `force=True`** → `logging.warning` + devam
- **CLI**: `--force` flag'i (bkz. `hga --help` → `kronecker-rank`)

### Regresyon Koruması

`tests/test_kronecker_rank_guvenlik.py` — 11 test:

- Sınır kabul: `(64, 8)` geçer
- 7 farklı güvensiz durum reddedilir:
  - `n<2`, `K<1`, `n>64`, `K>8`
  - `(256, 32)` ve `(128, 16)` log sınırı aşımı
- Orijinal hata senaryosu `(128, 16)` artık `ValueError` veriyor
- Güvenli konfigürasyon `(4, 1)` çalışıyor

### İlgili Commit

`f73723c` — *fix(kronecker): n^(2K) taşma koruması ve --force flag (P0-8)*
