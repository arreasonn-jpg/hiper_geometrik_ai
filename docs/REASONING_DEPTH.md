# Güvenilir Çıkarım Derinliği — C_R ve C_RD (P0-7)

Protokol: `reliable_reasoning_depth_v1` · profil: `deep` · veri imzası: `d09dc2d62c4c`
Eşik: 1 · tohumlar: [1, 2, 3] · bellek: 262144 slot × 2 tablo

**C_R = 32**  *(ızgara tavanı — gerçek derinlik daha büyük olabilir)*

| Dolgu | C_RD | C_RD / C_R |
|---:|---:|---:|
| 0 | 32 | 1.000 |
| 64 | 32 | 1.000 |
| 256 | 32 | 1.000 |
| 1024 | 32 | 1.000 |
| 4096 | 32 | 1.000 |
| 16384 | 8 | 0.250 |

## Doğruluk ızgarası

| hop \ dolgu | 0 | 64 | 256 | 1024 | 4096 | 16384 |
|---|---:|---:|---:|---:|---:|---:|
| 1 adım (geri çağırma) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 8 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 16 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 |
| 32 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 |

## Bellek geri çağırma ızgarası

| hop \ dolgu | 0 | 64 | 256 | 1024 | 4096 | 16384 |
|---|---:|---:|---:|---:|---:|---:|
| 1 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 8 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 16 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| 32 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## Kabul kapıları

| kapı | sonuç |
|---|---|
| c_r_at_least_8 | GEÇTİ |
| retains_half_depth_under_max_distractors | KALDI |
| multi_hop_beats_degenerate | GEÇTİ |
| c_r_not_grid_limited | KALDI |
| depth_monotone_in_distractors | GEÇTİ |

## Bulgular

- C_R = 32 (dolgu yok, eşik 1). UYARI: bu değer ızgaranın en derin hop'una (32) eşit — ölçüm tavana çarptı, gerçek derinlik daha büyük olabilir.
- C_RD: 64→32, 256→32, 1024→32, 4096→32, 16384→8.
- Derinlik ilk olarak 16384 dolgu seviyesinde düşüyor (32 → 8). Bu ölçülen bir sınırdır; eşik gevşetilerek gizlenmedi.
- En düşük bellek geri çağırma oranı 0.0000; 262144 slot × 2 tablo ile ölçüldü. Derinlik kaybı ile bellek kaybı bu sayede ayrı okunabilir.

## Sınırlar

- Bu bir dil modeli 'context window' testi DEĞİLDİR; token penceresi, attention span veya doğal metin anlama ölçülmez. Sembolik depo + seyrek bellek üzerinde zincir takibi ölçülür.
- Zincir tek bir geçişli ilişki (öncesi) üzerinden kurulur. Bu gerçek ama DAR bir çıkarımdır; çok ilişkili karma akıl yürütme kapsam dışıdır.
- hop=1 sütunu çıkarım değil geri çağırmadır ve negatif kontrol olarak okunmalıdır; çıkarım iddiası yalnız hop>=2 için geçerlidir.
- Dolgu olgular sentetiktir ve zincirle aynı ilişkiyi kullanır; doğal metindeki anlamsal karışıklığı temsil etmez.
- Zincir takibi genişlik-öncelikli aramadır; maliyeti kenar sayısıyla büyür. Bu bir öğrenilmiş yetenek değil, deterministik bir çıkarımdır.
- C_R/C_RD deterministik zincir takibini ölçer; öğrenilmiş akıl yürütme değildir.
- Güvenilirlik eşiği rapora yazılır; eşik düşürülürse derinlik yapay olarak artar — tablolar eşiksiz okunmamalıdır.
- Profil `deep` ızgarası sonlu: tarananın ötesinde bir çökme noktası olabilir, bulunmaması yokluğu kanıtlamaz.
