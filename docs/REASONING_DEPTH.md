# Güvenilir Çıkarım Derinliği — C_R ve C_RD (P0-7)

Protokol: `reliable_reasoning_depth_v1` · profil: `deep` · veri imzası: `5c8e421c20f1`
Eşik: 1 · tohumlar: [1, 2, 3] · bellek: 262144 slot × 2 tablo

**C_R = 2048**

| Dolgu | C_RD | C_RD / C_R |
|---:|---:|---:|
| 0 | 2048 | 1.000 |
| 64 | 2048 | 1.000 |
| 256 | 2048 | 1.000 |
| 1024 | 2048 | 1.000 |
| 4096 | 2048 | 1.000 |
| 16384 | 256 | 0.125 |

## Doğruluk ızgarası

| hop \ dolgu | 0 | 64 | 256 | 1024 | 4096 | 16384 |
|---|---:|---:|---:|---:|---:|---:|
| 1 adım (geri çağırma) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 8 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 64 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 256 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 1024 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 |
| 2048 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 |
| 4096 adım | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |

## Bellek geri çağırma ızgarası

| hop \ dolgu | 0 | 64 | 256 | 1024 | 4096 | 16384 |
|---|---:|---:|---:|---:|---:|---:|
| 1 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 8 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 64 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 256 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 1024 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| 2048 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| 4096 adım | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Kabul kapıları

| kapı | sonuç |
|---|---|
| c_r_at_least_8 | GEÇTİ |
| retains_half_depth_under_max_distractors | KALDI |
| multi_hop_beats_degenerate | GEÇTİ |
| c_r_not_grid_limited | GEÇTİ |
| depth_monotone_in_distractors | GEÇTİ |

## Bulgular

- C_R = 2048 (dolgu yok, eşik 1).
- C_RD: 64→2048, 256→2048, 1024→2048, 4096→2048, 16384→256.
- Derinlik ilk olarak 16384 dolgu seviyesinde düşüyor (2048 → 256). Bu ölçülen bir sınırdır; eşik gevşetilerek gizlenmedi.
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

