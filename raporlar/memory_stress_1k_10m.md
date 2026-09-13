# Aktif Dynamic KV + Fixed Memory Stress

- Seedler: `[1, 2, 3, 4, 5]`
- Dataset hash: `c46da1ee227948e65743b42980bd3c7e57b9ac8fed5f5bda11a6b93065a1a5d2`
- Config hash: `2ef4f1d96e7d0e4e5312d3716da38159306fb4de7c43bc9d9cf1acbfc887bd6f`
- Git commit: `94fe3f6757d6b8b08e240f11862e4795daae6aea`
- Temiz manifestler: `True`

## 5-seed aggregate

| Context | Fixed history recall | Dynamic history recall | Dynamic active recall | Eviction | RSS MiB | ctx-pair/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.991600 ± 0.002154 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0 ± 0 | 436.59 ± 36.81 | 92,207 ± 5,707 |
| 10,000 | 0.926120 ± 0.000773 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0 ± 0 | 435.35 ± 32.65 | 86,162 ± 9,326 |
| 100,000 | 0.512728 ± 0.001186 | 0.655360 ± 0.000000 | 1.000000 ± 0.000000 | 34,464 ± 0 | 473.12 ± 4.69 | 60,163 ± 4,281 |
| 1,000,000 | 0.065536 ± 0.000000 | 0.065536 ± 0.000000 | 1.000000 ± 0.000000 | 934,464 ± 0 | 576.83 ± 0.59 | 27,277 ± 1,749 |
| 10,000,000 | 0.006554 ± 0.000000 | 0.006554 ± 0.000000 | 1.000000 ± 0.000000 | 9,934,464 ± 0 | 577.30 ± 1.01 | 27,409 ± 564 |

Tüm seed/kapılar: `True`
1K→10M kabulü: `True`

## Seed 1

Süre: `373.306727` saniye

| Context | Fixed recall | Dynamic history recall | Dynamic active recall | Eviction | RSS MiB | ctx-pair/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.993000 | 1.000000 | 1.000000 | 0 | 363.59 | 91,487 |
| 10,000 | 0.927200 | 1.000000 | 1.000000 | 0 | 373.39 | 85,756 |
| 100,000 | 0.511460 | 0.655360 | 1.000000 | 34,464 | 467.99 | 54,984 |
| 1,000,000 | 0.065536 | 0.065536 | 1.000000 | 934,464 | 577.49 | 27,591 |
| 10,000,000 | 0.006554 | 0.006554 | 1.000000 | 9,934,464 | 578.90 | 26,752 |

Tüm kapılar: `True`

## Seed 2

Süre: `358.122342` saniye

| Context | Fixed recall | Dynamic history recall | Dynamic active recall | Eviction | RSS MiB | ctx-pair/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.994000 | 1.000000 | 1.000000 | 0 | 447.93 | 93,002 |
| 10,000 | 0.924900 | 1.000000 | 1.000000 | 0 | 447.93 | 91,312 |
| 100,000 | 0.511370 | 0.655360 | 1.000000 | 34,464 | 470.28 | 61,758 |
| 1,000,000 | 0.065536 | 0.065536 | 1.000000 | 934,464 | 576.11 | 27,541 |
| 10,000,000 | 0.006554 | 0.006554 | 1.000000 | 9,934,464 | 576.08 | 28,006 |

Tüm kapılar: `True`

## Seed 3

Süre: `357.036147` saniye

| Context | Fixed recall | Dynamic history recall | Dynamic active recall | Eviction | RSS MiB | ctx-pair/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.989000 | 1.000000 | 1.000000 | 0 | 457.72 | 100,756 |
| 10,000 | 0.926400 | 1.000000 | 1.000000 | 0 | 459.01 | 94,304 |
| 100,000 | 0.512680 | 0.655360 | 1.000000 | 34,464 | 475.99 | 65,745 |
| 1,000,000 | 0.065536 | 0.065536 | 1.000000 | 934,464 | 577.54 | 28,947 |
| 10,000,000 | 0.006554 | 0.006554 | 1.000000 | 9,934,464 | 577.55 | 27,956 |

Tüm kapılar: `True`

## Seed 4

Süre: `372.792109` saniye

| Context | Fixed recall | Dynamic history recall | Dynamic active recall | Eviction | RSS MiB | ctx-pair/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.989000 | 1.000000 | 1.000000 | 0 | 451.88 | 82,825 |
| 10,000 | 0.925700 | 1.000000 | 1.000000 | 0 | 433.32 | 68,342 |
| 100,000 | 0.514050 | 0.655360 | 1.000000 | 34,464 | 470.44 | 55,350 |
| 1,000,000 | 0.065536 | 0.065536 | 1.000000 | 934,464 | 576.34 | 28,367 |
| 10,000,000 | 0.006554 | 0.006554 | 1.000000 | 9,934,464 | 576.35 | 26,727 |

Tüm kapılar: `True`

## Seed 5

Süre: `367.678118` saniye

| Context | Fixed recall | Dynamic history recall | Dynamic active recall | Eviction | RSS MiB | ctx-pair/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.993000 | 1.000000 | 1.000000 | 0 | 461.82 | 92,965 |
| 10,000 | 0.926400 | 1.000000 | 1.000000 | 0 | 463.10 | 91,095 |
| 100,000 | 0.514080 | 0.655360 | 1.000000 | 34,464 | 480.89 | 62,980 |
| 1,000,000 | 0.065536 | 0.065536 | 1.000000 | 934,464 | 576.65 | 23,938 |
| 10,000,000 | 0.006554 | 0.006554 | 1.000000 | 9,934,464 | 577.63 | 27,603 |

Tüm kapılar: `True`

## Sınırlar

- Sentetik unique context tek streaming geçişte üretilir; giriş listesi tutulmaz.
- FIRST_WINS table-1 exact history recall = occupied/context; her slot yalnız ilk kimliği tutar.
- Bounded Dynamic KV exact history recall = active/context; aktif küme exact recall'ı ayrıca örneklenir.
- Dynamic KV collision-free olsa da kapasite sonrası LRU eviction nedeniyle tüm tarih recall'ı düşer.
- RSS Linux /proc current resident set'tir; storage değerleri CPython nesne tahminidir.
- Bu exact-ID stress'tir; semantic/learned retrieval, neural kalite veya genel dil ölçmez.
