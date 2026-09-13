# HGA Milestone Tablosu — K₀ → E₀ → V₀ → K₁ → … → Kₙ

Protokol: `versioned-ledgered-closed-loop-milestone-v1`

- seed'ler: [1, 2, 3, 4, 5] (5 koşu, mean ± std)
- döngü: 100 (tamamlanan: 100), batch: 32
- ground truth: bağımsız aritmetik oracle (evaluator DEĞİL)
- bilgi zinciri geçerli: True
- defter zinciri geçerli: True
- test izolasyonu temiz: True

## Metrik × Cycle

| Metrik | Cycle 0 | Cycle 10 | Cycle 50 | Cycle 100 |
| --- | ---: | ---: | ---: | ---: |
| Verified Knowledge | 100.0 ± 0.0 | 136.4 ± 4.2 | 295.8 ± 5.0 | 497.6 ± 12.5 |
| New Knowledge | 0.0 ± 0.0 | 36.4 ± 4.2 | 195.8 ± 5.0 | 397.6 ± 12.5 |
| Invalid | 0.0 ± 0.0 | 283.6 ± 4.2 | 1404.2 ± 5.0 | 2802.4 ± 12.5 |
| Uncertain | 0.0 ± 0.0 | 20.0 ± 0.0 | 100.0 ± 0.0 | 200.0 ± 0.0 |
| Conflict | 0.0 ± 0.0 | 10.0 ± 0.0 | 50.0 ± 0.0 | 100.0 ± 0.0 |
| FAR | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| FRR | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| Novelty | 0.0000 ± 0.0000 | 0.9143 ± 0.0000 | 0.9143 ± 0.0000 | 0.9143 ± 0.0000 |
| Experience Yield | 0.0000 ± 0.0000 | 0.1040 ± 0.0121 | 0.1119 ± 0.0029 | 0.1136 ± 0.0036 |
| Memory Collision | 0.0 ± 0.0 | 0.2 ± 0.4 | 5.2 ± 1.7 | 19.0 ± 4.1 |
| Memory Recall | 1.0000 ± 0.0000 | 0.9950 ± 0.0100 | 0.9735 ± 0.0086 | 0.9523 ± 0.0099 |
| Incorrect Knowledge | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |

## Kapasite Çerçevesi (Faz 13–14)

- C_E (üretilebilir)     : 1,184,832
- C_V (doğrulanabilir)   : 1,180,685
- C_V / C_E              : 0.99649993
- sıralama C_V ≤ C_E ≤ C_M: True

## Rollback Tatbikatı (Faz 23)

- sağlam sürüm      : `K100`
- bozulmuş sürüm    : `K101` (yanlış olgu: 1)
- geri yüklenen     : `K102` (yanlış olgu: 0)
- içerik birebir geri geldi: True
- hatalı sürüm geçmişte korundu: True

## Immutable Ledger (Faz 24)

- toplam kayıt (seed başına): 3,500
- her aday — VERIFIED, INVALID, UNCERTAIN, CONFLICT — hash-zincirli deftere yazılır; hiçbir kayıt silinmez.

## Sınırlar

- Ground truth bağımsız ama sentetik aritmetik environment'tan gelir.
- Yalnız VERIFIED olgular Kₙ'e yazılır; bu bir üst sınır protokolüdür.
- Tablo genel dilde otonom bilgi keşfi kanıtı değildir.
- Rollback tatbikatı kasıtlı failure injection'dır; doğal hata oranı değildir.

## Tekrarlanabilirlik

- deney kimlikleri: EXP-0001, EXP-0002, EXP-0003, EXP-0004, EXP-0005
- dataset_hash: `18859f26d4b3e3c19218a0408a186de612b2f8252a9b9c2ee633fa4151e39a32`
- config_hash: `96a0b4a41a2905d755623e7ecd191c759f7b0e8580e802613c3066a738f0fa87`
- seed'ler arası sonuç özdeşliği: False
