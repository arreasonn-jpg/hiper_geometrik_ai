# Hierarchical Kronecker PoC — Sonuçlar

## Test 1: Ölçek (KANITLANDI)

| n_base | FLAT eff_dim | HIER eff_dim | Kazanç |
|---|---|---|---|
| 16 | 89.63 | 2,461.07 | 27.46× |
| 32 | 365.72 | 9,905.39 | 27.08× |
| 64 | 1,497.51 | 39,852.06 | 26.61× |
| 128 | 5,960.22 | 158,836.93 | 26.65× |

Hiyerarşik yapı her ölçekte sabit ~27× kazanç sağlıyor.

## Test 2: Low-rank seyreklik (BAŞARISIZ)

| rank_budget | params | eff_dim | eff_dim/param |
|---|---|---|---|
| full | 86,016 | 9,905 | 0.115 |
| 16 | 21,504 | 214 | 0.0099 |
| 32 | 40,960 | 716 | 0.0175 |
| 64 | 69,632 | 2,005 | 0.0288 |

Verimlilik 11.5× düşüyor. Low-rank seyreklik doğru araç değil.
Alternatifler: block-sparse, MoE, quantization.

## Sonuç

Vizyonun %80'i doğru: hiyerarşik genişleme ÇALIŞIYOR.
Seyreklik için farklı strateji gerekli (low-rank değil).
