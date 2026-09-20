# Hierarchical Kronecker PoC — v1 Sonuçları

## Test 1: Ölçek Genişlemesi

| n_base | FLAT eff_dim | HIER eff_dim | Kazanç |
|---|---|---|---|
| 16 | 89.63 | 2,461.07 | 27.46× |
| 32 | 365.72 | 9,905.39 | 27.08× |
| 64 | 1,497.51 | 39,852.06 | 26.61× |
| 128 | 5,960.22 | 158,836.93 | 26.65× |

**Bulgu:** Hiyerarşik Kronecker, flat yapıya göre sabit ~27× etkin boyut kazancı sağlıyor.
Kazanç ölçekle büyümüyor (sabit kalıyor) ama her ölçekte var.

## Test 2: Seyreklik (BUG)

| rank_budget | params | rank | eff_dim |
|---|---|---|---|
| full | 86,016 | 16,384 | 9,905 |
| 16 | 86,016 | 3,840 | 214 |
| 32 | 86,016 | 7,168 | 716 |
| 64 | 86,016 | 12,288 | 2,005 |

**BUG:** Parametre sayısı düşmüyor. Düşük rank faktörleri U@V.T ile
tam matrise çevrildiği için numel() tam matrisi sayıyor.

**Düzeltme:** `params = r*(n_in + n_out)` kullan.
