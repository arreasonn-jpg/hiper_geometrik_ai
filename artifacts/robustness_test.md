# Robustness — Hibrit Kazanç Görev Büyüdükçe Artıyor

## Test A: Entity Count
| Entity | Neural | Hybrid | Gain |
|---|---|---|---|
| 60 | 0.8658 | 0.9275 | +0.0617 |
| 120 | 0.7617 | 0.8850 | +0.1233 |
| 240 | 0.7167 | 0.8583 | +0.1417 |
**Trend: Gain 2.3× büyüyor**

## Test B: Relation Count
| Relation | Neural | Hybrid | Gain |
|---|---|---|---|
| 4 | 0.8442 | 0.9067 | +0.0625 |
| 8 | 0.7617 | 0.8850 | +0.1233 |
| 16 | 0.7075 | 0.8567 | +0.1492 |
**Trend: Gain 2.4× büyüyor**

## Test C: Unseen Ratio
| Unseen | Neural | Hybrid | Gain |
|---|---|---|---|
| 0.10 | 0.7417 | 0.8250 | +0.0833 |
| 0.20 | 0.5500 | 0.7958 | +0.2458 |
| 0.40 | 0.5813 | 0.7729 | +0.1917 |

## Ana Bulgu

**Hibrit avantajı görev büyüklüğüyle artıyor** (0.06 → 0.15).
Klasik neural kapasitesiz kalıp düşerken (0.87 → 0.72), hibrit dayanıklı kalıyor (0.93 → 0.86).

Bu, "hibrit mimari büyük problemlerde daha değerli" iddiasını destekler.
