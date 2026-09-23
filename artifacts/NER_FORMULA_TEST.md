# NER Formula Testi — 4. Görev Tipi

**Tarih:** 2026-09-23
**Görev:** Named Entity Recognition (PER/LOC/ORG)
**Metrik:** Token accuracy (linear)
**Kurulum:** 5 seed × 4 hidden_ratio = 20 konfigürasyon

## Sonuç

| Formül | Ortalama ε | Maksimum ε | ε < 0.01 |
|---|---|---|---|
| Basit | 0.0083 | 0.0175 | 11/20 |
| **Kesin** | **0.0000** | **0.0000** | **20/20** |

**Kesin formül, NER + token accuracy'de makine hassasiyetinde doğrulandı.**

## Kanıt Matrisi (4 Görev Tipi)

| Görev Tipi | Metrik | Ortalama ε |
|---|---|---|
| Binary (dependency) | Accuracy | 0.003 |
| Multiclass (K=4) | Accuracy | 0.008 |
| Multi-label | Bit-accuracy | 0.005 |
| **NER** | **Token accuracy** | **0.0000** |

**4/4 görev tipi, tüm lineer metriklerde formül geçerli.**
