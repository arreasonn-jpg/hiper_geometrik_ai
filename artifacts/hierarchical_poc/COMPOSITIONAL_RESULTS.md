# Compositional Görev Sonuçları

| Model | Params | Test |
|---|---|---|
| Hierarchical | 87,044 | 0.795 |
| Dense MLP | 20,868 | 0.782 |
| Flat Kronecker | 2,564 | 0.781 |

## Bulgu

- Hierarchical kazanıyor ama sadece +1.3 puan
- Flat en verimli (2.5K param, neredeyse aynı sonuç)
- Kazanç görev zorluğuna bağlı:
  * Zor görev (random lineer): hierarchical +19 puan
  * Kolay görev (compositional): +1 puan

## Sonraki: Gerçek veri testi (TWT)
