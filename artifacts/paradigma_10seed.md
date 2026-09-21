# Faz 21 — Neural vs Symbolic vs Hybrid

Tohumlar: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Görev: `{'entities': 120, 'relations': 8, 'train': 1200, 'test': 400, 'test_unseen': 80, 'test_symbolic_decidable': 199}`

| Metrik | symbolic | neural | hybrid |
|---|---:|---:|---:|
| Accuracy | 0.490 ± 0.034 | 0.790 ± 0.031 | 0.896 ± 0.020 |
| Coverage | 0.490 ± 0.034 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Precision | 1.000 ± 0.000 | 0.803 ± 0.035 | 0.902 ± 0.028 |
| Recall | 1.000 ± 0.000 | 0.768 ± 0.033 | 0.888 ± 0.019 |
| F1 | 1.000 ± 0.000 | 0.785 ± 0.031 | 0.895 ± 0.020 |
| FAR | 0.000 ± 0.000 | 0.189 ± 0.035 | 0.097 ± 0.029 |
| FRR | 0.000 ± 0.000 | 0.232 ± 0.033 | 0.112 ± 0.019 |
| Acc (görülen) | 0.487 ± 0.046 | 0.855 ± 0.036 | 0.927 ± 0.027 |
| Acc (görülmemiş) | 0.501 ± 0.071 | 0.530 ± 0.044 | 0.772 ± 0.031 |

- Sembolik kol cevapladığı örneklerde çok güçlü (accuracy_on_answered=1.000) ama coverage=0.490: gizli özellikli örneklerde çekimser kalır. Tek başına 'yüksek doğruluk' iddiası bu yüzden kapsam belirtilmeden anlamsızdır.
- Nöral kol tam kapsam verir (coverage=1.000) ama doğruluğu düşüktür (accuracy=0.790) ve görülmemiş varlıklarda 0.530'e çöker (görülen: 0.855).
- Hibrit kol her iki zaafı da kapatır: accuracy=0.896 ± 0.020, coverage=1.000.
- Hibritin sembolik üzerine net kazancı +0.406, nöral üzerine +0.106 accuracy puanıdır.
- FAR tek başına raporlanmaz: hibrit FAR=0.097, FRR=0.112, F1=0.895.
- Nöral kol eğitim doğruluğu 1.000, test doğruluğu 0.790 → genelleme açığı +0.210. Görülmemiş varlıkta 0.530 ≈ şans (0.5): kural öğrenmiyor, varlık kimliği ezberliyor.
