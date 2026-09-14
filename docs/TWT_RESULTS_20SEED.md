# TWT Gerçek Sonuç Tablosu (P0-3)

- Protokol: `twt_real_results_v1` v1 (imza `300a5cded2eb`)
- Görev: binary dependency arc validation — **dil modelleme değildir**
- Veri: TWT v1 `66b13a898efa8899…`, train/dev/test aday sayısı 127602/17030/16036
- Profil / tohumlar: `smoke` / `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]`

## Maliyet bütçesi

| Model | Parametre | Gövde | Bayt | İleri FLOP/örnek | Eğitim (s) | Çıkarım (s) |
|---|---:|---:|---:|---:|---:|---:|
| dense | 291,808 | 10,496 | 1,167,232 | 10.39K | 0.123 | 0.011 |
| transformer | 291,816 | 10,504 | 1,167,264 | 60.70K | 0.523 | 0.145 |
| kronecker | 291,514 | 10,202 | 1,166,056 | 13.64K | 0.216 | 0.074 |
| hga | 291,827 | 10,515 | 1,167,308 | 117.85K | 0.834 | 0.286 |

FLOP max/min oranı: **11.345×** (kapı 2.0×) → KALDI

> Parametre eşitliği FLOP eşitliğini GARANTİ ETMEZ. Bu oran kapıyı geçmezse, sonuç farkı kısmen işlem bütçesi farkına atfedilebilir ve öyle okunmalıdır.

## Ana sonuç tablosu (test, ortalama ± std)

### `all` (n=16036)

| Model | accuracy ↑ | f1 ↑ | precision ↑ | recall ↑ | far ↓ | frr ↓ | coverage ↑ |
|---|---|---|---|---|---|---|---|
| dense | 0.8978 ±0.0064 | 0.8992 ±0.0070 | 0.8872 ±0.0108 | 0.9118 ±0.0181 | 0.1162 ±0.0142 | 0.0882 ±0.0181 | 1.0000 ±0.0000 |
| transformer | 0.8210 ±0.0139 | 0.8196 ±0.0166 | 0.8262 ±0.0216 | 0.8147 ±0.0363 | 0.1726 ±0.0307 | 0.1853 ±0.0363 | 1.0000 ±0.0000 |
| kronecker **←** | 0.8991 ±0.0051 | 0.9008 ±0.0051 | 0.8854 ±0.0089 | 0.9170 ±0.0108 | 0.1188 ±0.0114 | 0.0830 ±0.0108 | 1.0000 ±0.0000 |
| hga | 0.8978 ±0.0066 | 0.8986 ±0.0065 | 0.8916 ±0.0107 | 0.9059 ±0.0120 | 0.1103 ±0.0127 | 0.0941 ±0.0120 | 1.0000 ±0.0000 |

### `entity_disjoint` (n=2704)

| Model | accuracy ↑ | f1 ↑ | precision ↑ | recall ↑ | far ↓ | frr ↓ | coverage ↑ |
|---|---|---|---|---|---|---|---|
| dense | 0.8715 ±0.0124 | 0.8720 ±0.0148 | 0.8676 ±0.0169 | 0.8777 ±0.0353 | 0.1348 ±0.0238 | 0.1223 ±0.0353 | 1.0000 ±0.0000 |
| transformer | 0.7752 ±0.0566 | 0.7525 ±0.1025 | 0.8106 ±0.0335 | 0.7242 ±0.1662 | 0.1739 ±0.0691 | 0.2758 ±0.1662 | 1.0000 ±0.0000 |
| kronecker **←** | 0.8752 ±0.0109 | 0.8760 ±0.0113 | 0.8701 ±0.0152 | 0.8826 ±0.0217 | 0.1322 ±0.0191 | 0.1174 ±0.0217 | 1.0000 ±0.0000 |
| hga | 0.8719 ±0.0105 | 0.8717 ±0.0106 | 0.8739 ±0.0200 | 0.8703 ±0.0234 | 0.1264 ±0.0256 | 0.1297 ±0.0234 | 1.0000 ±0.0000 |

### `relation_disjoint` (n=168)

| Model | accuracy ↑ | f1 ↑ | precision ↑ | recall ↑ | far ↓ | frr ↓ | coverage ↑ |
|---|---|---|---|---|---|---|---|
| dense **←** | 0.7429 ±0.1066 | 0.6861 ±0.1826 | 0.7950 ±0.0335 | 0.6524 ±0.2696 | 0.1667 ±0.0663 | 0.3476 ±0.2696 | 1.0000 ±0.0000 |
| transformer | 0.6848 ±0.0644 | 0.6283 ±0.1441 | 0.7385 ±0.0468 | 0.5952 ±0.2372 | 0.2256 ±0.1225 | 0.4048 ±0.2372 | 1.0000 ±0.0000 |
| kronecker | 0.7190 ±0.1043 | 0.6551 ±0.1820 | 0.7713 ±0.0609 | 0.6149 ±0.2693 | 0.1768 ±0.0748 | 0.3851 ±0.2693 | 1.0000 ±0.0000 |
| hga | 0.7065 ±0.1115 | 0.6193 ±0.1913 | 0.7900 ±0.0722 | 0.5452 ±0.2570 | 0.1321 ±0.0491 | 0.4548 ±0.2570 | 1.0000 ±0.0000 |

### `composition_disjoint` (n=7086)

| Model | accuracy ↑ | f1 ↑ | precision ↑ | recall ↑ | far ↓ | frr ↓ | coverage ↑ |
|---|---|---|---|---|---|---|---|
| dense | 0.8773 ±0.0073 | 0.8792 ±0.0082 | 0.8658 ±0.0122 | 0.8935 ±0.0212 | 0.1389 ±0.0169 | 0.1065 ±0.0212 | 1.0000 ±0.0000 |
| transformer | 0.7978 ±0.0131 | 0.7952 ±0.0127 | 0.8061 ±0.0193 | 0.7852 ±0.0181 | 0.1895 ±0.0242 | 0.2148 ±0.0181 | 1.0000 ±0.0000 |
| kronecker **←** | 0.8795 ±0.0063 | 0.8821 ±0.0063 | 0.8638 ±0.0101 | 0.9014 ±0.0138 | 0.1424 ±0.0137 | 0.0986 ±0.0138 | 1.0000 ±0.0000 |
| hga | 0.8762 ±0.0087 | 0.8773 ±0.0090 | 0.8695 ±0.0114 | 0.8855 ±0.0171 | 0.1331 ±0.0140 | 0.1145 ±0.0171 | 1.0000 ±0.0000 |

### `wording_disjoint` (n=6036)

| Model | accuracy ↑ | f1 ↑ | precision ↑ | recall ↑ | far ↓ | frr ↓ | coverage ↑ |
|---|---|---|---|---|---|---|---|
| dense | 0.9395 ±0.0062 | 0.9406 ±0.0060 | 0.9237 ±0.0096 | 0.9583 ±0.0079 | 0.0793 ±0.0110 | 0.0417 ±0.0079 | 1.0000 ±0.0000 |
| transformer | 0.8743 ±0.0162 | 0.8773 ±0.0160 | 0.8574 ±0.0205 | 0.8986 ±0.0238 | 0.1500 ±0.0252 | 0.1014 ±0.0238 | 1.0000 ±0.0000 |
| kronecker | 0.9394 ±0.0053 | 0.9407 ±0.0050 | 0.9206 ±0.0089 | 0.9618 ±0.0040 | 0.0831 ±0.0101 | 0.0382 ±0.0040 | 1.0000 ±0.0000 |
| hga **←** | 0.9420 ±0.0068 | 0.9430 ±0.0066 | 0.9272 ±0.0081 | 0.9594 ±0.0069 | 0.0754 ±0.0087 | 0.0406 ±0.0069 | 1.0000 ±0.0000 |

### `sentence_disjoint` (n=16036)

| Model | accuracy ↑ | f1 ↑ | precision ↑ | recall ↑ | far ↓ | frr ↓ | coverage ↑ |
|---|---|---|---|---|---|---|---|
| dense | 0.8978 ±0.0064 | 0.8992 ±0.0070 | 0.8872 ±0.0108 | 0.9118 ±0.0181 | 0.1162 ±0.0142 | 0.0882 ±0.0181 | 1.0000 ±0.0000 |
| transformer | 0.8210 ±0.0139 | 0.8196 ±0.0166 | 0.8262 ±0.0216 | 0.8147 ±0.0363 | 0.1726 ±0.0307 | 0.1853 ±0.0363 | 1.0000 ±0.0000 |
| kronecker **←** | 0.8991 ±0.0051 | 0.9008 ±0.0051 | 0.8854 ±0.0089 | 0.9170 ±0.0108 | 0.1188 ±0.0114 | 0.0830 ±0.0108 | 1.0000 ±0.0000 |
| hga | 0.8978 ±0.0066 | 0.8986 ±0.0065 | 0.8916 ±0.0107 | 0.9059 ±0.0120 | 0.1103 ±0.0127 | 0.0941 ±0.0120 | 1.0000 ±0.0000 |

### `seen_composition` (n=6036)

| Model | accuracy ↑ | f1 ↑ | precision ↑ | recall ↑ | far ↓ | frr ↓ | coverage ↑ |
|---|---|---|---|---|---|---|---|
| dense | 0.9395 ±0.0062 | 0.9406 ±0.0060 | 0.9237 ±0.0096 | 0.9583 ±0.0079 | 0.0793 ±0.0110 | 0.0417 ±0.0079 | 1.0000 ±0.0000 |
| transformer | 0.8743 ±0.0162 | 0.8773 ±0.0160 | 0.8574 ±0.0205 | 0.8986 ±0.0238 | 0.1500 ±0.0252 | 0.1014 ±0.0238 | 1.0000 ±0.0000 |
| kronecker | 0.9394 ±0.0053 | 0.9407 ±0.0050 | 0.9206 ±0.0089 | 0.9618 ±0.0040 | 0.0831 ±0.0101 | 0.0382 ±0.0040 | 1.0000 ±0.0000 |
| hga **←** | 0.9420 ±0.0068 | 0.9430 ±0.0066 | 0.9272 ±0.0081 | 0.9594 ±0.0069 | 0.0754 ±0.0087 | 0.0406 ±0.0069 | 1.0000 ±0.0000 |

## Kalibrasyon (dev'de fit, testte ölçüm — `all` dilimi)

| Model | ece ↓ | brier ↓ | nll ↓ | aurc ↓ |
|---|---|---|---|---|
| dense | 0.0125 ±0.0038 | 0.0763 ±0.0037 | 0.2594 ±0.0099 | 0.0279 ±0.0021 |
| transformer | 0.0119 ±0.0054 | 0.1252 ±0.0090 | 0.3938 ±0.0250 | 0.0703 ±0.0113 |
| kronecker | 0.0080 ±0.0034 | 0.0767 ±0.0041 | 0.2621 ±0.0125 | 0.0291 ±0.0029 |
| hga | 0.0061 ±0.0024 | 0.0761 ±0.0044 | 0.2560 ±0.0125 | 0.0277 ±0.0027 |

## HGA vs en güçlü rakip (eşleşmiş, F1)

| Dilim | Rakip | HGA | Rakip | Fark %95 CI | Hedges g | p | Karar |
|---|---|---:|---:|---|---:|---:|---|
| all | kronecker | 0.8986 | 0.9008 | [-0.0052, 0.0006] | -0.311 | 0.1623 | AYRIM YOK |
| entity_disjoint | kronecker | 0.8717 | 0.8760 | [-0.0082, -0.0003] | -0.443 | 0.0531 | KARARSIZ |
| relation_disjoint | dense | 0.6193 | 0.6861 | [-0.1177, -0.0170] | -0.552 | 0.0196 | AYRIŞMA |
| composition_disjoint | kronecker | 0.8773 | 0.8821 | [-0.0095, -0.0004] | -0.434 | 0.0544 | KARARSIZ |
| wording_disjoint | kronecker | 0.9430 | 0.9407 | [-0.0007, 0.0052] | 0.316 | 0.1565 | AYRIM YOK |
| sentence_disjoint | kronecker | 0.8986 | 0.9008 | [-0.0052, 0.0006] | -0.311 | 0.1623 | AYRIM YOK |
| seen_composition | kronecker | 0.9430 | 0.9407 | [-0.0007, 0.0052] | 0.316 | 0.1565 | AYRIM YOK |

## Kabul kapıları

| Kapı | Sonuç |
|---|---|
| all_seeds_completed | GEÇTİ |
| all_dimensions_reported | GEÇTİ |
| all_headline_metrics_present | GEÇTİ |
| calibration_metrics_present | GEÇTİ |
| flops_reported_for_all_models | GEÇTİ |
| flop_budget_within_tolerance | KALDI |
| analytic_flops_match_measured | GEÇTİ |
| parameter_budget_within_one_percent | GEÇTİ |
| sentence_disjoint_reported | GEÇTİ |
| seed_count_sufficient_for_inference | GEÇTİ |
| underlying_fairness_gates_pass | GEÇTİ |
| comparison_direction_reported | GEÇTİ |

## Bulgular

- Profil `smoke`, 20 tohum, 4 mimari, 7 disjoint dilim; gerçek veri TWT v1 (`66b13a898efa…`).
- FLOP oranı 11.34× (kapı 2.0×): mimariler eşit parametrede ama eşit işlem maliyetinde DEĞİL. Performans farkı kısmen hesap bütçesine atfedilebilir.
- `dense`: `all` diliminden belirgin düşüş → relation_disjoint (−0.213). Toplam skor bu zorluğu gizler; dilim tablosu bu yüzden var.
- `transformer`: `all` diliminden belirgin düşüş → entity_disjoint (−0.067), relation_disjoint (−0.191). Toplam skor bu zorluğu gizler; dilim tablosu bu yüzden var.
- `kronecker`: `all` diliminden belirgin düşüş → relation_disjoint (−0.246). Toplam skor bu zorluğu gizler; dilim tablosu bu yüzden var.
- `hga`: `all` diliminden belirgin düşüş → relation_disjoint (−0.279). Toplam skor bu zorluğu gizler; dilim tablosu bu yüzden var.
- HGA en güçlü rakibinin şu dilimlerde İSTATİSTİKSEL OLARAK GERİSİNDE KALDI: ['relation_disjoint (-0.0668 vs dense)']. Bu, eşit parametre bütçesinde HGA lehine bir üstünlük iddiasının VERİYLE ÇELİŞTİĞİ anlamına gelir ve gizlenmez.
- Şu dilimlerde HGA ile en güçlü rakip AYRIŞTIRILAMADI (fark CI'si sıfırı içeriyor): ['all', 'wording_disjoint', 'sentence_disjoint', 'seen_composition']. Bu bir üstünlük iddiasının reddidir, eksik ölçüm değil.
- F1 ortalamasına göre dilim kazanımları: kronecker×4, hga×2, dense×1

## Sınırlar

- Görev ikili bağımlılık-yayı doğrulamasıdır; dil modelleme, metin üretimi veya perplexity iddiası İÇERMEZ.
- FLOP sayıları analitik kapalı formdur; gerçek donanım verimliliği (bellek bant genişliği, çekirdek füzyonu) hesaba katılmaz.
- Gömme parametreleri modeller arasında paylaşılır; 'body' oranı bu yüzden toplam orandan daha anlamlı bir adillik göstergesidir.
- Kalibrasyon sıcaklığı yalnız dev'de fit edilir ve argmax'ı değiştirmez; bu yüzden accuracy/F1 kalibrasyondan etkilenmez.
- Tohum sayısı 20; 20'nin altındaki her sonuç betimleyicidir.
