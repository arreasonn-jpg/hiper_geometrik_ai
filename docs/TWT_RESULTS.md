# TWT Gerçek Sonuç Tablosu (P0-3)

- Protokol: `twt_real_results_v1` v1 (imza `24de8912e159`)
- Görev: binary dependency arc validation — **dil modelleme değildir**
- Veri: TWT v1 `66b13a898efa8899…`, train/dev/test aday sayısı 127602/17030/16036
- Profil / tohumlar: `full` / `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]`

## Maliyet bütçesi

| Model | Parametre | Gövde | Bayt | İleri FLOP/örnek | Eğitim (s) | Çıkarım (s) |
|---|---:|---:|---:|---:|---:|---:|
| dense | 291,808 | 10,496 | 1,167,232 | 10.39K | 0.856 | 0.024 |
| transformer | 291,816 | 10,504 | 1,167,264 | 60.70K | 4.788 | 0.184 |
| kronecker | 291,514 | 10,202 | 1,166,056 | 13.64K | 1.835 | 0.063 |
| hga | 291,827 | 10,515 | 1,167,308 | 117.85K | 8.297 | 0.330 |

FLOP max/min oranı: **11.345×** (kapı 2.0×) → KALDI

> Parametre eşitliği FLOP eşitliğini GARANTİ ETMEZ. Bu oran kapıyı geçmezse, sonuç farkı kısmen işlem bütçesi farkına atfedilebilir ve öyle okunmalıdır.

## Ana sonuç tablosu (test, ortalama ± std)

### `all` (n=16036)

| Model | accuracy ↑ | f1 ↑ | precision ↑ | recall ↑ | far ↓ | frr ↓ | coverage ↑ |
|---|---|---|---|---|---|---|---|
| dense **←** | 0.9387 ±0.0029 | 0.9394 ±0.0032 | 0.9299 ±0.0052 | 0.9491 ±0.0092 | 0.0716 ±0.0063 | 0.0509 ±0.0092 | 1.0000 ±0.0000 |
| transformer | 0.9237 ±0.0058 | 0.9242 ±0.0067 | 0.9179 ±0.0086 | 0.9310 ±0.0188 | 0.0835 ±0.0108 | 0.0690 ±0.0188 | 1.0000 ±0.0000 |
| kronecker | 0.9386 ±0.0025 | 0.9392 ±0.0027 | 0.9301 ±0.0046 | 0.9486 ±0.0083 | 0.0714 ±0.0056 | 0.0514 ±0.0083 | 1.0000 ±0.0000 |
| hga | 0.9350 ±0.0023 | 0.9356 ±0.0026 | 0.9275 ±0.0063 | 0.9440 ±0.0103 | 0.0739 ±0.0076 | 0.0560 ±0.0103 | 1.0000 ±0.0000 |

### `entity_disjoint` (n=2704)

| Model | accuracy ↑ | f1 ↑ | precision ↑ | recall ↑ | far ↓ | frr ↓ | coverage ↑ |
|---|---|---|---|---|---|---|---|
| dense **←** | 0.9135 ±0.0088 | 0.9141 ±0.0097 | 0.9078 ±0.0128 | 0.9210 ±0.0241 | 0.0940 ±0.0162 | 0.0790 ±0.0241 | 1.0000 ±0.0000 |
| transformer | 0.8899 ±0.0244 | 0.8875 ±0.0316 | 0.8995 ±0.0152 | 0.8790 ±0.0639 | 0.0992 ±0.0210 | 0.1210 ±0.0639 | 1.0000 ±0.0000 |
| kronecker | 0.9138 ±0.0066 | 0.9140 ±0.0076 | 0.9113 ±0.0090 | 0.9171 ±0.0208 | 0.0896 ±0.0119 | 0.0829 ±0.0208 | 1.0000 ±0.0000 |
| hga | 0.9115 ±0.0065 | 0.9117 ±0.0067 | 0.9093 ±0.0102 | 0.9143 ±0.0140 | 0.0914 ±0.0119 | 0.0857 ±0.0140 | 1.0000 ±0.0000 |

### `relation_disjoint` (n=168)

| Model | accuracy ↑ | f1 ↑ | precision ↑ | recall ↑ | far ↓ | frr ↓ | coverage ↑ |
|---|---|---|---|---|---|---|---|
| dense | 0.7101 ±0.1414 | 0.5882 ±0.2588 | 0.8229 ±0.0924 | 0.5065 ±0.2970 | 0.0863 ±0.0334 | 0.4935 ±0.2970 | 1.0000 ±0.0000 |
| transformer **←** | 0.7432 ±0.0950 | 0.6727 ±0.1654 | 0.8529 ±0.0518 | 0.5917 ±0.2302 | 0.1054 ±0.0591 | 0.4083 ±0.2302 | 1.0000 ±0.0000 |
| kronecker | 0.7140 ±0.1263 | 0.6007 ±0.2351 | 0.8427 ±0.0856 | 0.5149 ±0.2792 | 0.0869 ±0.0457 | 0.4851 ±0.2792 | 1.0000 ±0.0000 |
| hga | 0.6839 ±0.1351 | 0.5422 ±0.2608 | 0.8048 ±0.1334 | 0.4548 ±0.2850 | 0.0869 ±0.0514 | 0.5452 ±0.2850 | 1.0000 ±0.0000 |

### `composition_disjoint` (n=7086)

| Model | accuracy ↑ | f1 ↑ | precision ↑ | recall ↑ | far ↓ | frr ↓ | coverage ↑ |
|---|---|---|---|---|---|---|---|
| dense **←** | 0.9301 ±0.0021 | 0.9311 ±0.0023 | 0.9171 ±0.0048 | 0.9457 ±0.0081 | 0.0856 ±0.0061 | 0.0543 ±0.0081 | 1.0000 ±0.0000 |
| transformer | 0.9115 ±0.0044 | 0.9126 ±0.0051 | 0.9019 ±0.0098 | 0.9238 ±0.0163 | 0.1007 ±0.0125 | 0.0762 ±0.0163 | 1.0000 ±0.0000 |
| kronecker | 0.9298 ±0.0026 | 0.9308 ±0.0030 | 0.9165 ±0.0042 | 0.9457 ±0.0094 | 0.0862 ±0.0055 | 0.0543 ±0.0094 | 1.0000 ±0.0000 |
| hga | 0.9252 ±0.0024 | 0.9263 ±0.0028 | 0.9134 ±0.0068 | 0.9396 ±0.0113 | 0.0892 ±0.0086 | 0.0604 ±0.0113 | 1.0000 ±0.0000 |

### `wording_disjoint` (n=6036)

| Model | accuracy ↑ | f1 ↑ | precision ↑ | recall ↑ | far ↓ | frr ↓ | coverage ↑ |
|---|---|---|---|---|---|---|---|
| dense **←** | 0.9685 ±0.0020 | 0.9689 ±0.0020 | 0.9564 ±0.0043 | 0.9817 ±0.0035 | 0.0447 ±0.0047 | 0.0183 ±0.0035 | 1.0000 ±0.0000 |
| transformer | 0.9601 ±0.0027 | 0.9607 ±0.0027 | 0.9463 ±0.0052 | 0.9756 ±0.0064 | 0.0554 ±0.0059 | 0.0244 ±0.0064 | 1.0000 ±0.0000 |
| kronecker | 0.9682 ±0.0021 | 0.9687 ±0.0020 | 0.9559 ±0.0047 | 0.9818 ±0.0029 | 0.0453 ±0.0052 | 0.0182 ±0.0029 | 1.0000 ±0.0000 |
| hga | 0.9661 ±0.0019 | 0.9666 ±0.0018 | 0.9535 ±0.0054 | 0.9801 ±0.0042 | 0.0478 ±0.0059 | 0.0199 ±0.0042 | 1.0000 ±0.0000 |

### `sentence_disjoint` (n=16036)

| Model | accuracy ↑ | f1 ↑ | precision ↑ | recall ↑ | far ↓ | frr ↓ | coverage ↑ |
|---|---|---|---|---|---|---|---|
| dense **←** | 0.9387 ±0.0029 | 0.9394 ±0.0032 | 0.9299 ±0.0052 | 0.9491 ±0.0092 | 0.0716 ±0.0063 | 0.0509 ±0.0092 | 1.0000 ±0.0000 |
| transformer | 0.9237 ±0.0058 | 0.9242 ±0.0067 | 0.9179 ±0.0086 | 0.9310 ±0.0188 | 0.0835 ±0.0108 | 0.0690 ±0.0188 | 1.0000 ±0.0000 |
| kronecker | 0.9386 ±0.0025 | 0.9392 ±0.0027 | 0.9301 ±0.0046 | 0.9486 ±0.0083 | 0.0714 ±0.0056 | 0.0514 ±0.0083 | 1.0000 ±0.0000 |
| hga | 0.9350 ±0.0023 | 0.9356 ±0.0026 | 0.9275 ±0.0063 | 0.9440 ±0.0103 | 0.0739 ±0.0076 | 0.0560 ±0.0103 | 1.0000 ±0.0000 |

### `seen_composition` (n=6036)

| Model | accuracy ↑ | f1 ↑ | precision ↑ | recall ↑ | far ↓ | frr ↓ | coverage ↑ |
|---|---|---|---|---|---|---|---|
| dense **←** | 0.9685 ±0.0020 | 0.9689 ±0.0020 | 0.9564 ±0.0043 | 0.9817 ±0.0035 | 0.0447 ±0.0047 | 0.0183 ±0.0035 | 1.0000 ±0.0000 |
| transformer | 0.9601 ±0.0027 | 0.9607 ±0.0027 | 0.9463 ±0.0052 | 0.9756 ±0.0064 | 0.0554 ±0.0059 | 0.0244 ±0.0064 | 1.0000 ±0.0000 |
| kronecker | 0.9682 ±0.0021 | 0.9687 ±0.0020 | 0.9559 ±0.0047 | 0.9818 ±0.0029 | 0.0453 ±0.0052 | 0.0182 ±0.0029 | 1.0000 ±0.0000 |
| hga | 0.9661 ±0.0019 | 0.9666 ±0.0018 | 0.9535 ±0.0054 | 0.9801 ±0.0042 | 0.0478 ±0.0059 | 0.0199 ±0.0042 | 1.0000 ±0.0000 |

## Kalibrasyon (dev'de fit, testte ölçüm — `all` dilimi)

| Model | ece ↓ | brier ↓ | nll ↓ | aurc ↓ |
|---|---|---|---|---|
| dense | 0.0042 ±0.0019 | 0.0458 ±0.0021 | 0.1554 ±0.0094 | 0.0086 ±0.0013 |
| transformer | 0.0050 ±0.0019 | 0.0561 ±0.0039 | 0.1873 ±0.0115 | 0.0129 ±0.0016 |
| kronecker | 0.0053 ±0.0020 | 0.0460 ±0.0018 | 0.1563 ±0.0075 | 0.0087 ±0.0011 |
| hga | 0.0053 ±0.0025 | 0.0485 ±0.0021 | 0.1662 ±0.0103 | 0.0102 ±0.0018 |

## HGA vs en güçlü rakip (eşleşmiş, F1)

| Dilim | Rakip | HGA | Rakip | Fark %95 CI | Hedges g | p | Karar |
|---|---|---:|---:|---|---:|---:|---|
| all | dense | 0.9356 | 0.9394 | [-0.0046, -0.0028] | -1.753 | 0.0000 | AYRIŞMA |
| entity_disjoint | dense | 0.9117 | 0.9141 | [-0.0054, 0.0010] | -0.302 | 0.1760 | AYRIM YOK |
| relation_disjoint | transformer | 0.5422 | 0.6727 | [-0.2501, -0.0209] | -0.470 | 0.0400 | AYRIŞMA |
| composition_disjoint | dense | 0.9263 | 0.9311 | [-0.0061, -0.0037] | -1.639 | 0.0000 | AYRIŞMA |
| wording_disjoint | dense | 0.9666 | 0.9689 | [-0.0030, -0.0015] | -1.280 | 0.0000 | AYRIŞMA |
| sentence_disjoint | dense | 0.9356 | 0.9394 | [-0.0046, -0.0028] | -1.753 | 0.0000 | AYRIŞMA |
| seen_composition | dense | 0.9666 | 0.9689 | [-0.0030, -0.0015] | -1.280 | 0.0000 | AYRIŞMA |

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

- Profil `full`, 20 tohum, 4 mimari, 7 disjoint dilim; gerçek veri TWT v1 (`66b13a898efa…`).
- FLOP oranı 11.34× (kapı 2.0×): mimariler eşit parametrede ama eşit işlem maliyetinde DEĞİL. Performans farkı kısmen hesap bütçesine atfedilebilir.
- `dense`: `all` diliminden belirgin düşüş → relation_disjoint (−0.351). Toplam skor bu zorluğu gizler; dilim tablosu bu yüzden var.
- `transformer`: `all` diliminden belirgin düşüş → relation_disjoint (−0.251). Toplam skor bu zorluğu gizler; dilim tablosu bu yüzden var.
- `kronecker`: `all` diliminden belirgin düşüş → relation_disjoint (−0.339). Toplam skor bu zorluğu gizler; dilim tablosu bu yüzden var.
- `hga`: `all` diliminden belirgin düşüş → relation_disjoint (−0.393). Toplam skor bu zorluğu gizler; dilim tablosu bu yüzden var.
- HGA en güçlü rakibinin şu dilimlerde İSTATİSTİKSEL OLARAK GERİSİNDE KALDI: ['all (-0.0038 vs dense)', 'relation_disjoint (-0.1305 vs transformer)', 'composition_disjoint (-0.0049 vs dense)', 'wording_disjoint (-0.0023 vs dense)', 'sentence_disjoint (-0.0038 vs dense)', 'seen_composition (-0.0023 vs dense)']. Bu, eşit parametre bütçesinde HGA lehine bir üstünlük iddiasının VERİYLE ÇELİŞTİĞİ anlamına gelir ve gizlenmez.
- Şu dilimlerde HGA ile en güçlü rakip AYRIŞTIRILAMADI (fark CI'si sıfırı içeriyor): ['entity_disjoint']. Bu bir üstünlük iddiasının reddidir, eksik ölçüm değil.
- F1 ortalamasına göre dilim kazanımları: dense×6, transformer×1

## Sınırlar

- Görev ikili bağımlılık-yayı doğrulamasıdır; dil modelleme, metin üretimi veya perplexity iddiası İÇERMEZ.
- FLOP sayıları analitik kapalı formdur; gerçek donanım verimliliği (bellek bant genişliği, çekirdek füzyonu) hesaba katılmaz.
- Gömme parametreleri modeller arasında paylaşılır; 'body' oranı bu yüzden toplam orandan daha anlamlı bir adillik göstergesidir.
- Kalibrasyon sıcaklığı yalnız dev'de fit edilir ve argmax'ı değiştirmez; bu yüzden accuracy/F1 kalibrasyondan etkilenmez.
- Tohum sayısı 20; 20'nin altındaki her sonuç betimleyicidir.
