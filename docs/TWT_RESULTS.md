# TWT Gerçek Sonuç Tablosu (P0-3)

- Protokol: `twt_real_results_v1` v2 (imza `e0d2593e5db4`)
- Görev: binary dependency arc validation — **dil modelleme değildir**
- Veri: TWT v1 `66b13a898efa8899…`, train/dev/test aday sayısı 127602/17030/16036
- Profil / tohumlar: `smoke` / `[1, 2, 3, 4, 5]`

## Maliyet bütçesi

| Model | Parametre | Gövde | Bayt | İleri FLOP/örnek | Eğitim (s) | Çıkarım (s) |
|---|---:|---:|---:|---:|---:|---:|
| dense | 291,808 | 10,496 | 1,167,232 | 10.39K | 0.103 | 0.008 |
| transformer | 291,816 | 10,504 | 1,167,264 | 60.70K | 0.391 | 0.115 |
| kronecker | 291,514 | 10,202 | 1,166,056 | 13.64K | 0.169 | 0.116 |
| hga | 291,827 | 10,515 | 1,167,308 | 117.85K | 0.644 | 0.215 |

FLOP max/min oranı: **11.345×** (eşik 2.0×) → eşik DIŞINDA

> Parametre eşitliği FLOP eşitliğini GARANTİ ETMEZ. Bu oran kapıyı geçmezse, sonuç farkı kısmen işlem bütçesi farkına atfedilebilir ve öyle okunmalıdır.

## FLOP-eşli kontrol rejimi

> Kontrol kolu: HGA aynen, baseline gövdeleri HGA'nın MAC bütçesine ölçekli. Parametre paritesi BİLEREK bırakıldı ve aşağıda raporlandı; iki rejim birlikte okunmalıdır.

- MAC max/min oranı: **1.041×** (tolerans 1.05×) → GEÇTİ
- Parametre oranı (bilerek serbest): 1.372×

| Model | MAC/örnek | f1@all |
|---|---:|---:|
| dense | 117,796 | 0.9128 ±0.0046 |
| transformer | 117,728 | 0.8309 ±0.0185 |
| kronecker | 122,600 | 0.9108 ±0.0036 |
| hga | 117,848 | 0.8975 ±0.0107 |

HGA vs `dense` (f1@all, eşleşmiş): fark -0.0153 — YETERSİZ GÜÇ: n=5 ile p<0.05 ulaşılamaz (en küçük p=0.06250). Fark ortalaması -0.0153122, %95 CI [-0.0204477, -0.0106809] — sıfırı içermiyor. Anlamlılık İDDİA EDİLMEZ; sonuç betimleyicidir.

## Ana sonuç tablosu (test, ortalama ± std)

### `all` (n=16036)

| Model | accuracy ↑ | f1 ↑ | precision ↑ | recall ↑ | far ↓ | frr ↓ | coverage ↑ |
|---|---|---|---|---|---|---|---|
| dense | 0.8938 ±0.0068 | 0.8943 ±0.0079 | 0.8893 ±0.0027 | 0.8995 ±0.0178 | 0.1120 ±0.0049 | 0.1005 ±0.0178 | 1.0000 ±0.0000 |
| transformer | 0.8207 ±0.0186 | 0.8206 ±0.0167 | 0.8226 ±0.0320 | 0.8198 ±0.0267 | 0.1784 ±0.0419 | 0.1802 ±0.0267 | 1.0000 ±0.0000 |
| kronecker **←** | 0.8977 ±0.0051 | 0.8992 ±0.0051 | 0.8859 ±0.0059 | 0.9130 ±0.0072 | 0.1176 ±0.0067 | 0.0870 ±0.0072 | 1.0000 ±0.0000 |
| hga | 0.8968 ±0.0112 | 0.8975 ±0.0107 | 0.8919 ±0.0183 | 0.9035 ±0.0163 | 0.1099 ±0.0215 | 0.0965 ±0.0163 | 1.0000 ±0.0000 |

### `entity_disjoint` (n=2704)

| Model | accuracy ↑ | f1 ↑ | precision ↑ | recall ↑ | far ↓ | frr ↓ | coverage ↑ |
|---|---|---|---|---|---|---|---|
| dense | 0.8693 ±0.0180 | 0.8686 ±0.0211 | 0.8713 ±0.0069 | 0.8666 ±0.0378 | 0.1280 ±0.0077 | 0.1334 ±0.0378 | 1.0000 ±0.0000 |
| transformer | 0.7928 ±0.0322 | 0.7864 ±0.0564 | 0.8059 ±0.0477 | 0.7854 ±0.1448 | 0.1997 ±0.0948 | 0.2146 ±0.1448 | 1.0000 ±0.0000 |
| kronecker **←** | 0.8807 ±0.0139 | 0.8816 ±0.0140 | 0.8747 ±0.0134 | 0.8888 ±0.0157 | 0.1274 ±0.0137 | 0.1112 ±0.0157 | 1.0000 ±0.0000 |
| hga | 0.8769 ±0.0104 | 0.8775 ±0.0093 | 0.8752 ±0.0320 | 0.8817 ±0.0331 | 0.1278 ±0.0426 | 0.1183 ±0.0331 | 1.0000 ±0.0000 |

### `relation_disjoint` (n=168)

| Model | accuracy ↑ | f1 ↑ | precision ↑ | recall ↑ | far ↓ | frr ↓ | coverage ↑ |
|---|---|---|---|---|---|---|---|
| dense | 0.6929 ±0.1030 | 0.6008 ±0.1818 | 0.7918 ±0.0356 | 0.5262 ±0.2786 | 0.1405 ±0.0764 | 0.4738 ±0.2786 | 1.0000 ±0.0000 |
| transformer **←** | 0.6750 ±0.0497 | 0.6277 ±0.1356 | 0.7213 ±0.0535 | 0.6095 ±0.2634 | 0.2595 ±0.1663 | 0.3905 ±0.2634 | 1.0000 ±0.0000 |
| kronecker | 0.6857 ±0.0993 | 0.5929 ±0.1625 | 0.7907 ±0.0528 | 0.5095 ±0.2601 | 0.1381 ±0.0727 | 0.4905 ±0.2601 | 1.0000 ±0.0000 |
| hga | 0.6690 ±0.1244 | 0.5543 ±0.2201 | 0.7504 ±0.0911 | 0.4738 ±0.2866 | 0.1357 ±0.0442 | 0.5262 ±0.2866 | 1.0000 ±0.0000 |

### `composition_disjoint` (n=7086)

| Model | accuracy ↑ | f1 ↑ | precision ↑ | recall ↑ | far ↓ | frr ↓ | coverage ↑ |
|---|---|---|---|---|---|---|---|
| dense | 0.8720 ±0.0109 | 0.8727 ±0.0125 | 0.8673 ±0.0081 | 0.8785 ±0.0258 | 0.1346 ±0.0108 | 0.1215 ±0.0258 | 1.0000 ±0.0000 |
| transformer | 0.7938 ±0.0203 | 0.7905 ±0.0193 | 0.8041 ±0.0259 | 0.7776 ±0.0163 | 0.1900 ±0.0287 | 0.2224 ±0.0163 | 1.0000 ±0.0000 |
| kronecker **←** | 0.8779 ±0.0051 | 0.8800 ±0.0052 | 0.8650 ±0.0080 | 0.8957 ±0.0111 | 0.1399 ±0.0105 | 0.1043 ±0.0111 | 1.0000 ±0.0000 |
| hga | 0.8741 ±0.0143 | 0.8749 ±0.0142 | 0.8700 ±0.0195 | 0.8802 ±0.0222 | 0.1319 ±0.0228 | 0.1198 ±0.0222 | 1.0000 ±0.0000 |

### `wording_disjoint` (n=6036)

| Model | accuracy ↑ | f1 ↑ | precision ↑ | recall ↑ | far ↓ | frr ↓ | coverage ↑ |
|---|---|---|---|---|---|---|---|
| dense | 0.9376 ±0.0047 | 0.9385 ±0.0049 | 0.9251 ±0.0030 | 0.9524 ±0.0098 | 0.0771 ±0.0035 | 0.0476 ±0.0098 | 1.0000 ±0.0000 |
| transformer | 0.8705 ±0.0280 | 0.8733 ±0.0273 | 0.8551 ±0.0314 | 0.8928 ±0.0294 | 0.1518 ±0.0350 | 0.1072 ±0.0294 | 1.0000 ±0.0000 |
| kronecker | 0.9364 ±0.0046 | 0.9378 ±0.0045 | 0.9175 ±0.0050 | 0.9591 ±0.0047 | 0.0863 ±0.0055 | 0.0409 ±0.0047 | 1.0000 ±0.0000 |
| hga **←** | 0.9406 ±0.0098 | 0.9415 ±0.0095 | 0.9274 ±0.0122 | 0.9561 ±0.0087 | 0.0750 ±0.0133 | 0.0439 ±0.0087 | 1.0000 ±0.0000 |

### `sentence_disjoint` (n=16036)

| Model | accuracy ↑ | f1 ↑ | precision ↑ | recall ↑ | far ↓ | frr ↓ | coverage ↑ |
|---|---|---|---|---|---|---|---|
| dense | 0.8938 ±0.0068 | 0.8943 ±0.0079 | 0.8893 ±0.0027 | 0.8995 ±0.0178 | 0.1120 ±0.0049 | 0.1005 ±0.0178 | 1.0000 ±0.0000 |
| transformer | 0.8207 ±0.0186 | 0.8206 ±0.0167 | 0.8226 ±0.0320 | 0.8198 ±0.0267 | 0.1784 ±0.0419 | 0.1802 ±0.0267 | 1.0000 ±0.0000 |
| kronecker **←** | 0.8977 ±0.0051 | 0.8992 ±0.0051 | 0.8859 ±0.0059 | 0.9130 ±0.0072 | 0.1176 ±0.0067 | 0.0870 ±0.0072 | 1.0000 ±0.0000 |
| hga | 0.8968 ±0.0112 | 0.8975 ±0.0107 | 0.8919 ±0.0183 | 0.9035 ±0.0163 | 0.1099 ±0.0215 | 0.0965 ±0.0163 | 1.0000 ±0.0000 |

### `seen_composition` (n=6036)

| Model | accuracy ↑ | f1 ↑ | precision ↑ | recall ↑ | far ↓ | frr ↓ | coverage ↑ |
|---|---|---|---|---|---|---|---|
| dense | 0.9376 ±0.0047 | 0.9385 ±0.0049 | 0.9251 ±0.0030 | 0.9524 ±0.0098 | 0.0771 ±0.0035 | 0.0476 ±0.0098 | 1.0000 ±0.0000 |
| transformer | 0.8705 ±0.0280 | 0.8733 ±0.0273 | 0.8551 ±0.0314 | 0.8928 ±0.0294 | 0.1518 ±0.0350 | 0.1072 ±0.0294 | 1.0000 ±0.0000 |
| kronecker | 0.9364 ±0.0046 | 0.9378 ±0.0045 | 0.9175 ±0.0050 | 0.9591 ±0.0047 | 0.0863 ±0.0055 | 0.0409 ±0.0047 | 1.0000 ±0.0000 |
| hga **←** | 0.9406 ±0.0098 | 0.9415 ±0.0095 | 0.9274 ±0.0122 | 0.9561 ±0.0087 | 0.0750 ±0.0133 | 0.0439 ±0.0087 | 1.0000 ±0.0000 |

## Kalibrasyon (dev'de fit, testte ölçüm — `all` dilimi)

| Model | ece ↓ | brier ↓ | nll ↓ | aurc ↓ |
|---|---|---|---|---|
| dense | 0.0096 ±0.0035 | 0.0781 ±0.0033 | 0.2636 ±0.0091 | 0.0286 ±0.0023 |
| transformer | 0.0140 ±0.0072 | 0.1237 ±0.0105 | 0.3882 ±0.0277 | 0.0672 ±0.0111 |
| kronecker | 0.0077 ±0.0031 | 0.0785 ±0.0047 | 0.2677 ±0.0158 | 0.0301 ±0.0046 |
| hga | 0.0070 ±0.0037 | 0.0771 ±0.0071 | 0.2581 ±0.0193 | 0.0274 ±0.0041 |

## HGA vs en güçlü rakip (eşleşmiş, F1)

| Dilim | Rakip | HGA | Rakip | Fark %95 CI | Hedges g | p | Karar |
|---|---|---:|---:|---|---:|---:|---|
| all | kronecker | 0.8975 | 0.8992 | [-0.0090, 0.0040] | -0.160 | 0.8750 | YETERSİZ GÜÇ |
| entity_disjoint | kronecker | 0.8775 | 0.8816 | [-0.0103, 0.0024] | -0.407 | 0.3750 | YETERSİZ GÜÇ |
| relation_disjoint | transformer | 0.5543 | 0.6277 | [-0.3029, 0.1614] | -0.190 | 0.6250 | YETERSİZ GÜÇ |
| composition_disjoint | kronecker | 0.8749 | 0.8800 | [-0.0157, 0.0025] | -0.335 | 0.5625 | YETERSİZ GÜÇ |
| wording_disjoint | dense | 0.9415 | 0.9385 | [-0.0018, 0.0084] | 0.366 | 0.3125 | YETERSİZ GÜÇ |
| sentence_disjoint | kronecker | 0.8975 | 0.8992 | [-0.0090, 0.0040] | -0.160 | 0.8750 | YETERSİZ GÜÇ |
| seen_composition | dense | 0.9415 | 0.9385 | [-0.0018, 0.0084] | 0.366 | 0.3125 | YETERSİZ GÜÇ |

## Kabul kapıları

| Kapı | Sonuç |
|---|---|
| all_seeds_completed | GEÇTİ |
| all_dimensions_reported | GEÇTİ |
| all_headline_metrics_present | GEÇTİ |
| calibration_metrics_present | GEÇTİ |
| flops_reported_for_all_models | GEÇTİ |
| flop_budget_controlled | GEÇTİ |
| flop_matched_control_reported | GEÇTİ |
| analytic_flops_match_measured | GEÇTİ |
| parameter_budget_within_one_percent | GEÇTİ |
| sentence_disjoint_reported | GEÇTİ |
| seed_count_sufficient_for_inference | KALDI |
| underlying_fairness_gates_pass | GEÇTİ |
| comparison_direction_reported | GEÇTİ |

## Bulgular

- Profil `smoke`, 5 tohum, 4 mimari, 7 disjoint dilim; gerçek veri TWT v1 (`66b13a898efa…`).
- Tohum sayısı 5 < 20. Bu tablo ENGINEERING kanıtıdır; çekirdek bilimsel iddia için yeterli değildir. Aşağıdaki p-değerleri ve güven aralıkları bu sınırla okunmalıdır.
- FLOP oranı 11.34× (eşik 2.0×): mimariler eşit parametrede ama eşit işlem maliyetinde DEĞİL. Bu yüzden FLOP-eşli kontrol rejimi koşuldu (aşağıda); iki rejim birlikte okunmalıdır.
- FLOP-eşli kontrol rejimi (MAC oranı 1.041×, parametre oranı 1.37× — bilerek serbest): HGA f1@all = 0.8975.
- FLOP-eşli rejimde HGA vs dense: fark -0.0153 (YETERSİZ GÜÇ: n=5 ile p<0.05 ulaşılamaz (en küçük p=0.06250). Fark ortalaması -0.0153122, %95 CI [-0.0204477, -0.0106809] — sıfırı içermiyor. Anlamlılık İDDİA EDİLMEZ; sonuç betimleyicidir.). Baseline'lar HGA'nın işlem bütçesine ölçeklenince de tablo değişmiyorsa fark hesap bütçesiyle açıklanamaz; değişiyorsa bütçe etkisi budur.
- `dense`: `all` diliminden belirgin düşüş → relation_disjoint (−0.294). Toplam skor bu zorluğu gizler; dilim tablosu bu yüzden var.
- `transformer`: `all` diliminden belirgin düşüş → relation_disjoint (−0.193). Toplam skor bu zorluğu gizler; dilim tablosu bu yüzden var.
- `kronecker`: `all` diliminden belirgin düşüş → relation_disjoint (−0.306). Toplam skor bu zorluğu gizler; dilim tablosu bu yüzden var.
- `hga`: `all` diliminden belirgin düşüş → relation_disjoint (−0.343). Toplam skor bu zorluğu gizler; dilim tablosu bu yüzden var.
- Hiçbir dilimde kesin ayrışma yok; tüm karşılaştırmalar kararsız ya da yetersiz güçte.
- F1 ortalamasına göre dilim kazanımları: kronecker×4, hga×2, transformer×1

## Sınırlar

- Görev ikili bağımlılık-yayı doğrulamasıdır; dil modelleme, metin üretimi veya perplexity iddiası İÇERMEZ.
- FLOP sayıları analitik kapalı formdur; gerçek donanım verimliliği (bellek bant genişliği, çekirdek füzyonu) hesaba katılmaz.
- Gömme parametreleri modeller arasında paylaşılır; 'body' oranı bu yüzden toplam orandan daha anlamlı bir adillik göstergesidir.
- Kalibrasyon sıcaklığı yalnız dev'de fit edilir ve argmax'ı değiştirmez; bu yüzden accuracy/F1 kalibrasyondan etkilenmez.
- Tohum sayısı 5; 20'nin altındaki her sonuç betimleyicidir.
