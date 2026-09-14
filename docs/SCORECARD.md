# HGA RESEARCH SCORECARD

Üretim zamanı (UTC): `2026-09-14T07:44:03+00:00`

> Bu skorlar benchmark kabul kapılarından **otomatik** hesaplanır. Elle yazılmış bir puan bu tabloya giremez; kanıtı olmayan bölüm `n/a` döner.

| Bölüm | Skor | Kanıt |
|---|---:|---|
| architecture | 10.0 | operator_baseline_family_v1@81610cdf041b |
| memory | 10.0 | reliable_reasoning_depth_v1@526c6a88dea7, hierarchical_memory_v1@72b18bd90ab7 |
| verification | 5.0 | priority_causal_chain_ablation_v1@f86dfc8878cb |
| generalization | 8.3 | compositional_generalization_v2_raw_text@f59f990d1591 |
| reasoning | 6.0 | reliable_reasoning_depth_v1@526c6a88dea7 |
| turkish_nlp | 8.9 | turkish_semantic_extraction_v1@525654cb978c, twt_real_results_v1@66b13a898efa |
| language_modeling | n/a | — |
| reproducibility | n/a | — |
| scientific_evidence | 7.6 | priority_causal_chain_ablation_v1@f86dfc8878cb, operator_baseline_family_v1@81610cdf041b, reliable_reasoning_depth_v1@526c6a88dea7, hga_signature_benchmark_v1@291da16d8787, turkish_semantic_extraction_v1@525654cb978c, compositional_generalization_v2_raw_text@f59f990d1591 |
| engineering | n/a | — |
| **Overall Research Readiness** | **8.0** | 7/10 bölüm |

## HGA Capability Vector

| Sembol | Ad | Tür | Değer | Birim | Kanıt |
|---|---|---|---:|---|---|
| `P` | Physical Parameters | ölçülen | 4.05249e+07 | parametre | capacity_framework |
| `C_I^UB` | Interaction Upper Bound | ÜST SINIR | 1.84467e+19 | operatör girdisi | capacity_framework |
| `C_M^UB` | Memory Address Upper Bound | ÜST SINIR | 2.81475e+62 | adres | capacity_framework |
| `C_E` | Measured Experience Capacity | ölçülen | 960 | deneyim | capacity_framework |
| `C_V` | Measured Verified Capacity | ölçülen | 960 | deneyim | capacity_framework |
| `C_G` | Measured Generalization Capacity | ölçülen | 0.857143 | oran | compositional_generalization_v2_raw_text@f59f990d1591 |
| `C_R` | Measured Reliable Reasoning Depth | ölçülen | 4 | hop | reliable_reasoning_depth_v1@526c6a88dea7 |
| `C_RD` | Distractor-Resistant Reasoning Depth | ölçülen | 4 | hop @ 64 dolgu | reliable_reasoning_depth_v1@526c6a88dea7 |
| `C_MR` | Measured Memory Recall | ölçülen | 1 | oran | hierarchical_memory_v1@72b18bd90ab7 |
| `C_H` | Hallucination Resistance | ölçülen | 0.772461 | oran | hga_signature_benchmark_v1@291da16d8787 |
| `C_U` | Uncertainty Calibration | ölçülen | n/a | 1−ECE | — |

## Uyarılar

- `language_modeling` bölümü için kanıt yok; skor üretilmedi.
- `reproducibility` bölümü için kanıt yok; skor üretilmedi.
- `engineering` bölümü için kanıt yok; skor üretilmedi.
- Genel skor yalnız 7/10 bölüm üzerinden hesaplandı; ['language_modeling', 'reproducibility', 'engineering'] kanıtsız. Bu ortalama eksik kanıtı gizlemez, onu işaretler.

## Bölüm gerekçeleri

- **architecture**: Kronecker/rank-1/low-rank/full-dense kolları eşit parametre ve eşit FLOP rejimlerinde; skor kabul kapılarının geçme oranıdır.
- **memory**: Hiyerarşik bellek (hot/warm/cold/archive) recall, gecikme, tahliye ve çökme kurtarma kapıları; kanıt yoksa çıkarım derinliği ızgarasındaki en düşük geri çağırma oranına düşülür.
- **verification**: Priority(E) ağırlıklarının skor→sıralama→seçim→downstream zincirini taşıyıp taşımadığı ölçülür.
- **generalization**: Şema ve ontoloji önceden verilmeden, ham metinden keşif + kompozisyon başarısı.
- **reasoning**: Güvenilir çıkarım derinliği ve dolgu baskısı altındaki dayanıklılık.
- **turkish_nlp**: Elle etiketli altın sette varlık/ilişki/özellik/zaman/olumsuzluk çıkarımı ve gerçek Türkçe treebank (TWT) üzerinde arc doğrulama.
- **language_modeling**: Gerçek Türkçe korpusta perplexity ve üretim kalitesi. Kanıt yoksa skor üretilmez — 'tiny smoke' bir dil modeli iddiası değildir.
- **reproducibility**: Manifest, veri/konfig hash'i ve çoklu tohum tamamlanması.
- **scientific_evidence**: Tüm protokollerin kabul kapılarının birleşik geçme oranı. Bu skor yalnızca ölçüm iyileşerek yükselir.
- **engineering**: CI matrisi, lint/type kapıları, paketleme sözleşmesi.
