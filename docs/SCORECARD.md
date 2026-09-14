# HGA RESEARCH SCORECARD

Üretim zamanı (UTC): `2026-09-14T22:34:49+00:00`

> Bu skorlar benchmark kabul kapılarından **otomatik** hesaplanır. Elle yazılmış bir puan bu tabloya giremez; kanıtı olmayan bölüm `n/a` döner.

| Bölüm | Skor | Kanıt |
|---|---:|---|
| architecture | 10.0 | operator_baseline_family_v1@81610cdf041b |
| memory | 10.0 | reliable_reasoning_depth_v1@5c8e421c20f1, hierarchical_memory_v1@72b18bd90ab7 |
| verification | 8.3 | priority_causal_chain_ablation_v1@f86dfc8878cb, multi_environment_verifier_isolation_v1, priority_weight_optimization_v1 |
| generalization | 8.3 | compositional_generalization_v2_raw_text@f59f990d1591 |
| reasoning | 9.1 | reliable_reasoning_depth_v1@5c8e421c20f1, reasoning_depth_root_cause_v1 |
| self_learning | 7.8 | self_learning_scaling_v1 |
| statistical_rigor | 7.5 | core_seed_statistics_v1 |
| human_evaluation | n/a | human_evaluation_protocol_v1 |
| turkish_nlp | 8.9 | turkish_semantic_extraction_v1@525654cb978c, twt_real_results_v1@66b13a898efa |
| language_modeling | 10.0 | turkish_lm_v1@d278278dd310 |
| reproducibility | 8.9 | reproducibility_audit_v1 |
| scientific_evidence | 8.2 | priority_causal_chain_ablation_v1@f86dfc8878cb, operator_baseline_family_v1@81610cdf041b, reliable_reasoning_depth_v1@5c8e421c20f1, hga_signature_benchmark_v1@291da16d8787, turkish_semantic_extraction_v1@525654cb978c, compositional_generalization_v2_raw_text@f59f990d1591, self_learning_scaling_v1, core_seed_statistics_v1, reasoning_depth_root_cause_v1, priority_weight_optimization_v1, turkish_lm_v1@d278278dd310 |
| engineering | 10.0 | engineering_contract_audit_v1 |
| **Overall Research Readiness** | **8.9** | 12/13 bölüm |

## HGA Capability Vector

| Sembol | Ad | Tür | Değer | Birim | Kanıt |
|---|---|---|---:|---|---|
| `P` | Physical Parameters | ölçülen | 4.05249e+07 | parametre | capacity_framework |
| `C_I^UB` | Interaction Upper Bound | ÜST SINIR | 1.84467e+19 | operatör girdisi | capacity_framework |
| `C_M^UB` | Memory Address Upper Bound | ÜST SINIR | 2.81475e+62 | adres | capacity_framework |
| `C_E` | Measured Experience Capacity | ölçülen | 960 | deneyim | capacity_framework |
| `C_V` | Measured Verified Capacity | ölçülen | 960 | deneyim | capacity_framework |
| `C_G` | Measured Generalization Capacity | ölçülen | 0.857143 | oran | compositional_generalization_v2_raw_text@f59f990d1591 |
| `C_R` | Measured Reliable Reasoning Depth | ölçülen | 2048 | hop | reliable_reasoning_depth_v1@5c8e421c20f1 |
| `C_RD` | Distractor-Resistant Reasoning Depth | ölçülen | 256 | hop @ 16384 dolgu | reliable_reasoning_depth_v1@5c8e421c20f1 |
| `C_MR` | Measured Memory Recall | ölçülen | 1 | oran | hierarchical_memory_v1@72b18bd90ab7 |
| `C_H` | Hallucination Resistance | ölçülen | 0.772461 | oran | hga_signature_benchmark_v1@291da16d8787 |
| `C_U` | Uncertainty Calibration | ölçülen | n/a | 1−ECE | — |

## Uyarılar

- `human_evaluation` bölümü için kanıt yok; skor üretilmedi.
- Genel skor yalnız 12/13 bölüm üzerinden hesaplandı; ['human_evaluation'] kanıtsız. Bu ortalama eksik kanıtı gizlemez, onu işaretler.

## Bölüm gerekçeleri

- **architecture**: Kronecker/rank-1/low-rank/full-dense kolları eşit parametre ve eşit FLOP rejimlerinde; skor kabul kapılarının geçme oranıdır.
- **memory**: Hiyerarşik bellek (hot/warm/cold/archive) recall, gecikme, tahliye ve çökme kurtarma kapıları; kanıt yoksa çıkarım derinliği ızgarasındaki en düşük geri çağırma oranına düşülür.
- **verification**: Priority(E) ağırlıklarının skor→sıralama→seçim→downstream zincirini taşıyıp taşımadığı ve doğrulayıcıların alan dışında çekimser kalıp kalmadığı (cross-domain kontaminasyon) ölçülür.
- **generalization**: Şema ve ontoloji önceden verilmeden, ham metinden keşif + kompozisyon başarısı.
- **reasoning**: Güvenilir çıkarım derinliği ve dolgu baskısı altındaki dayanıklılık.
- **self_learning**: Uzun kapalı döngüde bilgi ölçeklemesi ve yanlış bilgi birikmemesi (self-training çöküşüne direnç).
- **statistical_rigor**: Çekirdek protokollerde 20 tohum, eşleşmiş tasarım, %95 bootstrap CI, etki büyüklüğü ve iki bağımsız anlamlılık testi. Çıplak p-değeri kabul edilmez.
- **human_evaluation**: 50–100 Türkçe prompt, 10–20 kör değerlendirici ve Krippendorff α ile kodlayıcılar arası güvenilirlik. Protokol ve araç hazır olsa bile gerçek insan puanı yoksa skor üretilmez.
- **turkish_nlp**: Elle etiketli altın sette varlık/ilişki/özellik/zaman/olumsuzluk çıkarımı ve gerçek Türkçe treebank (TWT) üzerinde arc doğrulama.
- **language_modeling**: Gerçek Türkçe korpusta (full: tr_corpus_v1 1.11M kelime; smoke: TWT) belge-ayrık held-out perplexity ve next-token doğruluğu; n-gram kontrolleri zorunlu zemindir. Kanıt yoksa skor üretilmez — 'tiny smoke' bir dil modeli iddiası değildir.
- **reproducibility**: Manifest üretimi, veri/konfig hash'i, ÖLÇÜLEN determinizm (aynı tohum → bayt-eş çıktı) ve 20 tohum kuralına uyum.
- **scientific_evidence**: Tüm protokollerin kabul kapılarının birleşik geçme oranı. Bu skor yalnızca ölçüm iyileşerek yükselir.
- **engineering**: CI matrisi, lint/type kapıları, paketleme sözleşmesi.
