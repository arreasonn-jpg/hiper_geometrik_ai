# HGA Research Benchmark Report

- Rapor tipi: `hga-research-benchmark-v1`
- Profil: `smoke`
- Sonuç: `COMPLETED`
- Seedler: `1, 2, 3, 4, 5`
- Dataset hash: `eaf76fd4a7dd81865f803ab60d92f3b3baa6e22aa9694c19cd59dc5001ee786c`
- Config hash: `3e9bc1e4a923927b66e3142a98ed31912bb2605243563f7a968b4d1f399047d1`
- Deneyler: `EXP-0001, EXP-0002, EXP-0003, EXP-0004, EXP-0005`
- Genel diagnostic skor: `0.95126546`

> Diagnostic skor bir zekâ/SOTA skoru değildir; heterojen protokol sağlık ve görev metriklerini tek görünümde toplar.

| Bölüm | Durum | Skor mean ± std | Kontroller |
|---|---|---:|---:|
| Architecture | COMPLETED | 1.000 ± 0.000 | 4/4 |
| Kronecker | COMPLETED | 1.000 ± 0.000 | 4/4 |
| Memory | COMPLETED | 0.782 ± 0.015 | 21/21 |
| Verification | COMPLETED | 1.000 ± 0.000 | 20/20 |
| Compositional Generalization | COMPLETED | 1.000 ± 0.000 | 4/4 |
| Neural / Symbolic / Hybrid | COMPLETED | 0.847 ± 0.054 | 4/4 |
| Self Learning | COMPLETED | 1.000 ± 0.000 | 20/20 |
| OOD | COMPLETED | 1.000 ± 0.000 | 3/3 |
| Turkish NLP | COMPLETED | 0.933 ± 0.000 | 45/45 |
| Reproducibility | COMPLETED | 5 seed | 5+ seed |

## Bölüm Bulguları

### Architecture

Fiziksel parametreler ile teorik kapasite üst sınırları ayrıştırıldı.

- PASS — `c_i_not_parameter_count`
- PASS — `c_m_not_physical_table_size`
- PASS — `physical_parameter_estimate_positive`
- PASS — `sparse_and_dense_counts_separated`

Sınırlar:
- Parametre sayısı modeli kurmadan formülle tahmin edilmiştir; kesin model introspection değildir.
- C_I ve C_M kalite ya da öğrenilebilir kapasite skoru değildir.

### Kronecker

Çöküş sözleşmesi ve eşit-bütçeli iki-öğretmen kıyası tamamlandı.

- PASS — `both_teacher_tasks_present`
- PASS — `equal_physical_parameter_budget`
- PASS — `linear_chain_collapses`
- PASS — `silu_breaks_linear_collapse`

Sınırlar:
- Teacher görevleri sentetiktir; dil öğrenme üstünlüğü göstermez.
- Rank-1 bottleneck, unrestricted dense/Transformer baseline değildir.

### Memory

Fixed-table exact retrieval, aktif Engine DYNAMIC_KV yaşam döngüsü, adversarial politikalar ve ölçek maliyeti birlikte ölçüldü.

- PASS — `accounting_consistent`
- PASS — `active_lifecycle_active_policy_is_dynamic_kv`
- PASS — `active_lifecycle_bounded_lru_eviction_exact`
- PASS — `active_lifecycle_compaction_preserves_active_cardinality`
- PASS — `active_lifecycle_compaction_reclaims_journal_events`
- PASS — `active_lifecycle_legacy_loss_not_invented`
- PASS — `active_lifecycle_migration_target_collision_free`
- PASS — `active_lifecycle_persistence_hash_rejects_tamper`
- PASS — `active_lifecycle_persistence_roundtrip_exact`
- PASS — `active_lifecycle_replay_contains_only_active_keys`
- PASS — `active_lifecycle_snapshot_chain_linked`
- PASS — `active_lifecycle_snapshot_content_hash_present`
- PASS — `active_lifecycle_snapshot_restore_exact`
- PASS — `active_lifecycle_store_version_monotonic_after_restore`
- PASS — `collision_samples_bounded`
- PASS — `dynamic_cost_growth_reported`
- PASS — `dynamic_kv_exact_recall_at_all_scales`
- PASS — `dynamic_kv_retains_both`
- PASS — `false_positive_rate_bounded`
- PASS — `fixed_policies_cannot_retain_both`
- PASS — `retrieval_bounded`

Sınırlar:
- Context akışı sentetiktir; semantic retrieval ölçülmez.
- Dynamic KV tam anahtarlı exact lookup'tır; learned/semantic retrieval değildir.
- Bounded modda çakışma yerine açık LRU/FIFO eviction vardır; eski bilgi kapasite dolunca kaybolabilir.
- Persistence tek süreç atomik dosya değiştirmesini doğrular; dağıtık consensus veya çok-yazarlı transaction değildir.
- Legacy migration yalnız replay/payload'da bulunan ve FIRST_WINS'te tutulmuş kayıtları taşır; geçmiş collision kaybını uydurmaz.
- Diagnostic score fixed-table exact retrieval accuracy'dir; aktif Dynamic KV'nin exact lookup başarısı düşük fixed skoru maskelemez.

### Verification

Golden kararlar, verifier fault-injection, katı proof attack suite ve gerçek kaynak artifact'ında STALE yaşam döngüsü birlikte raporlandı.

- PASS — `attack_far_detected`
- PASS — `attack_frr_detected`
- PASS — `control_far_zero`
- PASS — `control_frr_zero`
- PASS — `golden_leakage_clean`
- PASS — `knowledge_lifecycle_changed_hash_creates_new_revision_and_supersedes_old`
- PASS — `knowledge_lifecycle_dependency_staleness_propagates`
- PASS — `knowledge_lifecycle_event_chain_integrity`
- PASS — `knowledge_lifecycle_explicit_clock_no_wall_clock_dependency`
- PASS — `knowledge_lifecycle_explicit_withdrawal_is_retracted`
- PASS — `knowledge_lifecycle_freshness_expiry_marks_stale_not_false_or_retracted`
- PASS — `knowledge_lifecycle_real_twt_urls_hashes_and_revision_pinned`
- PASS — `knowledge_lifecycle_same_hash_revalidation_restores_active`
- PASS — `knowledge_lifecycle_source_outage_is_stale_not_retracted`
- PASS — `knowledge_lifecycle_stale_excluded_from_default_query_but_auditable`
- PASS — `proof_attack_far_zero`
- PASS — `proof_attack_frr_zero`
- PASS — `proof_attack_robustness_complete`
- PASS — `unknown_and_conflict_preserved`
- PASS — `unsupported_rule_abstains`

Real-artifact STALE lifecycle:

- Protocol: `real-artifact-knowledge-lifecycle-v1`
- Source revision: `40838e5cbe3f2882d4e768a3d782e6219e50b52a`
- Final states: `{'ACTIVE': 0, 'STALE': 2, 'SUPERSEDED': 1, 'RETRACTED': 1}`
- Hash-chained events: `14`; head `90307a5a99739f3fbb2aaa5ebecf9637a7d1fa828c5f01867b484a88cfe5fa8b`
- STALE is freshness/dependency uncertainty; it is not FALSE or RETRACTED.

Sınırlar:
- Proof attack suite katı tam sayı toplama şemasıyla sınırlıdır; genel theorem prover değildir.
- Golden v1 küçüktür; üretim verifier soundness kanıtı değildir.
- TWT files and provenance are real; lifecycle changes after ingest are controlled interventions.
- No live HTTP fetch is performed, so network nondeterminism cannot affect the benchmark.
- STALE blocks default use but is not evidence that the underlying proposition is false.

### Compositional Generalization

C_G=1.000 (5/5); teorik kapasite değildir.

- PASS — `candidate_generation_complete`
- PASS — `ood_unknown_not_false`
- PASS — `semantic_leakage_clean`
- PASS — `surface_leakage_clean`

- **C_G:** `5/5` (`1.000`)
- Üretim exact: `4/4`
- Tanım: Sabit hga-compositional-tr-v1 held-out setinde, en az bir unseen boyutu taşıyan VALID örneklerden doğru çözülenlerin sayısı/oranı.

Sınırlar:
- Bu küçük ve elle sabitlenmiş bir Türkçe semantik regresyon setidir; gerçek corpus ölçeği veya genel dil kalitesi kanıtı değildir.
- Unseen entity, varlığın ontoloji kaydı ve özellikleri bilinirken eğitim üçlülerinde görülmemesi anlamına gelir; sıfırdan entity discovery değildir.
- Unseen relation, ilişki şeması bilinirken eğitim olgularında görülmemesi anlamına gelir; bilinmeyen ilişki indüksiyonu değildir.
- Metin üretimi ilişkiye özel morfolojik şablondur; neural serbest üretim değildir.
- Unseen wording ölçümü küratörlü sözlük tabanlı ayıklayıcıyı test eder; açık alan dependency parsing testi değildir.

### Neural / Symbolic / Hybrid

Üç paradigma aynı split ve metrik sözleşmesinde karşılaştırıldı.

- PASS — `all_error_metrics_present`
- PASS — `hybrid_full_coverage`
- PASS — `same_test_count`
- PASS — `symbolic_abstention_visible`

Sınırlar:
- Görev prosedürel ve sentetiktir; Transformer baseline içermez.
- Smoke profilindeki az epoch yayınlanabilir model kıyası değildir.

### Self Learning

Kapalı aritmetik kontrol ile arithmetic/logic/consistency shared store+DynamicKV self-learning izolasyonu birlikte ölçüldü.

- PASS — `far_zero_after_verifier`
- PASS — `frr_zero_after_verifier`
- PASS — `holdout_isolation_clean`
- PASS — `multi_environment_all_durable_knowledge_correct`
- PASS — `multi_environment_candidate_sets_disjoint`
- PASS — `multi_environment_cross_verifiers_abstain`
- PASS — `multi_environment_cross_verifiers_never_accept`
- PASS — `multi_environment_environment_namespaces_disjoint`
- PASS — `multi_environment_environment_schedule_balanced`
- PASS — `multi_environment_every_environment_grows`
- PASS — `multi_environment_generation_memory_knowledge_holdouts_clean`
- PASS — `multi_environment_no_false_acceptance_after_verifier`
- PASS — `multi_environment_no_false_rejection_after_verifier`
- PASS — `multi_environment_shared_memory_contains_no_holdout`
- PASS — `multi_environment_shared_memory_exact_retrieval`
- PASS — `multi_environment_shared_memory_is_active_dynamic_kv`
- PASS — `multi_environment_shared_store_growth_accounted`
- PASS — `multi_environment_shared_store_has_no_cross_environment_facts`
- PASS — `multi_environment_three_independent_environments_present`
- PASS — `no_incorrect_durable_knowledge`

Multi-environment closed self-learning (mean ± std):

| Environment | Generated | Verified | Rejected | FAR | FRR | Durable growth | Incorrect durable | Memory recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arithmetic | 40.0000 ± 0.0000 | 13.4000 ± 0.4899 | 26.6000 ± 0.4899 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 13.4000 ± 0.4899 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| logic | 40.0000 ± 0.0000 | 20.4000 ± 2.0591 | 19.6000 ± 2.0591 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 20.4000 ± 2.0591 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| consistency | 40.0000 ± 0.0000 | 19.4000 ± 1.3565 | 20.6000 ± 1.3565 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 19.4000 ± 1.3565 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 |

- Shared final facts: `62.2000 ± 2.7129`
- Cross-verifier acceptances (all seeds): `0`
- Tüm seed kapıları: `True`

Sınırlar:
- Ground truth bağımsız ama sentetik aritmetik environment tarafından sağlanır.
- Entity uzayı başlangıçta sabittir; ölçülen büyüme yeni doğrulanmış relation fact büyümesidir.
- Bu deney genel dilde otonom yeni bilgi keşfi kanıtı değildir.
- Üç environment deterministik ve sentetiktir; gerçek dünya veya genel dil self-learning kanıtı değildir.
- Verifier router relation namespace'ini önceden bilir; environment discovery/induction ölçülmez.
- Shared store büyümesi yeni entity öğrenimi değil, bağımsız doğrulanmış relation fact eklenmesidir.
- Neural model eğitimi yoktur; bu epistemik durum, contamination ve yaşam döngüsü protokolüdür.
- Holdout yalnız sızıntı kontrolüdür; held-out task generalization skoru değildir.

### OOD

Kanıtı olmayan OOD nesne FALSE yerine UNCERTAIN olarak ölçüldü.

- PASS — `ood_cases_present`
- PASS — `ood_expected_state_accuracy`
- PASS — `unknown_separated_from_false`

Sınırlar:
- OOD bölümü tek küçük ontoloji-dışı özellik vakasıdır; dağılım kayması benchmarkı değildir.

### Turkish NLP

TWT: 4849 gerçek, insan anotasyonlu cümle; schema F1=0.933; dört neural kol fiziksel parametre oranı=1.0011; HGA C_G_N=0.879.

- PASS — `baseline_all_four_model_families_present`
- PASS — `baseline_all_models_full_test_coverage`
- PASS — `baseline_all_models_optimization_finite`
- PASS — `baseline_all_trainable_parameters_active`
- PASS — `baseline_architecture_body_parameters_within_five_percent`
- PASS — `baseline_calibration_is_dev_only_and_argmax_preserving`
- PASS — `baseline_calibration_reports_all_disjoint_slices`
- PASS — `baseline_no_unused_parameter_padding`
- PASS — `baseline_physical_parameters_within_one_percent`
- PASS — `baseline_relation_holdout_not_in_model_visible_train_dev`
- PASS — `baseline_required_metrics_all_models`
- PASS — `baseline_same_examples_optimizer_loss_schedule`
- PASS — `baseline_same_real_dataset_and_splits`
- PASS — `baseline_train_only_vocabulary_has_no_unknown`
- PASS — `controlled_heldout_wording_cases_present`
- PASS — `neural_compositional_all_ablation_arms_present`
- PASS — `neural_compositional_all_arms_full_test_coverage`
- PASS — `neural_compositional_all_arms_optimization_finite`
- PASS — `neural_compositional_all_remaining_trainable_parameters_active`
- PASS — `neural_compositional_c_g_n_is_measured_not_theoretical`
- PASS — `neural_compositional_composition_disjoint_balanced_and_nonempty`
- PASS — `neural_compositional_full_and_additive_geometry_parameter_matched`
- PASS — `neural_compositional_removed_components_reduce_parameters_without_padding`
- PASS — `neural_compositional_same_real_twt_split_and_candidates`
- PASS — `neural_compositional_same_training_protocol_all_arms`
- PASS — `neural_compositional_shared_initialization_equal`
- PASS — `neural_compositional_train_only_vocabulary_clean`
- PASS — `turkish_characters_roundtrip`
- PASS — `twt_balanced_binary_test`
- PASS — `twt_candidate_hashes_present`
- PASS — `twt_composition_disjoint_slice_present`
- PASS — `twt_deterministic_nonempty_splits`
- PASS — `twt_disjoint_dimension_contracts_clean`
- PASS — `twt_duplicate_and_leakage_audit_clean`
- PASS — `twt_entity_disjoint_slice_present`
- PASS — `twt_human_annotations_documented`
- PASS — `twt_open_redistributable_license`
- PASS — `twt_raw_source_hashes_verified`
- PASS — `twt_real_turkish_sources_documented`
- PASS — `twt_relation_disjoint_slice_present`
- PASS — `twt_required_metrics_reported`
- PASS — `twt_split_hashes_present`
- PASS — `twt_upstream_revision_pinned`
- PASS — `twt_wording_disjoint_slice_present`
- PASS — `unk_rate_below_five_percent`

Parameter-matched TWT architecture test özeti:

| Model | Parametre | Body parametre | Accuracy mean ± std | F1 mean ± std | FAR mean | FRR mean | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 291808 | 10496 | 0.8938 ± 0.0061 | 0.8943 ± 0.0071 | 0.1120 | 0.1005 | 1.0000 |
| transformer | 291816 | 10504 | 0.8207 ± 0.0166 | 0.8206 ± 0.0149 | 0.1784 | 0.1802 | 1.0000 |
| kronecker | 291514 | 10202 | 0.8977 ± 0.0046 | 0.8992 ± 0.0046 | 0.1176 | 0.0870 | 1.0000 |
| hga | 291827 | 10515 | 0.8996 ± 0.0095 | 0.8997 ± 0.0092 | 0.1014 | 0.0994 | 1.0000 |

- Seedler: `[1, 2, 3, 4, 5]`
- Dataset hash: `66b13a898efa88998a9329f1551530f5241835a8085a0e5f26e0eb374d7e3276`
- Not: Mean/std aynı gerçek TWT test splitindeki seed koşularıdır; mimari üstünlük veya genel dil/SOTA sonucu değildir.

Dev-only temperature calibration (test all; mean ± std):

| Model | Temperature | ECE before | ECE after | NLL before | NLL after | Brier after | AURC after |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 1.0279 ± 0.0359 | 0.0101 ± 0.0030 | 0.0096 ± 0.0032 | 0.2639 ± 0.0083 | 0.2636 ± 0.0081 | 0.0781 ± 0.0029 | 0.0286 ± 0.0020 |
| transformer | 0.8547 ± 0.0240 | 0.0228 ± 0.0039 | 0.0140 ± 0.0065 | 0.3904 ± 0.0256 | 0.3882 ± 0.0248 | 0.1237 ± 0.0094 | 0.0672 ± 0.0099 |
| kronecker | 0.7389 ± 0.0240 | 0.0416 ± 0.0064 | 0.0077 ± 0.0027 | 0.2808 ± 0.0134 | 0.2677 ± 0.0141 | 0.0785 ± 0.0042 | 0.0301 ± 0.0041 |
| hga | 0.9803 ± 0.0796 | 0.0108 ± 0.0034 | 0.0064 ± 0.0028 | 0.2546 ± 0.0152 | 0.2537 ± 0.0149 | 0.0753 ± 0.0055 | 0.0268 ± 0.0032 |

Neural compositional HGA ablation özeti:

| Kol | Parametre | C_G_N mean ± std | All F1 mean ± std | Çıkarılan/değiştirilen bileşen |
|---|---:|---:|---:|---|
| full | 291827 | 0.8753 ± 0.0119 | 0.8956 ± 0.0085 | yok |
| no_attention | 290770 | 0.8767 ± 0.0100 | 0.8972 ± 0.0056 | HiperGeometrikAttention |
| additive_geometry | 291827 | 0.8856 ± 0.0031 | 0.9051 ± 0.0025 | multiplicative outer product (same projections retained) |
| no_kronecker_chain | 289019 | 0.8550 ± 0.0078 | 0.8771 ± 0.0098 | KureselZincir |

- Tanım: C_G_N = gerçek TWT composition_disjoint dengeli arc testindeki accuracy; teorik kapasite veya kontrollü fixture C_G değildir.
- Tüm seed kapıları: `True`

Sınırlar:
- TWT dependency etiketleri morphosyntactic'tir; semantik knowledge triple değildir.
- Entity-disjoint anahtarı lemma/form token kimliğidir; named entity annotation değildir.
- Negatifler insan tarafından tek tek yazılmamıştır; insan anotasyonlu single-head basic tree sözleşmesinden deterministik türetilir.
- Lemma-set Jaccard denetimi muhafazakâr bir lexical-semantic yakınlık proxy'sidir; semantik eşdeğerlik kanıtı değildir.
- Schema-frequency baseline benchmark hattını doğrular; HGA/Transformer üstünlüğü veya genel Türkçe yeterlilik iddiası taşımaz.
- Görev TWT basic morphosyntactic dependency-arc doğrulamasıdır; genel dil modelleme değildir.
- Structured 6-field encoder tüm kollarda ortaktır; ham cümleden end-to-end dependency parsing ölçülmez.
- Smoke profili yarım epoch'tan az sabit adım kullanır; yayınlanabilir convergence/SOTA sonucu değildir.
- Parameter matching gerçek trainable numel üzerinden ±%1 içindedir; FLOP, aktivasyon belleği ve duvar süresi eşitlenmez, ayrıca raporlanır.
- HGA kolu repository'nin attention/outer-product/Kronecker/fraktal çekirdeğidir; tam autoregressive HiperGeometrikAI dil modeli değildir.
- Unseen relation train-only vocabulary'de UNKNOWN olur; bu sıfırdan relation-semantics induction görevidir ve yüksek skor beklenmez.
- Temperature yalnız dev NLL'yi optimize eder; test calibration metriklerinin iyileşmesi garanti edilmez ve argmax doğruluğu değişmez.
- Selective risk sabit coverage noktalarında diagnostic ölçümdür; gerçek deployment threshold'u veya domain-shift garantisi değildir.
- C_G_N yalnız gerçek TWT composition-disjoint arc-verification accuracy'sidir; mevcut kontrollü C_G veya teorik kapasite değildir.
- Structured altı-alan girdisi ham cümleden end-to-end parsing ya da language modeling ölçmez.
- Component removal kollarında parametre sayısı doğal olarak azalır; sonuç yalnız accuracy değil numel/süre ile birlikte okunmalıdır.
- Smoke profili 32 adımdır; nedensel mimari üstünlük veya convergence/SOTA iddiası taşımaz.
- Ablation sonuçları TWT morphosyntactic relation'larıyla sınırlıdır; semantik compositional reasoning kanıtı değildir.
- Tokenizer ölçümü küçük smoke corpusundadır ve TWT test skoruna katılmaz.
- Kontrollü lexicon parser sonucu TWT dependency sonucu olarak sunulmaz.

## Reproducibility

- seed_count: `5`
- five_or_more_seeds: `True`
- all_manifests_completed: `True`
- dataset_hash_identical_across_seeds: `True`
- byte_identical_results: `False`
- Not: Byte-identical sonuç zorunlu değildir: stochastic modeller ve süre metrikleri değişir. Bilimsel değerlendirme seed-bazlı skor dağılımını kullanır.

## Suite Sınırları

- overall_diagnostic_score, tamamlanan heterojen bölüm skorlarının basit ortalamasıdır; zekâ veya SOTA skoru değildir.
- Smoke profil CI/protokol doğrulaması içindir; yayınlanabilir ölçek sonucu değildir.
- Sentetik bölümler, kontrollü compositional fixture ve gerçek insan-anotasyonlu TWT sonucu raporda ayrı tutulur.
- TWT görevi morphosyntactic dependency arc doğrulamasıdır; semantik relation extraction veya genel dil iddiası değildir.
- SKIPPED bölümler başarı sayılmaz ve overall_diagnostic_score hesabına girmez.
