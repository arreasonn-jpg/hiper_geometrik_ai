# -*- coding: utf-8 -*-
"""Evaluation metrikleri ve birleşik araştırma benchmark protokolleri."""
from .calibration import (
    CALIBRATION_PROTOCOL,
    COVERAGE_TARGETS,
    calibration_metrics,
    fit_temperature,
    temperature_scaling_report,
)
from .capability_vector import (
    SCORECARD_SECTIONS,
    TERMINOLOGY,
    CapabilityEntry,
    Scorecard,
    ScorecardSection,
    build_capability_vector,
    build_scorecard,
    save_scorecard,
    scorecard_markdown,
)
from .capacity import (
    CapacityReport,
    capacity_contract,
    measure_experience_capacity,
    run_capacity_benchmark,
)
from .compositional import (
    CompositionalDataset,
    CompositionalMetrics,
    CompositionalReport,
    DimensionMetrics,
    GeneralizationCapacity,
    run_compositional_benchmark,
)
from .compositional_v2 import (
    AXES,
    TEST_CASES,
    TRAIN_CORPUS,
    AxisMetrics,
    CompositionalV2Report,
    V2Case,
    compositional_v2_markdown,
    run_compositional_v2_benchmark,
)
from .english_ewt import (
    ARCHITECTURE as ENGLISH_EWT_ARCHITECTURE,
    EnglishEWT,
    EnglishEWTReport,
    EWTTaskData,
    english_hga_scaling_markdown,
    english_ewt_markdown,
    prepare_english_ewt_task,
    run_english_hga_scaling_probe,
    run_english_ewt_baselines,
)
from .experiment import (
    ExperimentRun,
    SeedSweepReport,
    canonical_hash,
    file_sha256,
    run_seed_sweep,
    seed_everything,
)
from .golden import (
    GoldenDataset,
    GoldenMetrics,
    GoldenReport,
    run_golden_benchmark,
    run_golden_seed_sweep,
)
from .hallucination import HallucinationReport, hallucination_metrics
from .kronecker import kronecker_capacity_contract, run_kronecker_dense_trial
from .kronecker_rank import (
    CokusOlcumu,
    NKTaramaRaporu,
    RankOlcumu,
    measure_chain_collapse,
    measure_single_layer_rank,
    run_nk_rank_sweep,
    theoretical_contract,
)
from .leakage import LeakageAuditReport, audit_partitions, semantic_fingerprint
from .lifecycle import LifecycleBenchmarkReport, run_lifecycle_benchmark
from .long_context import (
    LONG_CONTEXT_TARGETS,
    LongContextReport,
    long_context_markdown,
    run_long_context_benchmark,
)
from .memory_hierarchy import (
    PROFILES as MEMORY_HIERARCHY_PROFILES,
)
from .memory_hierarchy import (
    STREAMING_PROTOCOL as MEMORY_STREAMING_PROTOCOL,
)
from .memory_hierarchy import (
    STREAMING_TARGET_RECORDS,
    MemoryHierarchyReport,
    MemoryStreamingHarnessReport,
    memory_hierarchy_markdown,
    memory_streaming_harness_markdown,
    run_memory_hierarchy_benchmark,
    run_memory_streaming_harness,
)
from .neural_compositional import (
    ABLATION_ORDER,
    NeuralCompositionalReport,
    run_neural_compositional_ablation,
)
from .operator_baselines import (
    ARMS as OPERATOR_ARMS,
)
from .operator_baselines import (
    OFFICIAL_SEED_REQUIREMENT,
    OFFICIAL_SEEDS,
    STRUCTURED_TEACHERS,
    TEACHERS,
    UNSTRUCTURED_TEACHERS,
    ArmResult,
    OperatorBaselineReport,
    arm_budget,
    budget_table,
    operator_baseline_markdown,
    run_operator_baseline_benchmark,
    run_operator_baseline_official_report,
)
from .paradigma import (
    KOLLAR,
    ParadigmaRaporu,
    ParadigmaSweepRaporu,
    gorev_uret,
    hibrit_tahmin,
    noral_tahmin,
    run_paradigm_ablation,
    run_paradigm_sweep,
    sembolik_tahmin,
)
from .priority_ablation import (
    TERIMLER,
    PriorityAblationReport,
    PriorityPool,
    VariantResult,
    havuz_uret,
    kendall_tau,
    priority_ablation_markdown,
    run_priority_weight_ablation,
    spearman_rho,
)
from .provenance import (
    PROVENANCE_CHAIN_KINDS,
    HashDogrulamaRaporu,
    ProvenanceChain,
    ProvenanceEdge,
    ProvenanceNode,
    ProvenanceRaporu,
    audit_provenance,
    build_provenance_chain,
    document_hash,
    ingest_with_provenance,
    verify_document_hashes,
)
from .real_turkish import (
    ArcCandidate,
    BinaryMetrics,
    RealTurkishReport,
    RealTurkishTaskData,
    SelectiveArcSchemaVerifier,
    TurkishWebTreebank,
    evaluate_arc_predictions,
    prepare_real_turkish_task,
    run_real_turkish_benchmark,
)
from .reasoning_depth import (
    PROFILES as DEPTH_PROFILES,
)
from .reasoning_depth import (
    DepthReport,
    measure_reasoning_depth,
    reasoning_depth_markdown,
)
from .reporting import benchmark_raporu_kaydet, benchmark_raporu_markdown, benchmark_raporu_olustur
from .research import (
    PROFILE_CONFIGS,
    SECTION_LABELS,
    SECTION_ORDER,
    ResearchBenchmarkReport,
    research_report_html,
    research_report_markdown,
    run_research_benchmark,
    save_research_report,
    validate_sections,
)
from .scaled_golden import (
    OlcekliGoldenRaporu,
    OlcekliGoldenSweep,
    run_scaled_golden,
    run_scaled_golden_sweep,
    veri_seti_uret,
)
from .self_learning_transfer import (
    SelfLearningTransferReport,
    TransferPairResult,
    run_self_learning_transfer_benchmark,
    self_learning_transfer_markdown,
)
from .semantic_extraction import (
    GOLD_SET,
    GOLD_SET_V1,
    GOLD_SET_V2,
    LAYERS,
    PROTOCOL_V1,
    PROTOCOL_V2,
    GoldSentence,
    SemanticExtractionReport,
    build_semantic_gold_v2,
    run_semantic_extraction_benchmark,
    run_semantic_extraction_v2_benchmark,
    semantic_extraction_markdown,
)
from .signature import (
    ARMS as SIGNATURE_ARMS,
)
from .signature import (
    MEMORY_CRITICAL_TASKS,
    NEURAL_ARMS,
    RELEASE_GATE_PROTOCOL,
    SIGNATURE_RELEASE_MIN_SEEDS,
    SIGNATURE_RELEASE_PROFILE,
    TASKS,
    Instance,
    SignatureReleaseGateReport,
    SignatureReport,
    build_dataset,
    leakage_report,
    run_signature_benchmark,
    run_signature_release_gate,
    signature_markdown,
    signature_release_gate_markdown,
    symbolic_predict,
)
from .sweep import nk_taramasi, parametre_tahmini
from .turkish_benchmark import (
    MINI_TURKCE_CUMLELER,
    TurkceBenchmarkRaporu,
    mini_turkce_corpus,
    perplexity_benchmark,
    tokenizer_kapsami,
)
from .turkish_corpus_expansion import (
    ALLOWED_LICENSES as CORPUS_EXPANSION_ALLOWED_LICENSES,
)
from .turkish_corpus_expansion import (
    DEFAULT_TARGET_WORDS as CORPUS_EXPANSION_TARGET_WORDS,
)
from .turkish_corpus_expansion import (
    CorpusCandidate,
    CorpusExpansionReport,
    run_turkish_corpus_expansion_pipeline,
    turkish_corpus_expansion_markdown,
)
from .turkish_lm import (
    NEURAL_ARMS as TURKISH_LM_ARMS,
)
from .turkish_lm import (
    NGRAM_ARMS,
    LMCorpus,
    LMDocument,
    NgramBaselines,
    TurkishLMReport,
    build_lm_models,
    document_splits_disjoint,
    prepare_lm_corpus,
    run_turkish_lm_benchmark,
    turkish_lm_markdown,
)
from .twt_baselines import (
    ARCHITECTURE_CONFIG,
    BASELINE_PROFILES,
    MODEL_ORDER,
    ArcFeatureVocabulary,
    TWTBaselineReport,
    build_twt_models,
    run_twt_architecture_baselines,
)
from .twt_results import (
    CALIBRATION_METRICS,
    COST_FIELDS,
    HEADLINE_METRICS,
    TWTResultsReport,
    analytic_forward_flops,
    flop_fairness,
    measured_forward_flops,
    reconcile_flops,
    results_markdown,
    run_twt_results,
)
from .verifier_adversarial import (
    DATA_FILE_V2,
    REQUIRED_ATTACK_CLASSES,
    REQUIRED_ATTACK_CLASSES_V2,
    ProofDecision,
    VerifierAttackDataset,
    VerifierAttackMetrics,
    VerifierAttackReport,
    run_verifier_adversarial_benchmark,
    run_verifier_adversarial_v2_benchmark,
    verify_arithmetic_proof,
)
from .verifier_ensemble import (
    ENSEMBLE_POLICY,
    ENSEMBLE_PROTOCOL,
    VerifierEnsembleReport,
    VerifierMember,
    default_verifier_members,
    run_verifier_ensemble_benchmark,
    verifier_ensemble_markdown,
    verify_normal_form_addition,
    verify_trace_replay_addition,
)

__all__ = [
    # P1 gerçek Türkçe dil modelleme (TWT held-out)
    "ENGLISH_EWT_ARCHITECTURE", "EnglishEWT", "EnglishEWTReport", "EWTTaskData",
    "prepare_english_ewt_task", "run_english_ewt_baselines", "english_ewt_markdown",
    "run_english_hga_scaling_probe", "english_hga_scaling_markdown",
    "CORPUS_EXPANSION_ALLOWED_LICENSES", "CORPUS_EXPANSION_TARGET_WORDS",
    "CorpusCandidate", "CorpusExpansionReport",
    "run_turkish_corpus_expansion_pipeline",
    "turkish_corpus_expansion_markdown",
    "TURKISH_LM_ARMS", "NGRAM_ARMS", "LMCorpus", "LMDocument",
    "NgramBaselines", "TurkishLMReport", "prepare_lm_corpus",
    "document_splits_disjoint", "build_lm_models",
    "run_turkish_lm_benchmark", "turkish_lm_markdown",
    # P0-3 TWT gerçek sonuç tablosu
    "HEADLINE_METRICS", "CALIBRATION_METRICS", "COST_FIELDS",
    "TWTResultsReport", "analytic_forward_flops", "measured_forward_flops",
    "reconcile_flops", "flop_fairness", "run_twt_results", "results_markdown",
    # P0-8 hiyerarşik bellek
    "MEMORY_HIERARCHY_PROFILES", "MEMORY_STREAMING_PROTOCOL",
    "STREAMING_TARGET_RECORDS", "MemoryHierarchyReport",
    "MemoryStreamingHarnessReport", "run_memory_hierarchy_benchmark",
    "run_memory_streaming_harness", "memory_hierarchy_markdown",
    "memory_streaming_harness_markdown",
    # P0-1 Priority(E) nedensel zincir
    "TERIMLER", "PriorityAblationReport", "PriorityPool", "VariantResult",
    "havuz_uret", "kendall_tau", "spearman_rho",
    "run_priority_weight_ablation", "priority_ablation_markdown",
    # P0-2 operatör baseline ailesi
    "OPERATOR_ARMS", "OFFICIAL_SEED_REQUIREMENT", "OFFICIAL_SEEDS",
    "TEACHERS", "STRUCTURED_TEACHERS", "UNSTRUCTURED_TEACHERS", "ArmResult",
    "OperatorBaselineReport", "arm_budget", "budget_table",
    "run_operator_baseline_benchmark", "run_operator_baseline_official_report",
    "operator_baseline_markdown",
    # P0-4 signature benchmark
    "SIGNATURE_ARMS", "NEURAL_ARMS", "TASKS", "MEMORY_CRITICAL_TASKS",
    "RELEASE_GATE_PROTOCOL", "SIGNATURE_RELEASE_MIN_SEEDS",
    "SIGNATURE_RELEASE_PROFILE", "Instance", "SignatureReport",
    "SignatureReleaseGateReport", "build_dataset", "leakage_report",
    "symbolic_predict", "run_signature_benchmark", "run_signature_release_gate",
    "signature_markdown", "signature_release_gate_markdown",
    # P0-5 semantik çıkarım
    "GOLD_SET", "GOLD_SET_V1", "GOLD_SET_V2", "PROTOCOL_V1", "PROTOCOL_V2",
    "LAYERS", "GoldSentence", "SemanticExtractionReport",
    "build_semantic_gold_v2", "run_semantic_extraction_benchmark",
    "run_semantic_extraction_v2_benchmark", "semantic_extraction_markdown",
    # P0-6 C_G v2
    "AXES", "TRAIN_CORPUS", "TEST_CASES", "V2Case", "AxisMetrics",
    "CompositionalV2Report", "run_compositional_v2_benchmark",
    "compositional_v2_markdown",
    # P0-7 C_R / C_RD
    "DEPTH_PROFILES", "DepthReport", "measure_reasoning_depth",
    "reasoning_depth_markdown",
    # P3 Capability Vector + otomatik karne
    "TERMINOLOGY", "SCORECARD_SECTIONS", "CapabilityEntry", "ScorecardSection",
    "Scorecard", "build_capability_vector", "build_scorecard",
    "scorecard_markdown", "save_scorecard",
    "CALIBRATION_PROTOCOL", "COVERAGE_TARGETS", "calibration_metrics",
    "fit_temperature", "temperature_scaling_report",
    "ABLATION_ORDER", "NeuralCompositionalReport", "run_neural_compositional_ablation",
    "ArcCandidate", "BinaryMetrics", "RealTurkishReport", "RealTurkishTaskData",
    "SelectiveArcSchemaVerifier", "TurkishWebTreebank", "evaluate_arc_predictions",
    "prepare_real_turkish_task", "run_real_turkish_benchmark",
    "ARCHITECTURE_CONFIG", "BASELINE_PROFILES", "MODEL_ORDER", "ArcFeatureVocabulary",
    "TWTBaselineReport", "build_twt_models", "run_twt_architecture_baselines",
    "DATA_FILE_V2", "REQUIRED_ATTACK_CLASSES", "REQUIRED_ATTACK_CLASSES_V2",
    "ProofDecision", "VerifierAttackDataset", "VerifierAttackMetrics",
    "VerifierAttackReport", "run_verifier_adversarial_benchmark",
    "run_verifier_adversarial_v2_benchmark",
    "ENSEMBLE_POLICY", "ENSEMBLE_PROTOCOL", "VerifierMember",
    "VerifierEnsembleReport", "default_verifier_members",
    "run_verifier_ensemble_benchmark", "verifier_ensemble_markdown",
    "verify_normal_form_addition", "verify_trace_replay_addition",
    "LONG_CONTEXT_TARGETS", "LongContextReport", "long_context_markdown",
    "run_long_context_benchmark",
    "verify_arithmetic_proof",
    "CompositionalDataset", "CompositionalMetrics", "CompositionalReport",
    "DimensionMetrics", "GeneralizationCapacity", "run_compositional_benchmark",
    "PROFILE_CONFIGS", "SECTION_LABELS", "SECTION_ORDER", "ResearchBenchmarkReport",
    "research_report_html", "research_report_markdown", "run_research_benchmark",
    "save_research_report", "validate_sections",
    "PROVENANCE_CHAIN_KINDS", "HashDogrulamaRaporu", "ProvenanceChain",
    "ProvenanceEdge", "ProvenanceNode", "ProvenanceRaporu", "audit_provenance",
    "build_provenance_chain", "document_hash", "ingest_with_provenance",
    "verify_document_hashes",
    "CokusOlcumu", "NKTaramaRaporu", "RankOlcumu", "measure_chain_collapse",
    "measure_single_layer_rank", "run_nk_rank_sweep", "theoretical_contract",
    "OlcekliGoldenRaporu", "OlcekliGoldenSweep", "run_scaled_golden",
    "run_scaled_golden_sweep", "veri_seti_uret",
    "SelfLearningTransferReport", "TransferPairResult",
    "run_self_learning_transfer_benchmark", "self_learning_transfer_markdown",
    "KOLLAR", "ParadigmaRaporu", "ParadigmaSweepRaporu", "gorev_uret",
    "sembolik_tahmin", "noral_tahmin", "hibrit_tahmin",
    "run_paradigm_ablation", "run_paradigm_sweep",
    "HallucinationReport", "hallucination_metrics",
    "parametre_tahmini", "nk_taramasi",
    "MINI_TURKCE_CUMLELER", "TurkceBenchmarkRaporu",
    "mini_turkce_corpus", "tokenizer_kapsami", "perplexity_benchmark",
    "benchmark_raporu_olustur", "benchmark_raporu_markdown", "benchmark_raporu_kaydet",
    "GoldenDataset", "GoldenMetrics", "GoldenReport", "run_golden_benchmark",
    "run_golden_seed_sweep",
    "LeakageAuditReport", "audit_partitions", "semantic_fingerprint",
    "LifecycleBenchmarkReport", "run_lifecycle_benchmark",
    "ExperimentRun", "SeedSweepReport", "canonical_hash", "file_sha256",
    "run_seed_sweep", "seed_everything", "kronecker_capacity_contract",
    "run_kronecker_dense_trial",
    "CapacityReport", "capacity_contract", "measure_experience_capacity",
    "run_capacity_benchmark",
]
