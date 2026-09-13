# -*- coding: utf-8 -*-
"""Evaluation metrikleri ve birleşik araştırma benchmark protokolleri."""
from .calibration import (
    CALIBRATION_PROTOCOL,
    COVERAGE_TARGETS,
    calibration_metrics,
    fit_temperature,
    temperature_scaling_report,
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
from .neural_compositional import (
    ABLATION_ORDER,
    NeuralCompositionalReport,
    run_neural_compositional_ablation,
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
from .provenance import (
    HashDogrulamaRaporu,
    ProvenanceRaporu,
    audit_provenance,
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
from .sweep import nk_taramasi, parametre_tahmini
from .turkish_benchmark import (
    MINI_TURKCE_CUMLELER,
    TurkceBenchmarkRaporu,
    mini_turkce_corpus,
    perplexity_benchmark,
    tokenizer_kapsami,
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
from .verifier_adversarial import (
    REQUIRED_ATTACK_CLASSES,
    ProofDecision,
    VerifierAttackDataset,
    VerifierAttackMetrics,
    VerifierAttackReport,
    run_verifier_adversarial_benchmark,
    verify_arithmetic_proof,
)

__all__ = [
    "CALIBRATION_PROTOCOL", "COVERAGE_TARGETS", "calibration_metrics",
    "fit_temperature", "temperature_scaling_report",
    "ABLATION_ORDER", "NeuralCompositionalReport", "run_neural_compositional_ablation",
    "ArcCandidate", "BinaryMetrics", "RealTurkishReport", "RealTurkishTaskData",
    "SelectiveArcSchemaVerifier", "TurkishWebTreebank", "evaluate_arc_predictions",
    "prepare_real_turkish_task", "run_real_turkish_benchmark",
    "ARCHITECTURE_CONFIG", "BASELINE_PROFILES", "MODEL_ORDER", "ArcFeatureVocabulary",
    "TWTBaselineReport", "build_twt_models", "run_twt_architecture_baselines",
    "REQUIRED_ATTACK_CLASSES", "ProofDecision", "VerifierAttackDataset",
    "VerifierAttackMetrics", "VerifierAttackReport", "run_verifier_adversarial_benchmark",
    "verify_arithmetic_proof",
    "CompositionalDataset", "CompositionalMetrics", "CompositionalReport",
    "DimensionMetrics", "GeneralizationCapacity", "run_compositional_benchmark",
    "PROFILE_CONFIGS", "SECTION_LABELS", "SECTION_ORDER", "ResearchBenchmarkReport",
    "research_report_html", "research_report_markdown", "run_research_benchmark",
    "save_research_report", "validate_sections",
    "HashDogrulamaRaporu", "ProvenanceRaporu", "audit_provenance",
    "document_hash", "ingest_with_provenance", "verify_document_hashes",
    "CokusOlcumu", "NKTaramaRaporu", "RankOlcumu", "measure_chain_collapse",
    "measure_single_layer_rank", "run_nk_rank_sweep", "theoretical_contract",
    "OlcekliGoldenRaporu", "OlcekliGoldenSweep", "run_scaled_golden",
    "run_scaled_golden_sweep", "veri_seti_uret",
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
