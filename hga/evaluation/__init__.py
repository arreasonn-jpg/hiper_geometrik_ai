# -*- coding: utf-8 -*-
"""Evaluation metrikleri: halüsinasyon, sweep, Türkçe tokenizer/perplexity."""
from .capacity import (
    CapacityReport,
    capacity_contract,
    measure_experience_capacity,
    run_capacity_benchmark,
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
from .reporting import benchmark_raporu_kaydet, benchmark_raporu_markdown, benchmark_raporu_olustur
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

__all__ = [
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
    "ExperimentRun", "SeedSweepReport", "canonical_hash", "file_sha256",
    "run_seed_sweep", "seed_everything", "kronecker_capacity_contract",
    "run_kronecker_dense_trial",
    "CapacityReport", "capacity_contract", "measure_experience_capacity",
    "run_capacity_benchmark",
]
