"""Birleşik, manifestli HGA Research Benchmark Suite.

Suite mevcut benchmarkları ortak bir sözleşmede toplar; yeni bir kalite iddiası
üretmez. Her seed ayrı ``EXP-NNNN`` koşusudur. Optional neural/Kronecker
ölçümleri PyTorch yoksa sessizce başarı sayılmaz, açıkça ``SKIPPED`` olur.
"""
from __future__ import annotations

import html
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .compositional import CompositionalDataset, CompositionalReport, run_compositional_benchmark
from .english_ewt import prepare_english_ewt_task, run_english_ewt_baselines
from .experiment import SeedSweepReport, canonical_hash, run_seed_sweep
from .golden import GoldenDataset, run_golden_benchmark
from .real_turkish import TurkishWebTreebank, run_real_turkish_benchmark
from .verifier_adversarial import VerifierAttackDataset

SECTION_ORDER = (
    "architecture",
    "kronecker",
    "memory",
    "verification",
    "compositional_generalization",
    "neural_symbolic_hybrid",
    "self_learning",
    "ood",
    "turkish_nlp",
    "english_nlp",
)
SECTION_LABELS = {
    "architecture": "Architecture",
    "kronecker": "Kronecker",
    "memory": "Memory",
    "verification": "Verification",
    "compositional_generalization": "Compositional Generalization",
    "neural_symbolic_hybrid": "Neural / Symbolic / Hybrid",
    "self_learning": "Self Learning",
    "ood": "OOD",
    "turkish_nlp": "Turkish NLP",
    "english_nlp": "English NLP Controls",
    "reproducibility": "Reproducibility",
}
PROFILE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "kronecker": {"n": 4, "k": 2, "samples": 64, "steps": 8,
                        "batch_size": 4, "test_samples": 16},
        "memory": {
            "context_count": 256, "slot_count": 512, "table_count": 1,
            "forced_collisions": 8, "scaling_contexts": [64, 256, 1024],
            "lifecycle_capacity": 16,
        },
        "paradigm": {"entity_count": 40, "relation_count": 3, "train_size": 120,
                     "test_size": 60, "epochs": 5},
        "self_learning": {"cycles": 5, "batch_size": 16, "initial_facts": 4,
                          "operands_max": 7, "negatives_per_fact": 3,
                          "memory_slots": 128},
        "multi_environment": {"cycles": 5, "batch_per_environment": 8,
                              "memory_capacity": 256},
        "twt_baselines": {"profile": "smoke", "device": "cpu"},
        "twt_neural_compositional": {"profile": "smoke", "device": "cpu"},
    },
    "full": {
        "kronecker": {"n": 8, "k": 4, "samples": 256, "steps": 100,
                        "batch_size": 16, "test_samples": 128},
        "memory": {
            "context_count": 10_000, "slot_count": 16_384, "table_count": 1,
            "forced_collisions": 25,
            "scaling_contexts": [1_000, 10_000, 100_000, 1_000_000],
            "lifecycle_capacity": 256,
        },
        "paradigm": {"entity_count": 120, "relation_count": 8, "train_size": 1200,
                     "test_size": 400, "epochs": 60},
        "self_learning": {"cycles": 100, "batch_size": 64, "initial_facts": 100,
                          "operands_max": 31, "negatives_per_fact": 7,
                          "memory_slots": 4096},
        "multi_environment": {"cycles": 10, "batch_per_environment": 8,
                              "memory_capacity": 512},
        "twt_baselines": {"profile": "full", "device": "cpu"},
        "twt_neural_compositional": {"profile": "full", "device": "cpu"},
    },
}


class OptionalDependencyUnavailable(RuntimeError):
    """Bir bölümün opsiyonel çalışma zamanı bağımlılığı eksik."""


@dataclass
class ResearchBenchmarkReport:
    schema_version: int
    report_type: str
    generated_at_utc: str
    profile: str
    seeds: List[int]
    selected_sections: List[str]
    dataset_hash: str
    config_hash: str
    experiment_ids: List[str]
    manifests: List[Dict[str, Any]]
    sections: Dict[str, Dict[str, Any]]
    reproducibility: Dict[str, Any]
    overall_diagnostic_score: Optional[float]
    outcome: str
    seed_runs: List[Dict[str, Any]]
    limitations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _mean(values: Sequence[float]) -> float:
    return round(statistics.fmean(values), 8) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return round(statistics.pstdev(values), 8) if values else 0.0


def _score_checks(checks: Mapping[str, bool]) -> float:
    return _mean([1.0 if value else 0.0 for value in checks.values()]) if checks else 0.0


def _section(
    callback: Callable[[], Dict[str, Any]],
    optional: bool = False,
) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        result = dict(callback())
        checks = {str(key): bool(value) for key, value in result.get("checks", {}).items()}
        score = float(result.get("score", _score_checks(checks)))
        result["score"] = round(max(0.0, min(1.0, score)), 8)
        result["checks"] = checks
        result["status"] = "COMPLETED"
    except (ImportError, OptionalDependencyUnavailable) as error:
        if not optional:
            raise
        result = {
            "status": "SKIPPED",
            "score": None,
            "checks": {},
            "reason": str(error),
            "metrics": {},
            "limitations": ["Opsiyonel bağımlılık eksik; bu bölüm başarı sayılmadı."],
        }
    except Exception as error:  # Bölümler tek raporda birbirini gölgelememeli.
        result = {
            "status": "ERROR",
            "score": None,
            "checks": {},
            "error_type": type(error).__name__,
            "reason": str(error),
            "metrics": {},
            "limitations": ["Bölüm tamamlanamadı; sonuç kalite skoru olarak kullanılmadı."],
        }
    result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    return result


def _architecture_section() -> Dict[str, Any]:
    from mimari.model_config import yukle

    from .capacity import capacity_contract
    from .sweep import parametre_tahmini

    config = yukle()["model"]
    estimate = parametre_tahmini(
        config.n, config.katman_sayisi, config.baglam_penceresi,
        vocab=config.sozluk_boyutu, emb=config.emb_dim, heads=config.num_heads,
        seyrek_satir=config.seyrek_tablo_boyutu, seyrek_boyut=config.seyrek_boyut,
    )
    contract = capacity_contract(
        physical_parameters=estimate["tahmini_parametre"], n=config.n,
        k=config.katman_sayisi, vocab=config.sozluk_boyutu,
        window=config.baglam_penceresi,
    )
    checks = {
        "c_i_not_parameter_count": contract["c_i_is_parameter_count"] is False,
        "c_m_not_physical_table_size": contract["c_m_is_physical_table_size"] is False,
        "physical_parameter_estimate_positive": estimate["tahmini_parametre"] > 0,
        "sparse_and_dense_counts_separated": (
            estimate["seyrek_parametre"] > 0
            and estimate["tahmini_yogun_parametre"] > 0
        ),
    }
    return {
        "score": _score_checks(checks), "checks": checks,
        "headline": "Fiziksel parametreler ile teorik kapasite üst sınırları ayrıştırıldı.",
        "metrics": {
            "model_config": config.to_dict(),
            "estimated_physical_parameters": estimate["tahmini_parametre"],
            "estimated_dense_parameters": estimate["tahmini_yogun_parametre"],
            "sparse_parameters": estimate["seyrek_parametre"],
            "c_i_interaction_upper_bound": contract["c_i_interaction"],
            "c_m_conceptual_upper_bound": contract["c_m_conceptual"],
        },
        "limitations": [
            "Parametre sayısı modeli kurmadan formülle tahmin edilmiştir; kesin model introspection değildir.",
            "C_I^UB ve C_M^UB kalite ya da öğrenilebilir kapasite skoru değildir.",
        ],
    }


def _kronecker_section(config: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    try:
        import torch  # noqa: F401
    except ImportError as error:
        raise OptionalDependencyUnavailable("Kronecker ölçümleri için PyTorch gerekli") from error

    from .kronecker import run_kronecker_dense_trial
    from .kronecker_rank import measure_chain_collapse

    n, k, samples = int(config["n"]), int(config["k"]), int(config["samples"])
    linear = measure_chain_collapse(n=n, k=k, seed=seed, activation=None, samples=samples)
    nonlinear = measure_chain_collapse(n=n, k=k, seed=seed, activation="silu", samples=samples)
    trial = run_kronecker_dense_trial(
        n=n, steps=int(config["steps"]), batch_size=int(config["batch_size"]),
        test_samples=int(config["test_samples"]), seed=seed,
    )
    checks = {
        "equal_physical_parameter_budget": bool(
            trial["fairness"]["same_physical_parameter_budget"]
        ),
        "linear_chain_collapses": bool(linear.collapsed),
        "silu_breaks_linear_collapse": not bool(nonlinear.collapsed),
        "both_teacher_tasks_present": set(trial["tasks"]) == {
            "kronecker_teacher", "rank1_teacher"
        },
    }
    training_seconds = sum(
        model["training_seconds"]
        for task in trial["tasks"].values()
        for model in task["models"].values()
    )
    return {
        "score": _score_checks(checks), "checks": checks,
        "headline": "Çöküş sözleşmesi ve eşit-bütçeli iki-öğretmen kıyası tamamlandı.",
        "metrics": {
            "n": n, "k": k,
            "linear_collapse_residual": linear.collapse_residual,
            "silu_collapse_residual": nonlinear.collapse_residual,
            "linear_collapsed": linear.collapsed,
            "silu_collapsed": nonlinear.collapsed,
            "physical_parameter_budget_each": trial["capacity"]["physical_parameter_budget_each"],
            "winners_by_task": {
                name: value["winner_by_test_normalized_mse"]
                for name, value in trial["tasks"].items()
            },
            "training_seconds": round(training_seconds, 6),
            "trial": trial,
        },
        "limitations": [
            "Teacher görevleri sentetiktir; dil öğrenme üstünlüğü göstermez.",
            "Rank-1 bottleneck, unrestricted dense/Transformer baseline değildir.",
        ],
    }


def _memory_section(config: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    from hga.memory import (
        DYNAMIC_KV,
        FIRST_WINS,
        LAST_WINS,
        run_dynamic_kv_lifecycle,
        run_fixed_vs_dynamic_scaling,
        run_memory_benchmark,
        run_policy_comparison,
    )

    common = {
        "context_count": int(config["context_count"]),
        "slot_count": int(config["slot_count"]),
        "table_count": int(config["table_count"]),
    }
    result = run_memory_benchmark(seed=seed, **common)
    policy = run_policy_comparison(
        forced_collisions=int(config["forced_collisions"]),
        slot_count=common["slot_count"], table_count=common["table_count"],
    )
    scaling = run_fixed_vs_dynamic_scaling(
        context_counts=[int(value) for value in config["scaling_contexts"]],
        slot_count=common["slot_count"], table_count=common["table_count"], seed=seed,
    )
    lifecycle = run_dynamic_kv_lifecycle(
        seed=seed, capacity=int(config["lifecycle_capacity"])
    )
    checks = {
        "accounting_consistent": result.retained_contexts + result.interference_loss
        == result.context_count,
        "retrieval_bounded": 0.0 <= result.retrieval_accuracy <= 1.0,
        "collision_samples_bounded": result.collision_samples_stored <= 1000,
        "false_positive_rate_bounded": 0.0 <= result.false_positive_rate <= 1.0,
        "fixed_policies_cannot_retain_both": (
            policy.reports[FIRST_WINS]["both_survival_rate"] == 0.0
            and policy.reports[LAST_WINS]["both_survival_rate"] == 0.0
        ),
        "dynamic_kv_retains_both": (
            policy.best_both_survival == DYNAMIC_KV
            and policy.reports[DYNAMIC_KV]["both_survival_rate"] == 1.0
        ),
        "dynamic_kv_exact_recall_at_all_scales": all(
            point.dynamic_retrieval_accuracy == 1.0 for point in scaling.points
        ),
        "dynamic_cost_growth_reported": (
            scaling.points[-1].dynamic_storage_bytes
            > scaling.points[0].dynamic_storage_bytes
        ),
        **{
            f"active_lifecycle_{name}": passed
            for name, passed in lifecycle.checks.items()
        },
    }
    return {
        # Sabit belleğin zayıflığını dinamik KV kontrolüyle maskelememek için ana
        # diagnostic skor hâlâ mevcut fixed-table exact retrieval accuracy'dir.
        "score": result.retrieval_accuracy, "checks": checks,
        "headline": (
            "Fixed-table exact retrieval, aktif Engine DYNAMIC_KV yaşam döngüsü, "
            "adversarial politikalar ve ölçek maliyeti birlikte ölçüldü."
        ),
        "metrics": {
            "fixed_table": result.to_dict(),
            "policy_comparison": policy.to_dict(),
            "fixed_vs_dynamic_scaling": scaling.to_dict(),
            "active_dynamic_kv_lifecycle": lifecycle.to_dict(),
        },
        "limitations": [
            "Context akışı sentetiktir; semantic retrieval ölçülmez.",
            *lifecycle.limitations,
            "Diagnostic score fixed-table exact retrieval accuracy'dir; aktif Dynamic KV'nin exact lookup başarısı düşük fixed skoru maskelemez.",
        ],
    }


def _verification_section(seed: int) -> Dict[str, Any]:
    from hga.experience import run_verifier_fault_injection

    from .lifecycle import run_lifecycle_benchmark
    from .verifier_adversarial import run_verifier_adversarial_benchmark

    golden = run_golden_benchmark()
    adversarial = run_verifier_adversarial_benchmark()
    lifecycle = run_lifecycle_benchmark()
    control = run_verifier_fault_injection(
        sample_per_class=4, unknown_count=2, conflict_count=2,
        false_acceptance_rate=0.0, false_rejection_rate=0.0, seed=seed,
    )
    attacked = run_verifier_fault_injection(
        sample_per_class=4, unknown_count=2, conflict_count=2,
        false_acceptance_rate=0.25, false_rejection_rate=0.25, seed=seed,
    )
    checks = {
        "golden_leakage_clean": golden.leakage.clean,
        "control_far_zero": control.far == 0.0,
        "control_frr_zero": control.frr == 0.0,
        "attack_far_detected": attacked.far > 0.0,
        "attack_frr_detected": attacked.frr > 0.0,
        "unknown_and_conflict_preserved": (
            attacked.uncertain == attacked.truth_unknown
            and attacked.conflict == attacked.actual_conflicts
        ),
        "proof_attack_far_zero": adversarial.metrics.far == 0.0,
        "proof_attack_frr_zero": adversarial.metrics.frr == 0.0,
        "proof_attack_robustness_complete": adversarial.metrics.robustness == 1.0,
        "unsupported_rule_abstains": adversarial.metrics.uncertain > 0,
        **{f"knowledge_lifecycle_{key}": value for key, value in lifecycle.checks.items()},
    }
    return {
        "score": min(golden.metrics.accuracy, adversarial.metrics.accuracy),
        "checks": checks,
        "headline": (
            "Golden kararlar, verifier fault-injection, katı proof attack suite ve "
            "gerçek kaynak artifact'ında STALE yaşam döngüsü birlikte raporlandı."
        ),
        "metrics": {
            "golden": golden.to_dict(),
            "control": control.to_dict(),
            "adversarial_fault_injection": attacked.to_dict(),
            "proof_attack_suite": adversarial.to_dict(),
            "knowledge_lifecycle": lifecycle.to_dict(),
        },
        "limitations": [
            "Proof attack suite katı tam sayı toplama şemasıyla sınırlıdır; genel theorem prover değildir.",
            "Golden v1 küçüktür; üretim verifier soundness kanıtı değildir.",
            *lifecycle.limitations,
        ],
    }


def _compositional_section(report: CompositionalReport) -> Dict[str, Any]:
    checks = {
        "semantic_leakage_clean": report.leakage.clean,
        "surface_leakage_clean": report.surface_leakage_clean,
        "candidate_generation_complete": report.metrics.candidate_generation_coverage == 1.0,
        "ood_unknown_not_false": report.dimensions["ood"].accuracy == 1.0,
    }
    return {
        "score": report.generalization_capacity.score, "checks": checks,
        "headline": (
            f"C_G={report.generalization_capacity.score:.3f} "
            f"({report.generalization_capacity.correctly_generalized}/"
            f"{report.generalization_capacity.eligible_cases}); teorik kapasite değildir."
        ),
        "metrics": report.to_dict(),
        "limitations": list(report.limitations),
    }


def _paradigm_section(config: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    try:
        import torch  # noqa: F401
    except ImportError as error:
        raise OptionalDependencyUnavailable("Neural/symbolic/hybrid bölümü için PyTorch gerekli") from error

    from .paradigma import run_paradigm_ablation

    report = run_paradigm_ablation(seed=seed, **{key: int(value) for key, value in config.items()})
    symbolic = report.arms["symbolic"]
    neural = report.arms["neural"]
    hybrid = report.arms["hybrid"]
    checks = {
        "same_test_count": symbolic["total"] == neural["total"] == hybrid["total"],
        "symbolic_abstention_visible": symbolic["coverage"] < 1.0,
        "all_error_metrics_present": all(
            key in hybrid for key in ("precision", "recall", "f1", "far", "frr")
        ),
        "hybrid_full_coverage": hybrid["coverage"] == 1.0,
    }
    return {
        "score": hybrid["accuracy"], "checks": checks,
        "headline": "Üç paradigma aynı split ve metrik sözleşmesinde karşılaştırıldı.",
        "metrics": report.to_dict(),
        "limitations": [
            "Görev prosedürel ve sentetiktir; Transformer baseline içermez.",
            "Smoke profilindeki az epoch yayınlanabilir model kıyası değildir.",
        ],
    }


def _self_learning_section(
    config: Mapping[str, Any], multi_config: Mapping[str, Any], seed: int
) -> Dict[str, Any]:
    from hga.experience import (
        run_multi_environment_self_learning,
        run_self_learning_experiment,
    )

    report = run_self_learning_experiment(
        seed=seed, **{key: int(value) for key, value in config.items()}
    )
    multi = run_multi_environment_self_learning(
        seed=seed, **{key: int(value) for key, value in multi_config.items()}
    )
    knowledge_accuracy = report.correct_knowledge / max(
        1, report.correct_knowledge + report.incorrect_knowledge
    )
    checks = {
        "holdout_isolation_clean": report.isolation_clean,
        "no_incorrect_durable_knowledge": report.incorrect_knowledge == 0,
        "far_zero_after_verifier": report.far == 0.0,
        "frr_zero_after_verifier": report.frr == 0.0,
        **{
            f"multi_environment_{name}": passed
            for name, passed in multi.checks.items()
        },
    }
    return {
        "score": round(knowledge_accuracy, 8), "checks": checks,
        "headline": (
            "Kapalı aritmetik kontrol ile arithmetic/logic/consistency shared "
            "store+DynamicKV self-learning izolasyonu birlikte ölçüldü."
        ),
        "metrics": {**report.to_dict(), "multi_environment": multi.to_dict()},
        "limitations": [*report.limitations, *multi.limitations],
    }


def _ood_section(report: CompositionalReport) -> Dict[str, Any]:
    metric = report.dimensions["ood"]
    checks = {
        "ood_cases_present": metric.total > 0,
        "unknown_separated_from_false": metric.uncertain_total > 0,
        "ood_expected_state_accuracy": metric.accuracy == 1.0,
    }
    return {
        "score": metric.accuracy, "checks": checks,
        "headline": "Kanıtı olmayan OOD nesne FALSE yerine UNCERTAIN olarak ölçüldü.",
        "metrics": asdict(metric),
        "limitations": [
            "OOD bölümü tek küçük ontoloji-dışı özellik vakasıdır; dağılım kayması benchmarkı değildir."
        ],
    }


def _turkish_section(
    report: CompositionalReport,
    seed: int,
    baseline_config: Mapping[str, Any],
    ablation_config: Mapping[str, Any],
) -> Dict[str, Any]:
    from mimari.bpe_tokenizer import BPETokenizer

    from .neural_compositional import run_neural_compositional_ablation
    from .turkish_benchmark import mini_turkce_corpus, tokenizer_kapsami
    from .twt_baselines import run_twt_architecture_baselines

    real = run_real_turkish_benchmark(seed=seed)
    baselines = run_twt_architecture_baselines(
        seed=seed,
        profile=str(baseline_config["profile"]),
        device=str(baseline_config["device"]),
    )
    neural_compositional = run_neural_compositional_ablation(
        seed=seed,
        profile=str(ablation_config["profile"]),
        device=str(ablation_config["device"]),
    )
    corpus = mini_turkce_corpus()
    tokenizer = BPETokenizer(baglam_penceresi=16, max_vocab_size=512, min_freq=1)
    tokenizer.fit_on_text("\n".join(corpus), verbose=False)
    coverage = tokenizer_kapsami(tokenizer, corpus)
    parsing_accuracy = report.metrics.parsing_accuracy
    checks = {
        **{f"twt_{key}": value for key, value in real.checks.items()},
        **{f"baseline_{key}": value for key, value in baselines.checks.items()},
        **{
            f"neural_compositional_{key}": value
            for key, value in neural_compositional.checks.items()
        },
        "turkish_characters_roundtrip": bool(coverage["turkce_karakter_tam"]),
        "unk_rate_below_five_percent": float(coverage["unk_orani"]) < 0.05,
        "controlled_heldout_wording_cases_present": (
            report.dimensions["unseen_wording"].total > 0
        ),
    }
    return {
        # Ana skor gerçek TWT testindeki binary arc-verification F1'ıdır. Mini
        # tokenizer ve kontrollü parser smoke sonuçları bu skoru şişirmez.
        "score": real.metrics.f1,
        "checks": checks,
        "headline": (
            f"TWT: {real.corpus['effective_sentences']} gerçek, insan anotasyonlu "
            f"cümle; schema F1={real.metrics.f1:.3f}; dört neural kol fiziksel "
            f"parametre oranı={baselines.fairness['parameter_max_to_min_ratio']:.4f}; "
            f"HGA C_G_N={neural_compositional.arms['full']['neural_generalization']['score']:.3f}."
        ),
        "metrics": {
            "real_turkish_twt": real.to_dict(),
            "parameter_matched_architectures": baselines.to_dict(),
            "neural_compositional_ablation": neural_compositional.to_dict(),
            "training_seconds": round(
                sum(
                    float(model["training_seconds"])
                    for model in baselines.models.values()
                )
                + sum(
                    float(arm["training_seconds"])
                    for arm in neural_compositional.arms.values()
                ),
                6,
            ),
            "tokenizer_smoke": coverage,
            "controlled_compositional_parsing_accuracy": parsing_accuracy,
            "controlled_unseen_wording": asdict(report.dimensions["unseen_wording"]),
        },
        "limitations": [
            *real.limitations,
            *baselines.limitations,
            *neural_compositional.limitations,
            "Tokenizer ölçümü küçük smoke corpusundadır ve TWT test skoruna katılmaz.",
            "Kontrollü lexicon parser sonucu TWT dependency sonucu olarak sunulmaz.",
        ],
    }



def _english_section(seed: int, profile: str) -> Dict[str, Any]:
    """Pinned English UD EWT architecture controls for one suite seed.

    The research runner supplies five independent EXP manifests.  Therefore the
    single-run report's local ``five_or_more_seeds`` flag is intentionally
    replaced by an explicit contribution marker; the aggregate suite's
    reproducibility section is the authority for the five-seed gate.
    """
    report = run_english_ewt_baselines(seeds=(int(seed),), profile=profile)
    raw = report.to_dict()
    checks = {
        name: bool(value)
        for name, value in raw["checks"].items()
        if name != "five_or_more_seeds"
    }
    checks["contributes_one_seed_to_suite_aggregate"] = True
    f1s = {name: raw["aggregate"][name]["f1"]["mean"] for name in raw["aggregate"]}
    return {
        "status": "COMPLETED",
        "score": _score_checks(checks),
        "checks": checks,
        "headline": "Pinned English UD EWT: dense/Transformer/BERT-style/GPT-style controls.",
        "metrics": {
            "english_ewt": raw,
            "f1_by_model": f1s,
            "training_seconds": round(sum(
                float(model["training_seconds"])
                for model in raw["per_seed"][0]["models"].values()
            ), 6),
        },
        "limitations": list(raw["limitations"]),
    }

def _run_trial(
    seed: int,
    profile: str,
    selected_sections: Sequence[str],
) -> Dict[str, Any]:
    config = PROFILE_CONFIGS[profile]
    selected = set(selected_sections)
    sections: Dict[str, Dict[str, Any]] = {}
    compositional: Optional[CompositionalReport] = None

    def compositional_report() -> CompositionalReport:
        nonlocal compositional
        if compositional is None:
            compositional = run_compositional_benchmark()
        return compositional

    runners: Dict[str, Tuple[Callable[[], Dict[str, Any]], bool]] = {
        "architecture": (_architecture_section, False),
        "kronecker": (lambda: _kronecker_section(config["kronecker"], seed), True),
        "memory": (lambda: _memory_section(config["memory"], seed), False),
        "verification": (lambda: _verification_section(seed), False),
        "compositional_generalization": (
            lambda: _compositional_section(compositional_report()), False
        ),
        "neural_symbolic_hybrid": (
            lambda: _paradigm_section(config["paradigm"], seed), True
        ),
        "self_learning": (
            lambda: _self_learning_section(
                config["self_learning"], config["multi_environment"], seed
            ),
            False,
        ),
        "ood": (lambda: _ood_section(compositional_report()), False),
        "turkish_nlp": (
            lambda: _turkish_section(
                compositional_report(),
                seed,
                config["twt_baselines"],
                config["twt_neural_compositional"],
            ),
            False,
        ),
        "english_nlp": (lambda: _english_section(seed, profile), True),
    }
    started = time.perf_counter()
    for name in SECTION_ORDER:
        if name in selected:
            callback, optional = runners[name]
            sections[name] = _section(callback, optional=optional)
            print(
                f"  {SECTION_LABELS[name]:35s} {sections[name]['status']:9s} "
                f"score={sections[name]['score']}"
            )
    elapsed = round(time.perf_counter() - started, 6)
    training_seconds = sum(
        float(section.get("metrics", {}).get("training_seconds", 0.0) or 0.0)
        for section in sections.values()
    )
    return {
        "seed": int(seed),
        "profile": profile,
        "sections": sections,
        "timings": {
            "total_seconds": elapsed,
            "training_seconds": round(training_seconds, 6),
            "inference_seconds": None,
        },
    }


def _multi_environment_summary(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    multi_runs = [
        run["metrics"]["multi_environment"]
        for run in runs
        if run.get("status") == "COMPLETED"
        and "multi_environment" in run.get("metrics", {})
    ]
    if not multi_runs:
        return {}
    environments: Dict[str, Any] = {}
    for name in ("arithmetic", "logic", "consistency"):
        environment: Dict[str, Any] = {}
        for metric_name in (
            "generated", "verified", "rejected", "false_acceptance",
            "false_rejection", "durable_new_knowledge",
            "incorrect_durable_knowledge", "memory_retrieval_accuracy",
        ):
            values = [
                float(run["per_environment"][name][metric_name])
                for run in multi_runs
            ]
            environment[metric_name] = {
                "mean": _mean(values),
                "std": _std(values),
                "min": min(values),
                "max": max(values),
            }
        environments[name] = environment
    shared_final = [float(run["shared_store_final_facts"]) for run in multi_runs]
    return {
        "protocol": multi_runs[0]["protocol"],
        "seed_count": len(multi_runs),
        "seeds": [run["seed"] for run in multi_runs],
        "environments": environments,
        "shared_store_final_facts": {
            "mean": _mean(shared_final),
            "std": _std(shared_final),
            "min": min(shared_final),
            "max": max(shared_final),
        },
        "cross_verifier_acceptances_all_seeds": sum(
            int(run["cross_verifier_acceptances"]) for run in multi_runs
        ),
        "all_checks_all_seeds": all(
            all(run["checks"].values()) for run in multi_runs
        ),
    }


def _twt_architecture_summary(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    baseline_runs = [
        run["metrics"]["parameter_matched_architectures"]
        for run in runs
        if run.get("status") == "COMPLETED"
    ]
    if not baseline_runs:
        return {}
    metric_names = ("accuracy", "precision", "recall", "f1", "far", "frr", "coverage")
    model_names = ("dense", "transformer", "kronecker", "hga")
    models: Dict[str, Any] = {}
    for model_name in model_names:
        dimension_names = tuple(baseline_runs[0]["models"][model_name]["test"])
        dimensions: Dict[str, Any] = {}
        for dimension in dimension_names:
            dimensions[dimension] = {
                metric: {
                    "mean": _mean([
                        float(run["models"][model_name]["test"][dimension][metric])
                        for run in baseline_runs
                    ]),
                    "std": _std([
                        float(run["models"][model_name]["test"][dimension][metric])
                        for run in baseline_runs
                    ]),
                    "min": min(
                        float(run["models"][model_name]["test"][dimension][metric])
                        for run in baseline_runs
                    ),
                    "max": max(
                        float(run["models"][model_name]["test"][dimension][metric])
                        for run in baseline_runs
                    ),
                }
                for metric in metric_names
            }
        training_values = [
            float(run["models"][model_name]["training_seconds"])
            for run in baseline_runs
        ]
        calibration_metrics = (
            "temperature", "ece_before", "ece_after", "adaptive_ece_before",
            "adaptive_ece_after", "nll_before", "nll_after", "brier_before",
            "brier_after", "aurc_after",
        )
        calibration_values: Dict[str, List[float]] = {
            metric: [] for metric in calibration_metrics
        }
        for run in baseline_runs:
            calibration = run["models"][model_name]["calibration"]
            before = calibration["slices"]["all"]["before"]
            after = calibration["slices"]["all"]["after"]
            calibration_values["temperature"].append(float(calibration["fit"]["temperature"]))
            for metric in ("ece", "adaptive_ece", "nll", "brier"):
                calibration_values[f"{metric}_before"].append(float(before[metric]))
                calibration_values[f"{metric}_after"].append(float(after[metric]))
            calibration_values["aurc_after"].append(float(after["aurc"]))
        models[model_name] = {
            "physical_parameters": baseline_runs[0]["models"][model_name][
                "physical_parameters"
            ],
            "architecture_body_parameters": baseline_runs[0]["models"][model_name][
                "architecture_body_parameters"
            ],
            "implementation": baseline_runs[0]["models"][model_name]["implementation"],
            "training_seconds": {
                "mean": _mean(training_values),
                "std": _std(training_values),
                "min": min(training_values),
                "max": max(training_values),
            },
            "calibration": {
                metric: {
                    "mean": _mean(values),
                    "std": _std(values),
                    "min": min(values),
                    "max": max(values),
                }
                for metric, values in calibration_values.items()
            },
            "test": dimensions,
            "per_seed": [
                {
                    "seed": run["seed"],
                    "accuracy": run["models"][model_name]["test"]["all"]["accuracy"],
                    "f1": run["models"][model_name]["test"]["all"]["f1"],
                }
                for run in baseline_runs
            ],
        }
    return {
        "protocol": baseline_runs[0]["protocol"],
        "seed_count": len(baseline_runs),
        "seeds": [run["seed"] for run in baseline_runs],
        "dataset_hash": baseline_runs[0]["dataset_hash"],
        "split_hashes": baseline_runs[0]["split_hashes"],
        "candidate_hashes": baseline_runs[0]["candidate_hashes"],
        "fairness_reference": baseline_runs[0]["fairness"],
        "all_checks_all_seeds": all(
            all(run["checks"].values()) for run in baseline_runs
        ),
        "models": models,
        "note": (
            "Mean/std aynı gerçek TWT test splitindeki seed koşularıdır; mimari "
            "üstünlük veya genel dil/SOTA sonucu değildir."
        ),
    }


def _twt_neural_compositional_summary(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ablation_runs = [
        run["metrics"]["neural_compositional_ablation"]
        for run in runs
        if run.get("status") == "COMPLETED"
    ]
    if not ablation_runs:
        return {}
    arms: Dict[str, Any] = {}
    for arm_name in ("full", "no_attention", "additive_geometry", "no_kronecker_chain"):
        c_g_values = [
            float(run["arms"][arm_name]["neural_generalization"]["score"])
            for run in ablation_runs
        ]
        f1_values = [
            float(run["arms"][arm_name]["test"]["all"]["f1"])
            for run in ablation_runs
        ]
        accuracy_values = [
            float(run["arms"][arm_name]["test"]["all"]["accuracy"])
            for run in ablation_runs
        ]
        arms[arm_name] = {
            "physical_parameters": ablation_runs[0]["arms"][arm_name][
                "physical_parameters"
            ],
            "removed_component": ablation_runs[0]["arms"][arm_name][
                "removed_component"
            ],
            "c_g_n": {
                "mean": _mean(c_g_values),
                "std": _std(c_g_values),
                "min": min(c_g_values),
                "max": max(c_g_values),
            },
            "all_accuracy": {
                "mean": _mean(accuracy_values),
                "std": _std(accuracy_values),
                "min": min(accuracy_values),
                "max": max(accuracy_values),
            },
            "all_f1": {
                "mean": _mean(f1_values),
                "std": _std(f1_values),
                "min": min(f1_values),
                "max": max(f1_values),
            },
            "per_seed": [
                {
                    "seed": run["seed"],
                    "c_g_n": run["arms"][arm_name]["neural_generalization"]["score"],
                    "all_f1": run["arms"][arm_name]["test"]["all"]["f1"],
                }
                for run in ablation_runs
            ],
        }
    delta_summary: Dict[str, Any] = {}
    for arm_name in ("no_attention", "additive_geometry", "no_kronecker_chain"):
        values = [
            float(run["deltas_from_full"][arm_name]["full_minus_arm_c_g_n"])
            for run in ablation_runs
        ]
        delta_summary[arm_name] = {
            "full_minus_arm_c_g_n_mean": _mean(values),
            "full_minus_arm_c_g_n_std": _std(values),
        }
    return {
        "protocol": ablation_runs[0]["protocol"],
        "seed_count": len(ablation_runs),
        "seeds": [run["seed"] for run in ablation_runs],
        "dataset_hash": ablation_runs[0]["dataset_hash"],
        "candidate_hashes": ablation_runs[0]["candidate_hashes"],
        "all_checks_all_seeds": all(
            all(run["checks"].values()) for run in ablation_runs
        ),
        "arms": arms,
        "delta_summary": delta_summary,
        "definition": (
            "C_G_N = gerçek TWT composition_disjoint dengeli arc testindeki accuracy; "
            "teorik kapasite veya kontrollü fixture C_G değildir."
        ),
    }


def _summarize_sections(
    results: Sequence[Mapping[str, Any]], selected_sections: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for name in selected_sections:
        runs = [result["sections"][name] for result in results]
        statuses = [run["status"] for run in runs]
        scores = [float(run["score"]) for run in runs if run.get("score") is not None]
        if all(status == "SKIPPED" for status in statuses):
            status = "SKIPPED"
        elif any(status == "ERROR" for status in statuses):
            status = "ERROR"
        elif any(status == "SKIPPED" for status in statuses):
            status = "PARTIAL"
        else:
            status = "COMPLETED"
        check_names = sorted({key for run in runs for key in run.get("checks", {})})
        checks = {
            key: all(run.get("checks", {}).get(key, False) for run in runs
                     if run["status"] == "COMPLETED")
            for key in check_names
        }
        first_completed = next((run for run in runs if run["status"] == "COMPLETED"), runs[0])
        summary[name] = {
            "label": SECTION_LABELS[name],
            "status": status,
            "score_mean": _mean(scores) if scores else None,
            "score_std": _std(scores) if scores else None,
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
            "checks_all_seeds": checks,
            "headline": first_completed.get("headline"),
            "metrics_reference_seed": first_completed.get("metrics", {}),
            "limitations": list(first_completed.get("limitations", [])),
            "skip_or_error_reasons": sorted({
                str(run.get("reason")) for run in runs if run.get("reason")
            }),
        }
        if name == "self_learning":
            summary[name]["multi_environment_summary"] = (
                _multi_environment_summary(runs)
            )
        if name == "turkish_nlp":
            summary[name]["architecture_summary"] = _twt_architecture_summary(runs)
            summary[name]["neural_compositional_summary"] = (
                _twt_neural_compositional_summary(runs)
            )
    return summary


def _reproducibility(
    sweep: SeedSweepReport,
    sections: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    dispersion: Dict[str, Dict[str, Any]] = {}
    for name, section in sections.items():
        mean = section.get("score_mean")
        std = section.get("score_std")
        relative = None
        if mean not in (None, 0.0) and std is not None:
            relative = round(float(std) / abs(float(mean)), 8)
        dispersion[name] = {
            "mean": mean, "std": std, "relative_std": relative,
            "stable_within_10_percent": relative is None or relative <= 0.10,
        }
    completed = all(manifest.get("result") == "COMPLETED" for manifest in sweep.manifests)
    hashes_same = len({manifest.get("dataset_hash") for manifest in sweep.manifests}) == 1
    return {
        "label": SECTION_LABELS["reproducibility"],
        "seed_count": len(sweep.seeds),
        "five_or_more_seeds": len(sweep.seeds) >= 5,
        "all_manifests_completed": completed,
        "dataset_hash_identical_across_seeds": hashes_same,
        "byte_identical_results": sweep.deterministic_results,
        "score_dispersion": dispersion,
        "note": (
            "Byte-identical sonuç zorunlu değildir: stochastic modeller ve süre metrikleri değişir. "
            "Bilimsel değerlendirme seed-bazlı skor dağılımını kullanır."
        ),
    }


def _dataset_hash() -> str:
    return canonical_hash({
        "compositional_tr_v1": CompositionalDataset().dataset_hash(),
        "golden_v1": GoldenDataset().dataset_hash(),
        "verifier_adversarial_v1": VerifierAttackDataset().dataset_hash(),
        "real_turkish_twt_v1": TurkishWebTreebank().dataset_hash(),
        "real_english_ewt_v1": prepare_english_ewt_task().dataset_hash,
        "procedural_protocols": {
            "kronecker": "kronecker-vs-rank1-param-matched-v2",
            "memory": "sparse-memory-collision-v1",
            "self_learning": "closed-verified-arithmetic-v2",
            "paradigm": "neural-symbolic-hybrid-v1",
            "twt_architectures": "twt-parameter-matched-architectures-v1",
            "twt_neural_compositional": "twt-neural-compositional-ablation-v1",
            "english_architectures": "hga-english-ud-ewt-architecture-baselines-v1",
            "active_memory": "active-dynamic-kv-lifecycle-v1",
            "multi_environment": "multi-environment-closed-verified-self-learning-v1",
            "uncertainty_calibration": "dev-temperature-scaling-selective-risk-v1",
            "knowledge_lifecycle": "real-artifact-knowledge-lifecycle-v1",
        },
    })


def validate_sections(sections: Optional[Sequence[str]]) -> List[str]:
    if sections is None:
        return list(SECTION_ORDER)
    normalized: List[str] = []
    for name in sections:
        clean = str(name).strip().lower().replace("-", "_")
        if not clean:
            continue
        if clean not in SECTION_ORDER:
            raise ValueError(
                f"Bilinmeyen research bölümü '{name}'. Geçerli: {', '.join(SECTION_ORDER)}"
            )
        if clean not in normalized:
            normalized.append(clean)
    if not normalized:
        raise ValueError("En az bir research bölümü seçilmeli")
    return [name for name in SECTION_ORDER if name in normalized]


def run_research_benchmark(
    root: Union[str, os.PathLike] = "experiments",
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    profile: str = "smoke",
    sections: Optional[Sequence[str]] = None,
) -> ResearchBenchmarkReport:
    """Tüm seçili bölümleri her seed için ayrı manifestli koşuda çalıştır."""
    if profile not in PROFILE_CONFIGS:
        raise ValueError(f"profile şunlardan biri olmalı: {', '.join(PROFILE_CONFIGS)}")
    selected = validate_sections(sections)
    normalized_seeds = [int(seed) for seed in seeds]
    dataset_hash = _dataset_hash()
    config = {
        "benchmark": "hga-research-benchmark-v1",
        "profile": profile,
        "sections": selected,
        "profile_config": PROFILE_CONFIGS[profile],
        "seeds": normalized_seeds,
    }
    # Manifestteki sayı formül-tahminidir; Architecture bölümü bu sınırı açıklar.
    from mimari.model_config import yukle

    from .sweep import parametre_tahmini

    model_config = yukle()["model"]
    parameter_estimate = parametre_tahmini(
        model_config.n, model_config.katman_sayisi, model_config.baglam_penceresi,
        vocab=model_config.sozluk_boyutu, emb=model_config.emb_dim,
        heads=model_config.num_heads, seyrek_satir=model_config.seyrek_tablo_boyutu,
        seyrek_boyut=model_config.seyrek_boyut,
    )["tahmini_parametre"]
    sweep = run_seed_sweep(
        callback=lambda seed: _run_trial(seed, profile, selected),
        seeds=normalized_seeds, root=root, config=config, dataset_hash=dataset_hash,
        parameters={
            "parameter_count": parameter_estimate,
            "parameter_count_kind": "formula_estimate_from_model_config",
            "profile": profile,
            "sections": selected,
            "human_curated_fixture": True,
            "real_turkish_dataset": "Turkish Web Treebank",
            "real_turkish_upstream_revision": (
                "40838e5cbe3f2882d4e768a3d782e6219e50b52a"
            ),
            "real_turkish_human_annotated": True,
            "real_turkish_license": "Apache-2.0",
            "real_turkish_task": "basic morphosyntactic dependency-arc verification",
            "real_english_dataset": "UD English EWT",
            "real_english_upstream_revision": "4a4d77f599ea53cc405f85d0cec4b2f14f81d42b",
            "real_english_license": "CC-BY-SA-4.0",
            "real_english_task": "basic morphosyntactic dependency-arc verification",
        },
    )
    section_summary = _summarize_sections(sweep.results, selected)
    reproducibility = _reproducibility(sweep, section_summary)
    completed_scores = [
        float(section["score_mean"])
        for section in section_summary.values()
        if section["status"] == "COMPLETED" and section["score_mean"] is not None
    ]
    errors = any(section["status"] == "ERROR" for section in section_summary.values())
    skips = any(section["status"] in ("SKIPPED", "PARTIAL")
                for section in section_summary.values())
    outcome = "COMPLETED"
    if errors:
        outcome = "COMPLETED_WITH_ERRORS"
    elif skips:
        outcome = "COMPLETED_WITH_SKIPS"
    limitations = [
        "overall_diagnostic_score, tamamlanan heterojen bölüm skorlarının basit ortalamasıdır; zekâ veya SOTA skoru değildir.",
        "Smoke profil CI/protokol doğrulaması içindir; yayınlanabilir ölçek sonucu değildir.",
        "Sentetik bölümler, kontrollü compositional fixture ve gerçek insan-anotasyonlu TWT sonucu raporda ayrı tutulur.",
        "TWT ve EWT görevleri morphosyntactic dependency arc doğrulamasıdır; semantik relation extraction veya genel dil iddiası değildir.",
        "English kontrolü BERT-style/GPT-style küçük sıfırdan-eğitilmiş mimarilerdir; pretrained LLM kıyası değildir.",
        "SKIPPED bölümler başarı sayılmaz ve overall_diagnostic_score hesabına girmez.",
    ]
    return ResearchBenchmarkReport(
        schema_version=1, report_type="hga-research-benchmark-v1",
        generated_at_utc=datetime.now(timezone.utc).isoformat(), profile=profile,
        seeds=normalized_seeds, selected_sections=selected,
        dataset_hash=dataset_hash, config_hash=canonical_hash(config),
        experiment_ids=list(sweep.experiment_ids), manifests=list(sweep.manifests),
        sections=section_summary, reproducibility=reproducibility,
        overall_diagnostic_score=_mean(completed_scores) if completed_scores else None,
        outcome=outcome, seed_runs=list(sweep.results), limitations=limitations,
    )


def research_report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# HGA Research Benchmark Report",
        "",
        f"- Rapor tipi: `{report['report_type']}`",
        f"- Profil: `{report['profile']}`",
        f"- Sonuç: `{report['outcome']}`",
        f"- Seedler: `{', '.join(str(seed) for seed in report['seeds'])}`",
        f"- Dataset hash: `{report['dataset_hash']}`",
        f"- Config hash: `{report['config_hash']}`",
        f"- Deneyler: `{', '.join(report['experiment_ids'])}`",
        f"- Genel diagnostic skor: `{report.get('overall_diagnostic_score')}`",
        "",
        "> Diagnostic skor bir zekâ/SOTA skoru değildir; heterojen protokol sağlık ve görev metriklerini tek görünümde toplar.",
        "",
        "| Bölüm | Durum | Skor mean ± std | Kontroller |",
        "|---|---|---:|---:|",
    ]
    for name in report["selected_sections"]:
        section = report["sections"][name]
        score = "n/a"
        if section["score_mean"] is not None:
            score = f"{section['score_mean']:.3f} ± {section['score_std']:.3f}"
        checks = section.get("checks_all_seeds", {})
        passed = sum(bool(value) for value in checks.values())
        lines.append(
            f"| {section['label']} | {section['status']} | {score} | {passed}/{len(checks)} |"
        )
    reproduction = report["reproducibility"]
    lines.extend([
        f"| Reproducibility | {'COMPLETED' if reproduction['all_manifests_completed'] else 'ERROR'} | "
        f"{reproduction['seed_count']} seed | "
        f"{'5+ seed' if reproduction['five_or_more_seeds'] else '<5 seed'} |",
        "",
        "## Bölüm Bulguları",
        "",
    ])
    for name in report["selected_sections"]:
        section = report["sections"][name]
        lines.extend([f"### {section['label']}", ""])
        if section.get("headline"):
            lines.append(section["headline"])
            lines.append("")
        for key, value in section.get("checks_all_seeds", {}).items():
            lines.append(f"- {'PASS' if value else 'FAIL'} — `{key}`")
        for reason in section.get("skip_or_error_reasons", []):
            lines.append(f"- SKIP/ERROR — {reason}")
        if name == "self_learning" and section["status"] == "COMPLETED":
            multi = section.get("multi_environment_summary", {})
            if multi:
                lines.extend([
                    "",
                    "Multi-environment closed self-learning (mean ± std):",
                    "",
                    "| Environment | Generated | Verified | Rejected | FAR | FRR | "
                    "Durable growth | Incorrect durable | Memory recall |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ])
                for environment_name in ("arithmetic", "logic", "consistency"):
                    environment = multi["environments"][environment_name]
                    def cell(metric_name):
                        metric = environment[metric_name]
                        return f"{metric['mean']:.4f} ± {metric['std']:.4f}"
                    lines.append(
                        f"| {environment_name} | {cell('generated')} | "
                        f"{cell('verified')} | {cell('rejected')} | "
                        f"{cell('false_acceptance')} | {cell('false_rejection')} | "
                        f"{cell('durable_new_knowledge')} | "
                        f"{cell('incorrect_durable_knowledge')} | "
                        f"{cell('memory_retrieval_accuracy')} |"
                    )
                lines.extend([
                    "",
                    f"- Shared final facts: `{multi['shared_store_final_facts']['mean']:.4f} "
                    f"± {multi['shared_store_final_facts']['std']:.4f}`",
                    f"- Cross-verifier acceptances (all seeds): "
                    f"`{multi['cross_verifier_acceptances_all_seeds']}`",
                    f"- Tüm seed kapıları: `{multi['all_checks_all_seeds']}`",
                ])
        if name == "verification" and section["status"] == "COMPLETED":
            lifecycle = section["metrics_reference_seed"].get("knowledge_lifecycle", {})
            if lifecycle:
                lines.extend([
                    "",
                    "Real-artifact STALE lifecycle:",
                    "",
                    f"- Protocol: `{lifecycle['protocol']}`",
                    f"- Source revision: `{lifecycle['real_source']['upstream_revision']}`",
                    f"- Final states: `{lifecycle['final_counts']}`",
                    f"- Hash-chained events: `{lifecycle['event_count']}`; "
                    f"head `{lifecycle['event_chain_head']}`",
                    "- STALE is freshness/dependency uncertainty; it is not FALSE or RETRACTED.",
                ])
        if name == "compositional_generalization" and section["status"] == "COMPLETED":
            metrics = section["metrics_reference_seed"]
            capacity = metrics["generalization_capacity"]
            lines.extend([
                "",
                f"- **C_G:** `{capacity['correctly_generalized']}/{capacity['eligible_cases']}` "
                f"(`{capacity['score']:.3f}`)",
                f"- Üretim exact: `{capacity['exactly_generated']}/{capacity['generation_targets']}`",
                f"- Tanım: {capacity['definition']}",
            ])
        if name == "turkish_nlp" and section["status"] == "COMPLETED":
            architecture = section.get("architecture_summary", {})
            if architecture:
                lines.extend([
                    "",
                    "Parameter-matched TWT architecture test özeti:",
                    "",
                    "| Model | Parametre | Body parametre | Accuracy mean ± std | "
                    "F1 mean ± std | FAR mean | FRR mean | Coverage |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ])
                for model_name in ("dense", "transformer", "kronecker", "hga"):
                    model = architecture["models"][model_name]
                    metric = model["test"]["all"]
                    lines.append(
                        f"| {model_name} | {model['physical_parameters']} | "
                        f"{model['architecture_body_parameters']} | "
                        f"{metric['accuracy']['mean']:.4f} ± {metric['accuracy']['std']:.4f} | "
                        f"{metric['f1']['mean']:.4f} ± {metric['f1']['std']:.4f} | "
                        f"{metric['far']['mean']:.4f} | {metric['frr']['mean']:.4f} | "
                        f"{metric['coverage']['mean']:.4f} |"
                    )
                lines.extend([
                    "",
                    f"- Seedler: `{architecture['seeds']}`",
                    f"- Dataset hash: `{architecture['dataset_hash']}`",
                    f"- Not: {architecture['note']}",
                    "",
                    "Dev-only temperature calibration (test all; mean ± std):",
                    "",
                    "| Model | Temperature | ECE before | ECE after | NLL before | "
                    "NLL after | Brier after | AURC after |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ])
                for model_name in ("dense", "transformer", "kronecker", "hga"):
                    calibration = architecture["models"][model_name]["calibration"]
                    def calibration_cell(metric_name):
                        metric = calibration[metric_name]
                        return f"{metric['mean']:.4f} ± {metric['std']:.4f}"
                    lines.append(
                        f"| {model_name} | {calibration_cell('temperature')} | "
                        f"{calibration_cell('ece_before')} | "
                        f"{calibration_cell('ece_after')} | "
                        f"{calibration_cell('nll_before')} | "
                        f"{calibration_cell('nll_after')} | "
                        f"{calibration_cell('brier_after')} | "
                        f"{calibration_cell('aurc_after')} |"
                    )
            neural_compositional = section.get("neural_compositional_summary", {})
            if neural_compositional:
                lines.extend([
                    "",
                    "Neural compositional HGA ablation özeti:",
                    "",
                    "| Kol | Parametre | C_G_N mean ± std | All F1 mean ± std | "
                    "Çıkarılan/değiştirilen bileşen |",
                    "|---|---:|---:|---:|---|",
                ])
                for arm_name in (
                    "full", "no_attention", "additive_geometry", "no_kronecker_chain"
                ):
                    arm = neural_compositional["arms"][arm_name]
                    removed = arm["removed_component"] or "yok"
                    lines.append(
                        f"| {arm_name} | {arm['physical_parameters']} | "
                        f"{arm['c_g_n']['mean']:.4f} ± {arm['c_g_n']['std']:.4f} | "
                        f"{arm['all_f1']['mean']:.4f} ± {arm['all_f1']['std']:.4f} | "
                        f"{removed} |"
                    )
                lines.extend([
                    "",
                    f"- Tanım: {neural_compositional['definition']}",
                    f"- Tüm seed kapıları: `{neural_compositional['all_checks_all_seeds']}`",
                ])
        lines.extend(["", "Sınırlar:"])
        lines.extend(f"- {note}" for note in section.get("limitations", []))
        lines.append("")
    lines.extend(["## Reproducibility", ""])
    for key in ("seed_count", "five_or_more_seeds", "all_manifests_completed",
                "dataset_hash_identical_across_seeds", "byte_identical_results"):
        lines.append(f"- {key}: `{reproduction[key]}`")
    lines.append(f"- Not: {reproduction['note']}")
    lines.extend(["", "## Suite Sınırları", ""])
    lines.extend(f"- {note}" for note in report.get("limitations", []))
    return "\n".join(lines) + "\n"


def research_report_html(report: Mapping[str, Any]) -> str:
    rows = []
    for name in report["selected_sections"]:
        section = report["sections"][name]
        score = "n/a" if section["score_mean"] is None else (
            f"{section['score_mean']:.3f} ± {section['score_std']:.3f}"
        )
        status_class = section["status"].lower().replace("_", "-")
        rows.append(
            "<tr>"
            f"<td>{html.escape(section['label'])}</td>"
            f"<td><span class='status {status_class}'>{html.escape(section['status'])}</span></td>"
            f"<td>{html.escape(score)}</td>"
            f"<td>{html.escape(section.get('headline') or '')}</td>"
            "</tr>"
        )
    limitations = "".join(
        f"<li>{html.escape(str(note))}</li>" for note in report.get("limitations", [])
    )
    details = html.escape(json.dumps(report["sections"], ensure_ascii=False, indent=2))
    return f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HGA Research Benchmark</title>
<style>
:root{{--bg:#08111f;--panel:#101c2d;--ink:#e8f0fa;--muted:#9fb0c5;--accent:#61dafb;--line:#26384f;--ok:#43d17b;--warn:#f2bf5e;--bad:#ff6b6b}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(145deg,#07101d,#0d1b2e);color:var(--ink);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1120px;margin:auto;padding:48px 22px 80px}} h1{{font-size:clamp(30px,5vw,52px);margin:0}} .eyebrow{{color:var(--accent);letter-spacing:.14em;text-transform:uppercase}}
.meta{{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0 30px}} .chip{{background:#15243a;border:1px solid var(--line);padding:7px 11px;border-radius:999px}}
.card{{background:rgba(16,28,45,.92);border:1px solid var(--line);border-radius:16px;padding:22px;margin:18px 0;box-shadow:0 18px 55px #0005}}
table{{width:100%;border-collapse:collapse}} th,td{{text-align:left;padding:12px;border-bottom:1px solid var(--line);vertical-align:top}} th{{color:var(--muted)}}
.status{{font-weight:700}} .completed{{color:var(--ok)}} .skipped,.partial{{color:var(--warn)}} .error,.completed-with-errors{{color:var(--bad)}}
.score{{font-size:38px;font-weight:800;color:var(--accent)}} code,pre{{font-family:ui-monospace,monospace}} pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#07101d;padding:16px;border-radius:10px;max-height:560px;overflow:auto}}
small,.muted{{color:var(--muted)}} details summary{{cursor:pointer;color:var(--accent)}}
</style></head><body><main>
<div class="eyebrow">HGA · ölçülebilir araştırma platformu</div><h1>Research Benchmark</h1>
<div class="meta"><span class="chip">{html.escape(report['profile'])}</span><span class="chip">{html.escape(report['outcome'])}</span><span class="chip">{len(report['seeds'])} seed</span></div>
<section class="card"><div class="muted">Genel diagnostic skor · zekâ/SOTA skoru değildir</div><div class="score">{html.escape(str(report.get('overall_diagnostic_score')))}</div><small>dataset {html.escape(report['dataset_hash'][:16])}… · config {html.escape(report['config_hash'][:16])}…</small></section>
<section class="card"><h2>Bölümler</h2><table><thead><tr><th>Bölüm</th><th>Durum</th><th>Skor</th><th>Bulgu</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section class="card"><h2>Reproducibility</h2><p>{report['reproducibility']['seed_count']} seed · manifestler: <strong>{html.escape(str(report['reproducibility']['all_manifests_completed']))}</strong> · ortak dataset hash: <strong>{html.escape(str(report['reproducibility']['dataset_hash_identical_across_seeds']))}</strong></p><p class="muted">{html.escape(report['reproducibility']['note'])}</p></section>
<section class="card"><h2>Sınırlar</h2><ul>{limitations}</ul></section>
<section class="card"><details><summary>Makine-okunur bölüm ayrıntıları</summary><pre>{details}</pre></details></section>
</main></body></html>"""


def _atomic_text(path: Union[str, os.PathLike], content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, target)


def save_research_report(
    report: ResearchBenchmarkReport,
    json_path: Union[str, os.PathLike] = "research_report.json",
    markdown_path: Union[str, os.PathLike] = "research_report.md",
    html_path: Union[str, os.PathLike] = "research_report.html",
) -> Dict[str, str]:
    data = report.to_dict()
    _atomic_text(json_path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _atomic_text(markdown_path, research_report_markdown(data))
    _atomic_text(html_path, research_report_html(data))
    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
        "html": str(html_path),
    }


__all__ = [
    "PROFILE_CONFIGS", "SECTION_LABELS", "SECTION_ORDER", "ResearchBenchmarkReport",
    "research_report_html", "research_report_markdown", "run_research_benchmark",
    "save_research_report", "validate_sections",
]
