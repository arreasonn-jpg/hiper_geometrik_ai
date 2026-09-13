"""Gerçek TWT üzerinde neural compositional-generalization HGA ablasyonu.

Aynı model-visible train/dev ve sabit composition-disjoint test diliminde dört
kol çalışır: tam HGA, attention çıkarılmış HGA, çarpımsal dış-çarpım yerine aynı
parametreli toplamsal geometri ve Kronecker zinciri çıkarılmış HGA. Ortak kalan
parametreler byte-identical başlangıçtan, tüm kollar aynı batch schedule ve
optimizer ile eğitilir.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

from .experiment import canonical_hash
from .real_turkish import RealTurkishTaskData, evaluate_arc_predictions, prepare_real_turkish_task
from .twt_baselines import (
    ARCHITECTURE_CONFIG,
    BASELINE_PROFILES,
    ArcFeatureVocabulary,
    build_twt_models,
)

ABLATION_ORDER = (
    "full",
    "no_attention",
    "additive_geometry",
    "no_kronecker_chain",
)


@dataclass
class NeuralCompositionalReport:
    protocol: str
    schema_version: int
    profile: str
    seed: int
    dataset_hash: str
    split_hashes: Dict[str, str]
    candidate_hashes: Dict[str, str]
    config_hash: str
    fairness: Dict[str, Any]
    arms: Dict[str, Dict[str, Any]]
    deltas_from_full: Dict[str, Dict[str, float]]
    checks: Dict[str, bool]
    limitations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        lines = [
            "# TWT Neural Compositional Generalization Ablation",
            "",
            f"- Profil / seed: `{self.profile}` / `{self.seed}`",
            f"- Dataset hash: `{self.dataset_hash}`",
            f"- Config hash: `{self.config_hash}`",
            "- Ana ölçüm: `C_G_N = composition_disjoint accuracy`",
            "",
            "| Kol | Parametre | C_G_N | F1 (all) | FAR | FRR | Coverage |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name in ABLATION_ORDER:
            arm = self.arms[name]
            metric = arm["test"]["all"]
            lines.append(
                f"| {name} | {arm['physical_parameters']} | "
                f"{arm['neural_generalization']['score']:.4f} | {metric['f1']:.4f} | "
                f"{metric['far']:.4f} | {metric['frr']:.4f} | {metric['coverage']:.4f} |"
            )
        lines.extend(["", "## Kabul kapıları", ""])
        lines.extend(f"- {'PASS' if value else 'FAIL'} — `{key}`" for key, value in self.checks.items())
        lines.extend(["", "## Sınırlar", ""])
        lines.extend(f"- {note}" for note in self.limitations)
        return "\n".join(lines) + "\n"


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as error:  # pragma: no cover
        raise ImportError("Neural compositional TWT ablasyonu için PyTorch gerekli") from error
    return torch, nn


def _parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _predict(torch, model, features, batch_size: int, device) -> tuple[List[bool], float]:
    model.eval()
    predictions: List[bool] = []
    started = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, int(features.shape[0]), batch_size):
            logits = model(features[offset:offset + batch_size].to(device))
            predictions.extend((logits[:, 1] >= logits[:, 0]).cpu().tolist())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return predictions, time.perf_counter() - started


def _batch_schedule(torch, train_count: int, steps: int, batch_size: int, seed: int):
    if train_count % 2 or batch_size % 2:
        raise ValueError("Pair schedule için train ve batch sayıları çift olmalı")
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 7_919)
    pairs = torch.randint(
        0,
        train_count // 2,
        (steps, batch_size // 2),
        generator=generator,
    )
    schedule = torch.stack((2 * pairs, 2 * pairs + 1), dim=-1).reshape(steps, batch_size)
    schedule_hash = canonical_hash(schedule.tolist())
    return schedule, schedule_hash


def _metrics_dict(candidates, predictions) -> Dict[str, Dict[str, Any]]:
    return {
        name: asdict(metric)
        for name, metric in evaluate_arc_predictions(candidates, predictions).items()
    }


def _run_neural_compositional_ablation(
    task_data: RealTurkishTaskData,
    seed: int,
    profile: str,
    device: str,
) -> NeuralCompositionalReport:
    torch, nn = _torch()
    if profile not in BASELINE_PROFILES:
        raise ValueError(f"profile şunlardan biri olmalı: {', '.join(BASELINE_PROFILES)}")
    config = dict(BASELINE_PROFILES[profile])
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA istendi fakat kullanılamıyor")

    vocabulary = ArcFeatureVocabulary.fit(task_data.train)
    train_rows, train_unknown = vocabulary.encode(task_data.train)
    dev_rows, _ = vocabulary.encode(task_data.dev)
    test_rows, _ = vocabulary.encode(task_data.test)
    train_x = torch.tensor(train_rows, dtype=torch.long)
    dev_x = torch.tensor(dev_rows, dtype=torch.long)
    test_x = torch.tensor(test_rows, dtype=torch.long)
    train_y = torch.tensor(
        [int(candidate.expected_valid) for candidate in task_data.train], dtype=torch.long
    )
    schedule, schedule_hash = _batch_schedule(
        torch,
        len(task_data.train),
        int(config["steps"]),
        int(config["batch_size"]),
        int(seed),
    )

    initialization_seed = int(seed) + 400_009
    torch.manual_seed(initialization_seed)
    full_template = build_twt_models(vocabulary.size, hga_ablation="full")["hga"]()
    full_initial_state = {
        name: value.detach().cpu().clone()
        for name, value in full_template.state_dict().items()
    }
    del full_template

    arm_reports: Dict[str, Dict[str, Any]] = {}
    shared_initialization_checks: Dict[str, bool] = {}
    for arm in ABLATION_ORDER:
        torch.manual_seed(initialization_seed)
        model = build_twt_models(vocabulary.size, hga_ablation=arm)["hga"]()
        own_state = model.state_dict()
        shared_names = [
            name for name, value in own_state.items()
            if name in full_initial_state and value.shape == full_initial_state[name].shape
        ]
        for name in shared_names:
            own_state[name] = full_initial_state[name].clone()
        model.load_state_dict(own_state, strict=True)
        shared_equal = all(
            torch.equal(model.state_dict()[name].cpu(), full_initial_state[name])
            for name in shared_names
        )
        shared_initialization_checks[arm] = shared_equal
        model = model.to(target_device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        loss_fn = nn.CrossEntropyLoss()
        losses: List[float] = []
        nonfinite_steps = 0
        maximum_gradient_norm = 0.0
        model.train()
        if target_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(target_device)
            torch.cuda.synchronize(target_device)
        started = time.perf_counter()
        for indices in schedule:
            features = train_x[indices].to(target_device)
            labels = train_y[indices].to(target_device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), labels)
            if not bool(torch.isfinite(loss)):
                nonfinite_steps += 1
                continue
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip_norm"])
            )
            if not bool(torch.isfinite(gradient_norm)):
                nonfinite_steps += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm.item()))
            optimizer.step()
            losses.append(float(loss.item()))
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        training_seconds = time.perf_counter() - started
        gradient_coverage = {
            name: parameter.grad is not None
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        dev_predictions, dev_seconds = _predict(
            torch, model, dev_x, int(config["eval_batch_size"]), target_device
        )
        test_predictions, test_seconds = _predict(
            torch, model, test_x, int(config["eval_batch_size"]), target_device
        )
        test_metrics = _metrics_dict(task_data.test, test_predictions)
        composition = test_metrics["composition_disjoint"]
        arm_report = {
            "arm": arm,
            "physical_parameters": _parameter_count(model),
            "shared_initial_tensor_count": len(shared_names),
            "shared_initial_tensors_equal_to_full": shared_equal,
            "removed_component": {
                "full": None,
                "no_attention": "HiperGeometrikAttention",
                "additive_geometry": "multiplicative outer product (same projections retained)",
                "no_kronecker_chain": "KureselZincir",
            }[arm],
            "training_seconds": round(training_seconds, 6),
            "dev_inference_seconds": round(dev_seconds, 6),
            "test_inference_seconds": round(test_seconds, 6),
            "final_loss": round(losses[-1], 8) if losses else None,
            "minimum_loss": round(min(losses), 8) if losses else None,
            "maximum_gradient_norm_before_clip": round(maximum_gradient_norm, 8),
            "nonfinite_steps": nonfinite_steps,
            "all_trainable_parameters_received_gradient": all(gradient_coverage.values()),
            "parameters_without_gradient": sorted(
                name for name, covered in gradient_coverage.items() if not covered
            ),
            "neural_generalization": {
                "symbol": "C_G_N",
                "eligible_cases": composition["total"],
                "correctly_generalized": composition["correct"],
                "score": composition["accuracy"],
                "is_theoretical_capacity": False,
                "definition": (
                    "Sabit gerçek TWT testindeki composition_disjoint dengeli "
                    "arc adaylarından doğru sınıflandırılanların oranı."
                ),
            },
            "dev": _metrics_dict(task_data.dev, dev_predictions),
            "test": test_metrics,
        }
        if target_device.type == "cuda":
            arm_report["peak_cuda_bytes"] = int(torch.cuda.max_memory_allocated(target_device))
        arm_reports[arm] = arm_report

    full_score = arm_reports["full"]["neural_generalization"]["score"]
    full_f1 = arm_reports["full"]["test"]["all"]["f1"]
    deltas = {
        arm: {
            "full_minus_arm_c_g_n": round(
                full_score - arm_reports[arm]["neural_generalization"]["score"], 8
            ),
            "full_minus_arm_all_f1": round(
                full_f1 - arm_reports[arm]["test"]["all"]["f1"], 8
            ),
        }
        for arm in ABLATION_ORDER
        if arm != "full"
    }
    parameter_counts = {
        arm: report["physical_parameters"] for arm, report in arm_reports.items()
    }
    fairness = {
        "same_dataset_hash": task_data.dataset_hash,
        "same_split_hashes": dict(task_data.split_hashes),
        "same_candidate_hashes": dict(task_data.candidate_hashes),
        "same_train_dev_test_counts": {
            "train": len(task_data.train),
            "dev": len(task_data.dev),
            "test": len(task_data.test),
        },
        "same_train_only_vocabulary_hash": vocabulary.vocabulary_hash(),
        "same_batch_schedule_sha256": schedule_hash,
        "same_optimizer": "AdamW",
        "same_loss": "CrossEntropyLoss",
        "same_steps": int(config["steps"]),
        "same_batch_size": int(config["batch_size"]),
        "same_initialization_seed": initialization_seed,
        "matching_shared_tensors_copied_from_full": shared_initialization_checks,
        "physical_parameter_counts": parameter_counts,
        "parameter_max_to_min_ratio": round(
            max(parameter_counts.values()) / min(parameter_counts.values()), 8
        ),
        "parameter_padding_or_reserve": False,
        "note": (
            "Gerçek component removal parametre sayısını düşürür; fark unused "
            "reserve ile kapatılmaz. Additive geometry kolu full ile aynı numel'e sahiptir."
        ),
    }
    checks = {
        "all_ablation_arms_present": tuple(arm_reports) == ABLATION_ORDER,
        "same_real_twt_split_and_candidates": (
            task_data.dataset_hash
            == "66b13a898efa88998a9329f1551530f5241835a8085a0e5f26e0eb374d7e3276"
            and "test_challenge" in task_data.candidate_hashes
        ),
        "train_only_vocabulary_clean": train_unknown == 0,
        "same_training_protocol_all_arms": True,
        "shared_initialization_equal": all(shared_initialization_checks.values()),
        "full_and_additive_geometry_parameter_matched": (
            parameter_counts["full"] == parameter_counts["additive_geometry"]
        ),
        "removed_components_reduce_parameters_without_padding": (
            parameter_counts["no_attention"] < parameter_counts["full"]
            and parameter_counts["no_kronecker_chain"] < parameter_counts["full"]
            and not fairness["parameter_padding_or_reserve"]
        ),
        "all_arms_optimization_finite": all(
            report["nonfinite_steps"] == 0 and report["final_loss"] is not None
            for report in arm_reports.values()
        ),
        "all_remaining_trainable_parameters_active": all(
            report["all_trainable_parameters_received_gradient"]
            for report in arm_reports.values()
        ),
        "composition_disjoint_balanced_and_nonempty": all(
            report["test"]["composition_disjoint"]["positive"]
            == report["test"]["composition_disjoint"]["negative"]
            > 0
            for report in arm_reports.values()
        ),
        "all_arms_full_test_coverage": all(
            report["test"]["all"]["coverage"] == 1.0
            for report in arm_reports.values()
        ),
        "c_g_n_is_measured_not_theoretical": all(
            report["neural_generalization"]["is_theoretical_capacity"] is False
            for report in arm_reports.values()
        ),
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise ValueError(f"Neural compositional ablation kabul kapısı başarısız: {failed}")
    report_config = {
        "protocol": "twt-neural-compositional-ablation-v1",
        "profile": profile,
        "seed": int(seed),
        "device": str(target_device),
        "training": config,
        "architecture": ARCHITECTURE_CONFIG,
        "arms": list(ABLATION_ORDER),
        "vocabulary_hash": vocabulary.vocabulary_hash(),
        "schedule_hash": schedule_hash,
    }
    return NeuralCompositionalReport(
        protocol="twt-neural-compositional-ablation-v1",
        schema_version=1,
        profile=profile,
        seed=int(seed),
        dataset_hash=task_data.dataset_hash,
        split_hashes=dict(task_data.split_hashes),
        candidate_hashes=dict(task_data.candidate_hashes),
        config_hash=canonical_hash(report_config),
        fairness=fairness,
        arms=arm_reports,
        deltas_from_full=deltas,
        checks=checks,
        limitations=[
            "C_G_N yalnız gerçek TWT composition-disjoint arc-verification accuracy'sidir; mevcut kontrollü C_G veya teorik kapasite değildir.",
            "Structured altı-alan girdisi ham cümleden end-to-end parsing ya da language modeling ölçmez.",
            "Component removal kollarında parametre sayısı doğal olarak azalır; sonuç yalnız accuracy değil numel/süre ile birlikte okunmalıdır.",
            "Smoke profili 32 adımdır; nedensel mimari üstünlük veya convergence/SOTA iddiası taşımaz.",
            "Ablation sonuçları TWT morphosyntactic relation'larıyla sınırlıdır; semantik compositional reasoning kanıtı değildir.",
        ],
    )


@lru_cache(maxsize=16)
def _cached_default_report(seed: int, profile: str, device: str) -> NeuralCompositionalReport:
    return _run_neural_compositional_ablation(
        prepare_real_turkish_task(), seed=seed, profile=profile, device=device
    )


def run_neural_compositional_ablation(
    seed: int = 1,
    profile: str = "smoke",
    task_data: Optional[RealTurkishTaskData] = None,
    device: str = "cpu",
) -> NeuralCompositionalReport:
    """Kontrollü HGA component ablation'ını çalıştır."""
    if task_data is not None:
        return _run_neural_compositional_ablation(
            task_data, seed=int(seed), profile=profile, device=device
        )
    return copy.deepcopy(_cached_default_report(int(seed), profile, device))


__all__ = [
    "ABLATION_ORDER",
    "NeuralCompositionalReport",
    "run_neural_compositional_ablation",
]
