"""Aynı gerçek TWT splitinde parameter-matched neural mimari baseline'ları.

Dört kol aynı model-visible candidate kayıtlarını, train-only vocabulary'yi,
batch indekslerini, kaybı, optimizer'ı, gradient clipping'i ve karar eşiğini
kullanır. Yalnız architecture body değişir:

* Dense MLP,
* PyTorch TransformerEncoder,
* repository'nin gerçek ``KureselZincir`` Kronecker zinciri,
* repository'nin gerçek attention + geometrik encoder + Kronecker zinciri +
  fraktal decoder HGA çekirdeği.

Bu küçük sınıflandırıcılar dil modeli pretraining kıyası değildir. Amaç gerçek,
insan-anotasyonlu aynı görevde mimari inductive bias'ları fiziksel parametre
bütçesi karıştırılmadan karşılaştırmaktır.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
import unicodedata
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .calibration import temperature_scaling_report
from .experiment import canonical_hash
from .real_turkish import (
    ArcCandidate,
    RealTurkishTaskData,
    evaluate_arc_predictions,
    prepare_real_turkish_task,
)

MODEL_ORDER = ("dense", "transformer", "kronecker", "hga")
FEATURE_FIELDS = (
    "dependent_entity",
    "dependent_upos",
    "relation",
    "head_entity",
    "head_upos",
    "arc_geometry",
)
BASELINE_PROFILES: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "steps": 32,
        "batch_size": 512,
        "eval_batch_size": 2048,
        "learning_rate": 0.003,
        "weight_decay": 0.01,
        "gradient_clip_norm": 5.0,
    },
    "full": {
        "steps": 256,
        "batch_size": 512,
        "eval_batch_size": 2048,
        "learning_rate": 0.003,
        "weight_decay": 0.01,
        "gradient_clip_norm": 5.0,
    },
}
ARCHITECTURE_CONFIG: Dict[str, Any] = {
    "embedding_dim": 16,
    "sequence_length": len(FEATURE_FIELDS),
    "dense_hidden_dim": 106,
    "transformer_heads": 4,
    "transformer_feedforward_dim": 278,
    "transformer_layers": 1,
    "kronecker_n": 10,
    "kronecker_layers": 2,
    "hga_n": 26,
    "hga_layers": 2,
    "parameter_tolerance_max_to_min_ratio": 1.01,
    "body_parameter_tolerance_max_to_min_ratio": 1.05,
}

#: FLOP-eşli kontrol rejimi. Parametre eşitliği FLOP eşitliğini garanti
#: etmez (birincil rejimde oran ~11.3×). Tek deneyde ikisini birden eşitlemek
#: mimarileri bozmadan imkânsızdır; bu yüzden İKİNCİ bir rejim tanımlanır:
#: HGA kolu AYNEN kalır, baseline gövdeleri HGA'nın ileri geçiş MAC
#: bütçesine (117 848) ölçeklenir (oran ≤ 1.05). Bu rejimde parametre
#: eşitliği BİLEREK bozulur ve gizlenmez — her rejim tek bir bütçeyi kontrol
#: eder, sonuç çifti birlikte okunur.
FLOP_MATCHED_CONFIG: Dict[str, Any] = {
    **ARCHITECTURE_CONFIG,
    "dense_hidden_dim": 1202,          # 96·1202 + 2·1202 = 117 796 MAC
    "transformer_feedforward_dim": 575,  # dikkat + 2·6·16·575 + 32 = 117 728
    "kronecker_n": 25,                 # 96·625 + 4·25³ + 100 = 122 600
    "flop_tolerance_max_to_min_ratio": 1.05,
}

#: Bütçe rejimi → mimari yapılandırma eşlemesi.
BUDGET_REGIMES: Dict[str, Dict[str, Any]] = {
    "parameter_matched": ARCHITECTURE_CONFIG,
    "flop_matched": FLOP_MATCHED_CONFIG,
}



def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as error:  # pragma: no cover - ortam bağımlı
        raise ImportError("TWT mimari baseline'ları için PyTorch gereklidir") from error
    return torch, nn


def _arc_geometry(candidate: ArcCandidate) -> str:
    if candidate.head_id == 0:
        return "ROOT"
    direction = "LEFT" if candidate.head_id < candidate.dependent_id else "RIGHT"
    distance = abs(candidate.head_id - candidate.dependent_id)
    bucket = "1" if distance == 1 else ("2" if distance == 2 else "3+")
    return f"{direction}:{bucket}"


def _feature_tokens(candidate: ArcCandidate) -> Tuple[str, ...]:
    values = (
        candidate.dependent_entity,
        candidate.dependent_upos,
        candidate.relation,
        candidate.head_entity,
        candidate.head_upos,
        _arc_geometry(candidate),
    )
    return tuple(
        f"{field}={unicodedata.normalize('NFKC', str(value)).casefold()}"
        for field, value in zip(FEATURE_FIELDS, values)
    )


@dataclass(frozen=True)
class ArcFeatureVocabulary:
    token_to_id: Dict[str, int]
    unknown_id: int = 0

    @classmethod
    def fit(cls, candidates: Sequence[ArcCandidate]) -> "ArcFeatureVocabulary":
        tokens = sorted({token for candidate in candidates for token in _feature_tokens(candidate)})
        return cls({token: index + 1 for index, token in enumerate(tokens)})

    @property
    def size(self) -> int:
        return len(self.token_to_id) + 1

    def encode(self, candidates: Sequence[ArcCandidate]) -> Tuple[List[List[int]], int]:
        rows: List[List[int]] = []
        unknown = 0
        for candidate in candidates:
            encoded = []
            for token in _feature_tokens(candidate):
                value = self.token_to_id.get(token, self.unknown_id)
                unknown += int(value == self.unknown_id)
                encoded.append(value)
            rows.append(encoded)
        return rows, unknown

    def vocabulary_hash(self) -> str:
        return canonical_hash({"unknown_id": self.unknown_id, "tokens": self.token_to_id})


@dataclass
class TWTBaselineReport:
    protocol: str
    schema_version: int
    profile: str
    seed: int
    dataset_hash: str
    dataset_config_hash: str
    split_hashes: Dict[str, str]
    candidate_hashes: Dict[str, str]
    benchmark_config_hash: str
    feature_contract: Dict[str, Any]
    fairness: Dict[str, Any]
    models: Dict[str, Dict[str, Any]]
    checks: Dict[str, bool]
    limitations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        lines = [
            "# TWT Parameter-Matched Architecture Baselines",
            "",
            f"- Profil / seed: `{self.profile}` / `{self.seed}`",
            f"- Dataset hash: `{self.dataset_hash}`",
            f"- Config hash: `{self.benchmark_config_hash}`",
            f"- Toplam parametre oranı max/min: `{self.fairness['parameter_max_to_min_ratio']:.6f}`",
            f"- Architecture-body parametre oranı max/min: "
            f"`{self.fairness['body_parameter_max_to_min_ratio']:.6f}`",
            "",
            "| Model | Parametre | Accuracy | F1 | FAR | FRR | Coverage | Süre (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name in MODEL_ORDER:
            model = self.models[name]
            metric = model["test"]["all"]
            lines.append(
                f"| {name} | {model['physical_parameters']} | {metric['accuracy']:.4f} | "
                f"{metric['f1']:.4f} | {metric['far']:.4f} | {metric['frr']:.4f} | "
                f"{metric['coverage']:.4f} | {model['training_seconds']:.3f} |"
            )
        lines.extend([
            "",
            "## Dev-only uncertainty calibration",
            "",
            "Temperature yalnız sabit dev splitinde NLL ile seçilir; test yalnız ölçümdür.",
            "",
            "| Model | Temperature | ECE önce | ECE sonra | NLL önce | NLL sonra | "
            "Brier önce | Brier sonra | AURC sonra |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for name in MODEL_ORDER:
            calibration = self.models[name]["calibration"]
            before = calibration["slices"]["all"]["before"]
            after = calibration["slices"]["all"]["after"]
            lines.append(
                f"| {name} | {calibration['fit']['temperature']:.4f} | "
                f"{before['ece']:.4f} | {after['ece']:.4f} | "
                f"{before['nll']:.4f} | {after['nll']:.4f} | "
                f"{before['brier']:.4f} | {after['brier']:.4f} | "
                f"{after['aurc']:.4f} |"
            )
        lines.extend(["", "## Adillik kapıları", ""])
        lines.extend(f"- {'PASS' if value else 'FAIL'} — `{key}`" for key, value in self.checks.items())
        lines.extend(["", "## Sınırlar", ""])
        lines.extend(f"- {note}" for note in self.limitations)
        return "\n".join(lines) + "\n"


def build_twt_models(vocabulary_size: int, hga_ablation: str = "full",
                     config: Optional[Dict[str, Any]] = None):
    """Baseline modellerini ve kontrollü HGA ablation varyantını oluştur.

    ``config`` verilmezse birincil (parameter-matched) yapılandırma kullanılır;
    ``FLOP_MATCHED_CONFIG`` verilirse baseline gövdeleri HGA'nın MAC bütçesine
    ölçeklenmiş kontrol rejimi kurulur (HGA kolu iki rejimde de AYNIDIR).
    """
    allowed_ablations = {
        "full", "no_attention", "additive_geometry", "no_kronecker_chain"
    }
    if hga_ablation not in allowed_ablations:
        raise ValueError(f"Bilinmeyen HGA ablation: {hga_ablation}")
    cfg = dict(ARCHITECTURE_CONFIG if config is None else config)
    torch, nn = _torch()
    from mimari.decoder import FraktalDecoder
    from mimari.encoder import GeometrikVeriEncoder
    from mimari.hiper_attention import HiperGeometrikAttention
    from mimari.kuresel_bag import KureselZincir

    embedding_dim = int(cfg["embedding_dim"])
    sequence_length = int(cfg["sequence_length"])
    flat_dim = embedding_dim * sequence_length

    class DenseClassifier(nn.Module):  # type: ignore[name-defined]
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
            hidden = int(cfg["dense_hidden_dim"])
            self.body = nn.Sequential(
                nn.Linear(flat_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 2),
            )

        def forward(self, token_ids):
            return self.body(self.embedding(token_ids).flatten(1))

    class TransformerClassifier(nn.Module):  # type: ignore[name-defined]
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
            self.position = nn.Parameter(torch.empty(1, sequence_length, embedding_dim))
            nn.init.normal_(self.position, mean=0.0, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=int(cfg["transformer_heads"]),
                dim_feedforward=int(cfg["transformer_feedforward_dim"]),
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.body = nn.TransformerEncoder(
                layer,
                num_layers=int(cfg["transformer_layers"]),
                norm=nn.LayerNorm(embedding_dim),
                enable_nested_tensor=False,
            )
            self.readout = nn.Linear(embedding_dim, 2)

        def forward(self, token_ids):
            hidden = self.body(self.embedding(token_ids) + self.position)
            return self.readout(hidden.mean(dim=1))

    class KroneckerClassifier(nn.Module):  # type: ignore[name-defined]
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
            n = int(cfg["kronecker_n"])
            self.project = nn.Linear(flat_dim, n * n)
            self.body = KureselZincir(
                n=n,
                katman_sayisi=int(cfg["kronecker_layers"]),
                dropout=0.0,
                checkpoint_kullan=False,
                aktivasyon="silu",
            )
            self.norm = nn.LayerNorm(n)
            self.readout = nn.Linear(2 * n, 2)
            self.n = n

        def forward(self, token_ids):
            matrix = self.project(self.embedding(token_ids).flatten(1)).reshape(-1, self.n, self.n)
            matrix = self.norm(self.body(matrix))
            pooled = torch.cat([matrix.mean(dim=1), matrix.mean(dim=2)], dim=-1)
            return self.readout(pooled)

    class HGAClassifier(nn.Module):  # type: ignore[name-defined]
        def __init__(self):
            super().__init__()
            self.ablation = hga_ablation
            self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
            self.position = nn.Parameter(torch.empty(1, sequence_length, embedding_dim))
            nn.init.normal_(self.position, mean=0.0, std=0.02)
            if hga_ablation != "no_attention":
                self.attention = HiperGeometrikAttention(
                    embedding_dim,
                    int(cfg["transformer_heads"]),
                    dropout=0.0,
                    is_causal=False,
                )
            else:
                self.attention = None
            n = int(cfg["hga_n"])
            self.encoder = GeometrikVeriEncoder(flat_dim, n, aktivasyon="tanh")
            if hga_ablation != "no_kronecker_chain":
                self.body = KureselZincir(
                    n=n,
                    katman_sayisi=int(cfg["hga_layers"]),
                    dropout=0.0,
                    checkpoint_kullan=False,
                    aktivasyon="silu",
                )
            else:
                self.body = nn.Identity()
            self.norm = nn.LayerNorm(n)
            self.decoder = FraktalDecoder(n, 2)

        def forward(self, token_ids):
            hidden = self.embedding(token_ids) + self.position
            if self.attention is not None:
                hidden = self.attention(hidden)
            flat = hidden.flatten(1)
            if self.ablation == "additive_geometry":
                u = self.encoder.proj_u(flat)
                v = self.encoder.proj_v(flat)
                matrix = torch.tanh(u.unsqueeze(2) + v.unsqueeze(1))
            else:
                matrix = self.encoder(flat)
            return self.decoder(self.norm(self.body(matrix)))

    return {
        "dense": DenseClassifier,
        "transformer": TransformerClassifier,
        "kronecker": KroneckerClassifier,
        "hga": HGAClassifier,
    }


def _parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _parameter_bytes(model) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _metrics_to_dict(candidates, predictions) -> Dict[str, Dict[str, Any]]:
    return {
        name: asdict(metric)
        for name, metric in evaluate_arc_predictions(candidates, predictions).items()
    }


def _predict(
    torch, model, features, batch_size: int, device
) -> Tuple[List[bool], List[List[float]], float]:
    model.eval()
    predictions: List[bool] = []
    raw_logits: List[List[float]] = []
    started = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, int(features.shape[0]), batch_size):
            logits = model(features[offset:offset + batch_size].to(device))
            cpu_logits = logits.cpu()
            predictions.extend((cpu_logits[:, 1] >= cpu_logits[:, 0]).tolist())
            raw_logits.extend(cpu_logits.tolist())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return predictions, raw_logits, time.perf_counter() - started


def _candidate_sequence_hash(candidates: Sequence[ArcCandidate]) -> str:
    digest = hashlib.sha256()
    for candidate in candidates:
        digest.update(
            json.dumps(
                candidate.hash_record(), ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _batch_schedule(torch, train_count: int, steps: int, batch_size: int, seed: int):
    if train_count % 2 or batch_size % 2:
        raise ValueError("Dengeli pair schedule için train_count ve batch_size çift olmalı")
    pair_count = train_count // 2
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 7_919)
    pairs = torch.randint(0, pair_count, (steps, batch_size // 2), generator=generator)
    schedule = torch.stack((2 * pairs, 2 * pairs + 1), dim=-1).reshape(steps, batch_size)
    digest = hashlib.sha256(schedule.numpy().tobytes()).hexdigest()
    return schedule, digest


def _run_twt_architecture_baselines(
    task_data: RealTurkishTaskData,
    seed: int,
    profile: str,
    device: str,
    regime: str = "parameter_matched",
) -> TWTBaselineReport:
    """Önceden hazırlanmış task data üzerinde dört mimariyi çalıştır."""
    torch, nn = _torch()
    if profile not in BASELINE_PROFILES:
        raise ValueError(f"profile şunlardan biri olmalı: {', '.join(BASELINE_PROFILES)}")
    if regime not in BUDGET_REGIMES:
        raise ValueError(
            f"regime şunlardan biri olmalı: {', '.join(BUDGET_REGIMES)}")
    arch_cfg = dict(BUDGET_REGIMES[regime])
    config = dict(BASELINE_PROFILES[profile])
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA istendi fakat torch.cuda.is_available() false")

    model_visible_hashes = {
        "train": _candidate_sequence_hash(task_data.train),
        "dev": _candidate_sequence_hash(task_data.dev),
        "test": _candidate_sequence_hash(task_data.test),
    }
    vocabulary = ArcFeatureVocabulary.fit(task_data.train)
    train_rows, train_unknown = vocabulary.encode(task_data.train)
    dev_rows, dev_unknown = vocabulary.encode(task_data.dev)
    test_rows, test_unknown = vocabulary.encode(task_data.test)
    train_x = torch.tensor(train_rows, dtype=torch.long)
    dev_x = torch.tensor(dev_rows, dtype=torch.long)
    test_x = torch.tensor(test_rows, dtype=torch.long)
    train_y = torch.tensor(
        [int(candidate.expected_valid) for candidate in task_data.train], dtype=torch.long
    )
    schedule, schedule_hash = _batch_schedule(
        torch,
        train_count=len(task_data.train),
        steps=int(config["steps"]),
        batch_size=int(config["batch_size"]),
        seed=int(seed),
    )

    constructors = build_twt_models(vocabulary.size, config=arch_cfg)
    # Her kol aynı train-only embedding başlangıç matrisiyle başlar.
    common_generator = torch.Generator(device="cpu").manual_seed(int(seed) + 31_337)
    common_embedding = torch.randn(
        vocabulary.size,
        int(arch_cfg["embedding_dim"]),
        generator=common_generator,
    )
    model_reports: Dict[str, Dict[str, Any]] = {}
    parameter_counts: Dict[str, int] = {}
    body_parameter_counts: Dict[str, int] = {}
    implementation = {
        "dense": ["torch.nn.Embedding", "torch.nn.Linear", "torch.nn.GELU"],
        "transformer": ["torch.nn.Embedding", "torch.nn.TransformerEncoder"],
        "kronecker": ["mimari.kuresel_bag.KureselZincir"],
        "hga": [
            "mimari.hiper_attention.HiperGeometrikAttention",
            "mimari.encoder.GeometrikVeriEncoder",
            "mimari.kuresel_bag.KureselZincir",
            "mimari.decoder.FraktalDecoder",
        ],
    }
    shared_embedding_parameters = vocabulary.size * int(arch_cfg["embedding_dim"])
    for model_index, name in enumerate(MODEL_ORDER):
        torch.manual_seed(int(seed) + 100_003 * (model_index + 1))
        model = constructors[name]().to(target_device)
        with torch.no_grad():
            model.embedding.weight.copy_(common_embedding.to(target_device))
        parameter_counts[name] = _parameter_count(model)
        body_parameter_counts[name] = parameter_counts[name] - shared_embedding_parameters
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        loss_fn = nn.CrossEntropyLoss()
        losses: List[float] = []
        maximum_gradient_norm = 0.0
        nonfinite_steps = 0
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
            parameter_name: parameter.grad is not None
            for parameter_name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        dev_predictions, dev_logits, dev_seconds = _predict(
            torch, model, dev_x, int(config["eval_batch_size"]), target_device
        )
        test_predictions, test_logits, test_seconds = _predict(
            torch, model, test_x, int(config["eval_batch_size"]), target_device
        )
        calibration = temperature_scaling_report(
            dev_logits=dev_logits,
            dev_labels=[candidate.expected_valid for candidate in task_data.dev],
            test_logits=test_logits,
            test_labels=[candidate.expected_valid for candidate in task_data.test],
            test_dimensions=[candidate.dimensions for candidate in task_data.test],
        )
        model_report = {
            "architecture": name,
            "implementation": implementation[name],
            "physical_parameters": parameter_counts[name],
            "shared_embedding_parameters": shared_embedding_parameters,
            "architecture_body_parameters": body_parameter_counts[name],
            "parameter_bytes": _parameter_bytes(model),
            "estimated_adamw_training_bytes": _parameter_bytes(model) * 4,
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
            "dev": _metrics_to_dict(task_data.dev, dev_predictions),
            "test": _metrics_to_dict(task_data.test, test_predictions),
            "calibration": calibration,
        }
        if target_device.type == "cuda":
            model_report["peak_cuda_bytes"] = int(torch.cuda.max_memory_allocated(target_device))
        model_reports[name] = model_report

    minimum_parameters = min(parameter_counts.values())
    maximum_parameters = max(parameter_counts.values())
    parameter_ratio = maximum_parameters / minimum_parameters
    minimum_body_parameters = min(body_parameter_counts.values())
    maximum_body_parameters = max(body_parameter_counts.values())
    body_parameter_ratio = maximum_body_parameters / minimum_body_parameters
    input_token_count = len(FEATURE_FIELDS)
    feature_contract = {
        "fields": list(FEATURE_FIELDS),
        "sequence_length": input_token_count,
        "vocabulary_source": "model-visible TWT train candidates only",
        "vocabulary_size_including_unknown": vocabulary.size,
        "vocabulary_hash": vocabulary.vocabulary_hash(),
        "unknown_id": vocabulary.unknown_id,
        "unknown_tokens": {
            "train": train_unknown,
            "dev": dev_unknown,
            "test": test_unknown,
        },
        "total_tokens": {
            "train": len(task_data.train) * input_token_count,
            "dev": len(task_data.dev) * input_token_count,
            "test": len(task_data.test) * input_token_count,
        },
        "tokenizer_changed": False,
        "note": "Bu structured benchmark encoder'ıdır; repository BPE tokenizer'ını değiştirmez.",
    }
    protocol_name = ("twt-parameter-matched-architectures-v1"
                     if regime == "parameter_matched"
                     else "twt-flop-matched-architectures-v1")
    benchmark_config = {
        "protocol": protocol_name,
        "profile": profile,
        "regime": regime,
        "training": config,
        "architectures": arch_cfg,
        "feature_vocabulary_hash": vocabulary.vocabulary_hash(),
        "model_visible_candidate_hashes": model_visible_hashes,
        "seed": int(seed),
        "device": str(target_device),
    }
    fairness = {
        "physical_parameter_counts": parameter_counts,
        "shared_embedding_parameters_each": shared_embedding_parameters,
        "architecture_body_parameter_counts": body_parameter_counts,
        "parameter_min": minimum_parameters,
        "parameter_max": maximum_parameters,
        "parameter_max_to_min_ratio": round(parameter_ratio, 8),
        "parameter_tolerance": float(
            arch_cfg["parameter_tolerance_max_to_min_ratio"]
        ),
        "body_parameter_min": minimum_body_parameters,
        "body_parameter_max": maximum_body_parameters,
        "body_parameter_max_to_min_ratio": round(body_parameter_ratio, 8),
        "body_parameter_tolerance": float(
            arch_cfg["body_parameter_tolerance_max_to_min_ratio"]
        ),
        "same_dataset_hash": task_data.dataset_hash,
        "same_split_hashes": dict(task_data.split_hashes),
        "source_candidate_hashes": dict(task_data.candidate_hashes),
        "model_visible_candidate_hashes": model_visible_hashes,
        "same_train_candidate_count": len(task_data.train),
        "same_dev_candidate_count": len(task_data.dev),
        "same_test_candidate_count": len(task_data.test),
        "same_feature_vocabulary": vocabulary.vocabulary_hash(),
        "same_batch_schedule_sha256": schedule_hash,
        "same_optimizer": "AdamW",
        "same_loss": "CrossEntropyLoss",
        "same_decision_rule": "class-1 logit >= class-0 logit",
        "same_steps": int(config["steps"]),
        "same_batch_size": int(config["batch_size"]),
        "same_learning_rate": float(config["learning_rate"]),
        "same_weight_decay": float(config["weight_decay"]),
        "same_gradient_clip_norm": float(config["gradient_clip_norm"]),
        "same_initial_embedding_weights": True,
        "unused_parameter_padding": False,
        "budget_regime": regime,
    }
    # FLOP muhasebesi rejimden bağımsız olarak raporlanır; flop_matched
    # rejimde kapıya dönüşür (parametre kapıları o rejimde BİLEREK gevşer
    # ve bu fairness sözlüğünde gizlenmeden durur).
    from .twt_results import analytic_forward_flops as _analytic_flops
    mac_counts = {
        m: _analytic_flops(m, arch_cfg)["forward_flops_per_example"]
        for m in MODEL_ORDER
    }
    mac_ratio = max(mac_counts.values()) / max(1, min(mac_counts.values()))
    fairness["forward_macs_per_example"] = mac_counts
    fairness["flop_max_to_min_ratio"] = round(mac_ratio, 8)
    if regime == "flop_matched":
        # FLOP-eşli rejimde parametre paritesi BİLEREK bırakılır (gövdeler
        # MAC bütçesine ölçeklenir); bu fairness sözlüğünde açıkça durur ve
        # parametre kapıları bu rejimde denetlenmez (denetlenen: FLOP oranı).
        fairness["flop_tolerance"] = float(
            arch_cfg["flop_tolerance_max_to_min_ratio"])
        fairness["parameter_parity_intentionally_relaxed"] = True
    checks = {
        "all_four_model_families_present": tuple(model_reports) == MODEL_ORDER,
        # Parametre kapıları yalnız parameter_matched rejiminin iddiasıdır.
        "physical_parameters_within_one_percent": (
            regime != "parameter_matched"
            or parameter_ratio <= float(
                arch_cfg["parameter_tolerance_max_to_min_ratio"])),
        "architecture_body_parameters_within_five_percent": (
            regime != "parameter_matched"
            or body_parameter_ratio <= float(
                arch_cfg["body_parameter_tolerance_max_to_min_ratio"])),
        # FLOP kapısı yalnız flop_matched rejiminin iddiasıdır.
        "flop_budget_within_regime_tolerance": (
            regime != "flop_matched"
            or mac_ratio <= float(arch_cfg["flop_tolerance_max_to_min_ratio"])),
        "same_real_dataset_and_splits": (
            task_data.dataset_hash
            == "66b13a898efa88998a9329f1551530f5241835a8085a0e5f26e0eb374d7e3276"
            and "test_challenge" in task_data.candidate_hashes
        ),
        "same_examples_optimizer_loss_schedule": True,
        "train_only_vocabulary_has_no_unknown": train_unknown == 0,
        "relation_holdout_not_in_model_visible_train_dev": all(
            candidate.gold_relation not in set(task_data.heldout_relations)
            for candidate in (*task_data.train, *task_data.dev)
        ),
        "all_models_full_test_coverage": all(
            report["test"]["all"]["coverage"] == 1.0
            for report in model_reports.values()
        ),
        "all_models_optimization_finite": all(
            report["nonfinite_steps"] == 0 and report["final_loss"] is not None
            for report in model_reports.values()
        ),
        "all_trainable_parameters_active": all(
            report["all_trainable_parameters_received_gradient"]
            for report in model_reports.values()
        ),
        "no_unused_parameter_padding": not fairness["unused_parameter_padding"],
        "required_metrics_all_models": all(
            all(
                key in report["test"]["all"]
                for key in ("accuracy", "precision", "recall", "f1", "far", "frr", "coverage")
            )
            for report in model_reports.values()
        ),
        "calibration_is_dev_only_and_argmax_preserving": all(
            all(report["calibration"]["checks"].values())
            for report in model_reports.values()
        ),
        "calibration_reports_all_disjoint_slices": all(
            set(report["calibration"]["slices"])
            == set(report["test"])
            for report in model_reports.values()
        ),
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise ValueError(f"TWT baseline adillik kapısı başarısız: {failed}")
    regime_limitation = (
        "Parameter matching gerçek trainable numel üzerinden ±%1 içindedir; "
        "FLOP, aktivasyon belleği ve duvar süresi eşitlenmez, ayrıca raporlanır."
        if regime == "parameter_matched" else
        "FLOP-eşli rejimde baseline gövdeleri HGA'nın MAC bütçesine "
        "ölçeklenir; parametre paritesi BİLEREK bırakılır ve fairness "
        "sözlüğünde raporlanır. İki rejim birlikte okunmalıdır.")
    return TWTBaselineReport(
        protocol=protocol_name,
        schema_version=1,
        profile=profile,
        seed=int(seed),
        dataset_hash=task_data.dataset_hash,
        dataset_config_hash=task_data.config_hash,
        split_hashes=dict(task_data.split_hashes),
        candidate_hashes=dict(task_data.candidate_hashes),
        benchmark_config_hash=canonical_hash(benchmark_config),
        feature_contract=feature_contract,
        fairness=fairness,
        models=model_reports,
        checks=checks,
        limitations=[
            "Görev TWT basic morphosyntactic dependency-arc doğrulamasıdır; genel dil modelleme değildir.",
            "Structured 6-field encoder tüm kollarda ortaktır; ham cümleden end-to-end dependency parsing ölçülmez.",
            "Smoke profili yarım epoch'tan az sabit adım kullanır; yayınlanabilir convergence/SOTA sonucu değildir.",
            regime_limitation,
            "HGA kolu repository'nin attention/outer-product/Kronecker/fraktal çekirdeğidir; tam autoregressive HiperGeometrikAI dil modeli değildir.",
            "Unseen relation train-only vocabulary'de UNKNOWN olur; bu sıfırdan relation-semantics induction görevidir ve yüksek skor beklenmez.",
            "Temperature yalnız dev NLL'yi optimize eder; test calibration metriklerinin iyileşmesi garanti edilmez ve argmax doğruluğu değişmez.",
            "Selective risk sabit coverage noktalarında diagnostic ölçümdür; gerçek deployment threshold'u veya domain-shift garantisi değildir.",
        ],
    )


@lru_cache(maxsize=16)
def _cached_default_baseline(seed: int, profile: str, device: str,
                             regime: str = "parameter_matched") -> TWTBaselineReport:
    return _run_twt_architecture_baselines(
        prepare_real_turkish_task(), seed=seed, profile=profile, device=device,
        regime=regime,
    )


def run_twt_architecture_baselines(
    seed: int = 1,
    profile: str = "smoke",
    task_data: Optional[RealTurkishTaskData] = None,
    device: str = "cpu",
    regime: str = "parameter_matched",
) -> TWTBaselineReport:
    """Dört model ailesini aynı TWT task data ve fiziksel bütçede eğit/ölç.

    Varsayılan dataset sonuçları süreç içinde seed/profile/device anahtarıyla
    cache'lenir; çağıran bağımsız bir rapor kopyası alır. Özel task data hiçbir
    zaman cache'e girmez.
    """
    if task_data is not None:
        return _run_twt_architecture_baselines(
            task_data, seed=int(seed), profile=profile, device=device,
            regime=regime,
        )
    return copy.deepcopy(
        _cached_default_baseline(int(seed), profile, device, regime))


__all__ = [
    "ARCHITECTURE_CONFIG",
    "BUDGET_REGIMES",
    "FLOP_MATCHED_CONFIG",
    "BASELINE_PROFILES",
    "FEATURE_FIELDS",
    "MODEL_ORDER",
    "ArcFeatureVocabulary",
    "TWTBaselineReport",
    "build_twt_models",
    "run_twt_architecture_baselines",
]
