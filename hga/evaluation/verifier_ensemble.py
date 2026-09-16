"""Çoklu verifier ensemble protokolü.

Tek bir doğrulayıcının kararını "mutlak hakem" yapmak yerine aynı kanıtı üç
bağımsız kontrol hattından geçiririz:

1. ``strict_schema_oracle``: mevcut katı şema + aritmetik oracle.
2. ``normal_form_oracle``: claim/sonuç normal formunu tekrar hesaplayan ayrı
   uygulama.
3. ``trace_replay_oracle``: ispat adımlarını replay eden ayrı uygulama.

Ensemble politikası bilinçli olarak muhafazakârdır: herhangi bir üye ``INVALID``
derse sonuç ``INVALID``; tüm üyeler ``VERIFIED`` demedikçe ``VERIFIED`` üretilmez;
alan dışı/unsupported durum ``UNCERTAIN`` olarak kalır. Böylece yanlış kabul
(false acceptance) azaltılır, desteklenmeyen kurallar başarı gibi sunulmaz.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .verifier_adversarial import (
    DATA_FILE_V2,
    REQUIRED_ATTACK_CLASSES_V2,
    ProofDecision,
    VerifierAttackDataset,
    VerifierAttackMetrics,
    _metrics,
    _strict_integer,
    verify_arithmetic_proof,
)

ENSEMBLE_PROTOCOL = "multi_verifier_ensemble_v1"
ENSEMBLE_POLICY = "unanimous_accept_any_invalid_reject_uncertain_preserved"


@dataclass(frozen=True)
class VerifierMember:
    """Ensemble üyesi: isim + karar fonksiyonu."""

    name: str
    verifier: Callable[[Any, int, int, int], ProofDecision]


@dataclass
class VerifierEnsembleReport:
    protocol: str
    benchmark_id: str
    dataset_hash: str
    ensemble_hash: str
    ensemble_policy: str
    members: List[str]
    metrics: VerifierAttackMetrics
    member_metrics: Dict[str, VerifierAttackMetrics]
    disagreement_rate: float
    by_attack_class: Dict[str, Dict[str, Any]]
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    checks: Dict[str, bool] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        return verifier_ensemble_markdown(self)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


def _schema_guard(
    proof: Any,
    minimum_operand: int,
    maximum_operand: int,
    maximum_steps: int,
) -> Optional[ProofDecision]:
    """Ortak giriş şemasını doğrula; geçerse ``None`` döndür."""
    if not isinstance(proof, Mapping):
        return ProofDecision("INVALID", "proof nesnesi mapping olmalı")
    if set(proof) != {"claim", "steps", "conclusion"}:
        return ProofDecision("INVALID", "proof alanları eksik veya fazladır")
    claim = proof["claim"]
    steps = proof["steps"]
    if not isinstance(claim, Mapping):
        return ProofDecision("INVALID", "claim mapping olmalı")
    if set(claim) != {"left", "right", "operator", "result"}:
        return ProofDecision("INVALID", "claim alanları eksik veya fazladır")
    if not isinstance(steps, list):
        return ProofDecision("INVALID", "steps liste olmalı")
    if not steps:
        return ProofDecision("INVALID", "ispat en az bir adım içermeli")
    if len(steps) > maximum_steps:
        return ProofDecision("INVALID", "ispat adım sınırını aşıyor")
    values = (claim["left"], claim["right"], claim["result"], proof["conclusion"])
    if not all(_strict_integer(value, minimum_operand, maximum_operand)
               for value in values):
        return ProofDecision("INVALID", "claim/conclusion katı ve sınırlı int olmalı")
    operator = claim["operator"]
    if not isinstance(operator, str):
        return ProofDecision("INVALID", "operator string olmalı")
    if operator != "add":
        return ProofDecision("UNCERTAIN", f"desteklenmeyen operator: {operator}")
    return None


def verify_normal_form_addition(
    proof: Any,
    minimum_operand: int = -1_000_000,
    maximum_operand: int = 1_000_000,
    maximum_steps: int = 4,
) -> ProofDecision:
    """Claim/result/conclusion normal formunu bağımsız aritmetik ile denetle."""
    guard = _schema_guard(proof, minimum_operand, maximum_operand, maximum_steps)
    if guard is not None:
        return guard
    assert isinstance(proof, Mapping)  # guard sonrası tip daraltma
    claim = proof["claim"]
    steps = proof["steps"]
    assert isinstance(claim, Mapping) and isinstance(steps, list)
    for step in steps:
        if not isinstance(step, Mapping) or set(step) != {"rule", "inputs", "output"}:
            return ProofDecision("INVALID", "adım şeması geçersiz")
        if step["rule"] != "integer_addition":
            return ProofDecision("UNCERTAIN", f"desteklenmeyen rule: {step['rule']}")
    computed = int(claim["left"]) + int(claim["right"])
    if int(claim["result"]) != computed:
        return ProofDecision("INVALID", "normal form claim sonucunu reddetti")
    if int(proof["conclusion"]) != computed:
        return ProofDecision("INVALID", "normal form conclusion sonucunu reddetti")
    return ProofDecision("VERIFIED", "claim/result/conclusion normal formu doğrulandı")


def verify_trace_replay_addition(
    proof: Any,
    minimum_operand: int = -1_000_000,
    maximum_operand: int = 1_000_000,
    maximum_steps: int = 4,
) -> ProofDecision:
    """İspat adımlarını claim ile aynı sırada tekrar oynat."""
    guard = _schema_guard(proof, minimum_operand, maximum_operand, maximum_steps)
    if guard is not None:
        return guard
    assert isinstance(proof, Mapping)
    claim = proof["claim"]
    steps = proof["steps"]
    assert isinstance(claim, Mapping) and isinstance(steps, list)
    if len(steps) != 1:
        return ProofDecision("INVALID", "trace replay tek addition adımı bekler")
    step = steps[0]
    if not isinstance(step, Mapping) or set(step) != {"rule", "inputs", "output"}:
        return ProofDecision("INVALID", "adım şeması geçersiz")
    if step["rule"] != "integer_addition":
        return ProofDecision("UNCERTAIN", f"desteklenmeyen rule: {step['rule']}")
    inputs = step["inputs"]
    if not isinstance(inputs, list) or len(inputs) != 2:
        return ProofDecision("INVALID", "adım iki input içermeli")
    values = (inputs[0], inputs[1], step["output"])
    if not all(_strict_integer(value, minimum_operand, maximum_operand)
               for value in values):
        return ProofDecision("INVALID", "trace değerleri katı ve sınırlı int olmalı")
    if inputs != [claim["left"], claim["right"]]:
        return ProofDecision("INVALID", "trace girdileri claim ile çelişiyor")
    replayed = int(inputs[0]) + int(inputs[1])
    if int(step["output"]) != replayed:
        return ProofDecision("INVALID", "trace çıktısı oracle ile uyuşmuyor")
    if int(claim["result"]) != replayed or int(proof["conclusion"]) != replayed:
        return ProofDecision("INVALID", "trace sonucu claim/conclusion ile çelişiyor")
    return ProofDecision("VERIFIED", "integer_addition trace replay doğrulandı")


def default_verifier_members() -> List[VerifierMember]:
    return [
        VerifierMember("strict_schema_oracle", verify_arithmetic_proof),
        VerifierMember("normal_form_oracle", verify_normal_form_addition),
        VerifierMember("trace_replay_oracle", verify_trace_replay_addition),
    ]


def _combine_votes(votes: Mapping[str, ProofDecision]) -> ProofDecision:
    states = {decision.state for decision in votes.values()}
    if "INVALID" in states:
        invalid_members = [name for name, decision in votes.items()
                           if decision.state == "INVALID"]
        return ProofDecision("INVALID", "INVALID oyu: " + ",".join(invalid_members))
    if states == {"VERIFIED"}:
        return ProofDecision("VERIFIED", "tüm verifier üyeleri VERIFIED dedi")
    return ProofDecision("UNCERTAIN", "en az bir üye UNCERTAIN; VERIFIED için oybirliği yok")


def _dataset_vote_hash(dataset_hash: str, members: Sequence[str]) -> str:
    payload = {
        "protocol": ENSEMBLE_PROTOCOL,
        "policy": ENSEMBLE_POLICY,
        "dataset_hash": dataset_hash,
        "members": list(members),
    }
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _by_attack_class(records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_class: Dict[str, Dict[str, Any]] = {}
    for attack_class in sorted({record["attack_class"] for record in records}):
        selected = [prediction for prediction in predictions
                    if prediction["attack_class"] == attack_class]
        correct = sum(prediction["correct"] for prediction in selected)
        disagreements = sum(prediction["member_disagreement"] for prediction in selected)
        by_class[attack_class] = {
            "total": len(selected),
            "correct": correct,
            "accuracy": _ratio(correct, len(selected)),
            "disagreements": disagreements,
            "disagreement_rate": _ratio(disagreements, len(selected)),
        }
    return by_class


def run_verifier_ensemble_benchmark(
    dataset: Optional[VerifierAttackDataset] = None,
    members: Optional[Sequence[VerifierMember]] = None,
) -> VerifierEnsembleReport:
    """Adversarial proof suite üzerinde çoklu verifier ensemble'ını ölç."""
    dataset = dataset or VerifierAttackDataset(
        data_file=DATA_FILE_V2,
        required_attack_classes=tuple(REQUIRED_ATTACK_CLASSES_V2),
    )
    member_list = list(members or default_verifier_members())
    if len(member_list) < 3:
        raise ValueError("ensemble en az üç verifier üyesi ister")
    names = [member.name for member in member_list]
    if len(set(names)) != len(names):
        raise ValueError("verifier üye adları benzersiz olmalı")

    limits = dataset.limits
    predictions: List[Dict[str, Any]] = []
    member_predictions: Dict[str, List[Dict[str, Any]]] = {name: [] for name in names}
    for record in dataset.records:
        votes = {
            member.name: member.verifier(
                record.get("proof"),
                int(limits["minimum_operand"]),
                int(limits["maximum_operand"]),
                int(limits["maximum_steps"]),
            )
            for member in member_list
        }
        ensemble = _combine_votes(votes)
        states = {decision.state for decision in votes.values()}
        row = {
            "case_id": record["case_id"],
            "attack_class": record["attack_class"],
            "expected": record["expected_state"],
            "predicted": ensemble.state,
            "correct": ensemble.state == record["expected_state"],
            "reason": ensemble.reason,
            "votes": {name: asdict(decision) for name, decision in votes.items()},
            "member_disagreement": len(states) > 1,
        }
        predictions.append(row)
        for name, decision in votes.items():
            member_predictions[name].append({
                "case_id": record["case_id"],
                "attack_class": record["attack_class"],
                "expected": record["expected_state"],
                "predicted": decision.state,
                "correct": decision.state == record["expected_state"],
                "reason": decision.reason,
            })

    metrics = _metrics(dataset.records, predictions)
    member_metrics = {
        name: _metrics(dataset.records, rows)
        for name, rows in member_predictions.items()
    }
    disagreements = sum(row["member_disagreement"] for row in predictions)
    unsupported_rows = [row for row in predictions if row["expected"] == "UNCERTAIN"]
    checks = {
        "member_count_at_least_3": len(member_list) >= 3,
        "all_cases_have_all_votes": all(len(row["votes"]) == len(member_list) for row in predictions),
        "zero_false_acceptance": metrics.far == 0.0,
        "zero_false_rejection": metrics.frr == 0.0,
        "robustness_at_least_1_0": metrics.robustness >= 1.0,
        "ensemble_accuracy_at_least_best_member": metrics.accuracy >= max(
            m.accuracy for m in member_metrics.values()),
        "unsupported_rule_not_forced_verified": all(
            row["predicted"] == "UNCERTAIN" for row in unsupported_rows),
    }
    limitations = list(dataset.document.get("limitations", [])) + [
        "Ensemble yalnız aynı kapalı aritmetik proof sözleşmesini üç ayrı denetimle ölçer; genel theorem prover değildir.",
        "Oybirliği politikası false accept riskini azaltır ama desteklenmeyen doğru ispatları UNCERTAIN bırakabilir.",
    ]
    return VerifierEnsembleReport(
        protocol=ENSEMBLE_PROTOCOL,
        benchmark_id=dataset.document["benchmark_id"],
        dataset_hash=dataset.dataset_hash(),
        ensemble_hash=_dataset_vote_hash(dataset.dataset_hash(), names),
        ensemble_policy=ENSEMBLE_POLICY,
        members=names,
        metrics=metrics,
        member_metrics=member_metrics,
        disagreement_rate=_ratio(disagreements, len(predictions)),
        by_attack_class=_by_attack_class(dataset.records, predictions),
        predictions=predictions,
        checks=checks,
        limitations=limitations,
    )


def verifier_ensemble_markdown(report: VerifierEnsembleReport) -> str:
    lines = [
        "# Çoklu Verifier Ensemble Benchmarkı",
        "",
        f"- Protokol: `{report.protocol}`",
        f"- Dataset: `{report.benchmark_id}` · hash: `{report.dataset_hash[:12]}`",
        f"- Ensemble hash: `{report.ensemble_hash[:12]}`",
        f"- Politika: `{report.ensemble_policy}`",
        f"- Üyeler: {', '.join(report.members)}",
        f"- Accuracy: `{report.metrics.accuracy:.3f}`",
        f"- FAR: `{report.metrics.far:.3f}` · FRR: `{report.metrics.frr:.3f}`",
        f"- Coverage: `{report.metrics.coverage:.3f}` · Robustness: `{report.metrics.robustness:.3f}`",
        f"- Üye anlaşmazlık oranı: `{report.disagreement_rate:.3f}`",
        "",
        "## Üye metrikleri",
        "",
        "| üye | accuracy | FAR | FRR | coverage | robustness |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in report.member_metrics.items():
        lines.append(
            f"| {name} | {metrics.accuracy:.3f} | {metrics.far:.3f} | "
            f"{metrics.frr:.3f} | {metrics.coverage:.3f} | {metrics.robustness:.3f} |"
        )
    lines.extend([
        "", "## Saldırı sınıfları", "",
        "| sınıf | N | doğru | accuracy | disagreement |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, class_metrics in sorted(report.by_attack_class.items()):
        lines.append(
            f"| {name} | {class_metrics['total']} | {class_metrics['correct']} | "
            f"{class_metrics['accuracy']:.3f} | "
            f"{class_metrics['disagreement_rate']:.3f} |"
        )
    lines.extend(["", "## Kabul kapıları", "", "| kapı | sonuç |", "|---|---|"])
    for name, passed in report.checks.items():
        lines.append(f"| {name} | {'GEÇTİ' if passed else 'KALDI'} |")
    lines.extend(["", "## Sınırlar", ""])
    lines.extend(f"- {note}" for note in report.limitations)
    return "\n".join(lines) + "\n"


__all__ = [
    "ENSEMBLE_PROTOCOL", "ENSEMBLE_POLICY", "VerifierMember",
    "VerifierEnsembleReport", "default_verifier_members",
    "verify_normal_form_addition", "verify_trace_replay_addition",
    "run_verifier_ensemble_benchmark", "verifier_ensemble_markdown",
]
