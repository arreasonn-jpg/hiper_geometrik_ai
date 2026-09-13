"""Katı şemalı verifier attack suite.

Amaç evaluator kararını bozmak değil, bir proof verifier'ın yanlış, eksik,
çelişkili, bozuk ve adversarial girdilere verdiği kararı doğrudan ölçmektir.
Desteklenmeyen ama iyi biçimli kural ``UNCERTAIN`` olur; bozuk ispat hiçbir
zaman kanıt yokluğu gibi gizlenmez ve ``INVALID`` döner.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from importlib import resources
from typing import Any, Dict, List, Mapping, Optional, Sequence

DATA_PACKAGE = "hga.evaluation.datasets"
DATA_FILE = "verifier_adversarial_v1.json"
STATES = ("VERIFIED", "INVALID", "UNCERTAIN")
REQUIRED_ATTACK_CLASSES = {
    "false_proof",
    "incomplete_proof",
    "contradictory_proof",
    "malformed_proof",
    "boundary_case",
    "adversarial_input",
}


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


@dataclass
class ProofDecision:
    state: str
    reason: str


@dataclass
class VerifierAttackMetrics:
    total: int
    correct: int
    accuracy: float
    valid_proofs: int
    invalid_or_unsupported_proofs: int
    true_acceptance: int
    true_rejection: int
    false_acceptance: int
    false_rejection: int
    uncertain: int
    precision: float
    recall: float
    f1: float
    far: float
    frr: float
    coverage: float
    robustness: float


@dataclass
class VerifierAttackReport:
    benchmark_id: str
    dataset_hash: str
    metrics: VerifierAttackMetrics
    by_attack_class: Dict[str, Dict[str, Any]]
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        lines = [
            "# HGA Verifier Adversarial Suite v1",
            "",
            f"- Dataset hash: `{self.dataset_hash}`",
            f"- Accuracy: `{self.metrics.accuracy:.3f}`",
            f"- FAR: `{self.metrics.far:.3f}`",
            f"- FRR: `{self.metrics.frr:.3f}`",
            f"- Coverage: `{self.metrics.coverage:.3f}`",
            f"- Robustness: `{self.metrics.robustness:.3f}`",
            "",
            "| Saldırı sınıfı | N | Doğru | Accuracy |",
            "|---|---:|---:|---:|",
        ]
        for name, metrics in sorted(self.by_attack_class.items()):
            lines.append(
                f"| {name} | {metrics['total']} | {metrics['correct']} | "
                f"{metrics['accuracy']:.3f} |"
            )
        lines.extend(["", "## Sınırlar", ""])
        lines.extend(f"- {note}" for note in self.limitations)
        return "\n".join(lines) + "\n"


class VerifierAttackDataset:
    def __init__(self, document: Optional[Mapping[str, Any]] = None):
        if document is None:
            ref = resources.files(DATA_PACKAGE).joinpath(DATA_FILE)
            with ref.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        self.document = dict(document)
        if self.document.get("schema_version") != 1:
            raise ValueError("Desteklenmeyen verifier adversarial schema_version")
        self.records = list(self.document.get("records", []))
        self.limits = dict(self.document.get("limits", {}))
        self._validate()

    def _validate(self) -> None:
        if not self.records:
            raise ValueError("Verifier adversarial records boş olamaz")
        ids = [record.get("case_id") for record in self.records]
        if any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("Verifier case_id değerleri dolu ve benzersiz olmalı")
        classes = {record.get("attack_class") for record in self.records}
        missing = REQUIRED_ATTACK_CLASSES - classes
        if missing:
            raise ValueError(f"Verifier saldırı sınıfları eksik: {sorted(missing)}")
        for record in self.records:
            if record.get("expected_state") not in STATES:
                raise ValueError(f"Geçersiz expected_state: {record.get('case_id')}")
            if not isinstance(record.get("proof_valid"), bool):
                raise ValueError(f"proof_valid bool olmalı: {record.get('case_id')}")
        minimum = self.limits.get("minimum_operand")
        maximum = self.limits.get("maximum_operand")
        steps = self.limits.get("maximum_steps")
        if not all(isinstance(value, int) and not isinstance(value, bool)
                   for value in (minimum, maximum, steps)):
            raise ValueError("Verifier limitleri tam sayı olmalı")
        # Yukarıdaki kapı üçünün de bool olmayan int olduğunu garanti eder;
        # aşağıdaki karşılaştırma için tipi açıkça daraltıyoruz.
        minimum_int, maximum_int, steps_int = int(minimum), int(maximum), int(steps)  # type: ignore[arg-type]
        if minimum_int >= maximum_int or steps_int < 1:
            raise ValueError("Verifier limit aralığı geçersiz")

    def dataset_hash(self) -> str:
        payload = json.dumps(
            self.document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _strict_integer(value: Any, minimum: int, maximum: int) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and minimum <= value <= maximum
    )


def verify_arithmetic_proof(
    proof: Any,
    minimum_operand: int = -1_000_000,
    maximum_operand: int = 1_000_000,
    maximum_steps: int = 4,
) -> ProofDecision:
    """Tek adımlı tam sayı toplama ispatını kapalı ve katı şemayla doğrula."""
    if not isinstance(proof, Mapping):
        return ProofDecision("INVALID", "proof nesnesi mapping olmalı")
    if set(proof) != {"claim", "steps", "conclusion"}:
        return ProofDecision("INVALID", "proof alanları eksik veya fazladır")

    claim = proof["claim"]
    steps = proof["steps"]
    conclusion = proof["conclusion"]
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

    integer_values = (claim["left"], claim["right"], claim["result"], conclusion)
    if not all(_strict_integer(value, minimum_operand, maximum_operand)
               for value in integer_values):
        return ProofDecision("INVALID", "claim/conclusion katı ve sınırlı int olmalı")

    operator = claim["operator"]
    if not isinstance(operator, str):
        return ProofDecision("INVALID", "operator string olmalı")
    if operator != "add":
        return ProofDecision("UNCERTAIN", f"desteklenmeyen operator: {operator}")

    if len(steps) != 1:
        return ProofDecision("INVALID", "integer_addition tam bir adım olmalı")
    step = steps[0]
    if not isinstance(step, Mapping) or set(step) != {"rule", "inputs", "output"}:
        return ProofDecision("INVALID", "adım şeması geçersiz")
    inputs = step["inputs"]
    if not isinstance(inputs, list) or len(inputs) != 2:
        return ProofDecision("INVALID", "adım iki input içermeli")
    step_values = (inputs[0], inputs[1], step["output"])
    if not all(_strict_integer(value, minimum_operand, maximum_operand)
               for value in step_values):
        return ProofDecision("INVALID", "adım değerleri katı ve sınırlı int olmalı")
    if step["rule"] != "integer_addition":
        return ProofDecision("UNCERTAIN", f"desteklenmeyen rule: {step['rule']}")
    if inputs != [claim["left"], claim["right"]]:
        return ProofDecision("INVALID", "adım girdileri claim ile çelişiyor")

    computed = claim["left"] + claim["right"]
    if step["output"] != computed:
        return ProofDecision("INVALID", "adım çıktısı aritmetik oracle ile uyuşmuyor")
    if claim["result"] != computed:
        return ProofDecision("INVALID", "claim sonucu aritmetik oracle ile uyuşmuyor")
    if conclusion != step["output"] or conclusion != claim["result"]:
        return ProofDecision("INVALID", "conclusion claim/adım ile çelişiyor")
    return ProofDecision("VERIFIED", "katı şema ve bağımsız toplama oracle'ı doğruladı")


def _metrics(
    records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
) -> VerifierAttackMetrics:
    correct = sum(record["expected_state"] == prediction["predicted"]
                  for record, prediction in zip(records, predictions))
    valid = sum(record["proof_valid"] for record in records)
    invalid = len(records) - valid
    true_acceptance = sum(
        record["proof_valid"] and prediction["predicted"] == "VERIFIED"
        for record, prediction in zip(records, predictions)
    )
    false_acceptance = sum(
        not record["proof_valid"] and prediction["predicted"] == "VERIFIED"
        for record, prediction in zip(records, predictions)
    )
    false_rejection = sum(
        record["proof_valid"] and prediction["predicted"] != "VERIFIED"
        for record, prediction in zip(records, predictions)
    )
    true_rejection = sum(
        not record["proof_valid"] and prediction["predicted"] != "VERIFIED"
        for record, prediction in zip(records, predictions)
    )
    uncertain = sum(prediction["predicted"] == "UNCERTAIN" for prediction in predictions)
    precision = _ratio(true_acceptance, true_acceptance + false_acceptance)
    recall = _ratio(true_acceptance, valid)
    attacks = [
        (record, prediction) for record, prediction in zip(records, predictions)
        if record["attack_class"] not in {"valid_control", "boundary_case"}
    ]
    robust = sum(record["expected_state"] == prediction["predicted"]
                 for record, prediction in attacks)
    return VerifierAttackMetrics(
        total=len(records), correct=correct, accuracy=_ratio(correct, len(records)),
        valid_proofs=valid, invalid_or_unsupported_proofs=invalid,
        true_acceptance=true_acceptance, true_rejection=true_rejection,
        false_acceptance=false_acceptance, false_rejection=false_rejection,
        uncertain=uncertain, precision=precision, recall=recall,
        f1=round(2 * precision * recall / (precision + recall), 8)
        if precision + recall else 0.0,
        far=_ratio(false_acceptance, invalid), frr=_ratio(false_rejection, valid),
        coverage=_ratio(len(records) - uncertain, len(records)),
        robustness=_ratio(robust, len(attacks)),
    )


def run_verifier_adversarial_benchmark(
    dataset: Optional[VerifierAttackDataset] = None,
) -> VerifierAttackReport:
    dataset = dataset or VerifierAttackDataset()
    predictions: List[Dict[str, Any]] = []
    for record in dataset.records:
        decision = verify_arithmetic_proof(
            record.get("proof"),
            minimum_operand=int(dataset.limits["minimum_operand"]),
            maximum_operand=int(dataset.limits["maximum_operand"]),
            maximum_steps=int(dataset.limits["maximum_steps"]),
        )
        predictions.append({
            "case_id": record["case_id"],
            "attack_class": record["attack_class"],
            "expected": record["expected_state"],
            "predicted": decision.state,
            "correct": decision.state == record["expected_state"],
            "reason": decision.reason,
        })
    metrics = _metrics(dataset.records, predictions)
    by_class: Dict[str, Dict[str, Any]] = {}
    for attack_class in sorted({record["attack_class"] for record in dataset.records}):
        selected = [prediction for prediction in predictions
                    if prediction["attack_class"] == attack_class]
        correct = sum(prediction["correct"] for prediction in selected)
        by_class[attack_class] = {
            "total": len(selected), "correct": correct,
            "accuracy": _ratio(correct, len(selected)),
        }
    return VerifierAttackReport(
        benchmark_id=dataset.document["benchmark_id"],
        dataset_hash=dataset.dataset_hash(), metrics=metrics,
        by_attack_class=by_class, predictions=predictions,
        limitations=list(dataset.document.get("limitations", [])),
    )


__all__ = [
    "DATA_FILE", "REQUIRED_ATTACK_CLASSES", "ProofDecision", "VerifierAttackDataset",
    "VerifierAttackMetrics", "VerifierAttackReport", "run_verifier_adversarial_benchmark",
    "verify_arithmetic_proof",
]
