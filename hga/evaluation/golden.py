"""Elle sabitlenmiş HGA golden benchmark yükleyicisi ve ölçümleri."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from importlib import resources
from typing import Any, Dict, List, Mapping, Optional

from hga.experience.evaluator import ExperienceEvaluator
from hga.knowledge import DeneyimDurumu, ExperienceCandidate, KaynakTuru, KnowledgeStore

from .experiment import SeedSweepReport, run_seed_sweep
from .leakage import LeakageAuditReport, audit_partitions

DATA_FILES = (
    "entities.json",
    "properties.json",
    "relations.json",
    "experiences.json",
    "conflicts.json",
    "negative_cases.json",
    "expected_results.json",
)


def _read_json(name: str) -> Dict[str, Any]:
    ref = resources.files("golden_dataset").joinpath(name)
    with ref.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1 or not isinstance(payload.get("records"), list):
        raise ValueError(f"Geçersiz golden dosyası: {name}")
    return payload


def _safe_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


@dataclass
class GoldenMetrics:
    total: int
    correct: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    far: float
    frr: float
    invalid_total: int
    valid_total: int
    uncertain_total: int
    conflict_total: int
    confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class GoldenReport:
    dataset_hash: str
    leakage: LeakageAuditReport
    metrics: GoldenMetrics
    predictions: List[Dict[str, str]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GoldenDataset:
    """Golden kayıtları değişikliğe uğratmadan okuyan sürüm-1 veri görünümü."""

    def __init__(self):
        self.documents = {name: _read_json(name) for name in DATA_FILES}
        self.entities = self.documents["entities.json"]["records"]
        self.properties = self.documents["properties.json"]["records"]
        self.relations = self.documents["relations.json"]["records"]
        self.experiences = self.documents["experiences.json"]["records"]
        self.conflicts = self.documents["conflicts.json"]["records"]
        self.expected = self.documents["expected_results.json"]["records"]
        expected_ids = {row["experience_id"] for row in self.expected}
        test_ids = {row["experience_id"] for row in self.test_records}
        if expected_ids != test_ids:
            raise ValueError("Golden expected_results ile test experience kimlikleri eşleşmiyor")

    @property
    def train_records(self) -> List[Mapping[str, Any]]:
        return [row for row in self.experiences if row.get("split") == "train"]

    @property
    def test_records(self) -> List[Mapping[str, Any]]:
        return [row for row in self.experiences if row.get("split") == "test"]

    def dataset_hash(self) -> str:
        digest = hashlib.sha256()
        for name in sorted(DATA_FILES):
            canonical = json.dumps(
                self.documents[name], ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            digest.update(name.encode("utf-8") + b"\0" + canonical + b"\0")
        return digest.hexdigest()

    def leakage_audit(self, **partitions) -> LeakageAuditReport:
        return audit_partitions(self.train_records, self.test_records, **partitions)

    def build_store(self) -> KnowledgeStore:
        store = KnowledgeStore()
        for row in self.entities:
            store.varlik_ekle(
                row["token"],
                entity_type=row.get("entity_type", "kavram"),
                entity_id=row["entity_id"],
                ozel_isim=bool(row.get("is_ozel", False)),
            )
        for row in self.properties:
            store.ozellik_koy(
                row["entity_id"], row["name"], row["value"],
                source=KaynakTuru(row["source"]), confidence=float(row["confidence"]),
            )
        for row in self.relations:
            store.iliski_tanimla(
                row["token"], relation_id=row["relation_id"],
                subject_types=row.get("subject_types"), object_types=row.get("object_types"),
                requires_object_props=row.get("requires_object_props"),
                requires_subject_props=row.get("requires_subject_props"),
                source=KaynakTuru(row["source"]),
            )
        # Yalnız train gerçekleri bilgi tabanına girer. Test beklenen sonuçları
        # evaluator/verifier girdisine hiçbir zaman yüklenmez.
        for row in self.train_records:
            store.olgu_kaydet(
                row["subject_id"], row["relation_id"], row["object_id"],
                score=1.0 if row.get("truth") else 0.0,
                source=KaynakTuru.REAL_DATA, confidence=1.0,
            )
        # Çelişki fixture'ları yalnız kendi test üçlüsüne ait sabit kanıtlardır.
        for fixture in self.conflicts:
            for evidence in fixture.get("evidence", []):
                store.olgu_kaydet(
                    evidence["subject_id"], evidence["relation_id"], evidence["object_id"],
                    score=float(evidence["score"]), source=KaynakTuru(evidence["source"]),
                    confidence=float(evidence["confidence"]),
                )
        return store


def _metrics(expected: Dict[str, str], predicted: Dict[str, str]) -> GoldenMetrics:
    labels = [state.value for state in (
        DeneyimDurumu.VALID, DeneyimDurumu.INVALID,
        DeneyimDurumu.UNCERTAIN, DeneyimDurumu.CONFLICT,
    )]
    confusion = {truth: {guess: 0 for guess in labels} for truth in labels}
    for exp_id, truth in expected.items():
        confusion[truth][predicted[exp_id]] += 1

    correct = sum(expected[key] == predicted[key] for key in expected)
    valid_total = sum(value == DeneyimDurumu.VALID.value for value in expected.values())
    invalid_total = sum(value == DeneyimDurumu.INVALID.value for value in expected.values())
    tp = sum(
        expected[key] == DeneyimDurumu.VALID.value
        and predicted[key] == DeneyimDurumu.VALID.value
        for key in expected
    )
    false_positive = sum(
        expected[key] != DeneyimDurumu.VALID.value
        and predicted[key] == DeneyimDurumu.VALID.value
        for key in expected
    )
    false_negative = sum(
        expected[key] == DeneyimDurumu.VALID.value
        and predicted[key] != DeneyimDurumu.VALID.value
        for key in expected
    )
    false_accepted = sum(
        expected[key] == DeneyimDurumu.INVALID.value
        and predicted[key] == DeneyimDurumu.VALID.value
        for key in expected
    )
    false_rejected = sum(
        expected[key] == DeneyimDurumu.VALID.value
        and predicted[key] == DeneyimDurumu.INVALID.value
        for key in expected
    )
    precision = _safe_ratio(tp, tp + false_positive)
    recall = _safe_ratio(tp, tp + false_negative)
    return GoldenMetrics(
        total=len(expected), correct=correct, accuracy=_safe_ratio(correct, len(expected)),
        precision=precision, recall=recall,
        f1=round(2 * precision * recall / (precision + recall), 6)
        if precision + recall else 0.0,
        far=_safe_ratio(false_accepted, invalid_total),
        frr=_safe_ratio(false_rejected, valid_total),
        invalid_total=invalid_total, valid_total=valid_total,
        uncertain_total=sum(v == DeneyimDurumu.UNCERTAIN.value for v in expected.values()),
        conflict_total=sum(v == DeneyimDurumu.CONFLICT.value for v in expected.values()),
        confusion=confusion,
    )


def run_golden_benchmark(evaluator: Optional[ExperienceEvaluator] = None) -> GoldenReport:
    dataset = GoldenDataset()
    leakage = dataset.leakage_audit()
    leakage.assert_clean()
    store = dataset.build_store()
    evaluator = evaluator or ExperienceEvaluator()
    expected = {row["experience_id"]: row["expected_state"] for row in dataset.expected}
    predicted: Dict[str, str] = {}
    rows: List[Dict[str, str]] = []
    for row in dataset.test_records:
        candidate = ExperienceCandidate(
            row["experience_id"], row["subject_id"], row["relation_id"], row["object_id"]
        )
        evaluator.degerlendir(candidate, store)
        predicted[candidate.experience_id] = candidate.state.value
        rows.append({
            "experience_id": candidate.experience_id,
            "expected": expected[candidate.experience_id],
            "predicted": candidate.state.value,
        })
    return GoldenReport(dataset.dataset_hash(), leakage, _metrics(expected, predicted), rows)


def run_golden_seed_sweep(root, seeds) -> SeedSweepReport:
    """Golden v1'i manifestli çoklu-seed tekrarlanabilirlik koşusuna bağla."""
    dataset = GoldenDataset()
    config = {
        "benchmark": "hga-golden-v1",
        "dataset_files": list(DATA_FILES),
        "epistemic_states": ["VALID", "INVALID", "UNCERTAIN", "CONFLICT"],
    }

    def run_one(_seed: int) -> Dict[str, Any]:
        report = run_golden_benchmark()
        print(
            f"golden accuracy={report.metrics.accuracy} "
            f"FAR={report.metrics.far} FRR={report.metrics.frr}"
        )
        return report.to_dict()

    return run_seed_sweep(
        run_one, seeds=seeds, root=root, config=config,
        dataset_hash=dataset.dataset_hash(),
        parameters={"test_count": len(dataset.test_records), "deterministic": True},
    )


__all__ = [
    "GoldenDataset",
    "GoldenMetrics",
    "GoldenReport",
    "run_golden_benchmark",
    "run_golden_seed_sweep",
]
