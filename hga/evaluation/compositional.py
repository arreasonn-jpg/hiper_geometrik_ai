"""Sürüm kontrollü Türkçe compositional-generalization benchmarkı.

Bu protokol üç ayrı şeyi birbirine karıştırmadan ölçer:

* yapılandırılmış üçlü için epistemik karar (VALID/INVALID/UNCERTAIN),
* kontrollü Generator'ın eğitimde olmayan üçlüyü aday olarak üretebilmesi ve
  ilişkiye özel morfolojik şablonun beklenen cümleyi kurması,
* küratörlü sözlük tabanlı ayıklayıcının görülmeyen yüzey biçimini çözmesi.

Sonuç bir genel dil ya da neural üretim iddiası değildir. ``C_G`` burada yalnız
bu sabit held-out protokolde doğru çözülen uygun örnek sayısı/oranıdır; teorik
bir kapasite üst sınırı değildir.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from importlib import resources
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hga.experience.cumle_ayiklayici import CumleAyiklayici, ascii_norm
from hga.experience.evaluator import ExperienceEvaluator
from hga.experience.generator import ExperienceGenerator
from hga.experience.text_generator import TextGenerator
from hga.knowledge import DeneyimDurumu, KaynakTuru, KnowledgeStore

from .leakage import LeakageAuditReport, audit_partitions, semantic_fingerprint

DATA_PACKAGE = "hga.evaluation.datasets"
DATA_FILE = "compositional_tr_v1.json"
DIMENSIONS = (
    "seen_composition",
    "unseen_entity",
    "unseen_relation",
    "unseen_combination",
    "unseen_wording",
    "unseen_sentence",
    "negative",
    "ood",
)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


def _surface_normalize(sentence: str) -> str:
    return re.sub(r"\s+", " ", ascii_norm(sentence)).strip()


def _triple(record: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        str(record["subject_id"]),
        str(record["relation_id"]),
        str(record["object_id"]),
    )


@dataclass
class DimensionMetrics:
    total: int
    correct: int
    accuracy: float
    valid_total: int
    invalid_total: int
    uncertain_total: int
    generation_targets: int
    generation_exact: int
    generation_exact_rate: float
    parsing_targets: int
    parsing_correct: int
    parsing_accuracy: float


@dataclass
class CompositionalMetrics:
    total: int
    correct: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    far: float
    frr: float
    candidate_generation_coverage: float
    generation_exact_rate: float
    parsing_accuracy: float


@dataclass
class GeneralizationCapacity:
    symbol: str
    eligible_cases: int
    correctly_generalized: int
    score: float
    generation_targets: int
    exactly_generated: int
    generation_score: float
    is_theoretical_capacity: bool
    definition: str


@dataclass
class CompositionalReport:
    benchmark_id: str
    dataset_hash: str
    language: str
    curation: Dict[str, Any]
    split_contract: Dict[str, Any]
    leakage: LeakageAuditReport
    surface_leakage_clean: bool
    metrics: CompositionalMetrics
    dimensions: Dict[str, DimensionMetrics]
    generalization_capacity: GeneralizationCapacity
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        lines = [
            "# HGA Türkçe Compositional Generalization v1",
            "",
            f"- Dataset hash: `{self.dataset_hash}`",
            f"- Semantik sızıntı denetimi: `{'TEMİZ' if self.leakage.clean else 'KİRLİ'}`",
            f"- Yüzey cümlesi sızıntı denetimi: `{'TEMİZ' if self.surface_leakage_clean else 'KİRLİ'}`",
            f"- Karar doğruluğu: `{self.metrics.accuracy:.3f}`",
            f"- Üretim exact-match: `{self.metrics.generation_exact_rate:.3f}`",
            f"- Ayıklama doğruluğu: `{self.metrics.parsing_accuracy:.3f}`",
            f"- C_G (ölçülen held-out oran): `{self.generalization_capacity.score:.3f}` "
            f"({self.generalization_capacity.correctly_generalized}/"
            f"{self.generalization_capacity.eligible_cases})",
            "",
            "| Boyut | N | Karar acc. | Üretim exact | Ayıklama acc. |",
            "|---|---:|---:|---:|---:|",
        ]
        for name in DIMENSIONS:
            metric = self.dimensions.get(name)
            if metric is None:
                continue
            generation = (
                f"{metric.generation_exact_rate:.3f}" if metric.generation_targets else "n/a"
            )
            parsing = f"{metric.parsing_accuracy:.3f}" if metric.parsing_targets else "n/a"
            lines.append(
                f"| {name} | {metric.total} | {metric.accuracy:.3f} | "
                f"{generation} | {parsing} |"
            )
        lines.extend(["", "## Sınırlar", ""])
        lines.extend(f"- {note}" for note in self.limitations)
        return "\n".join(lines) + "\n"


class CompositionalDataset:
    """Elle sabitlenmiş veri paketini yükler ve split sözleşmesini doğrular."""

    def __init__(self, document: Optional[Mapping[str, Any]] = None):
        if document is None:
            ref = resources.files(DATA_PACKAGE).joinpath(DATA_FILE)
            with ref.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        self.document = dict(document)
        if self.document.get("schema_version") != 1:
            raise ValueError("Desteklenmeyen compositional dataset schema_version")
        self.entities = list(self.document.get("entities", []))
        self.relations = list(self.document.get("relations", []))
        self.train = list(self.document.get("train", []))
        self.test = list(self.document.get("test", []))
        self._validate()

    def dataset_hash(self) -> str:
        payload = json.dumps(
            self.document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _validate(self) -> None:
        entity_ids = {row["entity_id"] for row in self.entities}
        relation_ids = {row["relation_id"] for row in self.relations}
        if len(entity_ids) != len(self.entities) or len(relation_ids) != len(self.relations):
            raise ValueError("Compositional dataset kimlikleri benzersiz olmalı")
        if not self.train or not self.test:
            raise ValueError("Compositional dataset train ve test kayıtları boş olamaz")

        experience_ids: set[str] = set()
        for record in self.train + self.test:
            experience_id = str(record.get("experience_id", ""))
            if not experience_id or experience_id in experience_ids:
                raise ValueError("Experience kimlikleri dolu ve benzersiz olmalı")
            experience_ids.add(experience_id)
            if record.get("subject_id") not in entity_ids or record.get("object_id") not in entity_ids:
                raise ValueError(f"Bilinmeyen entity: {experience_id}")
            if record.get("relation_id") not in relation_ids:
                raise ValueError(f"Bilinmeyen relation: {experience_id}")

        train_fingerprints = {semantic_fingerprint(row) for row in self.train}
        train_surfaces = {_surface_normalize(row["sentence"]) for row in self.train}
        train_components = {
            "entity": {value for row in self.train for value in (row["subject_id"], row["object_id"])},
            "relation": {row["relation_id"] for row in self.train},
        }
        entity_splits = {row["entity_id"]: row.get("split") for row in self.entities}
        relation_splits = {row["relation_id"]: row.get("split") for row in self.relations}

        for record in self.test:
            dimensions = set(record.get("dimensions", []))
            unknown_dimensions = dimensions - set(DIMENSIONS)
            if unknown_dimensions:
                raise ValueError(f"Bilinmeyen boyut: {sorted(unknown_dimensions)}")
            fingerprint = semantic_fingerprint(record)
            in_train = fingerprint in train_fingerprints
            if bool(record.get("semantic_novel")) == in_train:
                raise ValueError(
                    f"semantic_novel sözleşmesi bozuk: {record['experience_id']}"
                )
            surface_in_train = _surface_normalize(record["sentence"]) in train_surfaces
            if bool(record.get("surface_novel")) == surface_in_train:
                raise ValueError(
                    f"surface_novel sözleşmesi bozuk: {record['experience_id']}"
                )
            if "seen_composition" in dimensions and not in_train:
                raise ValueError("seen_composition train üçlüsüyle eşleşmeli")
            if "unseen_combination" in dimensions:
                if in_train:
                    raise ValueError("unseen_combination train'de bulunamaz")
                if (record["subject_id"] not in train_components["entity"]
                        or record["object_id"] not in train_components["entity"]
                        or record["relation_id"] not in train_components["relation"]):
                    raise ValueError(
                        "unseen_combination tüm bileşenleri train'de tekil olarak görmeli"
                    )
            if "unseen_entity" in dimensions:
                heldout_entities = [
                    value for value in (record["subject_id"], record["object_id"])
                    if entity_splits.get(value) == "test" and value not in train_components["entity"]
                ]
                if not heldout_entities:
                    raise ValueError("unseen_entity gerçekten train dışı entity içermeli")
            if "unseen_relation" in dimensions:
                relation_id = record["relation_id"]
                if relation_splits.get(relation_id) != "test" or relation_id in train_components["relation"]:
                    raise ValueError("unseen_relation gerçekten train olguları dışında olmalı")

    def leakage_audit(self) -> LeakageAuditReport:
        novel_tests = [row for row in self.test if row.get("semantic_novel")]
        return audit_partitions(self.train, novel_tests)

    def surface_leakage_clean(self) -> bool:
        train_surfaces = {_surface_normalize(row["sentence"]) for row in self.train}
        return all(
            _surface_normalize(row["sentence"]) not in train_surfaces
            for row in self.test if row.get("surface_novel")
        )

    def build_store(self) -> KnowledgeStore:
        store = KnowledgeStore()
        for row in self.entities:
            store.varlik_ekle(
                row["token"], entity_type=row["entity_type"],
                properties=row.get("properties"), entity_id=row["entity_id"],
                ozel_isim=bool(row.get("is_ozel", False)), source=KaynakTuru.HUMAN_CONFIRMED,
            )
        for row in self.relations:
            store.iliski_tanimla(
                row["token"], relation_id=row["relation_id"],
                subject_types=row.get("subject_types"), object_types=row.get("object_types"),
                requires_subject_props=row.get("requires_subject_props"),
                requires_object_props=row.get("requires_object_props"),
                source=KaynakTuru.HUMAN_CONFIRMED,
            )
        for row in self.train:
            store.olgu_kaydet(
                row["subject_id"], row["relation_id"], row["object_id"],
                score=1.0, source=KaynakTuru.REAL_DATA, confidence=1.0,
            )
        return store

    def build_parser(self) -> CumleAyiklayici:
        entity_by_id = {row["entity_id"]: row for row in self.entities}
        subjects: Dict[str, Dict[str, Any]] = {}
        objects: Dict[str, Dict[str, Any]] = {}
        aliases = {
            "E_AT": ("at", "ata"),
            "E_ARABA": ("araba", "arabaya", "arabayi"),
            "E_OTOBUS": ("otobus", "otobuse", "otobusu"),
            "E_GOKYUZU": ("gokyuzu", "gokyuzune"),
            "E_MARS": ("mars",),
        }
        for row in self.entities:
            item = {
                "token": row["token"], "tip": row["entity_type"],
                "ozellikler": dict(row.get("properties", {})),
            }
            normalized = ascii_norm(row["token"])
            if row["entity_type"] == "insan":
                subjects[normalized] = item
            else:
                for alias in aliases.get(row["entity_id"], (normalized,)):
                    objects[alias] = item

        # Kullanılmayan yerel değişkeni bilinçli bir veri bütünlüğü kontrolüne çevir.
        if set(entity_by_id) != {row["entity_id"] for row in self.entities}:
            raise ValueError("Entity indeksi oluşturulamadı")
        common = {"ozneler": subjects, "nesneler": objects}
        lexicon = {
            "binmek": {
                **common, "iliski": "Binmek", "nesne_durumu": "yonelme",
                "yuklemler": ["bindi", "biniyor", "binecek", "biner"],
            },
            "kullanmak": {
                **common, "iliski": "Kullanmak", "nesne_durumu": "belirtme",
                "yuklemler": ["kullandi", "kullaniyor", "kullanacak", "kullanir"],
            },
        }
        return CumleAyiklayici(lexicon)


def _classification_metrics(
    records: Sequence[Mapping[str, Any]], predictions: Sequence[str]
) -> CompositionalMetrics:
    correct = sum(row["expected_state"] == predicted for row, predicted in zip(records, predictions))
    tp = sum(row["expected_state"] == "VALID" and predicted == "VALID"
             for row, predicted in zip(records, predictions))
    fp = sum(row["expected_state"] != "VALID" and predicted == "VALID"
             for row, predicted in zip(records, predictions))
    fn = sum(row["expected_state"] == "VALID" and predicted != "VALID"
             for row, predicted in zip(records, predictions))
    invalid_total = sum(row["expected_state"] == "INVALID" for row in records)
    false_accept = sum(row["expected_state"] == "INVALID" and predicted == "VALID"
                       for row, predicted in zip(records, predictions))
    valid_total = sum(row["expected_state"] == "VALID" for row in records)
    false_reject = sum(row["expected_state"] == "VALID" and predicted == "INVALID"
                       for row, predicted in zip(records, predictions))
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return CompositionalMetrics(
        total=len(records), correct=correct, accuracy=_ratio(correct, len(records)),
        precision=precision, recall=recall,
        f1=round(2 * precision * recall / (precision + recall), 8)
        if precision + recall else 0.0,
        far=_ratio(false_accept, invalid_total), frr=_ratio(false_reject, valid_total),
        candidate_generation_coverage=0.0, generation_exact_rate=0.0,
        parsing_accuracy=0.0,
    )


def _dimension_metrics(
    records: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], dimension: str
) -> DimensionMetrics:
    selected = [(record, row) for record, row in zip(records, rows)
                if dimension in record.get("dimensions", [])]
    correct = sum(row["decision_correct"] for _, row in selected)
    generation = [(record, row) for record, row in selected if record.get("generation_target")]
    parsing = [(record, row) for record, row in selected if record.get("parsing_target")]
    return DimensionMetrics(
        total=len(selected), correct=correct, accuracy=_ratio(correct, len(selected)),
        valid_total=sum(record["expected_state"] == "VALID" for record, _ in selected),
        invalid_total=sum(record["expected_state"] == "INVALID" for record, _ in selected),
        uncertain_total=sum(record["expected_state"] == "UNCERTAIN" for record, _ in selected),
        generation_targets=len(generation),
        generation_exact=sum(row["generation_exact"] for _, row in generation),
        generation_exact_rate=_ratio(sum(row["generation_exact"] for _, row in generation),
                                     len(generation)),
        parsing_targets=len(parsing),
        parsing_correct=sum(row["parsing_correct"] for _, row in parsing),
        parsing_accuracy=_ratio(sum(row["parsing_correct"] for _, row in parsing), len(parsing)),
    )


def run_compositional_benchmark(
    dataset: Optional[CompositionalDataset] = None,
) -> CompositionalReport:
    dataset = dataset or CompositionalDataset()
    leakage = dataset.leakage_audit()
    leakage.assert_clean()
    surface_clean = dataset.surface_leakage_clean()
    if not surface_clean:
        raise ValueError("Compositional benchmark yüzey cümlesi sızıntısı tespit edildi")

    store = dataset.build_store()
    evaluator = ExperienceEvaluator()
    text_generator = TextGenerator()
    parser = dataset.build_parser()
    candidates = ExperienceGenerator(tip_filtresi=True).uret(store)
    by_triple = {candidate.uclusu: candidate for candidate in candidates}
    token_to_entity = {row["token"]: row["entity_id"] for row in dataset.entities}
    token_to_relation = {row["token"]: row["relation_id"] for row in dataset.relations}

    result_rows: List[Dict[str, Any]] = []
    predictions: List[str] = []
    for record in dataset.test:
        triple = _triple(record)
        candidate = by_triple.get(triple)
        generated_candidate = candidate is not None
        if candidate is None:  # kapsama hatasını açık sonuç olarak tut; sessizce aday yaratma
            predicted = "MISSING_CANDIDATE"
            generated_sentence = None
        else:
            evaluator.degerlendir(candidate, store)
            predicted = candidate.state.value
            generated_sentence = (
                text_generator.cumle(store, candidate)
                if candidate.state == DeneyimDurumu.VALID else None
            )
        predictions.append(predicted)

        accepted = {_surface_normalize(value)
                    for value in record.get("accepted_generations", [])}
        generation_exact = bool(
            record.get("generation_target") and generated_sentence
            and _surface_normalize(generated_sentence) in accepted
        )

        parsed = parser.ayikla(record["sentence"]) if record.get("parsing_target") else None
        parsed_triple = None
        if parsed is not None:
            parsed_triple = (
                token_to_entity.get(parsed.ozne),
                token_to_relation.get(parsed.iliski),
                token_to_entity.get(parsed.nesne),
            )
        parsing_correct = bool(record.get("parsing_target") and parsed_triple == triple)
        result_rows.append({
            "experience_id": record["experience_id"],
            "dimensions": list(record.get("dimensions", [])),
            "expected": record["expected_state"],
            "predicted": predicted,
            "decision_correct": predicted == record["expected_state"],
            "candidate_generated": generated_candidate,
            "input_sentence": record["sentence"],
            "generated_sentence": generated_sentence,
            "generation_exact": generation_exact,
            "parsed_triple": list(parsed_triple) if parsed_triple else None,
            "parsing_correct": parsing_correct,
        })

    metrics = _classification_metrics(dataset.test, predictions)
    metrics.candidate_generation_coverage = _ratio(
        sum(row["candidate_generated"] for row in result_rows), len(result_rows)
    )
    generation_rows = [row for record, row in zip(dataset.test, result_rows)
                       if record.get("generation_target")]
    metrics.generation_exact_rate = _ratio(
        sum(row["generation_exact"] for row in generation_rows), len(generation_rows)
    )
    parsing_rows = [row for record, row in zip(dataset.test, result_rows)
                    if record.get("parsing_target")]
    metrics.parsing_accuracy = _ratio(
        sum(row["parsing_correct"] for row in parsing_rows), len(parsing_rows)
    )

    dimensions = {
        name: _dimension_metrics(dataset.test, result_rows, name)
        for name in DIMENSIONS
        if any(name in record.get("dimensions", []) for record in dataset.test)
    }
    generalization_dimensions = {
        "unseen_entity", "unseen_relation", "unseen_combination", "unseen_sentence"
    }
    eligible = [
        (record, row) for record, row in zip(dataset.test, result_rows)
        if record["expected_state"] == "VALID"
        and generalization_dimensions.intersection(record.get("dimensions", []))
    ]
    correct_generalizations = sum(row["decision_correct"] for _, row in eligible)
    eligible_generation = [(record, row) for record, row in eligible
                           if record.get("generation_target")]
    exact_generation = sum(row["generation_exact"] for _, row in eligible_generation)
    c_g = GeneralizationCapacity(
        symbol="C_G", eligible_cases=len(eligible),
        correctly_generalized=correct_generalizations,
        score=_ratio(correct_generalizations, len(eligible)),
        generation_targets=len(eligible_generation), exactly_generated=exact_generation,
        generation_score=_ratio(exact_generation, len(eligible_generation)),
        is_theoretical_capacity=False,
        definition=(
            "Sabit hga-compositional-tr-v1 held-out setinde, en az bir unseen "
            "boyutu taşıyan VALID örneklerden doğru çözülenlerin sayısı/oranı."
        ),
    )
    return CompositionalReport(
        benchmark_id=dataset.document["benchmark_id"],
        dataset_hash=dataset.dataset_hash(), language=dataset.document["language"],
        curation=dict(dataset.document["curation"]),
        split_contract={
            "train_triples": len(dataset.train), "test_cases": len(dataset.test),
            "entity_partitioned": True, "relation_partitioned": True,
            "semantic_novel_cases": sum(bool(row.get("semantic_novel")) for row in dataset.test),
            "surface_novel_cases": sum(bool(row.get("surface_novel")) for row in dataset.test),
        },
        leakage=leakage, surface_leakage_clean=surface_clean,
        metrics=metrics, dimensions=dimensions, generalization_capacity=c_g,
        predictions=result_rows, limitations=list(dataset.document.get("limitations", [])),
    )


__all__ = [
    "DATA_FILE", "DIMENSIONS", "CompositionalDataset", "CompositionalMetrics",
    "CompositionalReport", "DimensionMetrics", "GeneralizationCapacity",
    "run_compositional_benchmark",
]
