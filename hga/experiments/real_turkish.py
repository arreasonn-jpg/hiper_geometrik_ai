"""Gerçek ve insan anotasyonlu Türkçe dependency benchmarkı.

Veri, sabit upstream revision'daki Turkish Web Treebank'tir (TWT). Bu modül
TWT'nin basic dependency ağaçlarından doğrulanabilir arc adayları üretir. Görev
semantik ilişki çıkarımı ya da NER değildir: dependency etiketleri
morphosyntactic, ``entity`` anahtarı ise yalnız split için kullanılan
lemma/form kimliğidir.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .experiment import canonical_hash, file_sha256

DATA_DIR = Path(__file__).resolve().parent / "datasets" / "twt_v1"
PROVENANCE_FILE = "PROVENANCE.json"
SECTIONS = ("web", "wiki")
SPLITS = ("train", "dev", "test")
DIMENSIONS = (
    "all",
    "entity_disjoint",
    "relation_disjoint",
    "composition_disjoint",
    "wording_disjoint",
    "sentence_disjoint",
    "seen_composition",
)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


def _normalize_surface(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


def _split_name(index: int) -> str:
    if index % 10 < 8:
        return "train"
    if index % 10 == 8:
        return "dev"
    return "test"


def _parse_features(raw: str) -> Dict[str, str]:
    if raw == "_":
        return {}
    result: Dict[str, str] = {}
    for feature in raw.split("|"):
        if "=" not in feature:
            raise ValueError(f"Biçimsiz CoNLL-U feature: {feature!r}")
        key, value = feature.split("=", 1)
        result[key] = value
    return result


@dataclass(frozen=True)
class TWTToken:
    token_id: int
    form: str
    lemma: str
    upos: str
    xpos: str
    features: Dict[str, str]
    head: int
    relation: str
    misc: Dict[str, str]

    @property
    def entity(self) -> str:
        value = self.lemma if self.lemma != "_" else self.form
        return unicodedata.normalize("NFKC", value).casefold()


@dataclass(frozen=True)
class TWTSentence:
    sentence_id: str
    text: str
    section: str
    section_index: int
    raw_split: str
    tokens: Tuple[TWTToken, ...]


@dataclass(frozen=True)
class ArcCandidate:
    candidate_id: str
    split: str
    section: str
    sentence_id: str
    sentence_text: str
    dependent_id: int
    dependent_entity: str
    dependent_upos: str
    relation: str
    head_id: int
    head_entity: str
    head_upos: str
    expected_valid: bool
    gold_relation: str
    gold_head_id: int
    gold_head_entity: str
    gold_head_upos: str
    dimensions: Tuple[str, ...] = ()

    def hash_record(self) -> List[Any]:
        return [
            self.candidate_id,
            self.split,
            self.section,
            self.sentence_id,
            hashlib.sha256(self.sentence_text.encode("utf-8")).hexdigest(),
            self.dependent_id,
            self.dependent_entity,
            self.dependent_upos,
            self.relation,
            self.head_id,
            self.head_entity,
            self.head_upos,
            self.expected_valid,
            self.gold_relation,
            self.gold_head_id,
            self.gold_head_entity,
            self.gold_head_upos,
            list(self.dimensions),
        ]


@dataclass(frozen=True)
class RealTurkishTaskData:
    """Mimari baseline'ların ortak kullandığı model-visible TWT splitleri."""

    dataset_hash: str
    config_hash: str
    split_hashes: Dict[str, str]
    candidate_hashes: Dict[str, str]
    train: Tuple[ArcCandidate, ...]
    dev: Tuple[ArcCandidate, ...]
    test: Tuple[ArcCandidate, ...]
    all_train_candidate_count: int
    all_dev_candidate_count: int
    heldout_relations: Tuple[str, ...]


@dataclass
class BinaryMetrics:
    total: int
    positive: int
    negative: int
    correct: int
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    uncertain: int
    accuracy: float
    selective_accuracy: float
    precision: float
    recall: float
    f1: float
    far: float
    frr: float
    coverage: float


@dataclass
class RealTurkishReport:
    benchmark_id: str
    schema_version: int
    seed: int
    dataset_hash: str
    config_hash: str
    split_hashes: Dict[str, str]
    candidate_hashes: Dict[str, str]
    provenance: Dict[str, Any]
    task: Dict[str, Any]
    corpus: Dict[str, Any]
    split_contract: Dict[str, Any]
    leakage_audit: Dict[str, Any]
    baseline: Dict[str, Any]
    metrics: BinaryMetrics
    dimensions: Dict[str, BinaryMetrics]
    checks: Dict[str, bool]
    prediction_samples: List[Dict[str, Any]] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        source = self.provenance["upstream"]
        license_info = self.provenance["license"]
        lines = [
            "# HGA Real Turkish Benchmark — TWT v1",
            "",
            f"- Upstream revision: `{source['revision']}`",
            f"- Lisans: `{license_info['spdx_id']}`",
            f"- Dataset SHA-256: `{self.dataset_hash}`",
            f"- Config SHA-256: `{self.config_hash}`",
            f"- Seed: `{self.seed}` (baseline deterministiktir)",
            f"- Etkin cümleler: `{self.corpus['effective_sentences']}`",
            f"- Accuracy / F1: `{self.metrics.accuracy:.4f}` / `{self.metrics.f1:.4f}`",
            f"- FAR / FRR / coverage: `{self.metrics.far:.4f}` / "
            f"`{self.metrics.frr:.4f}` / `{self.metrics.coverage:.4f}`",
            "",
            "| Boyut | N | Accuracy | F1 | FAR | FRR | Coverage |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name in DIMENSIONS:
            metric = self.dimensions.get(name)
            if metric is None:
                continue
            lines.append(
                f"| {name} | {metric.total} | {metric.accuracy:.4f} | "
                f"{metric.f1:.4f} | {metric.far:.4f} | {metric.frr:.4f} | "
                f"{metric.coverage:.4f} |"
            )
        lines.extend(["", "## Bütünlük kapıları", ""])
        lines.extend(f"- {'PASS' if value else 'FAIL'} — `{key}`" for key, value in self.checks.items())
        lines.extend(["", "## Sınırlar", ""])
        lines.extend(f"- {note}" for note in self.limitations)
        return "\n".join(lines) + "\n"


class TurkishWebTreebank:
    """Vendored TWT dosyalarını hash doğrulamasıyla yükleyen sabit dataset."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
        manifest_path = self.data_dir / PROVENANCE_FILE
        self.provenance = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.provenance.get("schema_version") != 1:
            raise ValueError("Desteklenmeyen TWT provenance schema_version")
        self.config = dict(self.provenance["benchmark_config"])
        self._verify_source_files()
        self.sentences = self._load_all()
        self.raw_splits = self._partition(quarantine=False)
        self.splits = self._partition(quarantine=True)
        self._validate_corpus()

    def _verify_source_files(self) -> None:
        for source in self.provenance["upstream"]["source_files"]:
            path = self.data_dir / source["vendored_path"]
            actual = file_sha256(path)
            if actual != source["sha256"]:
                raise ValueError(
                    f"TWT source SHA-256 uyuşmazlığı: {path.name}: "
                    f"beklenen={source['sha256']} gerçek={actual}"
                )
        license_info = self.provenance["license"]
        license_hash = file_sha256(self.data_dir / license_info["vendored_license_path"])
        if license_hash != license_info["license_sha256"]:
            raise ValueError("TWT Apache-2.0 lisans dosyasının SHA-256 değeri bozuk")

    def _load_all(self) -> List[TWTSentence]:
        sentences: List[TWTSentence] = []
        for section in SECTIONS:
            text = (self.data_dir / f"{section}.conllu").read_text(encoding="utf-8")
            blocks = [block for block in text.split("\n\n") if block.strip()]
            for index, block in enumerate(blocks):
                sentences.append(self._parse_sentence(section, index, block))
        return sentences

    @staticmethod
    def _parse_sentence(section: str, index: int, block: str) -> TWTSentence:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3 or not lines[0].startswith("# sent_id = "):
            raise ValueError(f"Biçimsiz TWT sent_id: {section}:{index}")
        if not lines[1].startswith("# text = "):
            raise ValueError(f"Biçimsiz TWT text: {section}:{index}")
        sentence_id = lines[0].split(" = ", 1)[1]
        text = lines[1].split(" = ", 1)[1]
        tokens: List[TWTToken] = []
        for line in lines[2:]:
            if line.startswith("#"):
                continue
            columns = line.split("\t")
            if len(columns) != 10:
                raise ValueError(f"Biçimsiz CoNLL-U token satırı: {sentence_id}")
            try:
                token_id = int(columns[0])
                head = int(columns[6])
            except ValueError as error:
                raise ValueError(f"Basic integer token/head bekleniyordu: {sentence_id}") from error
            tokens.append(TWTToken(
                token_id=token_id,
                form=columns[1],
                lemma=columns[2],
                upos=columns[3],
                xpos=columns[4],
                features=_parse_features(columns[5]),
                head=head,
                relation=columns[7],
                misc=_parse_features(columns[9]),
            ))
        token_ids = [token.token_id for token in tokens]
        if token_ids != list(range(1, len(tokens) + 1)):
            raise ValueError(f"TWT token kimlikleri ardışık değil: {sentence_id}")
        if any(token.head < 0 or token.head > len(tokens) for token in tokens):
            raise ValueError(f"TWT head aralık dışında: {sentence_id}")
        if any(token.head == token.token_id for token in tokens):
            raise ValueError(f"TWT self-head içeriyor: {sentence_id}")
        if sum(token.head == 0 for token in tokens) != 1:
            raise ValueError(f"TWT basic tree tam bir root içermiyor: {sentence_id}")
        return TWTSentence(
            sentence_id=sentence_id,
            text=text,
            section=section,
            section_index=index,
            raw_split=_split_name(index),
            tokens=tuple(tokens),
        )

    def _partition(self, quarantine: bool) -> Dict[str, List[TWTSentence]]:
        quarantined = set(self.config["quarantined_sentence_ids"]) if quarantine else set()
        result: Dict[str, List[TWTSentence]] = {name: [] for name in SPLITS}
        for sentence in self.sentences:
            if sentence.sentence_id not in quarantined:
                result[sentence.raw_split].append(sentence)
        return result

    def _validate_corpus(self) -> None:
        ids = [sentence.sentence_id for sentence in self.sentences]
        if len(ids) != len(set(ids)):
            raise ValueError("TWT sentence_id değerleri benzersiz değil")
        expected = self.provenance["expected_integrity"]
        raw_counts = {name: len(self.raw_splits[name]) for name in SPLITS}
        effective_counts = {name: len(self.splits[name]) for name in SPLITS}
        if raw_counts != expected["raw_split_sentence_counts"]:
            raise ValueError(f"TWT raw split sayıları değişti: {raw_counts}")
        if effective_counts != expected["effective_split_sentence_counts"]:
            raise ValueError(f"TWT etkin split sayıları değişti: {effective_counts}")
        published = self.provenance["dataset"]["published_counts"]
        if len(self.sentences) != published["sentences"]:
            raise ValueError("TWT toplam cümle sayısı published metadata ile uyuşmuyor")
        if sum(len(sentence.tokens) for sentence in self.sentences) != published[
            "inflectional_group_tokens"
        ]:
            raise ValueError("TWT token sayısı published metadata ile uyuşmuyor")

    def source_hashes(self) -> Dict[str, str]:
        return {
            row["section"]: row["sha256"]
            for row in self.provenance["upstream"]["source_files"]
        }

    def config_hash(self) -> str:
        return canonical_hash(self.config)

    def dataset_hash(self) -> str:
        return canonical_hash({
            "benchmark_id": self.provenance["benchmark_id"],
            "upstream_revision": self.provenance["upstream"]["revision"],
            "source_sha256": self.source_hashes(),
            "config_sha256": self.config_hash(),
        })

    def split_hashes(self) -> Dict[str, str]:
        return {
            name: canonical_hash([
                [sentence.section, sentence.section_index, sentence.sentence_id,
                 hashlib.sha256(sentence.text.encode("utf-8")).hexdigest()]
                for sentence in self.splits[name]
            ])
            for name in SPLITS
        }

    def sentence_by_id(self) -> Dict[str, TWTSentence]:
        return {sentence.sentence_id: sentence for sentence in self.sentences}


class SelectiveArcSchemaVerifier:
    """Açık, deterministik ve neural olmayan benchmark sağlık baseline'ı."""

    def __init__(self, positive_threshold: float, negative_threshold: float, alpha: float):
        if not 0.5 <= positive_threshold <= 1.0:
            raise ValueError("positive_threshold [0.5, 1] içinde olmalı")
        if not 0.0 <= negative_threshold <= 0.5:
            raise ValueError("negative_threshold [0, 0.5] içinde olmalı")
        self.positive_threshold = float(positive_threshold)
        self.negative_threshold = float(negative_threshold)
        self.alpha = float(alpha)
        self.counts: DefaultDict[Tuple[str, ...], List[int]] = defaultdict(lambda: [0, 0])
        self.relations: set[str] = set()

    @staticmethod
    def _signature(candidate: ArcCandidate) -> Tuple[str, ...]:
        if candidate.head_id == 0:
            direction = "ROOT"
            distance = "ROOT"
        else:
            direction = "LEFT" if candidate.head_id < candidate.dependent_id else "RIGHT"
            delta = abs(candidate.head_id - candidate.dependent_id)
            distance = "NEAR" if delta <= 2 else "FAR"
        return (
            candidate.dependent_upos,
            candidate.relation,
            candidate.head_upos,
            direction,
            distance,
        )

    def fit(self, candidates: Iterable[ArcCandidate]) -> None:
        for candidate in candidates:
            label = 1 if candidate.expected_valid else 0
            self.counts[self._signature(candidate)][label] += 1
            self.relations.add(candidate.relation)

    def predict(self, candidate: ArcCandidate) -> Optional[bool]:
        if candidate.relation not in self.relations:
            return None
        counts = self.counts.get(self._signature(candidate))
        if counts is None:
            return None
        negative, positive = counts
        probability = (positive + self.alpha) / (negative + positive + 2 * self.alpha)
        if probability >= self.positive_threshold:
            return True
        if probability <= self.negative_threshold:
            return False
        return None

    @property
    def learned_schema_count(self) -> int:
        return len(self.counts)


def _candidate_hash(candidates: Sequence[ArcCandidate]) -> str:
    digest = hashlib.sha256()
    for candidate in candidates:
        payload = json.dumps(
            candidate.hash_record(), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        digest.update(payload)
        digest.update(b"\n")
    return digest.hexdigest()


def _head_fields(token_by_id: Mapping[int, TWTToken], head_id: int) -> Tuple[str, str]:
    if head_id == 0:
        return "ROOT", "ROOT"
    token = token_by_id[head_id]
    return token.entity, token.upos


def _negative_variant(sentence: TWTSentence, token: TWTToken) -> Tuple[int, str]:
    alternatives = [
        head for head in range(0, len(sentence.tokens) + 1)
        if head not in (token.token_id, token.head)
    ]
    if alternatives:
        key = f"{sentence.sentence_id}:{token.token_id}:negative-head".encode("utf-8")
        index = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(alternatives)
        return alternatives[index], token.relation
    # Tek-token cümlede farklı head yoktur; tek gold relation sözleşmesini boz.
    return token.head, "p" if token.relation != "p" else "root"


def _build_candidates(sentences: Sequence[TWTSentence], split: str) -> List[ArcCandidate]:
    result: List[ArcCandidate] = []
    for sentence in sentences:
        token_by_id = {token.token_id: token for token in sentence.tokens}
        for token in sentence.tokens:
            gold_head_entity, gold_head_upos = _head_fields(token_by_id, token.head)
            positive = ArcCandidate(
                candidate_id=f"{sentence.sentence_id}:{token.token_id}:P",
                split=split,
                section=sentence.section,
                sentence_id=sentence.sentence_id,
                sentence_text=sentence.text,
                dependent_id=token.token_id,
                dependent_entity=token.entity,
                dependent_upos=token.upos,
                relation=token.relation,
                head_id=token.head,
                head_entity=gold_head_entity,
                head_upos=gold_head_upos,
                expected_valid=True,
                gold_relation=token.relation,
                gold_head_id=token.head,
                gold_head_entity=gold_head_entity,
                gold_head_upos=gold_head_upos,
            )
            negative_head, negative_relation = _negative_variant(sentence, token)
            negative_head_entity, negative_head_upos = _head_fields(token_by_id, negative_head)
            negative = ArcCandidate(
                candidate_id=f"{sentence.sentence_id}:{token.token_id}:N",
                split=split,
                section=sentence.section,
                sentence_id=sentence.sentence_id,
                sentence_text=sentence.text,
                dependent_id=token.token_id,
                dependent_entity=token.entity,
                dependent_upos=token.upos,
                relation=negative_relation,
                head_id=negative_head,
                head_entity=negative_head_entity,
                head_upos=negative_head_upos,
                expected_valid=False,
                gold_relation=token.relation,
                gold_head_id=token.head,
                gold_head_entity=gold_head_entity,
                gold_head_upos=gold_head_upos,
            )
            result.extend((positive, negative))
    return result


def _gold_composition(candidate: ArcCandidate) -> Tuple[str, str, str]:
    return (
        candidate.dependent_entity,
        candidate.gold_relation,
        candidate.gold_head_entity,
    )


def _assign_dimensions(
    candidates: Sequence[ArcCandidate],
    train_entities: set[str],
    train_relations: set[str],
    train_compositions: set[Tuple[str, str, str]],
    train_surfaces: set[str],
) -> List[ArcCandidate]:
    rows: List[ArcCandidate] = []
    for candidate in candidates:
        dependent_seen = candidate.dependent_entity in train_entities
        head_seen = (
            candidate.gold_head_entity == "ROOT"
            or candidate.gold_head_entity in train_entities
        )
        entities_seen = dependent_seen and head_seen
        relation_seen = candidate.gold_relation in train_relations
        composition_seen = _gold_composition(candidate) in train_compositions
        wording_unseen = _normalize_surface(candidate.sentence_text) not in train_surfaces
        dimensions = ["all", "sentence_disjoint"]
        if not entities_seen and relation_seen:
            dimensions.append("entity_disjoint")
        if entities_seen and not relation_seen:
            dimensions.append("relation_disjoint")
        if entities_seen and relation_seen and not composition_seen:
            dimensions.append("composition_disjoint")
        if entities_seen and relation_seen and composition_seen:
            dimensions.append("seen_composition")
            if wording_unseen:
                dimensions.append("wording_disjoint")
        rows.append(ArcCandidate(**{**asdict(candidate), "dimensions": tuple(dimensions)}))
    return rows


def _metrics(
    candidates: Sequence[ArcCandidate], predictions: Sequence[Optional[bool]]
) -> BinaryMetrics:
    positive = sum(candidate.expected_valid for candidate in candidates)
    negative = len(candidates) - positive
    tp = sum(candidate.expected_valid and prediction is True
             for candidate, prediction in zip(candidates, predictions))
    tn = sum(not candidate.expected_valid and prediction is False
             for candidate, prediction in zip(candidates, predictions))
    fp = sum(not candidate.expected_valid and prediction is True
             for candidate, prediction in zip(candidates, predictions))
    fn = sum(candidate.expected_valid and prediction is False
             for candidate, prediction in zip(candidates, predictions))
    uncertain = sum(prediction is None for prediction in predictions)
    answered = len(candidates) - uncertain
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, positive)
    return BinaryMetrics(
        total=len(candidates),
        positive=positive,
        negative=negative,
        correct=tp + tn,
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        uncertain=uncertain,
        accuracy=_ratio(tp + tn, len(candidates)),
        selective_accuracy=_ratio(tp + tn, answered),
        precision=precision,
        recall=recall,
        f1=_ratio(2 * tp, 2 * tp + fp + (positive - tp)),
        far=_ratio(fp, negative),
        frr=_ratio(fn, positive),
        coverage=_ratio(answered, len(candidates)),
    )


def evaluate_arc_predictions(
    candidates: Sequence[ArcCandidate],
    predictions: Sequence[Optional[bool]],
) -> Dict[str, BinaryMetrics]:
    """Ortak overall/disjoint metrik sözleşmesini herhangi bir predictor'a uygula."""
    if len(candidates) != len(predictions):
        raise ValueError("Candidate ve prediction sayıları eşit olmalı")
    result = {"all": _metrics(candidates, predictions)}
    for dimension in DIMENSIONS:
        if dimension == "all":
            continue
        selected = [
            (candidate, prediction)
            for candidate, prediction in zip(candidates, predictions)
            if dimension in candidate.dimensions
        ]
        if selected:
            result[dimension] = _metrics(
                [row[0] for row in selected], [row[1] for row in selected]
            )
    return result


def _lemma_set(sentence: TWTSentence) -> set[str]:
    return {token.entity for token in sentence.tokens if token.upos != "PUNCT"}


def _near_duplicate_pairs(
    reference: Sequence[TWTSentence],
    targets: Sequence[TWTSentence],
    threshold: float,
) -> List[Dict[str, Any]]:
    reference_sets = [_lemma_set(sentence) for sentence in reference]
    inverted: DefaultDict[str, List[int]] = defaultdict(list)
    for index, lemmas in enumerate(reference_sets):
        for lemma in lemmas:
            inverted[lemma].append(index)
    findings: List[Dict[str, Any]] = []
    for target in targets:
        target_set = _lemma_set(target)
        possible: set[int] = set()
        for lemma in target_set:
            possible.update(inverted.get(lemma, ()))
        for index in possible:
            reference_set = reference_sets[index]
            union = target_set | reference_set
            similarity = len(target_set & reference_set) / len(union) if union else 1.0
            if similarity >= threshold:
                findings.append({
                    "reference_id": reference[index].sentence_id,
                    "target_id": target.sentence_id,
                    "lemma_set_jaccard": round(similarity, 8),
                })
    return sorted(findings, key=lambda row: (row["target_id"], row["reference_id"]))


def _leakage_audit(dataset: TurkishWebTreebank) -> Dict[str, Any]:
    surfaces: Dict[str, Dict[str, List[str]]] = {}
    for split in SPLITS:
        indexed: Dict[str, List[str]] = defaultdict(list)
        for sentence in dataset.splits[split]:
            indexed[_normalize_surface(sentence.text)].append(sentence.sentence_id)
        surfaces[split] = dict(indexed)
    cross_surface: List[Dict[str, Any]] = []
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            for surface in set(surfaces[left]) & set(surfaces[right]):
                cross_surface.append({
                    "left_split": left,
                    "right_split": right,
                    "left_ids": surfaces[left][surface],
                    "right_ids": surfaces[right][surface],
                    "surface_sha256": hashlib.sha256(surface.encode("utf-8")).hexdigest(),
                })
    threshold = float(dataset.config["near_duplicate_threshold"])
    train = dataset.splits["train"]
    dev = dataset.splits["dev"]
    test = dataset.splits["test"]
    near_pairs = _near_duplicate_pairs(train, dev, threshold)
    near_pairs.extend(_near_duplicate_pairs([*train, *dev], test, threshold))

    sentence_map = dataset.sentence_by_id()
    quarantine_evidence: List[Dict[str, Any]] = []
    evidence_pairs = (
        ("tr-forum:00001297:S012", "tr-forum:00001297:S029"),
        ("tr-forum:00002420:S010", "tr-forum:00000353:S041"),
    )
    for reference_id, quarantined_id in evidence_pairs:
        reference_set = _lemma_set(sentence_map[reference_id])
        quarantined_set = _lemma_set(sentence_map[quarantined_id])
        union = reference_set | quarantined_set
        similarity = len(reference_set & quarantined_set) / len(union) if union else 1.0
        quarantine_evidence.append({
            "reference_id": reference_id,
            "quarantined_id": quarantined_id,
            "exact_surface": (
                _normalize_surface(sentence_map[reference_id].text)
                == _normalize_surface(sentence_map[quarantined_id].text)
            ),
            "lemma_set_jaccard": round(similarity, 8),
        })
    ids_by_split = {
        split: {sentence.sentence_id for sentence in dataset.splits[split]}
        for split in SPLITS
    }
    id_overlap = any(
        ids_by_split[left] & ids_by_split[right]
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1:]
    )
    internal_surface_duplicates = sum(
        len(ids) - 1
        for split in SPLITS
        for ids in surfaces[split].values()
        if len(ids) > 1
    )
    clean = not id_overlap and not cross_surface and not near_pairs and not internal_surface_duplicates
    return {
        "clean": clean,
        "sentence_id_disjoint": not id_overlap,
        "internal_normalized_surface_duplicate_count": internal_surface_duplicates,
        "cross_split_normalized_surface_duplicate_count": len(cross_surface),
        "cross_split_normalized_surface_duplicates": cross_surface,
        "lexical_semantic_proxy": {
            "metric": dataset.config["near_duplicate_metric"],
            "threshold": threshold,
            "clean": not near_pairs,
            "effective_cross_split_finding_count": len(near_pairs),
            "effective_cross_split_findings": near_pairs,
            "quarantine_evidence": quarantine_evidence,
            "is_semantic_equivalence_proof": False,
        },
        "quarantined_sentence_ids": dict(dataset.config["quarantined_sentence_ids"]),
    }


def _validate_expected_hashes(
    expected: Mapping[str, Any],
    split_hashes: Mapping[str, str],
    candidate_hashes: Mapping[str, str],
) -> None:
    expected_splits = expected.get("split_hashes", {})
    expected_candidates = expected.get("candidate_hashes", {})
    if expected_splits and dict(split_hashes) != dict(expected_splits):
        raise ValueError("TWT split SHA-256 değerleri sabit provenance manifestiyle uyuşmuyor")
    if expected_candidates and dict(candidate_hashes) != dict(expected_candidates):
        raise ValueError("TWT candidate SHA-256 değerleri sabit provenance manifestiyle uyuşmuyor")


def _prepare_real_turkish_task(dataset: TurkishWebTreebank) -> RealTurkishTaskData:
    split_hashes = dataset.split_hashes()
    candidates = {
        split: _build_candidates(dataset.splits[split], split)
        for split in SPLITS
    }
    heldout_relations = set(dataset.config["relation_disjoint_labels"])
    model_train = [
        candidate for candidate in candidates["train"]
        if candidate.gold_relation not in heldout_relations
    ]
    model_dev = [
        candidate for candidate in candidates["dev"]
        if candidate.gold_relation not in heldout_relations
    ]
    train_positives = [candidate for candidate in model_train if candidate.expected_valid]
    train_entities = {
        entity
        for candidate in train_positives
        for entity in (candidate.dependent_entity, candidate.gold_head_entity)
        if entity != "ROOT"
    }
    train_relations = {candidate.gold_relation for candidate in train_positives}
    train_compositions = {_gold_composition(candidate) for candidate in train_positives}
    train_surfaces = {
        _normalize_surface(sentence.text) for sentence in dataset.splits["train"]
    }
    model_dev = _assign_dimensions(
        model_dev,
        train_entities=train_entities,
        train_relations=train_relations,
        train_compositions=train_compositions,
        train_surfaces=train_surfaces,
    )
    test_candidates = _assign_dimensions(
        candidates["test"],
        train_entities=train_entities,
        train_relations=train_relations,
        train_compositions=train_compositions,
        train_surfaces=train_surfaces,
    )
    candidate_hashes = {
        split: _candidate_hash(rows) for split, rows in candidates.items()
    }
    candidate_hashes["test_challenge"] = _candidate_hash(test_candidates)
    _validate_expected_hashes(
        dataset.provenance["expected_integrity"], split_hashes, candidate_hashes
    )
    return RealTurkishTaskData(
        dataset_hash=dataset.dataset_hash(),
        config_hash=dataset.config_hash(),
        split_hashes=split_hashes,
        candidate_hashes=candidate_hashes,
        train=tuple(model_train),
        dev=tuple(model_dev),
        test=tuple(test_candidates),
        all_train_candidate_count=len(candidates["train"]),
        all_dev_candidate_count=len(candidates["dev"]),
        heldout_relations=tuple(sorted(heldout_relations)),
    )


@lru_cache(maxsize=1)
def _cached_default_task() -> RealTurkishTaskData:
    return _prepare_real_turkish_task(TurkishWebTreebank())


def prepare_real_turkish_task(
    dataset: Optional[TurkishWebTreebank] = None,
) -> RealTurkishTaskData:
    """Aynı train/dev/test adaylarını tüm model aileleri için tek kez hazırla."""
    if dataset is None:
        return _cached_default_task()
    return _prepare_real_turkish_task(dataset)


def _run_real_turkish_benchmark(
    dataset: TurkishWebTreebank,
    seed: int,
) -> RealTurkishReport:
    """Önceden doğrulanmış dataset üzerinde benchmark çekirdeğini çalıştır."""
    task_data = _prepare_real_turkish_task(dataset)
    split_hashes = task_data.split_hashes
    candidate_hashes = task_data.candidate_hashes
    heldout_relations = set(task_data.heldout_relations)
    model_train = list(task_data.train)
    test_candidates = list(task_data.test)
    train_positives = [candidate for candidate in model_train if candidate.expected_valid]
    train_entities = {
        entity
        for candidate in train_positives
        for entity in (candidate.dependent_entity, candidate.gold_head_entity)
        if entity != "ROOT"
    }
    train_relations = {candidate.gold_relation for candidate in train_positives}
    train_compositions = {_gold_composition(candidate) for candidate in train_positives}
    train_surfaces = {
        _normalize_surface(sentence.text) for sentence in dataset.splits["train"]
    }

    verifier = SelectiveArcSchemaVerifier(
        positive_threshold=float(dataset.config["baseline_positive_threshold"]),
        negative_threshold=float(dataset.config["baseline_negative_threshold"]),
        alpha=float(dataset.config["laplace_alpha"]),
    )
    verifier.fit(model_train)
    predictions = [verifier.predict(candidate) for candidate in test_candidates]
    dimensions = evaluate_arc_predictions(test_candidates, predictions)
    metrics = dimensions["all"]

    leakage = _leakage_audit(dataset)
    raw_token_count = sum(len(sentence.tokens) for sentence in dataset.sentences)
    effective_token_counts = {
        split: sum(len(sentence.tokens) for sentence in dataset.splits[split])
        for split in SPLITS
    }
    positive_dimension_rows = {
        name: [
            candidate for candidate in test_candidates
            if candidate.expected_valid and name in candidate.dimensions
        ]
        for name in DIMENSIONS
    }
    train_dev_sentence_ids = {
        sentence.sentence_id
        for split in ("train", "dev")
        for sentence in dataset.splits[split]
    }
    dimension_audit = {
        "entity_disjoint_clean": all(
            (
                candidate.dependent_entity not in train_entities
                or (
                    candidate.gold_head_entity != "ROOT"
                    and candidate.gold_head_entity not in train_entities
                )
            )
            and candidate.gold_relation in train_relations
            for candidate in positive_dimension_rows["entity_disjoint"]
        ),
        "relation_disjoint_clean": all(
            candidate.dependent_entity in train_entities
            and (
                candidate.gold_head_entity == "ROOT"
                or candidate.gold_head_entity in train_entities
            )
            and candidate.gold_relation not in train_relations
            for candidate in positive_dimension_rows["relation_disjoint"]
        ),
        "composition_disjoint_clean": all(
            candidate.dependent_entity in train_entities
            and (
                candidate.gold_head_entity == "ROOT"
                or candidate.gold_head_entity in train_entities
            )
            and candidate.gold_relation in train_relations
            and _gold_composition(candidate) not in train_compositions
            for candidate in positive_dimension_rows["composition_disjoint"]
        ),
        "wording_disjoint_clean": all(
            _gold_composition(candidate) in train_compositions
            and _normalize_surface(candidate.sentence_text) not in train_surfaces
            for candidate in positive_dimension_rows["wording_disjoint"]
        ),
        "sentence_disjoint_clean": all(
            candidate.sentence_id not in train_dev_sentence_ids
            for candidate in positive_dimension_rows["sentence_disjoint"]
        ),
        "all_dimension_pairs_balanced": all(
            metric.positive == metric.negative for metric in dimensions.values()
        ),
    }
    checks = {
        "open_redistributable_license": (
            dataset.provenance["license"]["spdx_id"] == "Apache-2.0"
            and bool(dataset.provenance["license"]["redistributable"])
        ),
        "real_turkish_sources_documented": set(
            dataset.provenance["dataset"]["source_domains"]
        ) == set(SECTIONS),
        "human_annotations_documented": bool(dataset.provenance["dataset"]["human_annotated"]),
        "upstream_revision_pinned": len(dataset.provenance["upstream"]["revision"]) == 40,
        "raw_source_hashes_verified": True,
        "deterministic_nonempty_splits": all(dataset.splits[name] for name in SPLITS),
        "split_hashes_present": all(len(value) == 64 for value in split_hashes.values()),
        "candidate_hashes_present": all(len(value) == 64 for value in candidate_hashes.values()),
        "duplicate_and_leakage_audit_clean": bool(leakage["clean"]),
        "entity_disjoint_slice_present": (
            "entity_disjoint" in dimensions and dimensions["entity_disjoint"].total > 0
        ),
        "relation_disjoint_slice_present": (
            "relation_disjoint" in dimensions and dimensions["relation_disjoint"].total > 0
        ),
        "composition_disjoint_slice_present": (
            "composition_disjoint" in dimensions
            and dimensions["composition_disjoint"].total > 0
        ),
        "wording_disjoint_slice_present": (
            "wording_disjoint" in dimensions and dimensions["wording_disjoint"].total > 0
        ),
        "balanced_binary_test": metrics.positive == metrics.negative,
        "disjoint_dimension_contracts_clean": all(dimension_audit.values()),
        "required_metrics_reported": all(
            0.0 <= value <= 1.0
            for value in (
                metrics.accuracy, metrics.precision, metrics.recall, metrics.f1,
                metrics.far, metrics.frr, metrics.coverage,
            )
        ),
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise ValueError(f"Gerçek Türkçe benchmark kabul kapısı başarısız: {failed}")

    samples: List[Dict[str, Any]] = []
    for candidate, prediction in zip(test_candidates, predictions):
        if prediction == candidate.expected_valid:
            continue
        samples.append({
            "candidate_id": candidate.candidate_id,
            "sentence_id": candidate.sentence_id,
            "sentence": candidate.sentence_text,
            "dependent_id": candidate.dependent_id,
            "relation": candidate.relation,
            "head_id": candidate.head_id,
            "expected": "VALID" if candidate.expected_valid else "INVALID",
            "predicted": (
                "UNCERTAIN" if prediction is None
                else ("VALID" if prediction else "INVALID")
            ),
            "dimensions": list(candidate.dimensions),
        })
        if len(samples) >= 25:
            break

    relation_test_positive_counts = Counter(
        candidate.gold_relation
        for candidate in test_candidates
        if candidate.expected_valid and candidate.gold_relation in heldout_relations
    )
    return RealTurkishReport(
        benchmark_id=dataset.provenance["benchmark_id"],
        schema_version=1,
        seed=int(seed),
        dataset_hash=dataset.dataset_hash(),
        config_hash=dataset.config_hash(),
        split_hashes=split_hashes,
        candidate_hashes=candidate_hashes,
        provenance={
            "dataset": dict(dataset.provenance["dataset"]),
            "upstream": dict(dataset.provenance["upstream"]),
            "license": dict(dataset.provenance["license"]),
            "citation": dict(dataset.provenance["citation"]),
            "modification_notice": dataset.provenance["modification_notice"],
        },
        task=dict(dataset.provenance["task_scope"]),
        corpus={
            "raw_sentences": len(dataset.sentences),
            "effective_sentences": sum(len(rows) for rows in dataset.splits.values()),
            "raw_inflectional_group_tokens": raw_token_count,
            "effective_token_counts": effective_token_counts,
            "source_sections": {
                section: sum(sentence.section == section for sentence in dataset.sentences)
                for section in SECTIONS
            },
        },
        split_contract={
            "source_rule": dict(dataset.config["source_split"]),
            "raw_sentence_counts": {
                split: len(dataset.raw_splits[split]) for split in SPLITS
            },
            "effective_sentence_counts": {
                split: len(dataset.splits[split]) for split in SPLITS
            },
            "quarantined_sentence_ids": dict(dataset.config["quarantined_sentence_ids"]),
            "training_candidate_count_before_relation_holdout": (
                task_data.all_train_candidate_count
            ),
            "training_candidate_count_after_relation_holdout": len(model_train),
            "dev_candidate_count_before_relation_holdout": task_data.all_dev_candidate_count,
            "dev_candidate_count_after_relation_holdout": len(task_data.dev),
            "test_candidate_count": len(test_candidates),
            "relation_disjoint_labels": sorted(heldout_relations),
            "relation_disjoint_positive_test_counts": dict(relation_test_positive_counts),
            "dimension_audit": dimension_audit,
            "dimension_definitions": {
                "entity_disjoint": "At least one gold endpoint entity absent from model train; relation seen.",
                "relation_disjoint": "Both gold endpoint entities seen; gold relation absent from model train.",
                "composition_disjoint": "Endpoints and relation seen separately; exact gold triple absent.",
                "wording_disjoint": "Gold composition seen; normalized test sentence absent from train.",
                "sentence_disjoint": "Sentence ID belongs only to the fixed test split.",
            },
        },
        leakage_audit=leakage,
        baseline={
            "name": "selective-pos-relation-direction-schema-frequency",
            "neural": False,
            "seed_dependent": False,
            "learned_schema_count": verifier.learned_schema_count,
            "learned_relation_count": len(verifier.relations),
            "positive_threshold": verifier.positive_threshold,
            "negative_threshold": verifier.negative_threshold,
            "laplace_alpha": verifier.alpha,
            "purpose": "Benchmark plumbing/health baseline; not a competitive language model.",
        },
        metrics=metrics,
        dimensions=dimensions,
        checks=checks,
        prediction_samples=samples,
        limitations=[
            "TWT dependency etiketleri morphosyntactic'tir; semantik knowledge triple değildir.",
            "Entity-disjoint anahtarı lemma/form token kimliğidir; named entity annotation değildir.",
            "Negatifler insan tarafından tek tek yazılmamıştır; insan anotasyonlu single-head basic tree sözleşmesinden deterministik türetilir.",
            "Lemma-set Jaccard denetimi muhafazakâr bir lexical-semantic yakınlık proxy'sidir; semantik eşdeğerlik kanıtı değildir.",
            "Schema-frequency baseline benchmark hattını doğrular; HGA/Transformer üstünlüğü veya genel Türkçe yeterlilik iddiası taşımaz.",
        ],
    )


@lru_cache(maxsize=1)
def _cached_default_report() -> RealTurkishReport:
    """Beş-seed suite'te seed-bağımsız corpus hazırlığını yalnız bir kez yap."""
    return _run_real_turkish_benchmark(TurkishWebTreebank(), seed=0)


def run_real_turkish_benchmark(
    dataset: Optional[TurkishWebTreebank] = None,
    seed: int = 1,
) -> RealTurkishReport:
    """Sabit TWT splitinde deterministic selective schema baseline'ını ölç.

    Varsayılan vendored dataset seed'den bağımsızdır ve süreç içinde cache'lenir;
    her çağrı bağımsız bir rapor kopyası alır. Özel dataset nesneleri (örneğin
    bütünlük testleri) hiçbir zaman cache'e girmez.
    """
    if dataset is not None:
        return _run_real_turkish_benchmark(dataset, int(seed))
    report = copy.deepcopy(_cached_default_report())
    report.seed = int(seed)
    return report


__all__ = [
    "DATA_DIR",
    "DIMENSIONS",
    "ArcCandidate",
    "BinaryMetrics",
    "RealTurkishReport",
    "RealTurkishTaskData",
    "SelectiveArcSchemaVerifier",
    "TWTSentence",
    "TWTToken",
    "TurkishWebTreebank",
    "evaluate_arc_predictions",
    "prepare_real_turkish_task",
    "run_real_turkish_benchmark",
]
