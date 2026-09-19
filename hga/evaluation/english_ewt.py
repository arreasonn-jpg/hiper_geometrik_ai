"""Pinned UD English EWT benchmark and small architecture baselines.

This module closes the *English data absent* gap without relabelling a synthetic
fixture as a real corpus.  The vendored files are byte-identical to a pinned
UD English EWT revision and checked before parsing.  The task intentionally
matches HGA's existing TWT benchmark: binary verification of positive basic
UD arcs versus deterministic within-sentence head corruptions.

The baseline families are deliberately **small models trained from scratch**:
``dense``, a bidirectional Transformer encoder, BERT-style encoder and GPT-style
causal decoder.  They are architectural controls, not the released/pretrained
BERT or GPT checkpoints; no pretrained-model claim is made.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .experiment import canonical_hash, file_sha256, seed_everything


DATA_DIR = Path(__file__).resolve().parent / "datasets" / "ewt_v1"
SPLITS = ("train", "dev", "test")
MODEL_ORDER = ("dense", "transformer", "bert_style", "gpt_style")
FEATURE_FIELDS = ("dependent", "dependent_upos", "relation", "head", "head_upos", "geometry")

# The cap makes five-seed CPU experiments practical while all original UD files
# remain vendored and hash-checked.  Selection happens *within* official splits.
SENTENCE_CAP = {"train": 512, "dev": 128, "test": 128}
PROFILE = {
    "smoke": {"steps": 24, "batch_size": 128, "learning_rate": 0.003},
    "full": {"steps": 160, "batch_size": 128, "learning_rate": 0.003},
}
ARCHITECTURE = {
    "embedding_dim": 24,
    "heads": 4,
    "feedforward_dim": 72,
    "layers": 1,
    "dense_hidden_dim": 78,
    "parameter_ratio_max": 1.12,
}


@dataclass(frozen=True)
class UDToken:
    token_id: int
    form: str
    lemma: str
    upos: str
    head: int
    relation: str

    @property
    def entity(self) -> str:
        value = self.lemma if self.lemma != "_" else self.form
        return unicodedata.normalize("NFKC", value).casefold()


@dataclass(frozen=True)
class UDSentence:
    sentence_id: str
    text: str
    tokens: Tuple[UDToken, ...]


@dataclass(frozen=True)
class EWTCandidate:
    candidate_id: str
    split: str
    sentence_id: str
    dependent_id: int
    dependent: str
    dependent_upos: str
    relation: str
    head_id: int
    head: str
    head_upos: str
    expected_valid: bool

    def hash_record(self) -> Tuple[Any, ...]:
        return (
            self.candidate_id, self.split, self.sentence_id, self.dependent_id,
            self.dependent, self.dependent_upos, self.relation, self.head_id,
            self.head, self.head_upos, self.expected_valid,
        )


@dataclass(frozen=True)
class EWTTaskData:
    dataset_hash: str
    config_hash: str
    source_hashes: Dict[str, str]
    candidate_hashes: Dict[str, str]
    sentence_counts: Dict[str, int]
    train: Tuple[EWTCandidate, ...]
    dev: Tuple[EWTCandidate, ...]
    test: Tuple[EWTCandidate, ...]


@dataclass(frozen=True)
class BinarySummary:
    total: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_acceptance_rate: float
    false_rejection_rate: float


@dataclass
class EnglishEWTReport:
    protocol: str
    profile: str
    seeds: List[int]
    dataset_hash: str
    config_hash: str
    source_hashes: Dict[str, str]
    candidate_hashes: Dict[str, str]
    sentence_counts: Dict[str, int]
    parameter_counts: Dict[str, int]
    per_seed: List[Dict[str, Any]]
    aggregate: Dict[str, Dict[str, Dict[str, float]]]
    checks: Dict[str, bool]
    limitations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("English EWT neural baselines require PyTorch.") from error
    return torch, nn


def _hash_records(records: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _parse_conllu(path: Path) -> List[UDSentence]:
    sentences: List[UDSentence] = []
    for block in path.read_text(encoding="utf-8").split("\n\n"):
        if not block.strip():
            continue
        sent_id = None
        text = ""
        tokens: List[UDToken] = []
        for raw_line in block.splitlines():
            line = raw_line.rstrip()
            if line.startswith("# sent_id = "):
                sent_id = line.split(" = ", 1)[1]
                continue
            if line.startswith("# text = "):
                text = line.split(" = ", 1)[1]
                continue
            if not line or line.startswith("#"):
                continue
            columns = line.split("\t")
            if len(columns) != 10:
                raise ValueError(f"Malformed CoNLL-U row in {path.name}: {line!r}")
            # Multiword ranges and empty nodes do not have a basic integer arc.
            if "-" in columns[0] or "." in columns[0]:
                continue
            try:
                token_id, head = int(columns[0]), int(columns[6])
            except ValueError as error:
                raise ValueError(f"Non-integer basic arc in {path.name}: {line!r}") from error
            tokens.append(UDToken(token_id, columns[1], columns[2], columns[3], head, columns[7]))
        if sent_id is None or not tokens:
            raise ValueError(f"Sentence without sent_id/tokens in {path.name}")
        sentences.append(UDSentence(sent_id, text, tuple(tokens)))
    return sentences


class EnglishEWT:
    """Hash-verifying reader for the pinned UD English EWT source files."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.provenance = json.loads((self.data_dir / "PROVENANCE.json").read_text(encoding="utf-8"))
        if self.provenance.get("schema_version") != 1:
            raise ValueError("Unsupported EWT provenance schema")
        self._verify()

    def _verify(self) -> None:
        for source in self.provenance["upstream"]["source_files"]:
            actual = file_sha256(self.data_dir / source["vendored_path"])
            if actual != source["sha256"]:
                raise ValueError(
                    f"EWT source SHA-256 mismatch: {source['vendored_path']} "
                    f"expected={source['sha256']} actual={actual}"
                )
        license_info = self.provenance["license"]
        license_actual = file_sha256(self.data_dir / license_info["vendored_license_path"])
        if license_actual != license_info["license_sha256"]:
            raise ValueError("EWT CC-BY-SA license SHA-256 mismatch")

    def source_hashes(self) -> Dict[str, str]:
        return {row["vendored_path"]: row["sha256"] for row in self.provenance["upstream"]["source_files"]}

    def split_sentences(self, split: str) -> Tuple[UDSentence, ...]:
        if split not in SPLITS:
            raise ValueError(f"Unknown EWT split: {split}")
        all_sentences = _parse_conllu(self.data_dir / f"{split}.conllu")
        cap = int(self.provenance["benchmark_config"]["sentence_cap"][split])
        selected = sorted(
            all_sentences,
            key=lambda sentence: hashlib.sha256(sentence.sentence_id.encode("utf-8")).hexdigest(),
        )[:cap]
        if len(selected) != cap:
            raise ValueError(f"EWT {split} has fewer than required {cap} sentences")
        return tuple(selected)


def _head_choice(sentence: UDSentence, token: UDToken) -> int | None:
    """Dengeli negatif için farklı bir legal head döndür.

    Tek tokenlı bir cümlede root dışındaki legal alternatif yoktur. Bu nadir
    pozitif tek başına benchmark'a alınmaz; aksi halde sınıf dengesi ve pair
    schedule sözleşmesi bozulur.
    """
    legal = [0, *(entry.token_id for entry in sentence.tokens)]
    for candidate in legal:
        if candidate not in (token.head, token.token_id):
            return candidate
    return None


def _head_info(sentence: UDSentence, head_id: int) -> Tuple[str, str]:
    if head_id == 0:
        return "ROOT", "ROOT"
    by_id = {token.token_id: token for token in sentence.tokens}
    token = by_id.get(head_id)
    if token is None:
        raise ValueError(f"Missing head {head_id} in {sentence.sentence_id}")
    return token.entity, token.upos


def _candidates(split: str, sentences: Sequence[UDSentence]) -> Tuple[EWTCandidate, ...]:
    candidates: List[EWTCandidate] = []
    for sentence in sentences:
        for token in sentence.tokens:
            negative_head = _head_choice(sentence, token)
            if negative_head is None:
                continue
            gold_head, gold_upos = _head_info(sentence, token.head)
            base = (split, sentence.sentence_id, token.token_id)
            candidates.append(EWTCandidate(
                candidate_id="|".join(map(str, (*base, "positive"))), split=split,
                sentence_id=sentence.sentence_id, dependent_id=token.token_id,
                dependent=token.entity, dependent_upos=token.upos, relation=token.relation,
                head_id=token.head, head=gold_head, head_upos=gold_upos, expected_valid=True,
            ))
            head, upos = _head_info(sentence, negative_head)
            candidates.append(EWTCandidate(
                candidate_id="|".join(map(str, (*base, "negative"))), split=split,
                sentence_id=sentence.sentence_id, dependent_id=token.token_id,
                dependent=token.entity, dependent_upos=token.upos, relation=token.relation,
                head_id=negative_head, head=head, head_upos=upos, expected_valid=False,
            ))
    return tuple(candidates)


def prepare_english_ewt_task(data_dir: Path | None = None) -> EWTTaskData:
    dataset = EnglishEWT(data_dir)
    sentences = {split: dataset.split_sentences(split) for split in SPLITS}
    candidates = {split: _candidates(split, sentences[split]) for split in SPLITS}
    source_hashes = dataset.source_hashes()
    config = dataset.provenance["benchmark_config"]
    candidate_hashes = {split: _hash_records(item.hash_record() for item in candidates[split]) for split in SPLITS}
    dataset_hash = canonical_hash({"source_hashes": source_hashes, "config": config})
    return EWTTaskData(
        dataset_hash=dataset_hash,
        config_hash=canonical_hash(config),
        source_hashes=source_hashes,
        candidate_hashes=candidate_hashes,
        sentence_counts={split: len(sentences[split]) for split in SPLITS},
        train=candidates["train"], dev=candidates["dev"], test=candidates["test"],
    )


def _feature(candidate: EWTCandidate) -> Tuple[str, ...]:
    if candidate.head_id == 0:
        geometry = "ROOT"
    else:
        direction = "LEFT" if candidate.head_id < candidate.dependent_id else "RIGHT"
        distance = abs(candidate.head_id - candidate.dependent_id)
        geometry = f"{direction}:{'1' if distance == 1 else '2' if distance == 2 else '3+'}"
    raw = (candidate.dependent, candidate.dependent_upos, candidate.relation,
           candidate.head, candidate.head_upos, geometry)
    return tuple(f"{field}={value}" for field, value in zip(FEATURE_FIELDS, raw))


def _fit_vocabulary(candidates: Sequence[EWTCandidate]) -> Dict[str, int]:
    return {token: index + 1 for index, token in enumerate(sorted({
        feature for candidate in candidates for feature in _feature(candidate)
    }))}


def _encode(candidates: Sequence[EWTCandidate], vocabulary: Mapping[str, int]) -> Tuple[List[List[int]], int]:
    encoded: List[List[int]] = []
    unknown = 0
    for candidate in candidates:
        row = [vocabulary.get(feature, 0) for feature in _feature(candidate)]
        unknown += sum(value == 0 for value in row)
        encoded.append(row)
    return encoded, unknown


def _build_models(vocabulary_size: int):
    torch, nn = _torch()
    dim, heads, ff, layers = (int(ARCHITECTURE[key]) for key in (
        "embedding_dim", "heads", "feedforward_dim", "layers"))
    length = len(FEATURE_FIELDS)

    def encoder():
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ff, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        return nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(dim),
                                     enable_nested_tensor=False)

    class Dense(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocabulary_size, dim)
            self.body = nn.Sequential(
                nn.Linear(length * dim, int(ARCHITECTURE["dense_hidden_dim"])), nn.GELU(),
                nn.Linear(int(ARCHITECTURE["dense_hidden_dim"]), 2),
            )
        def forward(self, x):
            return self.body(self.embedding(x).flatten(1))

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocabulary_size, dim)
            self.position = nn.Parameter(torch.empty(1, length, dim))
            nn.init.normal_(self.position, std=0.02)
            self.body, self.readout = encoder(), nn.Linear(dim, 2)
        def forward(self, x):
            return self.readout(self.body(self.embedding(x) + self.position).mean(dim=1))

    class BertStyle(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocabulary_size, dim)
            self.cls = nn.Parameter(torch.empty(1, 1, dim))
            self.position = nn.Parameter(torch.empty(1, length + 1, dim))
            nn.init.normal_(self.cls, std=0.02); nn.init.normal_(self.position, std=0.02)
            self.body, self.pooler, self.readout = encoder(), nn.Linear(dim, dim), nn.Linear(dim, 2)
        def forward(self, x):
            batch = x.shape[0]
            hidden = torch.cat([self.cls.expand(batch, -1, -1), self.embedding(x)], dim=1)
            hidden = self.body(hidden + self.position)
            return self.readout(torch.tanh(self.pooler(hidden[:, 0])))

    class GPTStyle(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocabulary_size, dim)
            self.position = nn.Parameter(torch.empty(1, length, dim))
            nn.init.normal_(self.position, std=0.02)
            self.body, self.readout = encoder(), nn.Linear(dim, 2)
        def forward(self, x):
            mask = torch.full((length, length), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=1)
            hidden = self.body(self.embedding(x) + self.position, mask=mask, is_causal=True)
            return self.readout(hidden[:, -1])

    return {"dense": Dense, "transformer": Transformer,
            "bert_style": BertStyle, "gpt_style": GPTStyle}


def _summary(labels: Sequence[int], predictions: Sequence[int]) -> BinarySummary:
    tp = sum(label == prediction == 1 for label, prediction in zip(labels, predictions))
    tn = sum(label == prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return BinarySummary(
        total=len(labels), accuracy=round((tp + tn) / max(1, len(labels)), 8),
        precision=round(precision, 8), recall=round(recall, 8),
        f1=round(2 * precision * recall / (precision + recall), 8) if precision + recall else 0.0,
        false_acceptance_rate=round(fp / max(1, fp + tn), 8),
        false_rejection_rate=round(fn / max(1, fn + tp), 8),
    )


def _schedule(torch, pairs: int, steps: int, batch_size: int, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed + 919)
    chosen = torch.randint(0, pairs, (steps, batch_size // 2), generator=generator)
    return torch.stack((2 * chosen, 2 * chosen + 1), dim=-1).reshape(steps, batch_size)


def _run_seed(task: EWTTaskData, seed: int, profile: str) -> Dict[str, Any]:
    if profile not in PROFILE:
        raise ValueError(f"Unknown profile: {profile}")
    torch, nn = _torch()
    seed_everything(seed)
    config = PROFILE[profile]
    vocabulary = _fit_vocabulary(task.train)
    train_rows, train_unknown = _encode(task.train, vocabulary)
    dev_rows, _ = _encode(task.dev, vocabulary)
    test_rows, _ = _encode(task.test, vocabulary)
    train_x, dev_x, test_x = (torch.tensor(rows, dtype=torch.long)
                               for rows in (train_rows, dev_rows, test_rows))
    train_y = torch.tensor([int(row.expected_valid) for row in task.train], dtype=torch.long)
    dev_y = [int(row.expected_valid) for row in task.dev]
    test_y = [int(row.expected_valid) for row in task.test]
    schedule = _schedule(torch, len(task.train) // 2, int(config["steps"]),
                         int(config["batch_size"]), int(seed))
    constructors = _build_models(len(vocabulary) + 1)
    reports: Dict[str, Dict[str, Any]] = {}
    for index, name in enumerate(MODEL_ORDER):
        seed_everything(seed + 100 * index)
        model = constructors[name]()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                      weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()
        model.train()
        started = time.perf_counter()
        for batch in schedule:
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x[batch])
            loss = loss_fn(logits, train_y[batch])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss in {name}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            dev_predictions = model(dev_x).argmax(dim=1).tolist()
            test_predictions = model(test_x).argmax(dim=1).tolist()
        parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        reports[name] = {
            "physical_parameters": parameter_count,
            "dev": asdict(_summary(dev_y, dev_predictions)),
            "test": asdict(_summary(test_y, test_predictions)),
            "training_seconds": round(time.perf_counter() - started, 6),
            "trained_from_scratch": True,
            "architecture_note": (
                "BERT-style bidirectional encoder; not pretrained BERT." if name == "bert_style" else
                "GPT-style causal decoder; not pretrained GPT." if name == "gpt_style" else
                "parameter-matched architectural baseline."
            ),
        }
    counts = [reports[name]["physical_parameters"] for name in MODEL_ORDER]
    ratio = max(counts) / min(counts)
    if train_unknown:
        raise AssertionError("Train-only vocabulary produced unknown train features")
    return {
        "seed": int(seed), "vocabulary_hash": canonical_hash(vocabulary),
        "batch_schedule_hash": hashlib.sha256(schedule.numpy().tobytes()).hexdigest(),
        "models": reports, "parameter_max_to_min_ratio": round(ratio, 8),
    }


def run_english_ewt_baselines(seeds: Sequence[int] = (1, 2, 3, 4, 5),
                              profile: str = "smoke") -> EnglishEWTReport:
    normalized = [int(seed) for seed in seeds]
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("At least one unique seed is required")
    task = prepare_english_ewt_task()
    runs = [_run_seed(task, seed, profile) for seed in normalized]
    aggregate: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model in MODEL_ORDER:
        aggregate[model] = {}
        for metric in ("accuracy", "precision", "recall", "f1", "false_acceptance_rate", "false_rejection_rate"):
            values = [float(run["models"][model]["test"][metric]) for run in runs]
            aggregate[model][metric] = {
                "mean": round(statistics.fmean(values), 8),
                "std": round(statistics.pstdev(values), 8),
                "min": min(values), "max": max(values),
            }
    parameter_counts = runs[0]["models"]
    checks = {
        "official_english_ud_sources_hash_verified": True,
        "all_four_baseline_families_present": all(tuple(run["models"]) == MODEL_ORDER for run in runs),
        "all_models_trained_from_scratch_and_reported": all(
            all(item["trained_from_scratch"] for item in run["models"].values()) for run in runs),
        "parameter_budget_within_twelve_percent": all(
            run["parameter_max_to_min_ratio"] <= float(ARCHITECTURE["parameter_ratio_max"]) for run in runs),
        "five_or_more_seeds": len(normalized) >= 5,
        "train_dev_test_candidate_hashes_distinct": len(set(task.candidate_hashes.values())) == 3,
    }
    return EnglishEWTReport(
        protocol="hga-english-ud-ewt-architecture-baselines-v1", profile=profile,
        seeds=normalized, dataset_hash=task.dataset_hash, config_hash=task.config_hash,
        source_hashes=task.source_hashes, candidate_hashes=task.candidate_hashes,
        sentence_counts=task.sentence_counts,
        parameter_counts={name: int(parameter_counts[name]["physical_parameters"]) for name in MODEL_ORDER},
        per_seed=runs, aggregate=aggregate, checks=checks,
        limitations=[
            "This is English-only; together with the separate Turkish TWT protocol it is bilingual evidence, not a multilingual foundation-model evaluation.",
            "The task is basic UD dependency-arc verification, not end-to-end parsing, language modelling, semantic relation extraction or general reasoning.",
            "BERT-style and GPT-style arms are small random-initialized architectural baselines trained from scratch, not pretrained BERT/GPT checkpoints.",
            "The deterministic source-verified sentence cap controls CPU cost; it is not the full EWT corpus or a SOTA evaluation.",
        ],
    )


def english_ewt_markdown(report: EnglishEWTReport) -> str:
    lines = [
        "# English UD EWT Architecture Baselines", "",
        f"- Protocol: `{report.protocol}` · profile: `{report.profile}` · seeds: `{report.seeds}`",
        f"- Dataset hash: `{report.dataset_hash}` · config hash: `{report.config_hash}`",
        f"- Official-source selected sentences: train/dev/test = "
        f"`{report.sentence_counts['train']}/{report.sentence_counts['dev']}/{report.sentence_counts['test']}`",
        "", "| Model | Params | Test accuracy (mean ± std) | Test F1 (mean ± std) | FAR | FRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        values = report.aggregate[model]
        lines.append(
            f"| {model} | {report.parameter_counts[model]:,} | "
            f"{values['accuracy']['mean']:.4f} ± {values['accuracy']['std']:.4f} | "
            f"{values['f1']['mean']:.4f} ± {values['f1']['std']:.4f} | "
            f"{values['false_acceptance_rate']['mean']:.4f} | "
            f"{values['false_rejection_rate']['mean']:.4f} |")
    lines.extend(["", "## Integrity checks", ""])
    lines.extend(f"- {'PASS' if value else 'FAIL'} — `{name}`" for name, value in report.checks.items())
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {value}" for value in report.limitations)
    return "\n".join(lines) + "\n"


__all__ = [
    "ARCHITECTURE", "DATA_DIR", "EnglishEWT", "EnglishEWTReport", "EWTTaskData",
    "MODEL_ORDER", "prepare_english_ewt_task", "run_english_ewt_baselines", "english_ewt_markdown",
]

# ── HGA parameter-scaling probe ─────────────────────────────────────────────
# This is deliberately a *probe*, not a scaling-law claim.  It uses the same
# pinned English task and reports its narrow parameter range explicitly.
HGA_SCALE_CONFIGS: Dict[str, Dict[str, int]] = {
    "small": {"embedding_dim": 12, "heads": 3, "n": 8, "layers": 1},
    "base": {"embedding_dim": 24, "heads": 4, "n": 16, "layers": 2},
    "large": {"embedding_dim": 36, "heads": 4, "n": 24, "layers": 2},
}


def _build_hga_scale_model(vocabulary_size: int, config: Mapping[str, int]):
    """Gerçek repository HGA çekirdeğini EWT feature task'ına bağla."""
    torch, nn = _torch()
    from mimari.decoder import FraktalDecoder
    from mimari.encoder import GeometrikVeriEncoder
    from mimari.hiper_attention import HiperGeometrikAttention
    from mimari.kuresel_bag import KureselZincir

    embedding_dim, heads, n, layers = (int(config[key]) for key in (
        "embedding_dim", "heads", "n", "layers"))
    length = len(FEATURE_FIELDS)

    class ScaledHGA(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
            self.position = nn.Parameter(torch.empty(1, length, embedding_dim))
            nn.init.normal_(self.position, std=0.02)
            self.attention = HiperGeometrikAttention(
                embedding_dim, heads, dropout=0.0, is_causal=False)
            self.encoder = GeometrikVeriEncoder(length * embedding_dim, n, aktivasyon="tanh")
            self.chain = KureselZincir(n=n, katman_sayisi=layers, dropout=0.0,
                                       checkpoint_kullan=False, aktivasyon="silu")
            self.norm = nn.LayerNorm(n)
            self.decoder = FraktalDecoder(n, 2)

        def forward(self, x):
            hidden = self.attention(self.embedding(x) + self.position)
            return self.decoder(self.norm(self.chain(self.encoder(hidden.flatten(1)))))

    return ScaledHGA()


def _hga_scale_seed(task: EWTTaskData, seed: int, profile: str,
                    config: Mapping[str, int]) -> Dict[str, Any]:
    torch, nn = _torch()
    parameters = PROFILE[profile]
    seed_everything(seed)
    vocabulary = _fit_vocabulary(task.train)
    train_rows, unknown = _encode(task.train, vocabulary)
    test_rows, _ = _encode(task.test, vocabulary)
    if unknown:
        raise AssertionError("HGA scale probe train vocabulary has unknown values")
    train_x = torch.tensor(train_rows, dtype=torch.long)
    train_y = torch.tensor([int(row.expected_valid) for row in task.train], dtype=torch.long)
    test_x = torch.tensor(test_rows, dtype=torch.long)
    test_y = [int(row.expected_valid) for row in task.test]
    schedule = _schedule(torch, len(task.train) // 2, int(parameters["steps"]),
                         int(parameters["batch_size"]), seed)
    model = _build_hga_scale_model(len(vocabulary) + 1, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(parameters["learning_rate"]),
                                  weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for batch in schedule:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(train_x[batch]), train_y[batch])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite HGA scale-probe loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(test_x)
        predictions = logits.argmax(dim=1).tolist()
        test_loss = float(loss_fn(logits, torch.tensor(test_y, dtype=torch.long)).item())
    metrics = _summary(test_y, predictions)
    return {
        "physical_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "test_cross_entropy": round(test_loss, 8),
        "test": asdict(metrics),
    }


def _least_squares_slope(points: Sequence[Tuple[float, float]]) -> float | None:
    if len(points) < 3:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    if denominator == 0:
        return None
    return round(sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator, 8)


def run_english_hga_scaling_probe(seeds: Sequence[int] = (1, 2, 3, 4, 5),
                                  profile: str = "smoke") -> Dict[str, Any]:
    """EWT üzerinde HGA boyut taraması; scaling-law diye sunulmaz.

    Üç küçük model boyutu ve sınırlı eğitim adımı gerçek üretim ölçeklemesi için
    yeterli değildir. Fonksiyon, negatif/pozitif eğilimi tekrar üretilebilir
    biçimde göstermek ve 'scaling law' iddiasının neden kapalı kaldığını sayıya
    dökmek içindir.
    """
    if profile not in PROFILE:
        raise ValueError(f"Unknown profile: {profile}")
    normalized = [int(seed) for seed in seeds]
    if len(normalized) < 2 or len(set(normalized)) != len(normalized):
        raise ValueError("HGA scaling probe needs at least two unique seeds")
    task = prepare_english_ewt_task()
    rows: List[Dict[str, Any]] = []
    for label, config in HGA_SCALE_CONFIGS.items():
        runs = [_hga_scale_seed(task, seed, profile, config) for seed in normalized]
        parameter_count = int(runs[0]["physical_parameters"])
        f1 = [float(run["test"]["f1"]) for run in runs]
        cross_entropy = [float(run["test_cross_entropy"]) for run in runs]
        rows.append({
            "scale": label,
            "config": dict(config),
            "physical_parameters": parameter_count,
            "f1": {"mean": round(statistics.fmean(f1), 8), "std": round(statistics.pstdev(f1), 8)},
            "cross_entropy": {"mean": round(statistics.fmean(cross_entropy), 8),
                              "std": round(statistics.pstdev(cross_entropy), 8)},
            "per_seed": runs,
        })
    rows.sort(key=lambda row: int(row["physical_parameters"]))
    import math
    loss_slope = _least_squares_slope([
        (math.log(float(row["physical_parameters"])),
         math.log(max(float(row["cross_entropy"]["mean"]), 1e-12)))
        for row in rows
    ])
    parameter_ratio = float(rows[-1]["physical_parameters"]) / rows[0]["physical_parameters"]
    return {
        "protocol": "hga-english-ewt-parameter-scaling-probe-v1",
        "status": "EXPLORATORY_NOT_A_SCALING_LAW",
        "profile": profile,
        "seeds": normalized,
        "dataset_hash": task.dataset_hash,
        "candidate_hashes": task.candidate_hashes,
        "rows": rows,
        "log_parameter_to_cross_entropy_slope": loss_slope,
        "parameter_range_ratio": round(parameter_ratio, 8),
        "checks": {
            "same_pinned_english_task_all_sizes": True,
            "same_seed_set_all_sizes": True,
            "at_least_three_sizes": len(rows) >= 3,
            "at_least_two_seeds": len(normalized) >= 2,
            "parameter_count_strictly_increases": all(
                rows[index]["physical_parameters"] < rows[index + 1]["physical_parameters"]
                for index in range(len(rows) - 1)),
            "scaling_law_claim_blocked_by_narrow_range": parameter_ratio < 100.0,
        },
        "limitations": [
            "Three small configurations over less than two orders of magnitude cannot identify a scaling law.",
            "Training compute, token count and model size are not independently swept; the fitted log slope is descriptive only.",
            "This controlled dependency-arc task is not next-token language-model scaling and cannot predict foundation-model behaviour.",
            "A scaling-law claim requires a much wider compute/data/model grid, repeated convergence runs and held-out extrapolation.",
        ],
    }


def english_hga_scaling_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# HGA English EWT parameter-scaling probe", "",
        f"- Status: **{report['status']}**", f"- Seeds: `{report['seeds']}`",
        f"- Parameter range: `{report['parameter_range_ratio']:.2f}×`",
        f"- Descriptive log(params) → log(cross-entropy) slope: "
        f"`{report['log_parameter_to_cross_entropy_slope']}`", "",
        "| Size | Parameters | Test F1 (mean ± std) | Test cross-entropy (mean ± std) |",
        "|---|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['scale']} | {row['physical_parameters']:,} | "
            f"{row['f1']['mean']:.4f} ± {row['f1']['std']:.4f} | "
            f"{row['cross_entropy']['mean']:.4f} ± {row['cross_entropy']['std']:.4f} |")
    lines.extend(["", "## Guard rails", ""])
    lines.extend(f"- {'PASS' if value else 'FAIL'} — `{key}`" for key, value in report["checks"].items())
    lines.extend(["", "## Why this is not a scaling law", ""])
    lines.extend(f"- {value}" for value in report["limitations"])
    return "\n".join(lines) + "\n"


# Keep exports declared at the end: earlier __all__ is intentionally extended
# instead of duplicating the substantial public contract above.
__all__ += [
    "HGA_SCALE_CONFIGS", "run_english_hga_scaling_probe", "english_hga_scaling_markdown",
]
