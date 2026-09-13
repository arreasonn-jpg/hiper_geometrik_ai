"""Benchmark split'leri için içerik tabanlı veri sızıntısı denetimi."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence

TRIPLE_FIELDS = ("subject_id", "relation_id", "object_id")


def semantic_fingerprint(record: Mapping[str, Any]) -> str:
    """Kimlik/cümle biçiminden bağımsız, kanonik üçlü SHA-256 izi."""
    missing = [name for name in TRIPLE_FIELDS if name not in record]
    if missing:
        raise ValueError(f"Semantik kayıt alanları eksik: {missing}")
    payload = [str(record[name]).strip().casefold() for name in TRIPLE_FIELDS]
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class LeakageFinding:
    fingerprint: str
    test_ids: List[str]
    contaminated_partitions: List[str]


@dataclass
class LeakageAuditReport:
    clean: bool
    train_count: int
    test_count: int
    findings: List[LeakageFinding] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def assert_clean(self) -> None:
        if not self.clean:
            details = ", ".join(
                f"{f.test_ids}->{f.contaminated_partitions}" for f in self.findings
            )
            raise ValueError(f"Benchmark data leakage detected: {details}")


def audit_partitions(
    train: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    **generated_partitions: Iterable[Mapping[str, Any]],
) -> LeakageAuditReport:
    """Test üçlülerinin train/memory/knowledge/candidate alanlarına sızmasını bul.

    ``generated_partitions`` anahtarları raporda aynen kullanılır. Böylece koşu
    öncesi ``memory=...``, ``knowledge=...`` ve ``candidates=...`` ayrı ayrı
    denetlenebilir.
    """
    sources = {"train": train, **generated_partitions}
    source_hashes = {
        name: {semantic_fingerprint(record) for record in records}
        for name, records in sources.items()
    }
    test_by_hash: Dict[str, List[str]] = {}
    for record in test:
        fp = semantic_fingerprint(record)
        test_by_hash.setdefault(fp, []).append(str(record.get("experience_id", "<unknown>")))

    findings: List[LeakageFinding] = []
    for fp, test_ids in sorted(test_by_hash.items()):
        contaminated = [name for name, hashes in source_hashes.items() if fp in hashes]
        if contaminated:
            findings.append(LeakageFinding(fp, test_ids, contaminated))
    return LeakageAuditReport(
        clean=not findings,
        train_count=len(train),
        test_count=len(test),
        findings=findings,
    )


__all__ = [
    "LeakageFinding",
    "LeakageAuditReport",
    "semantic_fingerprint",
    "audit_partitions",
]
