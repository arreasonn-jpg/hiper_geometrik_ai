"""Time-aware provenance lifecycle for externally sourced knowledge.

STALE means that freshness evidence expired or a dependency is no longer
current.  It is deliberately distinct from RETRACTED (known withdrawn) and
never means that a proposition has been proven false.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class KnowledgeLifecycleState(str, Enum):
    ACTIVE = "ACTIVE"
    STALE = "STALE"
    SUPERSEDED = "SUPERSEDED"
    RETRACTED = "RETRACTED"


@dataclass
class LifecycleRecord:
    record_id: str
    subject_id: str
    relation_id: str
    object_id: str
    source_url: str
    document_hash: str
    source_revision: str
    retrieved_at: float
    validated_at: float
    max_age_seconds: float
    state: KnowledgeLifecycleState = KnowledgeLifecycleState.ACTIVE
    supersedes: Optional[str] = None
    derived_from: List[str] = field(default_factory=list)
    stale_reason: Optional[str] = None
    retracted_reason: Optional[str] = None

    @property
    def next_validation_due(self) -> float:
        return self.validated_at + self.max_age_seconds

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["state"] = self.state.value
        value["next_validation_due"] = self.next_validation_due
        return value


@dataclass(frozen=True)
class LifecycleEvent:
    sequence: int
    event: str
    record_id: str
    at: float
    previous_state: Optional[str]
    new_state: str
    reason: str
    chain_hash: str


class KnowledgeLifecycle:
    """Explicit-clock lifecycle registry with immutable hash-chained events."""

    def __init__(self):
        self.records: Dict[str, LifecycleRecord] = {}
        self.events: List[LifecycleEvent] = []
        self._source_heads: Dict[str, str] = {}

    def _event(
        self, event: str, record: LifecycleRecord, at: float,
        previous: Optional[KnowledgeLifecycleState], reason: str,
    ) -> None:
        parent = self.events[-1].chain_hash if self.events else "0" * 64
        # Alanlar önce tiplenmiş yerel değişkenlere alınır; payload yalnız
        # hash girdisidir. Böylece hash sözleşmesi bozulmadan LifecycleEvent
        # tip güvenli kurulur (dict[str, object] unpack'i yerine).
        sequence = len(self.events) + 1
        at_value = float(at)
        previous_state = previous.value if previous else None
        new_state = record.state.value
        payload: Dict[str, Any] = {
            "sequence": sequence,
            "event": event,
            "record_id": record.record_id,
            "at": at_value,
            "previous_state": previous_state,
            "new_state": new_state,
            "reason": reason,
            "parent": parent,
        }
        chain_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.events.append(LifecycleEvent(
            chain_hash=chain_hash,
            sequence=sequence,
            event=event,
            record_id=record.record_id,
            at=at_value,
            previous_state=previous_state,
            new_state=new_state,
            reason=reason,
        ))

    def ingest(
        self, *, record_id: str, subject_id: str, relation_id: str,
        object_id: str, source_url: str, document_hash: str,
        source_revision: str, retrieved_at: float, max_age_seconds: float,
        derived_from: Sequence[str] = (),
    ) -> LifecycleRecord:
        if record_id in self.records:
            raise ValueError(f"Lifecycle record zaten var: {record_id}")
        if len(document_hash) != 64 or max_age_seconds <= 0 or not source_url:
            raise ValueError("İzlenebilir source URL/SHA-256 ve pozitif max age gerekli")
        missing = sorted(set(derived_from) - set(self.records))
        if missing:
            raise ValueError(f"Bilinmeyen lifecycle dependency: {missing}")
        predecessor_id = self._source_heads.get(source_url)
        predecessor = self.records.get(predecessor_id) if predecessor_id else None
        if predecessor and predecessor.document_hash == document_hash:
            raise ValueError("Aynı source/hash yeni record değil revalidate edilmelidir")
        record = LifecycleRecord(
            record_id=record_id, subject_id=subject_id, relation_id=relation_id,
            object_id=object_id, source_url=source_url, document_hash=document_hash,
            source_revision=source_revision, retrieved_at=float(retrieved_at),
            validated_at=float(retrieved_at), max_age_seconds=float(max_age_seconds),
            supersedes=predecessor_id, derived_from=list(derived_from),
        )
        self.records[record_id] = record
        if predecessor and predecessor.state != KnowledgeLifecycleState.RETRACTED:
            previous = predecessor.state
            predecessor.state = KnowledgeLifecycleState.SUPERSEDED
            predecessor.stale_reason = f"superseded_by:{record_id}"
            self._event("SUPERSEDE", predecessor, retrieved_at, previous, predecessor.stale_reason)
        self._source_heads[source_url] = record_id
        self._event("INGEST", record, retrieved_at, None, "source revision ingested")
        self.propagate_dependencies(retrieved_at)
        return record

    def revalidate(self, record_id: str, *, at: float, document_hash: str) -> LifecycleRecord:
        record = self.records[record_id]
        if record.state in (
            KnowledgeLifecycleState.SUPERSEDED, KnowledgeLifecycleState.RETRACTED
        ):
            raise ValueError("Superseded/retracted record yeniden aktive edilemez")
        if document_hash != record.document_hash:
            raise ValueError("Değişen içerik revalidate değil yeni revision olarak ingest edilmelidir")
        previous = record.state
        record.state = KnowledgeLifecycleState.ACTIVE
        record.validated_at = float(at)
        record.stale_reason = None
        self._event("REVALIDATE", record, at, previous, "same content hash confirmed")
        return record

    def unavailable(self, record_id: str, *, at: float, reason: str) -> None:
        record = self.records[record_id]
        if record.state in (KnowledgeLifecycleState.SUPERSEDED, KnowledgeLifecycleState.RETRACTED):
            return
        previous = record.state
        record.state = KnowledgeLifecycleState.STALE
        record.stale_reason = f"source_unavailable:{reason}"
        self._event("SOURCE_UNAVAILABLE", record, at, previous, record.stale_reason)
        self.propagate_dependencies(at)

    def retract(self, record_id: str, *, at: float, reason: str) -> None:
        record = self.records[record_id]
        previous = record.state
        record.state = KnowledgeLifecycleState.RETRACTED
        record.retracted_reason = reason
        record.stale_reason = None
        self._event("RETRACT", record, at, previous, reason)
        self.propagate_dependencies(at)

    def audit(self, *, at: float) -> Dict[str, int]:
        for record in self.records.values():
            if record.state == KnowledgeLifecycleState.ACTIVE and at > record.next_validation_due:
                previous = record.state
                record.state = KnowledgeLifecycleState.STALE
                record.stale_reason = "freshness_window_expired"
                self._event("EXPIRE", record, at, previous, record.stale_reason)
        self.propagate_dependencies(at)
        return self.counts()

    def propagate_dependencies(self, at: float) -> None:
        changed = True
        while changed:
            changed = False
            for record in self.records.values():
                if record.state != KnowledgeLifecycleState.ACTIVE or not record.derived_from:
                    continue
                nonactive = [
                    dependency for dependency in record.derived_from
                    if self.records[dependency].state != KnowledgeLifecycleState.ACTIVE
                ]
                if nonactive:
                    previous = record.state
                    record.state = KnowledgeLifecycleState.STALE
                    record.stale_reason = f"nonactive_dependencies:{','.join(sorted(nonactive))}"
                    self._event("DEPENDENCY_STALE", record, at, previous, record.stale_reason)
                    changed = True

    def query(self, *, include_stale: bool = False) -> List[LifecycleRecord]:
        allowed = {KnowledgeLifecycleState.ACTIVE}
        if include_stale:
            allowed.add(KnowledgeLifecycleState.STALE)
        return sorted(
            (record for record in self.records.values() if record.state in allowed),
            key=lambda record: record.record_id,
        )

    def counts(self) -> Dict[str, int]:
        return {
            state.value: sum(record.state == state for record in self.records.values())
            for state in KnowledgeLifecycleState
        }

    def verify_event_chain(self) -> bool:
        parent = "0" * 64
        for event in self.events:
            payload = {
                "sequence": event.sequence, "event": event.event,
                "record_id": event.record_id, "at": event.at,
                "previous_state": event.previous_state, "new_state": event.new_state,
                "reason": event.reason, "parent": parent,
            }
            expected = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if expected != event.chain_hash:
                return False
            parent = event.chain_hash
        return True


__all__ = [
    "KnowledgeLifecycle", "KnowledgeLifecycleState", "LifecycleEvent", "LifecycleRecord"
]
