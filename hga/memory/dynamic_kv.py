# -*- coding: utf-8 -*-
"""Aktif, sürümlü ve kalıcı Dynamic KV deneyim belleği.

Anahtar tam ``(subject_id, relation_id, object_id)`` üçlüsüdür; 31-bit hash
yalnız gözlem/adres raporunda kullanılır ve kimlik olarak kullanılmaz. Bu
nedenle sabit slot çakışması yoktur. Bedeli, fiziksel kaydın tutulan deneyim
sayısıyla büyümesidir.
"""
from __future__ import annotations

import copy
import hashlib
import heapq
import json
import os
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .sparse_memory import parmak_izi

Triple = Tuple[str, str, str]
DYNAMIC_KV_FORMAT = "hga-dynamic-kv"
DYNAMIC_KV_SCHEMA_VERSION = 2


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _content_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _normalize_key(parts: Iterable[Any]) -> Triple:
    key = tuple(str(part) for part in parts)
    if len(key) != 3:
        raise ValueError("Dynamic KV anahtarı tam üç bileşenli olmalı")
    if any(not part for part in key):
        raise ValueError("Dynamic KV anahtar bileşenleri boş olamaz")
    return key  # type: ignore[return-value]


@dataclass
class DynamicKVRecord:
    key: Triple
    experience_id: str
    payload: Optional[Dict[str, Any]]
    created_tick: int
    updated_tick: int
    last_access_tick: int
    access_count: int
    record_version: int

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["key"] = list(self.key)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DynamicKVRecord":
        return cls(
            key=_normalize_key(data["key"]),
            experience_id=str(data["experience_id"]),
            payload=copy.deepcopy(data.get("payload")),
            created_tick=int(data.get("created_tick", 0)),
            updated_tick=int(data.get("updated_tick", data.get("created_tick", 0))),
            last_access_tick=int(data.get("last_access_tick", data.get("created_tick", 0))),
            access_count=int(data.get("access_count", 0)),
            record_version=int(data.get("record_version", 1)),
        )


@dataclass
class DynamicKVSnapshot:
    schema_version: int
    store_version: int
    tick: int
    max_entries: Optional[int]
    eviction_policy: str
    max_idle_ticks: Optional[int]
    records: List[Dict[str, Any]]
    journal: List[Dict[str, Any]]
    statistics: Dict[str, int]
    parent_snapshot_hash: Optional[str]
    label: Optional[str]
    created_at_utc: str
    content_hash: str = ""

    def _unhashed_dict(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in ("content_hash", "created_at_utc")
        }

    def seal(self) -> "DynamicKVSnapshot":
        self.content_hash = _content_hash(self._unhashed_dict())
        return self

    def to_dict(self) -> Dict[str, Any]:
        if not self.content_hash:
            self.seal()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DynamicKVSnapshot":
        raw = dict(data)
        expected = str(raw.pop("content_hash", ""))
        hashed_payload = {
            key: value for key, value in raw.items() if key != "created_at_utc"
        }
        actual = _content_hash(hashed_payload)
        if not expected or expected != actual:
            raise ValueError("Dynamic KV snapshot bütünlük hash'i eşleşmiyor")
        snapshot = cls(**raw, content_hash=expected)
        if snapshot.schema_version != DYNAMIC_KV_SCHEMA_VERSION:
            raise ValueError(
                f"Desteklenmeyen Dynamic KV snapshot şeması: {snapshot.schema_version}"
            )
        return snapshot


@dataclass
class EvictionEvent:
    key: Triple
    experience_id: str
    reason: str
    store_version: int

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["key"] = list(self.key)
        return data


@dataclass
class CompactionReport:
    journal_before: int
    journal_after: int
    reclaimed_events: int
    active_records: int
    compaction_count: int

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class MigrationReport:
    source_policy: str
    target_policy: str
    scanned_candidates: int
    migrated_records: int
    missing_from_source: int
    source_collision_events: int
    target_records: int
    fidelity: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DynamicKVMemory:
    """Tam üçlü anahtarlı, bounded/unbounded Dynamic KV deposu.

    ``max_entries`` dolduğunda deterministik LRU veya FIFO eviction uygulanır.
    ``max_idle_ticks`` verilirse erişimsiz kalan kayıtlar TTL-benzeri biçimde
    temizlenir. Her logical mutation ``store_version`` değerini artırır.
    """

    def __init__(
        self,
        max_entries: Optional[int] = None,
        eviction_policy: str = "lru",
        max_idle_ticks: Optional[int] = None,
        max_journal_entries: Optional[int] = None,
    ):
        if max_entries is not None and int(max_entries) < 1:
            raise ValueError("max_entries None veya >= 1 olmalı")
        eviction_policy = str(eviction_policy).lower()
        if eviction_policy not in ("lru", "fifo"):
            raise ValueError("eviction_policy 'lru' veya 'fifo' olmalı")
        if max_idle_ticks is not None and int(max_idle_ticks) < 0:
            raise ValueError("max_idle_ticks negatif olamaz")
        self.max_entries = int(max_entries) if max_entries is not None else None
        self.eviction_policy = eviction_policy
        self.max_idle_ticks = (
            int(max_idle_ticks) if max_idle_ticks is not None else None
        )
        default_journal_limit = max(1_000, 4 * (self.max_entries or 250))
        self.max_journal_entries = int(max_journal_entries or default_journal_limit)
        if self.max_journal_entries < 1:
            raise ValueError("max_journal_entries >= 1 olmalı")
        self._records: Dict[Triple, DynamicKVRecord] = {}
        self._journal: List[Dict[str, Any]] = []
        # Lazy-invalidated heap kapasite eviction'ını O(N) tarama yerine
        # amortize O(log N) yapar. Tuple: (policy tick, key).
        self._eviction_heap: List[Tuple[int, Triple]] = []
        self._tick = 0
        self.store_version = 0
        self._last_snapshot_hash: Optional[str] = None
        self._last_snapshot: Optional[DynamicKVSnapshot] = None
        self._last_evictions: List[EvictionEvent] = []
        self.cakisma_sayisi = 0
        self._stats: Dict[str, int] = {
            "reads": 0,
            "writes": 0,
            "hits": 0,
            "misses": 0,
            "updates": 0,
            "evictions": 0,
            "ttl_evictions": 0,
            "compactions": 0,
        }
        self._lock = threading.RLock()

    @property
    def okuma_sayisi(self) -> int:
        return self._stats["reads"]

    @property
    def yazma_sayisi(self) -> int:
        return self._stats["writes"]

    @property
    def last_evictions(self) -> List[EvictionEvent]:
        return list(self._last_evictions)

    @property
    def last_snapshot(self) -> Optional[DynamicKVSnapshot]:
        return copy.deepcopy(self._last_snapshot)

    def adresler(self, anahtar_bilesenleri) -> Tuple[int, ...]:
        return (parmak_izi(_normalize_key(anahtar_bilesenleri)),)

    def _append_journal(self, operation: str, record: DynamicKVRecord) -> None:
        self._journal.append({
            "operation": operation,
            "store_version": self.store_version,
            "key": list(record.key),
            "experience_id": record.experience_id,
            "record_version": record.record_version,
        })
        # Bir compact journal etkin kayıt başına en az bir event taşır. Etkin
        # cardinality sabit eşiği geçtiğinde her yazıda O(N) compact döngüsüne
        # girmemek için ancak journal hem mutlak eşiği hem 2× canlı kayıt
        # eşiğini aştığında otomatik sıkıştır.
        automatic_threshold = max(
            self.max_journal_entries, 4 * max(1, len(self._records))
        )
        if len(self._journal) > automatic_threshold:
            self._compact_locked()

    def _remove_locked(self, key: Triple, reason: str) -> Optional[DynamicKVRecord]:
        record = self._records.pop(key, None)
        if record is None:
            return None
        self.store_version += 1
        self._stats["evictions"] += 1
        if reason == "ttl":
            self._stats["ttl_evictions"] += 1
        event = EvictionEvent(key, record.experience_id, reason, self.store_version)
        self._last_evictions.append(event)
        self._append_journal("delete", record)
        return record

    def _evict_expired_locked(self) -> List[DynamicKVRecord]:
        if self.max_idle_ticks is None:
            return []
        threshold = self._tick - self.max_idle_ticks
        expired = sorted(
            (
                record for record in self._records.values()
                if record.last_access_tick < threshold
            ),
            key=lambda record: (record.last_access_tick, record.key),
        )
        removed = []
        for record in expired:
            deleted = self._remove_locked(record.key, "ttl")
            if deleted is not None:
                removed.append(deleted)
        return removed

    def _heap_metric(self, record: DynamicKVRecord) -> int:
        if self.eviction_policy == "fifo":
            return record.created_tick
        return record.last_access_tick

    def _heap_push_locked(self, record: DynamicKVRecord) -> None:
        heapq.heappush(self._eviction_heap, (self._heap_metric(record), record.key))
        # Çok sayıda LRU read aynı anahtar için stale heap düğümü üretmesin.
        if len(self._eviction_heap) > max(1_000, 4 * max(1, len(self._records))):
            self._rebuild_eviction_heap_locked()

    def _rebuild_eviction_heap_locked(self) -> None:
        self._eviction_heap = [
            (self._heap_metric(record), record.key)
            for record in self._records.values()
        ]
        heapq.heapify(self._eviction_heap)

    def _capacity_victim_locked(self) -> DynamicKVRecord:
        while self._eviction_heap:
            metric, key = heapq.heappop(self._eviction_heap)
            record = self._records.get(key)
            if record is not None and metric == self._heap_metric(record):
                return record
        # Heap yalnız restore/tamper kaynaklı eksik olabilir; yeniden kur ve
        # sessizce yanlış kayıt seçmek yerine açık invariant uygula.
        self._rebuild_eviction_heap_locked()
        if not self._eviction_heap:
            raise RuntimeError("Dynamic KV eviction heap boş fakat kapasite dolu")
        return self._capacity_victim_locked()

    def yaz(
        self,
        experience_id: str,
        anahtar_bilesenleri,
        payload: Optional[Mapping[str, Any]] = None,
        compute_address: bool = True,
    ) -> int:
        """Kaydı upsert et; istenirse gözlem için deterministik adres döndür.

        ``compute_address=False`` yalnız büyük streaming stresinde, çağıranın
        adresi kullanmadığı durumda hash maliyetini atlar; saklama/eviction/
        journal yolu aynıdır.
        """
        if not str(experience_id):
            raise ValueError("experience_id boş olamaz")
        key = _normalize_key(anahtar_bilesenleri)
        with self._lock:
            self._tick += 1
            self._stats["writes"] += 1
            self._last_evictions = []
            self._evict_expired_locked()
            previous = self._records.get(key)
            if previous is None and self.max_entries is not None:
                while len(self._records) >= self.max_entries:
                    victim = self._capacity_victim_locked()
                    self._remove_locked(victim.key, self.eviction_policy)
            self.store_version += 1
            if previous is None:
                record = DynamicKVRecord(
                    key=key,
                    experience_id=str(experience_id),
                    payload=copy.deepcopy(dict(payload)) if payload is not None else None,
                    created_tick=self._tick,
                    updated_tick=self._tick,
                    last_access_tick=self._tick,
                    access_count=1,
                    record_version=1,
                )
            else:
                self._stats["updates"] += 1
                record = DynamicKVRecord(
                    key=key,
                    experience_id=str(experience_id),
                    payload=copy.deepcopy(dict(payload)) if payload is not None else None,
                    created_tick=previous.created_tick,
                    updated_tick=self._tick,
                    last_access_tick=self._tick,
                    access_count=previous.access_count + 1,
                    record_version=previous.record_version + 1,
                )
            self._records[key] = record
            self._heap_push_locked(record)
            self._append_journal("upsert", record)
            return self.adresler(key)[0] if compute_address else -1

    def getir(self, anahtar_bilesenleri, touch: bool = True) -> Optional[DynamicKVRecord]:
        key = _normalize_key(anahtar_bilesenleri)
        with self._lock:
            self._tick += 1
            self._stats["reads"] += 1
            self._last_evictions = []
            self._evict_expired_locked()
            record = self._records.get(key)
            if record is None:
                self._stats["misses"] += 1
                return None
            self._stats["hits"] += 1
            if touch:
                record.last_access_tick = self._tick
                record.access_count += 1
                if self.eviction_policy == "lru":
                    self._heap_push_locked(record)
            return copy.deepcopy(record)

    def icerir(self, experience_id: str, anahtar_bilesenleri) -> bool:
        record = self.getir(anahtar_bilesenleri)
        return record is not None and record.experience_id == experience_id

    def kayitlar(self) -> List[DynamicKVRecord]:
        with self._lock:
            return [copy.deepcopy(self._records[key]) for key in sorted(self._records)]

    def lru_temizle(self, max_yas: int) -> int:
        max_yas = int(max_yas)
        if max_yas < 0:
            raise ValueError("max_yas negatif olamaz")
        with self._lock:
            self._last_evictions = []
            threshold = self._tick - max_yas
            keys = sorted(
                key for key, record in self._records.items()
                if record.last_access_tick < threshold
            )
            for key in keys:
                self._remove_locked(key, "manual_lru")
            return len(keys)

    def _compact_locked(self) -> CompactionReport:
        before = len(self._journal)
        self._journal = [
            {
                "operation": "upsert",
                "store_version": self.store_version,
                "key": list(record.key),
                "experience_id": record.experience_id,
                "record_version": record.record_version,
            }
            for record in self.kayitlar()
        ]
        self._stats["compactions"] += 1
        return CompactionReport(
            journal_before=before,
            journal_after=len(self._journal),
            reclaimed_events=max(0, before - len(self._journal)),
            active_records=len(self._records),
            compaction_count=self._stats["compactions"],
        )

    def sikistir(self) -> CompactionReport:
        """Eski upsert/tombstone journal olaylarını etkin kayıt başına teke indir."""
        with self._lock:
            return self._compact_locked()

    def snapshot_olustur(self, label: Optional[str] = None) -> DynamicKVSnapshot:
        with self._lock:
            snapshot = DynamicKVSnapshot(
                schema_version=DYNAMIC_KV_SCHEMA_VERSION,
                store_version=self.store_version,
                tick=self._tick,
                max_entries=self.max_entries,
                eviction_policy=self.eviction_policy,
                max_idle_ticks=self.max_idle_ticks,
                records=[record.to_dict() for record in self.kayitlar()],
                journal=copy.deepcopy(self._journal),
                statistics=dict(self._stats),
                parent_snapshot_hash=self._last_snapshot_hash,
                label=label,
                created_at_utc=datetime.now(timezone.utc).isoformat(),
            ).seal()
            self._last_snapshot_hash = snapshot.content_hash
            self._last_snapshot = copy.deepcopy(snapshot)
            return snapshot

    @classmethod
    def snapshotten_olustur(cls, snapshot: DynamicKVSnapshot) -> "DynamicKVMemory":
        memory = cls(
            max_entries=snapshot.max_entries,
            eviction_policy=snapshot.eviction_policy,
            max_idle_ticks=snapshot.max_idle_ticks,
        )
        memory._records = {
            record.key: record
            for record in (DynamicKVRecord.from_dict(item) for item in snapshot.records)
        }
        if memory.max_entries is not None and len(memory._records) > memory.max_entries:
            raise ValueError("Snapshot kayıt sayısı max_entries sınırını aşıyor")
        memory._journal = copy.deepcopy(snapshot.journal)
        memory._rebuild_eviction_heap_locked()
        memory._stats.update({key: int(value) for key, value in snapshot.statistics.items()})
        memory._tick = int(snapshot.tick)
        memory.store_version = int(snapshot.store_version)
        memory._last_snapshot_hash = snapshot.content_hash
        memory._last_snapshot = copy.deepcopy(snapshot)
        return memory

    def snapshot_geri_yukle(self, snapshot: DynamicKVSnapshot) -> int:
        """Snapshot verisini geri yükle; logical store version monoton kalır."""
        restored = self.snapshotten_olustur(snapshot)
        with self._lock:
            previous_version = self.store_version
            self.max_entries = restored.max_entries
            self.eviction_policy = restored.eviction_policy
            self.max_idle_ticks = restored.max_idle_ticks
            self._records = restored._records
            self._journal = restored._journal
            self._eviction_heap = restored._eviction_heap
            self._stats = restored._stats
            self._tick = max(self._tick, restored._tick) + 1
            self.store_version = max(previous_version, restored.store_version) + 1
            self._last_snapshot_hash = snapshot.content_hash
            self._last_snapshot = copy.deepcopy(snapshot)
            return self.store_version

    def kaydet(self, path: os.PathLike | str, label: Optional[str] = None) -> DynamicKVSnapshot:
        """Belleği atomik, hash-korumalı UTF-8 JSON snapshot olarak kaydet."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot = self.snapshot_olustur(label=label)
        envelope = {
            "format": DYNAMIC_KV_FORMAT,
            "schema_version": DYNAMIC_KV_SCHEMA_VERSION,
            "snapshot": snapshot.to_dict(),
        }
        serialized = json.dumps(
            envelope, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        return snapshot

    @classmethod
    def yukle(cls, path: os.PathLike | str) -> "DynamicKVMemory":
        with open(path, "r", encoding="utf-8") as handle:
            envelope = json.load(handle)
        if envelope.get("format") != DYNAMIC_KV_FORMAT:
            raise ValueError("Dosya HGA Dynamic KV formatında değil")
        schema = int(envelope.get("schema_version", 0))
        snapshot_data = envelope.get("snapshot", {})
        if schema == 1:
            snapshot_data = cls._migrate_v1_snapshot(snapshot_data)
        elif schema != DYNAMIC_KV_SCHEMA_VERSION:
            raise ValueError(f"Desteklenmeyen Dynamic KV dosya şeması: {schema}")
        snapshot = DynamicKVSnapshot.from_dict(snapshot_data)
        return cls.snapshotten_olustur(snapshot)

    @staticmethod
    def _migrate_v1_snapshot(data: Mapping[str, Any]) -> Dict[str, Any]:
        """Minimal v1 records formatını bütünlüklü v2 snapshot'a yükselt."""
        records = []
        tick = 0
        for item in data.get("records", []):
            tick += 1
            records.append(DynamicKVRecord(
                key=_normalize_key(item["key"]),
                experience_id=str(item["experience_id"]),
                payload=copy.deepcopy(item.get("payload")),
                created_tick=tick,
                updated_tick=tick,
                last_access_tick=tick,
                access_count=1,
                record_version=1,
            ).to_dict())
        snapshot = DynamicKVSnapshot(
            schema_version=DYNAMIC_KV_SCHEMA_VERSION,
            store_version=len(records),
            tick=tick,
            max_entries=data.get("max_entries"),
            eviction_policy=str(data.get("eviction_policy", "lru")),
            max_idle_ticks=data.get("max_idle_ticks"),
            records=records,
            journal=[],
            statistics={
                "reads": 0, "writes": len(records), "hits": 0, "misses": 0,
                "updates": 0, "evictions": 0, "ttl_evictions": 0,
                "compactions": 0,
            },
            parent_snapshot_hash=None,
            label="migrated-from-v1",
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        ).seal()
        return snapshot.to_dict()

    @classmethod
    def legacy_slotlardan_migre_et(
        cls,
        slots,
        candidates: Iterable[Any],
        max_entries: Optional[int] = None,
        eviction_policy: str = "lru",
    ) -> Tuple["DynamicKVMemory", MigrationReport]:
        """FIRST_WINS slot deposunda gerçekten tutulan adayları Dynamic KV'ye taşı."""
        candidate_list = list(candidates)
        retained = [
            candidate for candidate in candidate_list
            if slots.icerir(candidate.experience_id, candidate.uclusu)
        ]
        capacity = max_entries
        if capacity is not None and capacity < len(retained):
            raise ValueError("Migration max_entries, tutulan legacy kayıtlardan küçük")
        memory = cls(max_entries=capacity, eviction_policy=eviction_policy)
        for candidate in retained:
            payload = candidate.to_dict() if hasattr(candidate, "to_dict") else None
            memory.yaz(candidate.experience_id, candidate.uclusu, payload=payload)
        scanned = len(candidate_list)
        migrated = len(retained)
        report = MigrationReport(
            source_policy="FIRST_WINS",
            target_policy="DYNAMIC_KV",
            scanned_candidates=scanned,
            migrated_records=migrated,
            missing_from_source=scanned - migrated,
            source_collision_events=int(getattr(slots, "cakisma_sayisi", 0)),
            target_records=len(memory),
            fidelity=round(migrated / scanned, 8) if scanned else 1.0,
            notes=[
                "Legacy slot deposu orijinal anahtarı saklamaz; migration aday/replay payloadı üzerinden yapılır.",
                "Legacy çakışmada daha önce kaybolmuş kayıtlar uydurulmaz ve missing_from_source olarak raporlanır.",
            ],
        )
        return memory, report

    def depolama_bayt(self) -> int:
        """CPython container/record/journal/heap için yaklaşık fiziksel bayt."""
        with self._lock:
            total = (
                sys.getsizeof(self._records)
                + sys.getsizeof(self._journal)
                + sys.getsizeof(self._eviction_heap)
            )
            for key, record in self._records.items():
                total += sys.getsizeof(key) + sum(sys.getsizeof(part) for part in key)
                total += sys.getsizeof(record) + sys.getsizeof(record.experience_id)
                total += sys.getsizeof(record.payload)
            for event in self._journal:
                total += sys.getsizeof(event)
                total += sum(
                    sys.getsizeof(key) + sys.getsizeof(value)
                    for key, value in event.items()
                )
            for item in self._eviction_heap:
                total += sys.getsizeof(item)
            return total

    def doluluk_orani(self) -> Tuple[int, int]:
        total = self.max_entries if self.max_entries is not None else len(self._records)
        return len(self._records), total

    def okuma_yazma_raporu(self) -> Dict[str, float | int]:
        total = self.okuma_sayisi + self.yazma_sayisi
        return {
            "okuma": self.okuma_sayisi,
            "yazma": self.yazma_sayisi,
            "okuma_orani": self.okuma_sayisi / total if total else 0.0,
            "yazma_orani": self.yazma_sayisi / total if total else 0.0,
        }

    def kapasite(self) -> Dict[str, Any]:
        occupied, total = self.doluluk_orani()
        return {
            "policy": "DYNAMIC_KV",
            "slot_sayisi": self.max_entries,
            "tablo_sayisi": None,
            "dolu_slot": occupied,
            "toplam_slot": total,
            "load_factor": round(occupied / total, 8) if total else 0.0,
            "cakisma": 0,
            "tablo_basi_cakisma": [],
            "cakisma_ornekleri": 0,
            "cakisma_orani": 0.0,
            "store_version": self.store_version,
            "journal_events": len(self._journal),
            "eviction_policy": self.eviction_policy,
            "max_idle_ticks": self.max_idle_ticks,
            "estimated_storage_bytes": self.depolama_bayt(),
            "okuma_yazma": self.okuma_yazma_raporu(),
            "statistics": dict(self._stats),
        }

    def __len__(self) -> int:
        return len(self._records)


__all__ = [
    "DYNAMIC_KV_FORMAT",
    "DYNAMIC_KV_SCHEMA_VERSION",
    "DynamicKVRecord",
    "DynamicKVSnapshot",
    "DynamicKVMemory",
    "EvictionEvent",
    "CompactionReport",
    "MigrationReport",
]
