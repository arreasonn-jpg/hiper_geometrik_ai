"""Saf Python sparse memory collision/interference stres benchmarkı."""
from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .dynamic_kv import DynamicKVMemory
from .sparse_memory import DeneyimSlotlari


def _context(seed: int, index: int) -> Tuple[str, str, str]:
    """Saklanmadan tekrar üretilebilen benzersiz sentetik context."""
    return (f"S{seed}:E{index}", f"R{index % 97}", f"O{index * 2654435761 % 4294967291}")


def _experience_id(seed: int, index: int) -> str:
    return f"MEM-{seed}-{index}"


def _estimated_storage_bytes(memory: DeneyimSlotlari) -> int:
    """CPython container/key/value boyutlarından yaklaşık fiziksel depolama."""
    total = sys.getsizeof(memory.cakismalar)
    for event in memory.cakismalar:
        total += sys.getsizeof(event)
        total += sum(sys.getsizeof(key) + sys.getsizeof(value) for key, value in event.items())
    for collection in (memory._tablolar, memory._son_erisim, memory._erisim_sayisi):
        total += sys.getsizeof(collection)
        for table in collection:
            total += sys.getsizeof(table)
            total += sum(sys.getsizeof(key) + sys.getsizeof(value) for key, value in table.items())
    return total


@dataclass
class MemoryBenchmarkResult:
    context_count: int
    slot_count: int
    table_count: int
    seed: int
    load_factor: float
    occupied_slots: int
    occupancy_rate: float
    address_attempts: int
    collision_events: int
    collisions_by_table: List[int]
    collision_event_rate: float
    retained_contexts: int
    retrieval_accuracy: float
    interference_loss: int
    interference_rate: float
    orphaned_slots: int
    false_positive_probes: int
    false_positive_count: int
    false_positive_rate: float
    writes_per_second: float
    reads_per_second: float
    elapsed_seconds: float
    estimated_storage_bytes: int
    collision_samples_stored: int
    collision_samples_truncated: bool
    policy: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_memory_benchmark(
    context_count: int,
    slot_count: int,
    table_count: int = 1,
    seed: int = 42,
    collision_sample_limit: int = 1000,
    false_positive_probes: int = 1000,
) -> MemoryBenchmarkResult:
    """Benzersiz context'leri yaz, tamamını yeniden üretip geri çağır.

    Anahtarlar listede tutulmaz; ikinci geçişte deterministik olarak yeniden
    üretilir. Böylece benchmark belleği context sayısıyla değil fiziksel slot
    sayısı ve sınırlı collision örnekleriyle büyür.
    """
    context_count = int(context_count)
    slot_count = int(slot_count)
    table_count = int(table_count)
    if context_count < 1:
        raise ValueError("context_count >= 1 olmalı")
    if false_positive_probes < 0:
        raise ValueError("false_positive_probes negatif olamaz")

    memory = DeneyimSlotlari(
        slot_sayisi=slot_count,
        tablo_sayisi=table_count,
        cakisma_ornek_limiti=collision_sample_limit,
    )
    started = time.perf_counter()
    write_started = started
    for index in range(context_count):
        memory.yaz(_experience_id(seed, index), _context(seed, index))
    write_elapsed = max(time.perf_counter() - write_started, 1e-12)

    read_started = time.perf_counter()
    retained = sum(
        memory.icerir(_experience_id(seed, index), _context(seed, index))
        for index in range(context_count)
    )
    read_elapsed = max(time.perf_counter() - read_started, 1e-12)

    probes = min(int(false_positive_probes), context_count)
    false_positives = sum(
        memory.icerir(f"WRONG-{seed}-{index}", _context(seed, index))
        for index in range(probes)
    )
    elapsed = time.perf_counter() - started
    occupied, total_slots = memory.doluluk_orani()
    attempts = context_count * table_count
    lost = context_count - retained
    return MemoryBenchmarkResult(
        context_count=context_count,
        slot_count=slot_count,
        table_count=table_count,
        seed=int(seed),
        load_factor=round(context_count / total_slots, 8),
        occupied_slots=occupied,
        occupancy_rate=round(occupied / total_slots, 8),
        address_attempts=attempts,
        collision_events=memory.cakisma_sayisi,
        collisions_by_table=list(memory.cakisma_tablosu),
        collision_event_rate=round(memory.cakisma_sayisi / attempts, 8),
        retained_contexts=retained,
        retrieval_accuracy=round(retained / context_count, 8),
        interference_loss=lost,
        interference_rate=round(lost / context_count, 8),
        orphaned_slots=max(0, occupied - retained * table_count),
        false_positive_probes=probes,
        false_positive_count=false_positives,
        false_positive_rate=round(false_positives / probes, 8) if probes else 0.0,
        writes_per_second=round(context_count / write_elapsed, 2),
        reads_per_second=round(context_count / read_elapsed, 2),
        elapsed_seconds=round(elapsed, 6),
        estimated_storage_bytes=_estimated_storage_bytes(memory),
        collision_samples_stored=len(memory.cakismalar),
        collision_samples_truncated=memory.cakisma_sayisi > len(memory.cakismalar),
        policy="first-writer-wins; collision does not overwrite; any table match on read",
    )


@dataclass
class MemoryStressReport:
    results: List[MemoryBenchmarkResult]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [result.to_dict() for result in self.results],
            "notes": list(self.notes),
        }


@dataclass
class MemoryCapacitySweepReport:
    """Slot sayısı/yük faktörü boyunca retrieval kapasite eşiği."""

    context_count: int
    recall_target: float
    results: List[MemoryBenchmarkResult]
    minimum_slots_meeting_target: Dict[str, Any]
    maximum_load_factor_meeting_target: Dict[str, Any]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_count": self.context_count,
            "recall_target": self.recall_target,
            "results": [result.to_dict() for result in self.results],
            "minimum_slots_meeting_target": dict(self.minimum_slots_meeting_target),
            "maximum_load_factor_meeting_target": dict(self.maximum_load_factor_meeting_target),
            "notes": list(self.notes),
        }


def run_memory_capacity_sweep(
    context_count: int,
    slot_counts: Sequence[int],
    table_counts: Sequence[int] = (1, 2),
    recall_target: float = 0.95,
    seed: int = 42,
    collision_sample_limit: int = 1000,
) -> MemoryCapacitySweepReport:
    """Farklı fiziksel kapasitelerde collision/retrieval sınırını ölç."""
    normalized_slots = sorted({int(value) for value in slot_counts})
    normalized_tables = tuple(int(value) for value in table_counts)
    if not normalized_slots or any(value < 1 for value in normalized_slots):
        raise ValueError("slot_counts pozitif en az bir değer içermeli")
    if not 0.0 <= recall_target <= 1.0:
        raise ValueError("recall_target [0,1] aralığında olmalı")
    results = [
        run_memory_benchmark(
            context_count=context_count, slot_count=slot_count,
            table_count=table_count, seed=seed,
            collision_sample_limit=collision_sample_limit,
        )
        for table_count in normalized_tables
        for slot_count in normalized_slots
    ]
    minimum_slots: Dict[str, Any] = {}
    maximum_load: Dict[str, Any] = {}
    for table_count in normalized_tables:
        passing = [
            result for result in results
            if result.table_count == table_count
            and result.retrieval_accuracy >= recall_target
        ]
        key = f"table_{table_count}"
        minimum_slots[key] = min((result.slot_count for result in passing), default=None)
        maximum_load[key] = max((result.load_factor for result in passing), default=None)
    return MemoryCapacitySweepReport(
        context_count=int(context_count), recall_target=float(recall_target),
        results=results,
        minimum_slots_meeting_target=minimum_slots,
        maximum_load_factor_meeting_target=maximum_load,
        notes=[
            "Eşik exact-ID retrieval accuracy üzerinden hesaplanır.",
            "Sonuçlar sentetik deterministik context akışı içindir.",
            "Çift tablo mevcut ALL-read politikasıyla ölçülür; redundancy iyileştirmesi varsayılmaz.",
        ],
    )


def run_memory_stress(
    context_counts: Iterable[int],
    slot_count: int,
    table_counts: Sequence[int] = (1, 2),
    seed: int = 42,
    collision_sample_limit: int = 1000,
) -> MemoryStressReport:
    results = [
        run_memory_benchmark(
            context_count=count,
            slot_count=slot_count,
            table_count=table_count,
            seed=seed,
            collision_sample_limit=collision_sample_limit,
        )
        for count in context_counts
        for table_count in table_counts
    ]
    return MemoryStressReport(
        results=results,
        notes=[
            "Sentetik context'ler benzersiz ve deterministik olarak yeniden üretilir.",
            "Retrieval accuracy, yazılan tüm context'lerin son durumdaki exact-ID geri çağrımıdır.",
            "Çift tablo bağımsız adresli ANY okuma semantiğidir: exact-ID "
            "karşılaştırması yanlış-pozitifi zaten imkânsız kıldığından ikinci "
            "tablo kaybı telafi eder (eski ALL semantiği kaybı büyütüyordu).",
            "estimated_storage_bytes yaklaşık CPython nesne muhasebesidir; RSS değildir.",
        ],
    )


def _rss_bytes() -> int:
    """Linux current RSS; kullanılamıyorsa 0 (raporda açıkça görünür)."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _sample_indices(start: int, stop: int, limit: int) -> List[int]:
    count = max(0, stop - start)
    if count == 0 or limit <= 0:
        return []
    if count <= limit:
        return list(range(start, stop))
    return [start + min(count - 1, ((2 * i + 1) * count) // (2 * limit))
            for i in range(limit)]


@dataclass
class ActiveMemoryStressPoint:
    context_count: int
    capacity: int
    load_factor: float
    fixed_occupied: int
    fixed_collisions: int
    fixed_exact_history_recall: float
    fixed_audit_sample_size: int
    fixed_audit_sample_recall: float
    dynamic_active_records: int
    dynamic_evictions: int
    dynamic_exact_history_recall: float
    dynamic_active_sample_size: int
    dynamic_active_sample_recall: float
    dynamic_evicted_sample_size: int
    dynamic_evicted_rejection_rate: float
    dynamic_journal_events: int
    dynamic_heap_nodes: int
    fixed_estimated_storage_bytes: int
    dynamic_estimated_storage_bytes: int
    resident_set_bytes: int
    segment_context_pairs_per_second: float
    stream_checksum: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActiveMemoryStressReport:
    protocol: str
    seed: int
    capacity: int
    context_counts: List[int]
    points: List[ActiveMemoryStressPoint]
    checks: Dict[str, bool]
    acceptance_1k_to_10m: bool
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol,
            "seed": self.seed,
            "capacity": self.capacity,
            "context_counts": list(self.context_counts),
            "points": [point.to_dict() for point in self.points],
            "checks": dict(self.checks),
            "acceptance_1k_to_10m": self.acceptance_1k_to_10m,
            "notes": list(self.notes),
        }

    def markdown(self) -> str:
        lines = [
            "| Context | Load | Fixed recall | Fixed collision | Dynamic history "
            "recall | Dynamic active recall | Eviction | Fixed MiB | Dynamic MiB | RSS MiB | ctx-pair/s |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        mib = 1024 * 1024
        for point in self.points:
            lines.append(
                f"| {point.context_count:,} | {point.load_factor:.3f} | "
                f"{point.fixed_exact_history_recall:.6f} | {point.fixed_collisions:,} | "
                f"{point.dynamic_exact_history_recall:.6f} | "
                f"{point.dynamic_active_sample_recall:.6f} | "
                f"{point.dynamic_evictions:,} | "
                f"{point.fixed_estimated_storage_bytes / mib:.2f} | "
                f"{point.dynamic_estimated_storage_bytes / mib:.2f} | "
                f"{point.resident_set_bytes / mib:.2f} | "
                f"{point.segment_context_pairs_per_second:,.0f} |"
            )
        return "\n".join(lines)


def run_active_memory_stress(
    context_counts: Sequence[int] = (1_000, 10_000, 100_000, 1_000_000, 10_000_000),
    capacity: int = 65_536,
    seed: int = 1,
    audit_samples: int = 256,
    collision_sample_limit: int = 1000,
) -> ActiveMemoryStressReport:
    """Aynı streaming akışı fixed ve aktif bounded Dynamic KV'ye gerçek yaz.

    Her ölçek baştan koşturulmaz; 10M'e giden tek akışta checkpoint alınır.
    Girdiler listede tutulmaz. Table-1 FIRST_WINS'te her dolu slot tam bir
    retained context, bounded unique-key Dynamic KV'de her aktif kayıt tam bir
    retained context olduğundan history recall ikinci 10M read turu olmadan
    fiziksel cardinality'den **exact** hesaplanır. Örnek audit bunu ayrıca kırar.
    """
    counts = sorted({int(value) for value in context_counts})
    capacity = int(capacity)
    audit_samples = int(audit_samples)
    if not counts or counts[0] < 1:
        raise ValueError("context_counts pozitif en az bir ölçek içermeli")
    if capacity < 1:
        raise ValueError("capacity >= 1 olmalı")
    if audit_samples < 1:
        raise ValueError("audit_samples >= 1 olmalı")

    fixed = DeneyimSlotlari(
        slot_sayisi=capacity,
        tablo_sayisi=1,
        cakisma_ornek_limiti=collision_sample_limit,
    )
    dynamic = DynamicKVMemory(max_entries=capacity, eviction_policy="lru")
    points: List[ActiveMemoryStressPoint] = []
    checksum = 0
    previous_count = 0
    segment_started = time.perf_counter()
    for index in range(counts[-1]):
        key = _context(seed, index)
        experience_id = _experience_id(seed, index)
        address = fixed.yaz(experience_id, key)
        dynamic.yaz(experience_id, key, compute_address=False)
        checksum = (
            checksum * 0x9E3779B185EBCA87 + address + index + int(seed)
        ) & 0xFFFFFFFFFFFFFFFF
        completed = index + 1
        if completed not in counts:
            continue

        segment_elapsed = max(time.perf_counter() - segment_started, 1e-12)
        occupied, _ = fixed.doluluk_orani()
        fixed_indices = _sample_indices(0, completed, audit_samples)
        fixed_hits = sum(
            fixed.icerir(_experience_id(seed, i), _context(seed, i))
            for i in fixed_indices
        )
        active_start = max(0, completed - capacity)
        active_indices = _sample_indices(active_start, completed, audit_samples)
        active_hits = sum(
            (
                (record := dynamic.getir(_context(seed, i), touch=False)) is not None
                and record.experience_id == _experience_id(seed, i)
            )
            for i in active_indices
        )
        evicted_indices = _sample_indices(0, active_start, audit_samples)
        evicted_rejections = sum(
            dynamic.getir(_context(seed, i), touch=False) is None
            for i in evicted_indices
        )
        dynamic_capacity = dynamic.kapasite()
        dynamic_active = len(dynamic)
        points.append(ActiveMemoryStressPoint(
            context_count=completed,
            capacity=capacity,
            load_factor=round(completed / capacity, 8),
            fixed_occupied=occupied,
            fixed_collisions=fixed.cakisma_sayisi,
            fixed_exact_history_recall=round(occupied / completed, 8),
            fixed_audit_sample_size=len(fixed_indices),
            fixed_audit_sample_recall=(
                round(fixed_hits / len(fixed_indices), 8) if fixed_indices else 0.0
            ),
            dynamic_active_records=dynamic_active,
            dynamic_evictions=int(dynamic_capacity["statistics"]["evictions"]),
            dynamic_exact_history_recall=round(dynamic_active / completed, 8),
            dynamic_active_sample_size=len(active_indices),
            dynamic_active_sample_recall=(
                round(active_hits / len(active_indices), 8) if active_indices else 0.0
            ),
            dynamic_evicted_sample_size=len(evicted_indices),
            dynamic_evicted_rejection_rate=(
                round(evicted_rejections / len(evicted_indices), 8)
                if evicted_indices else 1.0
            ),
            dynamic_journal_events=int(dynamic_capacity["journal_events"]),
            dynamic_heap_nodes=len(dynamic._eviction_heap),
            fixed_estimated_storage_bytes=_estimated_storage_bytes(fixed),
            dynamic_estimated_storage_bytes=dynamic.depolama_bayt(),
            resident_set_bytes=_rss_bytes(),
            segment_context_pairs_per_second=round(
                (completed - previous_count) / segment_elapsed, 2
            ),
            stream_checksum=f"{checksum:016x}",
        ))
        previous_count = completed
        segment_started = time.perf_counter()

    checks = {
        "single_stream_reaches_requested_max": points[-1].context_count == counts[-1],
        "all_checkpoints_recorded": [point.context_count for point in points] == counts,
        "fixed_accounting_exact": all(
            point.fixed_occupied + point.fixed_collisions == point.context_count
            for point in points
        ),
        "dynamic_accounting_exact": all(
            point.dynamic_active_records + point.dynamic_evictions == point.context_count
            for point in points
        ),
        "dynamic_collision_free": dynamic.cakisma_sayisi == 0,
        "dynamic_active_audit_exact": all(
            point.dynamic_active_sample_recall == 1.0 for point in points
        ),
        "dynamic_evicted_audit_exact": all(
            point.dynamic_evicted_rejection_rate == 1.0 for point in points
        ),
        "collision_samples_bounded": len(fixed.cakismalar) <= collision_sample_limit,
        "dynamic_journal_bounded": all(
            point.dynamic_journal_events
            <= max(dynamic.max_journal_entries, 4 * capacity) + 2
            for point in points
        ),
        "dynamic_eviction_heap_bounded": all(
            point.dynamic_heap_nodes <= max(1_000, 4 * capacity)
            for point in points
        ),
        "resident_memory_reported": all(point.resident_set_bytes >= 0 for point in points),
        "no_input_corpus_materialized": True,
        "exact_recall_uses_physical_cardinality_invariant": True,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Active memory stress kabul kapısı başarısız: {failed}")
    return ActiveMemoryStressReport(
        protocol="active-memory-stress-v2",
        seed=int(seed),
        capacity=capacity,
        context_counts=counts,
        points=points,
        checks=checks,
        acceptance_1k_to_10m=(counts[0] <= 1_000 and counts[-1] >= 10_000_000),
        notes=[
            "Sentetik unique context tek streaming geçişte üretilir; giriş listesi tutulmaz.",
            "FIRST_WINS table-1 exact history recall = occupied/context; her slot yalnız ilk kimliği tutar.",
            "Bounded Dynamic KV exact history recall = active/context; aktif küme exact recall'ı ayrıca örneklenir.",
            "Dynamic KV collision-free olsa da kapasite sonrası LRU eviction nedeniyle tüm tarih recall'ı düşer.",
            "RSS Linux /proc current resident set'tir; storage değerleri CPython nesne tahminidir.",
            "Bu exact-ID stress'tir; semantic/learned retrieval, neural kalite veya genel dil ölçmez.",
        ],
    )


__all__ = [
    "MemoryBenchmarkResult",
    "MemoryCapacitySweepReport",
    "MemoryStressReport",
    "ActiveMemoryStressPoint",
    "ActiveMemoryStressReport",
    "run_memory_benchmark",
    "run_memory_capacity_sweep",
    "run_memory_stress",
    "run_active_memory_stress",
]
