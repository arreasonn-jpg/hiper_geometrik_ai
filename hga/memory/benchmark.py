"""Saf Python sparse memory collision/interference stres benchmarkı."""
from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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
        policy="first-writer-wins; collision does not overwrite; all tables required on read",
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
            "Mevcut çift tablo Bloom-benzeri ALL okuma semantiğidir; daha iyi sonuç varsayılmaz.",
            "estimated_storage_bytes yaklaşık CPython nesne muhasebesidir; RSS değildir.",
        ],
    )


__all__ = [
    "MemoryBenchmarkResult",
    "MemoryCapacitySweepReport",
    "MemoryStressReport",
    "run_memory_benchmark",
    "run_memory_capacity_sweep",
    "run_memory_stress",
]
