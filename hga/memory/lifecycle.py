# -*- coding: utf-8 -*-
"""Aktif Dynamic KV yaşam döngüsünün manifestlenebilir kabul benchmarkı."""
from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

from ..knowledge.schemas import ExperienceCandidate
from .entegrasyon import BellekEntegrasyonu
from .interference import carpisan_anahtar_bul


@dataclass
class DynamicKVLifecycleReport:
    protocol: str
    seed: int
    capacity: int
    writes: int
    active_records: int
    evictions: int
    exact_retrieval_accuracy: float
    replay_active_key_consistency: float
    store_version_before_restore: int
    store_version_after_restore: int
    snapshot_hash: str
    snapshot_parent_hash: str
    persistence_roundtrip_records: int
    tamper_rejected: bool
    compaction: Dict[str, int]
    migration: Dict[str, Any]
    checks: Dict[str, bool]
    limitations: list[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _candidate(seed: int, index: int, key=None) -> ExperienceCandidate:
    triple = key or (
        f"ENV-{seed}:SUBJECT-{index}",
        f"REL-{index % 7}",
        f"ENV-{seed}:OBJECT-{index}",
    )
    return ExperienceCandidate(
        experience_id=f"LIFECYCLE-{seed}-{index}",
        subject_id=triple[0],
        relation_id=triple[1],
        object_id=triple[2],
    )


def run_dynamic_kv_lifecycle(seed: int = 1, capacity: int = 16) -> DynamicKVLifecycleReport:
    """Eviction, replay, snapshot, persistence, compaction ve migration'ı kır."""
    capacity = int(capacity)
    if capacity < 4:
        raise ValueError("capacity >= 4 olmalı")
    integration = BellekEntegrasyonu(
        slot_sayisi=capacity,
        replay_kapasitesi=capacity,
        tohum=int(seed),
        politika="DYNAMIC_KV",
        eviction_policy="lru",
    )
    candidates = [_candidate(int(seed), index) for index in range(capacity + 4)]
    integration.coklu_yaz(candidates)
    retained = candidates[-capacity:]
    evicted = candidates[:-capacity]
    retained_hits = sum(integration.icerir(candidate) for candidate in retained)
    evicted_misses = sum(not integration.icerir(candidate) for candidate in evicted)
    exact_accuracy = round(
        (retained_hits + evicted_misses) / len(candidates), 8
    )
    active_keys = {record.key for record in integration.dynamic_kv.kayitlar()}
    replay = integration.replay.icerik()
    replay_consistent = sum(candidate.uclusu in active_keys for candidate in replay)
    replay_consistency = round(replay_consistent / len(replay), 8) if replay else 0.0

    first_snapshot = integration.snapshot_olustur("lifecycle-before-mutation")
    extra = _candidate(int(seed), capacity + 100)
    integration.yaz(extra)
    version_before_restore = integration.dynamic_kv.store_version
    restored_version = integration.snapshot_geri_yukle(first_snapshot)
    restored_keys = {record.key for record in integration.dynamic_kv.kayitlar()}
    restore_exact = restored_keys == active_keys

    updated = retained[-1]
    for update in range(5):
        replacement = ExperienceCandidate(
            experience_id=f"{updated.experience_id}-UPDATE-{update}",
            subject_id=updated.subject_id,
            relation_id=updated.relation_id,
            object_id=updated.object_id,
        )
        integration.yaz(replacement)
    compaction = integration.sikistir()

    with tempfile.TemporaryDirectory(prefix="hga-dynamic-kv-") as directory:
        path = Path(directory) / "memory.json"
        saved = integration.kaydet(path, label="lifecycle-persistence")
        loaded = BellekEntegrasyonu(
            slot_sayisi=capacity, replay_kapasitesi=capacity
        )
        loaded.yukle(path)
        current_pairs = {
            (record.key, record.experience_id)
            for record in integration.dynamic_kv.kayitlar()
        }
        loaded_pairs = {
            (record.key, record.experience_id)
            for record in loaded.dynamic_kv.kayitlar()
        }
        persistence_exact = current_pairs == loaded_pairs
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["snapshot"]["records"][0]["experience_id"] = "TAMPERED"
        tampered_path = Path(directory) / "tampered.json"
        tampered_path.write_text(json.dumps(envelope), encoding="utf-8")
        tamper_rejected = False
        try:
            loaded.yukle(tampered_path)
        except ValueError:
            tamper_rejected = True

    legacy = BellekEntegrasyonu(
        slot_sayisi=8, replay_kapasitesi=8, politika="FIRST_WINS"
    )
    victim_key = (f"MIGRATE-{seed}", "R", "VICTIM")
    attacker_key = carpisan_anahtar_bul(
        legacy.slotlar, victim_key, arama_limiti=10_000
    )
    if attacker_key is None:
        raise RuntimeError("Lifecycle migration için çakışan anahtar bulunamadı")
    victim = _candidate(int(seed), 10_000, victim_key)
    attacker = _candidate(int(seed), 10_001, attacker_key)
    legacy.yaz(victim)
    legacy.yaz(attacker)
    migration = legacy.dynamic_kvye_migre_et(max_entries=8)

    stats = integration.dynamic_kv.kapasite()["statistics"]
    checks = {
        "active_policy_is_dynamic_kv": integration.politika == "DYNAMIC_KV",
        "bounded_lru_eviction_exact": (
            len(active_keys) == capacity
            and int(stats["evictions"]) >= 4
            and exact_accuracy == 1.0
        ),
        "replay_contains_only_active_keys": replay_consistency == 1.0,
        "snapshot_content_hash_present": len(first_snapshot.content_hash) == 64,
        "snapshot_restore_exact": restore_exact,
        "store_version_monotonic_after_restore": restored_version > version_before_restore,
        "compaction_reclaims_journal_events": compaction.reclaimed_events > 0,
        "compaction_preserves_active_cardinality": (
            compaction.active_records == capacity
        ),
        "persistence_roundtrip_exact": persistence_exact,
        "persistence_hash_rejects_tamper": tamper_rejected,
        "snapshot_chain_linked": saved.parent_snapshot_hash == first_snapshot.content_hash,
        "legacy_loss_not_invented": (
            migration.scanned_candidates == 2
            and migration.migrated_records == 1
            and migration.missing_from_source == 1
            and migration.source_collision_events == 1
        ),
        "migration_target_collision_free": (
            legacy.dynamic_kv.cakisma_sayisi == 0 and len(legacy.dynamic_kv) == 1
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Dynamic KV lifecycle kabul kapısı başarısız: {failed}")
    return DynamicKVLifecycleReport(
        protocol="active-dynamic-kv-lifecycle-v1",
        seed=int(seed),
        capacity=capacity,
        writes=len(candidates),
        active_records=len(integration.dynamic_kv),
        evictions=int(stats["evictions"]),
        exact_retrieval_accuracy=exact_accuracy,
        replay_active_key_consistency=replay_consistency,
        store_version_before_restore=version_before_restore,
        store_version_after_restore=restored_version,
        snapshot_hash=first_snapshot.content_hash,
        snapshot_parent_hash=saved.parent_snapshot_hash or "",
        persistence_roundtrip_records=len(current_pairs),
        tamper_rejected=tamper_rejected,
        compaction=compaction.to_dict(),
        migration=migration.to_dict(),
        checks=checks,
        limitations=[
            "Dynamic KV tam anahtarlı exact lookup'tır; learned/semantic retrieval değildir.",
            "Bounded modda çakışma yerine açık LRU/FIFO eviction vardır; eski bilgi kapasite dolunca kaybolabilir.",
            "Persistence tek süreç atomik dosya değiştirmesini doğrular; dağıtık consensus veya çok-yazarlı transaction değildir.",
            "Legacy migration yalnız replay/payload'da bulunan ve FIRST_WINS'te tutulmuş kayıtları taşır; geçmiş collision kaybını uydurmaz.",
        ],
    )


__all__ = ["DynamicKVLifecycleReport", "run_dynamic_kv_lifecycle"]
