# -*- coding: utf-8 -*-
"""Aktif Engine Dynamic KV: eviction/persistence/version/compaction/migration."""
from __future__ import annotations

import json

import pytest

from hga.engine import ExperienceEngine
from hga.knowledge.schemas import ExperienceCandidate
from hga.memory import (
    BellekEntegrasyonu,
    DeneyimSlotlari,
    DynamicKVMemory,
    carpisan_anahtar_bul,
    run_dynamic_kv_lifecycle,
)


def _candidate(index: int, key=None) -> ExperienceCandidate:
    key = key or (f"E_S{index}", "R_TEST", f"E_O{index}")
    return ExperienceCandidate(
        experience_id=f"EXP-{index}",
        subject_id=key[0],
        relation_id=key[1],
        object_id=key[2],
    )


def test_dynamic_kv_hash_cakismasinda_tam_anahtarlari_birlikte_korur():
    adresleyici = DeneyimSlotlari(slot_sayisi=16)
    victim_key = ("VICTIM", "R", "OBJECT")
    attacker_key = carpisan_anahtar_bul(adresleyici, victim_key, arama_limiti=1000)
    assert attacker_key is not None
    assert adresleyici.adresler(victim_key) == adresleyici.adresler(attacker_key)

    memory = DynamicKVMemory(max_entries=4)
    memory.yaz("VICTIM-ID", victim_key)
    memory.yaz("ATTACKER-ID", attacker_key)

    assert memory.icerir("VICTIM-ID", victim_key)
    assert memory.icerir("ATTACKER-ID", attacker_key)
    assert len(memory) == 2
    assert memory.cakisma_sayisi == 0


def test_dynamic_kv_lru_fifo_ttl_eviction_deterministik():
    lru = DynamicKVMemory(max_entries=2, eviction_policy="lru")
    a, b, c = _candidate(1), _candidate(2), _candidate(3)
    lru.yaz(a.experience_id, a.uclusu)
    lru.yaz(b.experience_id, b.uclusu)
    assert lru.icerir(a.experience_id, a.uclusu)  # A en yeni erişilen
    lru.yaz(c.experience_id, c.uclusu)
    assert [(event.key, event.reason) for event in lru.last_evictions] == [
        (b.uclusu, "lru")
    ]
    assert lru.getir(b.uclusu) is None
    assert lru.icerir(a.experience_id, a.uclusu)
    assert lru.icerir(c.experience_id, c.uclusu)
    assert lru.kapasite()["statistics"]["evictions"] == 1

    fifo = DynamicKVMemory(max_entries=2, eviction_policy="fifo")
    fifo.yaz(a.experience_id, a.uclusu)
    fifo.yaz(b.experience_id, b.uclusu)
    assert fifo.icerir(a.experience_id, a.uclusu)
    fifo.yaz(c.experience_id, c.uclusu)
    assert fifo.getir(a.uclusu) is None
    assert fifo.icerir(b.experience_id, b.uclusu)

    ttl = DynamicKVMemory(max_entries=4, max_idle_ticks=1)
    ttl.yaz(a.experience_id, a.uclusu)
    ttl.yaz(b.experience_id, b.uclusu)
    ttl.yaz(c.experience_id, c.uclusu)  # A artık threshold'dan eski
    assert ttl.getir(a.uclusu) is None
    assert ttl.kapasite()["statistics"]["ttl_evictions"] >= 1


def test_dynamic_kv_eviction_stale_replay_kopyasini_ayni_islemde_siler():
    integration = BellekEntegrasyonu(
        slot_sayisi=2, replay_kapasitesi=8, politika="DYNAMIC_KV"
    )
    candidates = [_candidate(index) for index in range(3)]
    integration.coklu_yaz(candidates)

    replay_keys = {candidate.uclusu for candidate in integration.replay.icerik()}
    active_keys = {record.key for record in integration.dynamic_kv.kayitlar()}
    assert replay_keys == active_keys == {candidates[1].uclusu, candidates[2].uclusu}
    assert not integration.icerir(candidates[0])


def test_dynamic_kv_persistence_atomik_hashli_roundtrip_ve_tamper(tmp_path):
    path = tmp_path / "memory.json"
    memory = DynamicKVMemory(max_entries=3, eviction_policy="lru")
    candidates = [_candidate(1), _candidate(2)]
    for candidate in candidates:
        memory.yaz(candidate.experience_id, candidate.uclusu, candidate.to_dict())
    snapshot = memory.kaydet(path, label="checkpoint-1")

    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))
    loaded = DynamicKVMemory.yukle(path)
    assert loaded.store_version == memory.store_version
    assert loaded.max_entries == 3
    assert loaded.eviction_policy == "lru"
    assert all(loaded.icerir(c.experience_id, c.uclusu) for c in candidates)
    assert loaded.kayitlar()[0].payload is not None
    assert len(snapshot.content_hash) == 64

    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["snapshot"]["records"][0]["experience_id"] = "TAMPERED"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="bütünlük hash"):
        DynamicKVMemory.yukle(path)


def test_snapshot_chain_immutable_ve_restore_version_monoton():
    memory = DynamicKVMemory(max_entries=4)
    a, b = _candidate(1), _candidate(2)
    memory.yaz(a.experience_id, a.uclusu)
    first = memory.snapshot_olustur("first")
    memory.yaz(b.experience_id, b.uclusu)
    version_before_restore = memory.store_version
    second = memory.snapshot_olustur("second")

    assert second.parent_snapshot_hash == first.content_hash
    assert len(first.records) == 1  # sonradan store değişimi snapshot'ı değiştirmez
    restored_version = memory.snapshot_geri_yukle(first)
    assert restored_version > version_before_restore
    assert memory.icerir(a.experience_id, a.uclusu)
    assert memory.getir(b.uclusu) is None


def test_journal_compaction_logical_kayitlari_degistirmez():
    memory = DynamicKVMemory(max_entries=3, max_journal_entries=100)
    candidate = _candidate(1)
    for version in range(8):
        memory.yaz(f"EXP-{version}", candidate.uclusu, {"version": version})
    before_records = [record.to_dict() for record in memory.kayitlar()]
    report = memory.sikistir()

    assert report.journal_before == 8
    assert report.journal_after == 1
    assert report.reclaimed_events == 7
    assert [record.to_dict() for record in memory.kayitlar()] == before_records
    assert memory.kapasite()["statistics"]["compactions"] == 1


def test_legacy_first_wins_migration_kaybi_uydurmadan_raporlar():
    legacy = BellekEntegrasyonu(
        slot_sayisi=8,
        replay_kapasitesi=8,
        politika="FIRST_WINS",
    )
    victim_key = ("VICTIM", "R", "OBJECT")
    attacker_key = carpisan_anahtar_bul(
        legacy.slotlar, victim_key, arama_limiti=1000
    )
    assert attacker_key is not None
    victim = _candidate(1, victim_key)
    attacker = _candidate(2, attacker_key)
    legacy.yaz(victim)
    legacy.yaz(attacker)
    assert legacy.slotlar.cakisma_sayisi == 1

    report = legacy.dynamic_kvye_migre_et(max_entries=8)

    assert report.scanned_candidates == 2
    assert report.migrated_records == 1
    assert report.missing_from_source == 1
    assert report.source_collision_events == 1
    assert report.fidelity == 0.5
    assert legacy.politika == "DYNAMIC_KV"
    assert legacy.icerir(victim)
    assert not legacy.icerir(attacker)
    assert legacy.replay.rapor()["dolu"] == 1


def test_v1_persistence_schema_v2ye_migre_edilir(tmp_path):
    candidate = _candidate(7)
    path = tmp_path / "v1.json"
    path.write_text(json.dumps({
        "format": "hga-dynamic-kv",
        "schema_version": 1,
        "snapshot": {
            "max_entries": 4,
            "records": [{
                "key": list(candidate.uclusu),
                "experience_id": candidate.experience_id,
                "payload": candidate.to_dict(),
            }],
        },
    }), encoding="utf-8")

    memory = DynamicKVMemory.yukle(path)
    assert memory.icerir(candidate.experience_id, candidate.uclusu)
    assert memory.kapasite()["store_version"] == 1


def test_dynamic_kv_lifecycle_manifestli_ve_seed_deterministik():
    first = run_dynamic_kv_lifecycle(seed=3, capacity=8)
    second = run_dynamic_kv_lifecycle(seed=3, capacity=8)

    assert all(first.checks.values())
    assert first.to_dict() == second.to_dict()
    assert first.exact_retrieval_accuracy == 1.0
    assert first.replay_active_key_consistency == 1.0
    assert first.evictions == 4
    assert first.compaction["reclaimed_events"] > 0
    assert first.migration["fidelity"] == 0.5


def _engine() -> ExperienceEngine:
    engine = ExperienceEngine(slot_sayisi=2)
    engine.store.varlik_ekle(
        "Ali", entity_type="insan", properties={"canli": 1}, entity_id="E_001"
    )
    engine.store.varlik_ekle(
        "Ata", entity_type="hayvan", properties={"binilebilir": 1}, entity_id="E_002"
    )
    engine.store.varlik_ekle(
        "Araba", entity_type="tasit", properties={"binilebilir": 1}, entity_id="E_003"
    )
    engine.store.varlik_ekle(
        "Gökyüzü", entity_type="mekan", properties={"binilebilir": 0}, entity_id="E_004"
    )
    engine.store.iliski_tanimla(
        "Binmek", relation_id="R_001", subject_types=["insan"],
        requires_object_props={"binilebilir": 1.0}
    )
    return engine


def test_engine_default_dynamic_kv_konsolidasyon_replay_ve_persistence(tmp_path):
    engine = _engine()
    candidates = engine.uret(["R_001"])
    engine.degerlendir(candidates)
    engine.konsolide(candidates)
    valid = [candidate for candidate in candidates if candidate.state.value == "VALID"]
    invalid = [candidate for candidate in candidates if candidate.state.value == "INVALID"]

    assert engine.ozet()["bellek"]["politika"] == "DYNAMIC_KV"
    assert engine.ozet()["bellek"]["slot_dolu"] == 2
    assert all(engine.bellekte_mi(candidate) for candidate in valid)
    assert all(not engine.bellekte_mi(candidate) for candidate in invalid)
    assert engine.bellek.ornek_oynat(2)

    path = tmp_path / "engine-memory.json"
    engine.bellek_kaydet(path, "engine-checkpoint")
    restored = ExperienceEngine(slot_sayisi=2)
    restored.bellek_yukle(path)
    assert restored.ozet()["bellek"]["politika"] == "DYNAMIC_KV"
    assert restored.ozet()["bellek"]["replay_dolu"] == 2
    assert all(restored.bellekte_mi(candidate) for candidate in valid)
