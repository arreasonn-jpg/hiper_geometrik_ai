"""Sparse memory collision, interference ve streaming benchmark testleri."""
import json
import subprocess
import sys

from hga.memory import (
    DeneyimSlotlari,
    run_memory_benchmark,
    run_memory_capacity_sweep,
    run_memory_stress,
)


def test_collision_ornekleri_sinirli_sayac_eksiksiz():
    memory = DeneyimSlotlari(slot_sayisi=1, cakisma_ornek_limiti=2)
    for index in range(10):
        memory.yaz(f"X{index}", ("E", "R", str(index)))
    assert memory.cakisma_sayisi == 9
    assert len(memory.cakismalar) == 2
    assert memory.kapasite()["cakisma"] == 9
    assert memory.kapasite()["cakisma_ornekleri"] == 2


def test_collision_first_writeri_bozmaz_yeni_yaziyi_reddeder():
    memory = DeneyimSlotlari(slot_sayisi=1)
    first_key = ("E1", "R", "O1")
    second_key = ("E2", "R", "O2")
    memory.yaz("FIRST", first_key)
    memory.yaz("SECOND", second_key)
    assert memory.icerir("FIRST", first_key)
    assert not memory.icerir("SECOND", second_key)
    assert memory.cakisma_sayisi == 1


def test_idempotent_yazma_collision_degildir():
    memory = DeneyimSlotlari(slot_sayisi=1)
    key = ("E1", "R", "O1")
    memory.yaz("SAME", key)
    memory.yaz("SAME", key)
    assert memory.cakisma_sayisi == 0
    assert memory.icerir("SAME", key)


def test_benchmark_tam_dolulukte_interference_olcer():
    result = run_memory_benchmark(
        context_count=10, slot_count=1, table_count=1,
        collision_sample_limit=2, false_positive_probes=10,
    )
    assert result.collision_events == 9
    assert result.collision_event_rate == 0.9
    assert result.retained_contexts == 1
    assert result.retrieval_accuracy == 0.1
    assert result.interference_loss == 9
    assert result.interference_rate == 0.9
    assert result.false_positive_rate == 0.0
    assert result.collision_samples_stored == 2
    assert result.collision_samples_truncated


def test_benchmark_bos_kapasitede_tam_retrieval():
    result = run_memory_benchmark(context_count=20, slot_count=100_000, seed=7)
    assert result.retrieval_accuracy == 1.0
    assert result.interference_rate == 0.0
    assert result.false_positive_rate == 0.0
    assert result.estimated_storage_bytes > 0


def test_tek_ve_cift_tablo_ayri_raporlanir():
    report = run_memory_stress([50], slot_count=16, table_counts=(1, 2), seed=3)
    assert [(row.context_count, row.table_count) for row in report.results] == [(50, 1), (50, 2)]
    assert all(row.collision_events > 0 for row in report.results)
    assert len(report.results[1].collisions_by_table) == 2
    assert report.results[1].collisions_by_table[0] == report.results[0].collision_events
    # Mevcut ALL-table semantiğinde ikinci tablo ilk tablonun kaybını telafi
    # edemez; kesişim şartı nedeniyle retrieval ancak aynı veya daha düşüktür.
    assert report.results[1].retrieval_accuracy <= report.results[0].retrieval_accuracy
    assert report.results[0].orphaned_slots == 0
    assert report.results[1].orphaned_slots > 0
    assert "ALL okuma" in " ".join(report.notes)


def test_memory_capacity_sweep_recall_esigini_bulur():
    report = run_memory_capacity_sweep(
        context_count=256,
        slot_counts=[64, 128, 256, 512, 1024, 2048, 4096],
        table_counts=(1, 2), recall_target=0.95, seed=1,
    )
    assert report.minimum_slots_meeting_target == {
        "table_1": 2048,
        "table_2": 4096,
    }
    assert report.maximum_load_factor_meeting_target == {
        "table_1": 0.125,
        "table_2": 0.03125,
    }
    by_table = {
        table: [row for row in report.results if row.table_count == table]
        for table in (1, 2)
    }
    assert all(rows[0].retrieval_accuracy < rows[-1].retrieval_accuracy
               for rows in by_table.values())
    assert all(rows[0].collision_event_rate > rows[-1].collision_event_rate
               for rows in by_table.values())


def test_memory_capacity_sweep_ulasilamayan_esigi_none_raporlar():
    report = run_memory_capacity_sweep(
        context_count=100, slot_counts=[1, 2], table_counts=(1,),
        recall_target=1.0, seed=1,
    )
    assert report.minimum_slots_meeting_target["table_1"] is None
    assert report.maximum_load_factor_meeting_target["table_1"] is None


def test_memory_benchmark_cli_manifestli(tmp_path):
    output = tmp_path / "summary.json"
    completed = subprocess.run(
        [
            sys.executable, "-m", "hga", "memory-benchmark",
            "--scales", "20,40", "--slots", "16", "--tables", "1,2",
            "--seeds", "1,2", "--experiment-root", str(tmp_path / "runs"),
            "--out", str(output),
        ],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["experiment_ids"] == ["EXP-0001", "EXP-0002"]
    assert "metrics.n20_t1.retrieval_accuracy" in summary["aggregate"]
    assert "Sparse memory benchmark özeti" in completed.stdout
    for experiment_id in summary["experiment_ids"]:
        manifest = json.loads(
            (tmp_path / "runs" / experiment_id / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["result"] == "COMPLETED"
        assert manifest["parameters"]["synthetic"] is True
