"""Kapalı self-learning, Experience Yield ve collapse failure-injection testleri."""
import json
import subprocess
import sys
from pathlib import Path

from hga.experience import (
    run_self_learning_experiment,
    run_self_training_collapse_test,
)


def test_closed_loop_yalniz_verified_bilgiyle_buyur():
    report = run_self_learning_experiment(
        cycles=10, batch_size=8, initial_facts=5,
        operands_max=5, negatives_per_fact=3, seed=1, memory_slots=128,
    )
    assert report.protocol == "closed-verified-self-learning-v1"
    assert report.final_knowledge_size == (
        report.initial_knowledge_size + report.verified_new_knowledge
    )
    assert report.final_knowledge_size > report.initial_knowledge_size
    assert report.correct_knowledge == report.final_knowledge_size
    assert report.incorrect_knowledge == 0
    assert report.false_acceptance_before_verifier == report.invalid_generated
    assert report.false_acceptance == 0
    assert report.false_rejection == 0
    assert report.far_before_verifier == 1.0
    assert report.far == 0.0
    assert report.frr == 0.0
    assert report.experience_yield == round(
        report.verified_new_knowledge / report.generated_experiences, 8
    )


def test_closed_loop_100_cycle_k0_k100_sozlesmesi():
    report = run_self_learning_experiment(
        cycles=100, batch_size=8, initial_facts=100,
        operands_max=31, negatives_per_fact=7, seed=42,
    )
    assert report.cycles_requested == 100
    assert report.cycles_completed == 100
    assert len(report.cycles) == 100
    assert report.generated_experiences == 800
    assert report.verified_new_knowledge > 0
    assert report.final_knowledge_size > 100
    assert report.incorrect_knowledge == 0
    assert all(cycle.false_acceptance == 0 for cycle in report.cycles)
    assert all(cycle.cumulative_diversity == 1.0 for cycle in report.cycles)


def test_closed_loop_seed_deterministik():
    kwargs = dict(
        cycles=5, batch_size=8, initial_facts=4,
        operands_max=4, negatives_per_fact=3, seed=9, memory_slots=64,
    )
    first = run_self_learning_experiment(**kwargs).to_dict()
    second = run_self_learning_experiment(**kwargs).to_dict()
    assert first == second


def test_pool_tukenince_yanlis_bilgi_uretilmez():
    report = run_self_learning_experiment(
        cycles=20, batch_size=20, initial_facts=3,
        operands_max=2, negatives_per_fact=1, seed=2,
    )
    assert report.generated_experiences < 20 * 20
    assert report.cycles[-1].generated == 0
    assert report.incorrect_knowledge == 0
    assert report.experience_yield > 0.0


def test_unverified_self_training_collapse_sinyallerini_yakalar():
    report = run_self_training_collapse_test(
        cycles=10, batch_size=8, initial_facts=2,
        operands_max=4, negatives_per_fact=3, seed=1, memory_slots=32,
    )
    assert report.protocol == "unverified-self-training-collapse-v1"
    assert report.collapse_detected
    assert report.initial_novelty_rate == 1.0
    assert report.final_novelty_rate == 0.0
    assert report.final_evaluator_novelty < report.initial_evaluator_novelty
    assert report.repetition_rate == 0.9
    assert report.incorrect_model_facts > 0
    assert report.far == 1.0
    assert report.memory_collisions > 0
    assert all(report.collapse_signals.values())


def test_collapse_repetition_ve_memory_recall_zamanla_kotulesir():
    report = run_self_training_collapse_test(
        cycles=8, batch_size=8, initial_facts=2,
        operands_max=4, negatives_per_fact=3, seed=5, memory_slots=16,
    )
    first = report.cycle_metrics[0]
    last = report.cycle_metrics[-1]
    assert last.repetition_count > first.repetition_count
    assert last.model_fact_records > first.model_fact_records
    assert last.cumulative_diversity < first.cumulative_diversity
    assert last.memory_collisions > first.memory_collisions
    assert last.memory_retrieval_accuracy < first.memory_retrieval_accuracy


def test_collapse_100_cycle_tamamlar():
    report = run_self_training_collapse_test(cycles=100, batch_size=8, seed=42)
    assert len(report.cycle_metrics) == 100
    assert report.generated_experiences == 800
    assert report.unique_experiences == 8
    assert report.repetition_rate == 0.99
    assert report.collapse_detected


def test_kuratörlü_bes_seed_raporu_reproducibility_metadata_tasir():
    path = Path(__file__).resolve().parents[1] / "raporlar" / "self_learning_5seed_summary.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["seeds"] == [1, 2, 3, 4, 5]
    assert len(report["dataset_hash"]) == 64
    assert len(report["config_hash"]) == 64
    assert len(report["manifests"]) == 5
    assert len(report["results"]) == 5
    assert all(len(result["closed_verified"]["cycles"]) == 100 for result in report["results"])
    assert all(manifest["result"] == "COMPLETED" for manifest in report["manifests"])
    assert all(manifest["git_dirty"] is False for manifest in report["manifests"])
    assert report["aggregate"]["closed_verified.incorrect_knowledge"]["mean"] == 0.0


def test_self_learning_cli_manifestli(tmp_path):
    output = tmp_path / "summary.json"
    completed = subprocess.run(
        [
            sys.executable, "-m", "hga", "self-learning-benchmark",
            "--cycles", "5", "--batch", "8", "--initial-facts", "4",
            "--operands-max", "4", "--negatives-per-fact", "3",
            "--memory-slots", "64", "--seeds", "1,2",
            "--experiment-root", str(tmp_path / "runs"), "--out", str(output),
        ],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["experiment_ids"] == ["EXP-0001", "EXP-0002"]
    assert summary["aggregate"]["closed_verified.far"]["mean"] == 0.0
    assert summary["aggregate"]["collapse_probe.far"]["mean"] == 1.0
    assert "Self-learning + collapse benchmark özeti" in completed.stdout
