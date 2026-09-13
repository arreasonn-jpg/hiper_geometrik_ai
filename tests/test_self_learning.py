"""Kapalı self-learning, Experience Yield ve collapse failure-injection testleri."""
import json
import subprocess
import sys
from pathlib import Path

from hga.experience import (
    run_self_learning_experiment,
    run_self_training_collapse_test,
    run_verifier_fault_injection,
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
    assert report.test_holdout_size > 0
    assert report.generation_test_overlap == report.memory_test_overlap == 0
    assert report.isolation_clean
    assert report.experience_yield == round(
        report.verified_new_knowledge / report.generated_experiences, 8
    )


def test_verim_ayristirmasi_raporlanir_ve_uey_bellek_kaybina_duyarli():
    """P1-005: EY yanında NY ve UEY de raporlanmalı; UEY kaybı görmeli.

    Dar bellekte doğrulanan bilginin bir kısmı geri çağrılamaz hale gelir.
    EY bunu göremez (doğrulama sayısı değişmez), UEY düşer.
    """
    from hga.experience import run_self_learning_experiment as kos

    ortak = dict(cycles=8, batch_size=8, initial_facts=6, operands_max=6,
                 negatives_per_fact=3, seed=1)
    dar = kos(memory_slots=16, **ortak)
    genis = kos(memory_slots=4096, **ortak)

    for rapor in (dar, genis):
        assert 0.0 <= rapor.novelty_yield <= 1.0
        assert 0.0 <= rapor.useful_experience_yield <= 1.0
        # Kullanışlı bilgi doğrulanmış bilginin alt kümesidir.
        assert rapor.useful_experience_yield <= rapor.experience_yield

    # EY bellek darlığından etkilenmez; UEY etkilenir.
    assert dar.experience_yield == genis.experience_yield
    assert dar.useful_experience_yield < genis.useful_experience_yield
    assert dar.memory_collisions > genis.memory_collisions


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


def test_verifier_fault_injection_frr_uncertain_conflict_metriklerini_kirar():
    report = run_verifier_fault_injection(
        sample_per_class=8, unknown_count=2, conflict_count=2,
        false_acceptance_rate=0.25, false_rejection_rate=0.25, seed=7,
    )
    assert report.protocol == "verifier-epistemic-fault-injection-v1"
    assert (report.true_acceptance, report.false_rejection) == (6, 2)
    assert (report.true_rejection, report.false_acceptance) == (6, 2)
    assert report.precision == report.recall == report.f1 == 0.75
    assert report.far == report.frr == 0.25
    assert report.uncertain == report.truth_unknown == 2
    assert report.conflict == report.actual_conflicts == 2
    assert report.durable_new_knowledge == 8
    assert report.incorrect_knowledge == 2
    # UNCERTAIN ve CONFLICT epistemik durumları durable fact'e dönüşmez.
    assert report.final_knowledge_size == report.initial_knowledge_size + 8


def test_verifier_fault_injection_hatasiz_kontrol():
    report = run_verifier_fault_injection(
        false_acceptance_rate=0.0, false_rejection_rate=0.0, seed=3,
    )
    assert report.far == report.frr == 0.0
    assert report.precision == report.recall == report.f1 == 1.0
    assert report.incorrect_knowledge == 0
    assert report.uncertain == 2
    assert report.conflict == 2


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
    assert report.test_holdout_size > 0
    assert report.generation_test_overlap == report.memory_test_overlap == 0
    assert report.isolation_clean
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
    assert all(result["closed_verified"]["isolation_clean"] for result in report["results"])
    assert all(result["collapse_probe"]["isolation_clean"] for result in report["results"])
    assert all(result["verifier_robustness"]["frr"] == 0.25 for result in report["results"])
    assert all("table_1" in result["memory_capacity"]["minimum_slots_meeting_target"]
               for result in report["results"])
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
    assert summary["aggregate"][
        "multi_environment.shared_memory_retrieval_accuracy"
    ]["mean"] == 1.0
    assert summary["aggregate"][
        "multi_environment.cross_verifier_acceptances"
    ]["mean"] == 0.0
    assert all(summary["results"][0]["multi_environment"]["checks"].values())
    assert summary["aggregate"]["verifier_robustness.frr"]["mean"] == 0.25
    assert summary["results"][0]["verifier_robustness"]["uncertain"] == 2
    assert "table_1" in summary["results"][0]["memory_capacity"][
        "minimum_slots_meeting_target"
    ]
    assert "Self-learning + collapse benchmark özeti" in completed.stdout
