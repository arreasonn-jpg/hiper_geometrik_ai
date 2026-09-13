"""Elle sabitlenmiş golden benchmark ve epistemik sınır testleri."""
import pytest

from hga.evaluation import GoldenDataset, audit_partitions, run_golden_benchmark
from hga.experience import DogrulamaHatti, ExperienceEvaluator
from hga.knowledge import DeneyimDurumu, ExperienceCandidate, KaynakTuru


def test_golden_dataset_train_test_sizintisiz():
    dataset = GoldenDataset()
    report = dataset.leakage_audit()
    assert report.clean
    assert report.train_count == 3
    assert report.test_count == 5
    assert len(dataset.dataset_hash()) == 64


def test_golden_benchmark_dort_epistemik_durumu_ayirir():
    report = run_golden_benchmark()
    assert report.leakage.clean
    assert report.metrics.total == 5
    assert report.metrics.correct == 5
    assert report.metrics.accuracy == 1.0
    assert report.metrics.far == 0.0
    assert report.metrics.frr == 0.0
    assert report.metrics.uncertain_total == 2
    assert report.metrics.conflict_total == 1


def test_far_sifir_tek_basina_basari_sayilmaz_frr_bunu_yakalar():
    class RejectAll:
        def degerlendir(self, candidate, _store):
            candidate.state = DeneyimDurumu.INVALID
            return candidate

    metrics = run_golden_benchmark(evaluator=RejectAll()).metrics
    assert metrics.far == 0.0
    assert metrics.frr == 1.0
    assert metrics.recall == 0.0
    assert metrics.accuracy < 1.0


def test_accept_all_false_acceptance_yaratir():
    class AcceptAll:
        def degerlendir(self, candidate, _store):
            candidate.state = DeneyimDurumu.VALID
            return candidate

    metrics = run_golden_benchmark(evaluator=AcceptAll()).metrics
    assert metrics.far == 1.0
    assert metrics.frr == 0.0
    assert metrics.precision < 1.0


def test_leakage_audit_memory_sizintisini_yakalar():
    dataset = GoldenDataset()
    leaked = [dict(dataset.test_records[0])]
    report = dataset.leakage_audit(memory=leaked)
    assert not report.clean
    assert report.findings[0].contaminated_partitions == ["memory"]
    with pytest.raises(ValueError, match="data leakage"):
        report.assert_clean()


def test_semantik_hash_experience_id_degisse_de_sizintiyi_bulur():
    train = [{"experience_id": "A", "subject_id": "E1", "relation_id": "R", "object_id": "E2"}]
    test = [{"experience_id": "B", "subject_id": "e1", "relation_id": "r", "object_id": "e2"}]
    assert not audit_partitions(train, test).clean


def test_evaluator_harici_kaynagi_bile_verified_yapamaz():
    dataset = GoldenDataset()
    store = dataset.build_store()
    candidate = ExperienceCandidate(
        "X", "E_ALI", "R_BINMEK", "E_ARABA",
        source=KaynakTuru.REAL_DATA, source_confidence=1.0,
    )
    ExperienceEvaluator().degerlendir(candidate, store)
    assert candidate.state == DeneyimDurumu.VALID
    assert candidate.verified_by is None


def test_verifier_none_uncertain_ve_candidate_reddi():
    dataset = GoldenDataset()
    store = dataset.build_store()
    candidate = ExperienceCandidate("X", "E_ALI", "R_BINMEK", "E_ARABA")
    verifier = DogrulamaHatti(lambda _store, _candidate: None)
    with pytest.raises(ValueError, match="önce Evaluator"):
        verifier.isle(store, [candidate])

    ExperienceEvaluator().degerlendir(candidate, store)
    report = verifier.isle(store, [candidate])
    assert report.belirsiz == 1
    assert candidate.state == DeneyimDurumu.UNCERTAIN
    assert candidate.state != DeneyimDurumu.VERIFIED
