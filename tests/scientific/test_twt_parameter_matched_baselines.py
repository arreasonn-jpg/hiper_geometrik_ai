"""Aynı TWT splitindeki Dense/Transformer/Kronecker/HGA adillik testleri."""
from __future__ import annotations

import pytest

from hga.evaluation.real_turkish import prepare_real_turkish_task
from hga.evaluation.twt_baselines import (
    MODEL_ORDER,
    run_twt_architecture_baselines,
)


def test_twt_model_visible_splitler_ayni_hashli_ve_relation_holdout_temiz():
    task = prepare_real_turkish_task()

    assert task.dataset_hash == "66b13a898efa88998a9329f1551530f5241835a8085a0e5f26e0eb374d7e3276"
    assert len(task.train) == 127602
    assert len(task.dev) == 17030
    assert len(task.test) == 16036
    assert task.heldout_relations == ("csubj", "parataxis")
    assert all(
        candidate.gold_relation not in task.heldout_relations
        for candidate in (*task.train, *task.dev)
    )
    assert any(
        candidate.gold_relation in task.heldout_relations
        for candidate in task.test
    )
    assert task.candidate_hashes["test_challenge"] == (
        "877a788e8468ba7cbb676e73581fb938018b1ddc891034a2c3c2bcc5460dbb24"
    )


def test_dort_mimari_gercek_parametre_ve_body_butcesinde_eslesir():
    report = run_twt_architecture_baselines(seed=1, profile="smoke")

    assert tuple(report.models) == MODEL_ORDER
    assert report.fairness["physical_parameter_counts"] == {
        "dense": 291808,
        "transformer": 291816,
        "kronecker": 291514,
        "hga": 291827,
    }
    assert report.fairness["architecture_body_parameter_counts"] == {
        "dense": 10496,
        "transformer": 10504,
        "kronecker": 10202,
        "hga": 10515,
    }
    assert report.fairness["shared_embedding_parameters_each"] == 281312
    assert report.fairness["parameter_max_to_min_ratio"] == 1.0010737
    assert report.fairness["body_parameter_max_to_min_ratio"] == 1.03068026
    assert report.fairness["unused_parameter_padding"] is False
    assert report.models["kronecker"]["implementation"] == [
        "mimari.kuresel_bag.KureselZincir"
    ]
    assert report.models["hga"]["implementation"] == [
        "mimari.hiper_attention.HiperGeometrikAttention",
        "mimari.encoder.GeometrikVeriEncoder",
        "mimari.kuresel_bag.KureselZincir",
        "mimari.decoder.FraktalDecoder",
    ]
    assert all(report.checks.values())


def test_dort_mimari_ayni_veri_optimizer_loss_batch_schedule_ve_metrikleri_kullanir():
    report = run_twt_architecture_baselines(seed=1, profile="smoke")
    fairness = report.fairness

    assert fairness["same_train_candidate_count"] == 127602
    assert fairness["same_dev_candidate_count"] == 17030
    assert fairness["same_test_candidate_count"] == 16036
    assert len(fairness["same_batch_schedule_sha256"]) == 64
    assert len(fairness["same_feature_vocabulary"]) == 64
    assert fairness["same_optimizer"] == "AdamW"
    assert fairness["same_loss"] == "CrossEntropyLoss"
    assert fairness["same_steps"] == 32
    assert fairness["same_batch_size"] == 512
    assert fairness["same_initial_embedding_weights"] is True
    assert report.feature_contract["tokenizer_changed"] is False
    assert report.feature_contract["vocabulary_source"].endswith("train candidates only")
    assert report.feature_contract["unknown_tokens"]["train"] == 0

    for model in report.models.values():
        assert model["nonfinite_steps"] == 0
        assert model["all_trainable_parameters_received_gradient"] is True
        assert model["parameters_without_gradient"] == []
        assert model["test"]["all"]["total"] == 16036
        assert model["test"]["all"]["coverage"] == 1.0
        assert set(model["test"]) == {
            "all",
            "entity_disjoint",
            "relation_disjoint",
            "composition_disjoint",
            "wording_disjoint",
            "sentence_disjoint",
            "seen_composition",
        }
        for key in ("accuracy", "precision", "recall", "f1", "far", "frr", "coverage"):
            assert 0.0 <= model["test"]["all"][key] <= 1.0
        calibration = model["calibration"]
        assert calibration["fit_split"] == "dev"
        assert calibration["evaluation_split"] == "test"
        assert all(calibration["checks"].values())
        assert set(calibration["slices"]) == set(model["test"])
        assert calibration["slices"]["all"]["before"]["accuracy"] == (
            calibration["slices"]["all"]["after"]["accuracy"]
        )
        assert len(calibration["slices"]["all"]["after"]["risk_coverage_points"]) == 5


def test_seed1_smoke_bilimsel_regresyon_skorlari_sabit():
    report = run_twt_architecture_baselines(seed=1, profile="smoke")
    expected = {
        "dense": 0.88722749,
        "transformer": 0.84333228,
        "kronecker": 0.891133,
        "hga": 0.89583333,
    }
    for name, expected_f1 in expected.items():
        assert report.models[name]["test"]["all"]["f1"] == pytest.approx(
            expected_f1, abs=1e-7
        )
    # Tek seed sıralaması evrensel üstünlük olarak yorumlanmamalıdır.
    assert report.limitations
    assert "genel dil modelleme değildir" in report.limitations[0]


def test_ayni_seed_cache_kopyasi_sonucu_degistirmez():
    first = run_twt_architecture_baselines(seed=1, profile="smoke")
    second = run_twt_architecture_baselines(seed=1, profile="smoke")

    assert first is not second
    assert first.to_dict() == second.to_dict()
    assert "Adillik kapıları" in first.markdown()
    assert "Dev-only uncertainty calibration" in first.markdown()
