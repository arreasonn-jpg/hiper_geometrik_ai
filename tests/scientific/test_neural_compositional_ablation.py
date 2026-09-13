"""Gerçek TWT neural compositional HGA component ablation testleri."""
from __future__ import annotations

import pytest
import torch

from hga.evaluation.neural_compositional import (
    ABLATION_ORDER,
    run_neural_compositional_ablation,
)
from hga.evaluation.twt_baselines import build_twt_models


def test_hga_ablation_bilesenleri_fiziksel_olarak_degisir():
    full = build_twt_models(128, hga_ablation="full")["hga"]()
    no_attention = build_twt_models(128, hga_ablation="no_attention")["hga"]()
    additive = build_twt_models(128, hga_ablation="additive_geometry")["hga"]()
    no_chain = build_twt_models(128, hga_ablation="no_kronecker_chain")["hga"]()

    assert full.attention is not None
    assert no_attention.attention is None
    assert additive.ablation == "additive_geometry"
    assert isinstance(no_chain.body, torch.nn.Identity)
    assert sum(p.numel() for p in full.parameters()) == sum(
        p.numel() for p in additive.parameters()
    )
    assert sum(p.numel() for p in no_attention.parameters()) < sum(
        p.numel() for p in full.parameters()
    )
    assert sum(p.numel() for p in no_chain.parameters()) < sum(
        p.numel() for p in full.parameters()
    )


def test_neural_compositional_ayni_twt_split_schedule_ve_baslangici_kullanir():
    report = run_neural_compositional_ablation(seed=1, profile="smoke")
    fairness = report.fairness

    assert tuple(report.arms) == ABLATION_ORDER
    assert report.dataset_hash == "66b13a898efa88998a9329f1551530f5241835a8085a0e5f26e0eb374d7e3276"
    assert fairness["same_train_dev_test_counts"] == {
        "train": 127602,
        "dev": 17030,
        "test": 16036,
    }
    assert len(fairness["same_train_only_vocabulary_hash"]) == 64
    assert len(fairness["same_batch_schedule_sha256"]) == 64
    assert fairness["same_optimizer"] == "AdamW"
    assert fairness["same_loss"] == "CrossEntropyLoss"
    assert fairness["same_steps"] == 32
    assert all(fairness["matching_shared_tensors_copied_from_full"].values())
    assert fairness["parameter_padding_or_reserve"] is False
    assert report.candidate_hashes["test_challenge"] == (
        "877a788e8468ba7cbb676e73581fb938018b1ddc891034a2c3c2bcc5460dbb24"
    )


def test_neural_compositional_parametre_gradient_ve_c_g_n_sozlesmesi():
    report = run_neural_compositional_ablation(seed=1, profile="smoke")

    assert report.fairness["physical_parameter_counts"] == {
        "full": 291827,
        "no_attention": 290770,
        "additive_geometry": 291827,
        "no_kronecker_chain": 289019,
    }
    assert report.fairness["parameter_max_to_min_ratio"] == 1.00971562
    assert all(report.checks.values())
    for arm in report.arms.values():
        generalization = arm["neural_generalization"]
        composition = arm["test"]["composition_disjoint"]
        assert generalization["symbol"] == "C_G_N"
        assert generalization["eligible_cases"] == 7086
        assert generalization["correctly_generalized"] == composition["correct"]
        assert generalization["score"] == composition["accuracy"]
        assert generalization["is_theoretical_capacity"] is False
        assert composition["positive"] == composition["negative"] == 3543
        assert arm["all_trainable_parameters_received_gradient"] is True
        assert arm["parameters_without_gradient"] == []
        assert arm["nonfinite_steps"] == 0
        assert arm["test"]["all"]["coverage"] == 1.0


def test_neural_compositional_seed1_regresyon_ve_farklar_sabit():
    report = run_neural_compositional_ablation(seed=1, profile="smoke")
    expected_c_g_n = {
        "full": 0.87933954,
        "no_attention": 0.8827265,
        "additive_geometry": 0.88103302,
        "no_kronecker_chain": 0.85831216,
    }
    for name, expected in expected_c_g_n.items():
        assert report.arms[name]["neural_generalization"]["score"] == pytest.approx(
            expected, abs=1e-7
        )
    assert report.deltas_from_full == {
        "no_attention": {
            "full_minus_arm_c_g_n": -0.00338696,
            "full_minus_arm_all_f1": -0.00251498,
        },
        "additive_geometry": {
            "full_minus_arm_c_g_n": -0.00169348,
            "full_minus_arm_all_f1": -0.00072669,
        },
        "no_kronecker_chain": {
            "full_minus_arm_c_g_n": 0.02102738,
            "full_minus_arm_all_f1": 0.01627954,
        },
    }
    # Bu tek seed sonucu mimari üstünlük diye yorumlanmamalıdır.
    assert "teorik kapasite değildir" in report.limitations[0]
    assert "Kabul kapıları" in report.markdown()


def test_neural_compositional_ayni_seed_cache_kopyasi_deterministik():
    first = run_neural_compositional_ablation(seed=1, profile="smoke")
    second = run_neural_compositional_ablation(seed=1, profile="smoke")

    assert first is not second
    assert first.to_dict() == second.to_dict()
