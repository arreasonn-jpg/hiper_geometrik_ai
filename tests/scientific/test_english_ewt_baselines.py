# -*- coding: utf-8 -*-
"""Pinned English UD EWT data and BERT/GPT-style baseline contract."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hga.evaluation.english_ewt import (
    DATA_DIR,
    MODEL_ORDER,
    EnglishEWT,
    english_ewt_markdown,
    prepare_english_ewt_task,
    run_english_ewt_baselines,
)


def test_ewt_provenance_kaynak_ve_lisans_hashleri_sabit():
    dataset = EnglishEWT()
    provenance = dataset.provenance

    assert provenance["upstream"]["revision"] == "4a4d77f599ea53cc405f85d0cec4b2f14f81d42b"
    assert provenance["license"]["spdx_id"] == "CC-BY-SA-4.0"
    assert "Creative Commons" in (
        DATA_DIR / provenance["license"]["vendored_license_path"]
    ).read_text(encoding="utf-8")
    assert set(dataset.source_hashes()) == {"train.conllu", "dev.conllu", "test.conllu"}


def test_ewt_resmi_split_icin_deterministik_kucuk_benchmark_taski_uretir():
    task = prepare_english_ewt_task()

    assert task.sentence_counts == {"train": 512, "dev": 128, "test": 128}
    assert {name: len(getattr(task, name)) for name in ("train", "dev", "test")} == {
        "train": 16_130, "dev": 3_442, "test": 3_100,
    }
    assert task.dataset_hash == "2f852e91da70301e5a572f717b3e9d6998d14ca6bf38daa8bb77f4baa5b3ddd3"
    assert task.candidate_hashes == {
        "train": "ddb49412f74deb2c2c891f422329a5e80163500265e447cfb217559b9a1e72a7",
        "dev": "a904920d0806300390ebc9d47f81263be8cf8f9df998e0b4751e58b68ac6f286",
        "test": "8a12c629f35bf42ef420b5c97ef6fb3351b43d41292024ecce721ffd2c5c2a53",
    }
    for split in (task.train, task.dev, task.test):
        assert len(split) % 2 == 0
        for positive, negative in zip(split[::2], split[1::2]):
            assert positive.expected_valid is True
            assert negative.expected_valid is False
            assert positive.dependent_id == negative.dependent_id
            assert positive.head_id != negative.head_id


def test_english_baseline_dort_ailenin_parametre_ve_etiket_sozlesmesini_raporlar():
    pytest.importorskip("torch")
    report = run_english_ewt_baselines(seeds=(1,), profile="smoke")

    assert tuple(report.parameter_counts) == MODEL_ORDER
    assert report.parameter_counts == {
        "dense": 102_116,
        "transformer": 96_938,
        "bert_style": 97_586,
        "gpt_style": 96_938,
    }
    assert report.checks["official_english_ud_sources_hash_verified"]
    assert report.checks["all_four_baseline_families_present"]
    assert report.checks["all_models_trained_from_scratch_and_reported"]
    assert report.checks["parameter_budget_within_twelve_percent"]
    assert report.checks["five_or_more_seeds"] is False
    assert any("not pretrained BERT/GPT" in item for item in report.limitations)
    assert "BERT-style" in english_ewt_markdown(report)
    assert "GPT-style" in english_ewt_markdown(report)


def test_kuraturlenmis_bes_seed_raporu_tum_kabul_kapilarini_gecer():
    path = Path(__file__).resolve().parents[2] / "docs" / "english_ewt_baselines.json"
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["seeds"] == [1, 2, 3, 4, 5]
    assert all(report["checks"].values())
    assert set(report["aggregate"]) == set(MODEL_ORDER)
    assert all(metric["f1"]["std"] >= 0.0 for metric in report["aggregate"].values())
