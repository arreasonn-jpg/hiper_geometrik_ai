"""Gerçek, insan anotasyonlu Turkish Web Treebank benchmark regresyonları."""
from __future__ import annotations

import json
import shutil

import pytest

from hga.evaluation.real_turkish import (
    DATA_DIR,
    DIMENSIONS,
    TurkishWebTreebank,
    run_real_turkish_benchmark,
)

EXPECTED_DATASET_HASH = "66b13a898efa88998a9329f1551530f5241835a8085a0e5f26e0eb374d7e3276"
EXPECTED_CONFIG_HASH = "be1ee486a7ad291f0054e6c7897544cdd5e0860065268d0011aacb3ad8b9e1ca"
EXPECTED_SPLIT_HASHES = {
    "train": "a2aecea070216a6b09ec668deb2b4082808d651a676ab77f48df4cfdc223abf8",
    "dev": "6e47d21756c81e294ca6ff3873e2997fdee7b8af3a0c0f70d55bfbe903587b4b",
    "test": "6a205cbd2eac4107fcf18c58b21ac2a1233ad46d410627c4790fedfb270d17ab",
}
EXPECTED_CANDIDATE_HASHES = {
    "train": "451bc9cc138a35372f1e844de3a649184bfcae1e2e57ae7679782bdfae89b63d",
    "dev": "9c2ce46ea784424fc221c6d6a1e03798988bb7258d5e959f8fc748f40752eeed",
    "test": "cba9b99bbff93f404f0581107e4de4490a29983b44b255ddf4e4edc2cf6e23e6",
    "test_challenge": "877a788e8468ba7cbb676e73581fb938018b1ddc891034a2c3c2bcc5460dbb24",
}


def test_twt_provenance_lisans_revision_ve_ham_hashlar_sabit():
    dataset = TurkishWebTreebank()
    provenance = dataset.provenance

    assert provenance["benchmark_id"] == "hga-real-turkish-twt-v1"
    assert provenance["dataset"]["language"] == "tr"
    assert provenance["dataset"]["human_annotated"] is True
    assert provenance["dataset"]["published_counts"] == {
        "sentences": 4851,
        "words": 66466,
        "inflectional_group_tokens": 81370,
        "web_sentences": 2541,
        "wiki_sentences": 2310,
    }
    assert provenance["upstream"]["revision"] == (
        "40838e5cbe3f2882d4e768a3d782e6219e50b52a"
    )
    assert provenance["license"]["spdx_id"] == "Apache-2.0"
    assert provenance["license"]["redistributable"] is True
    assert dataset.source_hashes() == {
        "web": "d6f92b12b74f367a36e30dd3d644d7a0bb01a991584602e6e69f1191664f86bf",
        "wiki": "58f049fae8d0653ba0940a58168c17c7cebbc50f88cad2985d12d34ac1b9acde",
    }
    assert "Apache License" in (DATA_DIR / "LICENSE-APACHE-2.0.txt").read_text(
        encoding="utf-8"
    )


def test_twt_split_ve_candidate_hashlari_byte_stable():
    dataset = TurkishWebTreebank()

    assert {name: len(rows) for name, rows in dataset.raw_splits.items()} == {
        "train": 3881,
        "dev": 485,
        "test": 485,
    }
    assert {name: len(rows) for name, rows in dataset.splits.items()} == {
        "train": 3881,
        "dev": 484,
        "test": 484,
    }
    assert dataset.dataset_hash() == EXPECTED_DATASET_HASH
    assert dataset.config_hash() == EXPECTED_CONFIG_HASH
    assert dataset.split_hashes() == EXPECTED_SPLIT_HASHES

    report = run_real_turkish_benchmark(dataset, seed=17)
    assert report.candidate_hashes == EXPECTED_CANDIDATE_HASHES
    assert report.dataset_hash == EXPECTED_DATASET_HASH
    assert report.config_hash == EXPECTED_CONFIG_HASH


def test_twt_leakage_quarantine_duplicate_surface_ve_semantic_proxy_temiz():
    report = run_real_turkish_benchmark(seed=1)
    audit = report.leakage_audit

    assert audit["clean"] is True
    assert audit["sentence_id_disjoint"] is True
    assert audit["internal_normalized_surface_duplicate_count"] == 0
    assert audit["cross_split_normalized_surface_duplicate_count"] == 0
    proxy = audit["lexical_semantic_proxy"]
    assert proxy["threshold"] == 0.8
    assert proxy["clean"] is True
    assert proxy["effective_cross_split_finding_count"] == 0
    assert proxy["is_semantic_equivalence_proof"] is False
    assert proxy["quarantine_evidence"] == [
        {
            "reference_id": "tr-forum:00001297:S012",
            "quarantined_id": "tr-forum:00001297:S029",
            "exact_surface": True,
            "lemma_set_jaccard": 1.0,
        },
        {
            "reference_id": "tr-forum:00002420:S010",
            "quarantined_id": "tr-forum:00000353:S041",
            "exact_surface": False,
            "lemma_set_jaccard": 0.83333333,
        },
    ]


def test_twt_insan_ground_truth_metrikleri_ve_disjoint_boyutlari_raporlanir():
    report = run_real_turkish_benchmark(seed=5)

    assert report.task["ground_truth"].startswith("human-annotated TWT")
    assert report.task["is_semantic_relation_extraction"] is False
    assert report.task["is_named_entity_recognition"] is False
    assert report.corpus == {
        "raw_sentences": 4851,
        "effective_sentences": 4849,
        "raw_inflectional_group_tokens": 81370,
        "effective_token_counts": {"train": 64707, "dev": 8633, "test": 8018},
        "source_sections": {"web": 2541, "wiki": 2310},
    }
    assert report.metrics.total == 16036
    assert report.metrics.positive == report.metrics.negative == 8018
    assert report.metrics.accuracy == 0.91169868
    assert report.metrics.f1 == 0.93269112
    assert report.metrics.far == 0.07782489
    assert report.metrics.frr == 0.02905962
    assert report.metrics.coverage == 0.96514093
    assert set(report.dimensions) == set(DIMENSIONS)
    assert report.dimensions["entity_disjoint"].total == 2704
    assert report.dimensions["relation_disjoint"].total == 168
    assert report.dimensions["composition_disjoint"].total == 7086
    assert report.dimensions["wording_disjoint"].total == 6036
    assert report.dimensions["sentence_disjoint"].total == 16036
    assert report.split_contract["relation_disjoint_positive_test_counts"] == {
        "csubj": 52,
        "parataxis": 53,
    }
    assert all(report.split_contract["dimension_audit"].values())
    assert all(report.checks.values())
    assert "Accuracy" in report.markdown()
    assert "FAR / FRR / coverage" in report.markdown()


def test_twt_relation_holdout_abstentioni_basari_gibi_gostermez():
    report = run_real_turkish_benchmark(seed=2)
    metric = report.dimensions["relation_disjoint"]

    assert metric.positive == metric.negative == 84
    assert metric.uncertain == metric.total
    assert metric.coverage == 0.0
    assert metric.accuracy == 0.0
    assert report.baseline["purpose"].endswith("not a competitive language model.")


def test_twt_deterministik_baseline_seedler_arasinda_ozdes():
    first = run_real_turkish_benchmark(seed=101)
    second = run_real_turkish_benchmark(seed=202)

    assert first.seed != second.seed
    first_payload = first.to_dict()
    second_payload = second.to_dict()
    first_payload.pop("seed")
    second_payload.pop("seed")
    assert first_payload == second_payload


def test_twt_source_tamperi_hash_kapisinda_reddedilir(tmp_path):
    copied = tmp_path / "twt_v1"
    shutil.copytree(DATA_DIR, copied)
    web = copied / "web.conllu"
    web.write_bytes(web.read_bytes() + b"\n# tampered\n")

    with pytest.raises(ValueError, match="source SHA-256 uyuşmazlığı"):
        TurkishWebTreebank(copied)


def test_twt_provenance_manifest_json_ve_dataset_card_paketlenebilir():
    provenance = json.loads((DATA_DIR / "PROVENANCE.json").read_text(encoding="utf-8"))
    card = (DATA_DIR / "DATASET_CARD.md").read_text(encoding="utf-8")

    assert provenance["expected_integrity"]["split_hashes"] == EXPECTED_SPLIT_HASHES
    assert provenance["expected_integrity"]["candidate_hashes"] == EXPECTED_CANDIDATE_HASHES
    assert "byte-identical" in card
    assert "not semantic knowledge triples" in card
