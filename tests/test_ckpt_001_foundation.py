# -*- coding: utf-8 -*-
"""CKPT-001 reproducibility/theory contracts without pretending external work happened."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hga.evaluation.experiment import determinism_contract, seed_everything
from hga.evaluation.foundation_analysis import (
    HASH_KEY_BITS,
    foundation_analysis_markdown,
    measure_gradient_flow,
    run_foundation_analysis,
    sparse_memory_theory,
)
from hga.evaluation.reproduce import reproduce_all_markdown, run_reproduce_all

ROOT = Path(__file__).resolve().parents[1]


def test_sparse_memory_reference_model_prehash_ve_occupancy_ayrimini_korur():
    report = sparse_memory_theory(
        vocabulary_size=4, window_size=2, slot_count=4, table_count=1,
        embedding_dimension=2, queried_unique_contexts=2,
    )

    assert report["input_namespace"]["log2_context_count"] == 4.0
    assert report["hash_and_signature_bounds"]["prehash_key_bits"] == HASH_KEY_BITS
    assert report["hash_and_signature_bounds"]["implementation_signature_count_upper_bound"] == 4
    small_namespace = sparse_memory_theory(
        vocabulary_size=2, window_size=2, slot_count=1_000, table_count=1,
        embedding_dimension=2, queried_unique_contexts=2,
    )
    assert small_namespace["hash_and_signature_bounds"]["implementation_signature_count_upper_bound"] == 4
    # E[occupied] = 4 * (1 - (3/4)^2) = 1.75; it is an idealized model.
    assert report["ideal_uniform_occupancy"]["expected_occupied_signatures"] == 1.75
    assert report["ideal_uniform_occupancy"]["probability_of_zero_collision"] == 0.75
    assert "not proven uniform" in " ".join(report["assumptions"])


def test_gradient_diagnostic_seedli_ve_sonlu():
    pytest.importorskip("torch")
    row = measure_gradient_flow(n=4, k=2, seed=7, activation="silu")

    assert row.all_gradients_finite
    assert row.input_gradient_norm > 0.0
    assert row.mean_factor_gradient_norm > 0.0
    assert row.linear_spectral_norm_product > 0.0


def test_foundation_grid_n16_bulgu_ve_cokus_sozlesmesini_tekrarlar():
    pytest.importorskip("torch")
    report = run_foundation_analysis(n_values=(4, 16), k_values=(1,), seed=1, samples=512)
    data = report.to_dict()

    assert all(data["checks"].values())
    n16 = next(row for row in data["rank_grid"] if row["n"] == 16)
    assert n16["single_layer_spectrum"]["measured"]["entropy_effective_dimension"] == pytest.approx(
        98.379852, abs=1e-5
    )
    assert n16["linear_chain"]["collapsed"]
    assert n16["silu_chain"]["collapse_residual"] > 1e-3
    assert "idealized hash model" in foundation_analysis_markdown(data)


def test_reproduce_orchestrator_five_seed_ve_receipt_sozlesmesini_zorlar():
    with pytest.raises(ValueError, match="at least two unique seeds"):
        run_reproduce_all(root=ROOT / ".ignored", seeds=(1,), profile="smoke")

    receipt = {
        "protocol": "hga-reproduce-all-foundation-v1",
        "status": "COMPLETED",
        "seeds": [1, 2, 3, 4, 5],
        "profile": "smoke",
        "experiment_id": "EXP-0001",
        "deterministic_value_fingerprint": "abc",
        "checks": {"example": True},
        "artifact_sha256": {"example.json": "def"},
        "limitations": ["timestamps excluded"],
    }
    assert "CKPT-001 reproduction receipt" in reproduce_all_markdown(receipt)


def test_seed_kontrolu_ve_ckpt_manifesti_external_durumu_dogrular():
    pytest.importorskip("torch")
    seed_everything(11)
    controls = determinism_contract()
    assert controls["torch_deterministic_algorithms"] is True
    assert controls["cudnn_deterministic"] is True
    assert controls["cudnn_benchmark"] is False

    checkpoint = json.loads((ROOT / "docs" / "checkpoints" / "CKPT-001.json").read_text())
    assert checkpoint["release"] == "v0.2.0-foundation"
    assert checkpoint["status"] == "artifact-complete-external-validation-pending"
    assert checkpoint["theory_contract"]["full_chain_vc_pseudodimension"] == "NOT_ESTABLISHED"
    assert checkpoint["theory_contract"]["sparse_memory_prehash_key_bits"] == 31


def test_foundation_docker_make_docs_ve_community_artefaktlari_mevcut():
    makefile = (ROOT / "Makefile").read_text()
    dockerfile = (ROOT / "Dockerfile").read_text()
    cuda = (ROOT / "Dockerfile.cuda").read_text()
    workflow = (ROOT / ".github" / "workflows" / "container.yml").read_text()

    assert "reproduce:" in makefile and "reproduce-all" in makefile
    assert "PYTHONHASHSEED" in makefile and "CUBLAS_WORKSPACE_CONFIG" in makefile
    assert "docker-entrypoint.sh" in dockerfile and 'CMD ["reproduce-all"]' in dockerfile
    assert "pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime" in cuda
    assert "ghcr.io/${{ github.repository_owner }}/hiper_geometrik_ai" in workflow
    environment = (ROOT / "environment.yml").read_text()
    assert "python=3.11.9" in environment and "requirements-lock.txt" in environment
    assert (ROOT / "mkdocs.yml").is_file()
    assert (ROOT / "paper" / "technical_report.tex").is_file()
    assert (ROOT / "docs" / "FOUNDATION_THEORY_SCOPE.md").is_file()
    assert (ROOT / "CODE_OF_CONDUCT.md").is_file()
    assert (ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").is_file()


def test_kuratorlu_foundation_grid_ckpt_seed_ve_teori_bulguyle_uyumlu():
    report = json.loads((ROOT / "docs" / "foundation_analysis.json").read_text())

    assert report["protocol"] == "hga-ckpt-001-foundation-analysis-v1"
    assert all(report["checks"].values())
    n16_rows = [row for row in report["rank_grid"] if row["n"] == 16]
    assert len(n16_rows) == 3
    assert all(row["linear_chain"]["collapsed"] for row in n16_rows)
    assert n16_rows[0]["single_layer_spectrum"]["measured"]["entropy_effective_dimension"] == pytest.approx(
        98.379852, abs=1e-5
    )
