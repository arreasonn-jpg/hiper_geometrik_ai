"""Birleşik research benchmark, manifest ve üç-format rapor testleri."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from hga.evaluation.research import (
    research_report_markdown,
    run_research_benchmark,
    save_research_report,
    validate_sections,
)

PURE_SECTIONS = (
    "architecture",
    "memory",
    "verification",
    "compositional_generalization",
    "self_learning",
    "ood",
    "turkish_nlp",
)


def test_research_suite_her_seed_icin_tam_manifest_uretir(tmp_path):
    report = run_research_benchmark(
        root=tmp_path / "experiments", seeds=[11, 12], profile="smoke",
        sections=PURE_SECTIONS,
    )
    assert report.outcome == "COMPLETED"
    assert report.experiment_ids == ["EXP-0001", "EXP-0002"]
    assert report.reproducibility["seed_count"] == 2
    assert report.reproducibility["all_manifests_completed"] is True
    assert report.reproducibility["dataset_hash_identical_across_seeds"] is True
    assert report.sections["compositional_generalization"]["score_mean"] == 1.0
    assert report.sections["verification"]["score_mean"] == 1.0
    proof_suite = report.sections["verification"]["metrics_reference_seed"]["proof_attack_suite"]
    assert proof_suite["metrics"]["robustness"] == 1.0
    assert proof_suite["metrics"]["far"] == proof_suite["metrics"]["frr"] == 0.0
    assert report.sections["memory"]["score_mean"] < 1.0  # collision sınırı gizlenmez
    lifecycle = report.sections["memory"]["metrics_reference_seed"][
        "active_dynamic_kv_lifecycle"
    ]
    assert lifecycle["protocol"] == "active-dynamic-kv-lifecycle-v1"
    assert lifecycle["exact_retrieval_accuracy"] == 1.0
    assert lifecycle["replay_active_key_consistency"] == 1.0
    assert lifecycle["tamper_rejected"] is True
    assert all(lifecycle["checks"].values())
    multi_environment = report.sections["self_learning"]["metrics_reference_seed"][
        "multi_environment"
    ]
    assert multi_environment["environments"] == [
        "arithmetic", "logic", "consistency"
    ]
    assert multi_environment["shared_memory_policy"] == "DYNAMIC_KV"
    assert multi_environment["shared_memory_retrieval_accuracy"] == 1.0
    assert multi_environment["cross_verifier_acceptances"] == 0
    assert all(multi_environment["checks"].values())
    multi_summary = report.sections["self_learning"]["multi_environment_summary"]
    assert multi_summary["seed_count"] == 2
    assert multi_summary["all_checks_all_seeds"] is True
    assert multi_summary["cross_verifier_acceptances_all_seeds"] == 0
    assert multi_summary["environments"]["logic"]["verified"]["std"] >= 0.0
    lifecycle = report.sections["verification"]["metrics_reference_seed"][
        "knowledge_lifecycle"
    ]
    assert lifecycle["protocol"] == "real-artifact-knowledge-lifecycle-v1"
    assert lifecycle["real_source"]["license"] == "Apache-2.0"
    assert all(lifecycle["checks"].values())
    turkish = report.sections["turkish_nlp"]["metrics_reference_seed"]["real_turkish_twt"]
    assert turkish["benchmark_id"] == "hga-real-turkish-twt-v1"
    assert turkish["provenance"]["license"]["spdx_id"] == "Apache-2.0"
    assert turkish["provenance"]["dataset"]["human_annotated"] is True
    assert turkish["metrics"]["f1"] == 0.93269112
    assert turkish["dimensions"]["relation_disjoint"]["coverage"] == 0.0
    assert turkish["leakage_audit"]["clean"] is True
    architectures = report.sections["turkish_nlp"]["metrics_reference_seed"][
        "parameter_matched_architectures"
    ]
    assert set(architectures["models"]) == {"dense", "transformer", "kronecker", "hga"}
    assert architectures["fairness"]["parameter_max_to_min_ratio"] <= 1.01
    assert architectures["fairness"]["body_parameter_max_to_min_ratio"] <= 1.05
    assert architectures["fairness"]["unused_parameter_padding"] is False
    assert all(architectures["checks"].values())
    architecture_summary = report.sections["turkish_nlp"]["architecture_summary"]
    assert architecture_summary["seed_count"] == 2
    assert architecture_summary["seeds"] == [11, 12]
    assert architecture_summary["all_checks_all_seeds"] is True
    assert set(architecture_summary["models"]) == {
        "dense", "transformer", "kronecker", "hga"
    }
    assert architecture_summary["models"]["hga"]["test"]["all"]["f1"]["std"] >= 0.0
    calibration = architecture_summary["models"]["hga"]["calibration"]
    assert calibration["temperature"]["mean"] > 0.0
    assert calibration["ece_before"]["std"] >= 0.0
    assert calibration["ece_after"]["std"] >= 0.0
    assert calibration["aurc_after"]["std"] >= 0.0
    ablation = report.sections["turkish_nlp"]["metrics_reference_seed"][
        "neural_compositional_ablation"
    ]
    assert set(ablation["arms"]) == {
        "full", "no_attention", "additive_geometry", "no_kronecker_chain"
    }
    assert all(ablation["checks"].values())
    ablation_summary = report.sections["turkish_nlp"]["neural_compositional_summary"]
    assert ablation_summary["seed_count"] == 2
    assert ablation_summary["seeds"] == [11, 12]
    assert ablation_summary["all_checks_all_seeds"] is True
    assert ablation_summary["arms"]["full"]["c_g_n"]["std"] >= 0.0
    markdown = research_report_markdown(report.to_dict())
    assert "Parameter-matched TWT architecture test özeti" in markdown
    assert "Body parametre" in markdown
    assert "Dev-only temperature calibration" in markdown
    assert "ECE before" in markdown
    assert "Neural compositional HGA ablation özeti" in markdown
    assert "C_G_N mean" in markdown
    assert "Multi-environment closed self-learning" in markdown
    assert "Cross-verifier acceptances" in markdown
    assert "Real-artifact STALE lifecycle" in markdown
    assert "not FALSE or RETRACTED" in markdown

    for experiment_id in report.experiment_ids:
        manifest_path = tmp_path / "experiments" / experiment_id / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["result"] == "COMPLETED"
        assert manifest["dataset_hash"] == report.dataset_hash
        assert manifest["config_hash"] == report.config_hash
        assert manifest["parameter_count"] > 0
        assert manifest["python_version"]
        assert manifest["cpu"]
        assert manifest["cpu_count"] >= 1
        assert manifest["ram_total_bytes"] > 0
        assert "gpu" in manifest and "cuda_version" in manifest
        assert manifest["timings"]["total_seconds"] >= 0.0
        assert "training_seconds" in manifest["timings"]
        assert "inference_seconds" in manifest["timings"]


def test_pytorch_bolumleri_skip_olmadan_tamamlanir(tmp_path):
    import torch

    assert torch.__version__
    report = run_research_benchmark(
        root=tmp_path / "experiments", seeds=[3], profile="smoke",
        sections=["kronecker", "neural_symbolic_hybrid"],
    )
    assert report.outcome == "COMPLETED"
    assert report.sections["kronecker"]["status"] == "COMPLETED"
    assert report.sections["neural_symbolic_hybrid"]["status"] == "COMPLETED"
    assert not report.sections["kronecker"]["skip_or_error_reasons"]
    assert not report.sections["neural_symbolic_hybrid"]["skip_or_error_reasons"]
    assert report.manifests[0]["torch_version"] == torch.__version__


def test_bes_seed_reproducibility_sozlesmesi(tmp_path):
    report = run_research_benchmark(
        root=tmp_path / "experiments", seeds=[1, 2, 3, 4, 5], profile="smoke",
        sections=["compositional_generalization"],
    )
    assert report.reproducibility["five_or_more_seeds"] is True
    assert report.reproducibility["all_manifests_completed"] is True
    assert len(report.experiment_ids) == 5
    assert report.sections["compositional_generalization"]["score_std"] == 0.0


def test_research_raporu_json_markdown_html_kaydeder(tmp_path):
    report = run_research_benchmark(
        root=tmp_path / "experiments", seeds=[1], profile="smoke",
        sections=["compositional_generalization", "ood"],
    )
    paths = save_research_report(
        report, tmp_path / "research_report.json", tmp_path / "research_report.md",
        tmp_path / "research_report.html",
    )
    assert set(paths) == {"json", "markdown", "html"}
    payload = json.loads((tmp_path / "research_report.json").read_text(encoding="utf-8"))
    assert payload["report_type"] == "hga-research-benchmark-v1"
    assert "C_G" in (tmp_path / "research_report.md").read_text(encoding="utf-8")
    html = (tmp_path / "research_report.html").read_text(encoding="utf-8")
    assert "HGA Research Benchmark" in html
    assert "Compositional Generalization" in html


def test_research_cli_tek_komutla_uc_rapor_uretir(tmp_path):
    root = Path(__file__).resolve().parents[2]
    output = subprocess.check_output(
        [
            sys.executable, "-m", "hga", "research-benchmark",
            "--seeds", "7",
            "--sections", "compositional-generalization,ood",
            "--experiment-root", str(tmp_path / "experiments"),
            "--out", str(tmp_path / "research_report.json"),
            "--markdown", str(tmp_path / "research_report.md"),
            "--html", str(tmp_path / "research_report.html"),
        ],
        cwd=root, text=True,
    )
    assert "HGA RESEARCH BENCHMARK" in output
    assert "Compositional Generalization" in output
    assert (tmp_path / "research_report.json").is_file()
    assert (tmp_path / "research_report.md").is_file()
    assert (tmp_path / "research_report.html").is_file()


def test_section_adlari_normalize_edilir_ve_bilinmeyen_reddedilir():
    assert validate_sections(["ood", "compositional-generalization"]) == [
        "compositional_generalization", "ood"
    ]
    try:
        validate_sections(["not-a-section"])
    except ValueError as error:
        assert "Bilinmeyen research bölümü" in str(error)
    else:  # pragma: no cover
        raise AssertionError("bilinmeyen bölüm reddedilmeliydi")
