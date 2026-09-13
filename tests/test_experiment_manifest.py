"""EXP-ID, manifest ve çoklu-seed tekrarlanabilirlik testleri."""
import json
from pathlib import Path

import pytest

from hga.evaluation import (
    ExperimentRun,
    GoldenDataset,
    canonical_hash,
    file_sha256,
    run_golden_seed_sweep,
    run_seed_sweep,
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_experiment_run_tam_dosya_sozlesmesi(tmp_path):
    model = tmp_path / "model.bin"
    model.write_bytes(b"fixed-model")
    config = {"model": {"n": 8}, "name": "smoke"}
    run = ExperimentRun.create(
        tmp_path / "runs", config=config, seed=42,
        dataset_hash="dataset-abc", parameters={"batch": 4}, model_path=model,
    )
    with run.capture_stdout():
        print("captured output")
    run.complete({"accuracy": 0.75})

    assert run.experiment_id == "EXP-0001"
    assert {path.name for path in run.directory.iterdir()} == {
        "config.yaml", "manifest.json", "results.json", "stdout.log", "model_hash.txt"
    }
    # JSON biçimi YAML 1.2'nin alt kümesidir.
    assert _json(run.directory / "config.yaml") == config
    assert (run.directory / "model_hash.txt").read_text().strip() == file_sha256(model)
    assert "captured output" in (run.directory / "stdout.log").read_text(encoding="utf-8")

    manifest = _json(run.directory / "manifest.json")
    assert manifest["experiment_id"] == "EXP-0001"
    assert manifest["seed"] == 42
    assert manifest["dataset_hash"] == "dataset-abc"
    assert manifest["config_hash"] == canonical_hash(config)
    assert manifest["parameters"] == {"batch": 4}
    assert manifest["parameter_count"] is None
    assert manifest["python_version"]
    assert manifest["cpu"] and manifest["cpu_count"] >= 1
    assert "ram_total_bytes" in manifest and "gpu" in manifest
    assert "torch_version" in manifest and "cuda_version" in manifest
    assert "device" in manifest
    assert manifest["timings"]["total_seconds"] >= 0.0
    assert "training_seconds" in manifest["timings"]
    assert "inference_seconds" in manifest["timings"]
    assert len(manifest["git_commit"]) == 40
    assert manifest["result"] == "COMPLETED"
    assert _json(run.directory / "results.json") == {"accuracy": 0.75}


def test_experiment_ids_artan_ve_atomik(tmp_path):
    kwargs = dict(config={}, seed=1, dataset_hash="hash")
    first = ExperimentRun.create(tmp_path, **kwargs)
    second = ExperimentRun.create(tmp_path, **kwargs)
    assert first.experiment_id == "EXP-0001"
    assert second.experiment_id == "EXP-0002"


def test_seed_sweep_mean_std_ve_ayri_manifestler(tmp_path):
    report = run_seed_sweep(
        lambda seed: {"metrics": {"score": float(seed), "fixed": 1.0}},
        seeds=[1, 2, 3], root=tmp_path, config={"kind": "unit"}, dataset_hash="d",
    )
    assert report.experiment_ids == ["EXP-0001", "EXP-0002", "EXP-0003"]
    assert report.dataset_hash == "d"
    assert report.config_hash == canonical_hash({"kind": "unit"})
    assert len(report.manifests) == 3
    assert all(manifest["result"] == "COMPLETED" for manifest in report.manifests)
    assert [result["metrics"]["score"] for result in report.results] == [1.0, 2.0, 3.0]
    assert report.aggregate["metrics.score"]["mean"] == 2.0
    assert report.aggregate["metrics.score"]["std"] > 0.0
    assert report.aggregate["metrics.fixed"]["std"] == 0.0
    assert not report.deterministic_results
    for index, seed in enumerate([1, 2, 3], 1):
        assert _json(tmp_path / f"EXP-{index:04d}" / "manifest.json")["seed"] == seed


def test_seed_sweep_duplicate_ve_bos_seed_reddi(tmp_path):
    def callback(seed):
        return {"seed": seed}

    with pytest.raises(ValueError, match="En az bir"):
        run_seed_sweep(callback, [], tmp_path, {}, "d")
    with pytest.raises(ValueError, match="benzersiz"):
        run_seed_sweep(callback, [1, 1], tmp_path, {}, "d")


def test_basarisiz_kosu_failed_manifest_birakir(tmp_path):
    def fail(_seed):
        raise RuntimeError("intentional")

    with pytest.raises(RuntimeError, match="intentional"):
        run_seed_sweep(fail, [7], tmp_path, {}, "d")
    manifest = _json(tmp_path / "EXP-0001" / "manifest.json")
    result = _json(tmp_path / "EXP-0001" / "results.json")
    assert manifest["result"] == "FAILED"
    assert result["error_type"] == "RuntimeError"


def test_seed_sweep_cok_kucuk_bilimsel_metrigi_sifira_yuvarlamaz(tmp_path):
    report = run_seed_sweep(
        lambda seed: {"nmse": seed * 1e-12},
        seeds=[1, 2, 3], root=str(tmp_path), config={"kind": "tiny"},
        dataset_hash="tiny-data",
    )
    assert report.aggregate["nmse"]["mean"] == 2e-12
    assert report.aggregate["nmse"]["std"] > 0.0


def test_golden_bes_seed_manifestli_ve_deterministik(tmp_path):
    report = run_golden_seed_sweep(tmp_path, [1, 2, 3, 4, 5])
    assert len(report.experiment_ids) == 5
    assert report.deterministic_results
    assert report.aggregate["metrics.accuracy"] == {
        "mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0,
    }
    assert "kalitesi kanıtı değildir" in report.note
    dataset_hash = GoldenDataset().dataset_hash()
    for experiment_id in report.experiment_ids:
        manifest = _json(tmp_path / experiment_id / "manifest.json")
        assert manifest["dataset_hash"] == dataset_hash
        assert manifest["result"] == "COMPLETED"
