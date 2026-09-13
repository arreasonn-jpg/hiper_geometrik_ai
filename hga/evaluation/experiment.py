"""Tekrarlanabilir deney dizini, manifest ve çoklu-seed çalıştırıcısı.

Her koşu atomik olarak ``EXP-NNNN`` kimliği alır ve şu sözleşmeyi üretir::

    EXP-0001/
      config.yaml
      manifest.json
      results.json
      stdout.log
      model_hash.txt

Runtime çıktıları varsayılan olarak ``experiments/EXP-*`` altındadır ve Git'e
alınmaz. Bilimsel rapora eklenecek seçilmiş sonuçlar ayrıca küratörlenmelidir.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, TextIO


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata() -> Dict[str, Any]:
    candidates = [Path.cwd(), Path(__file__).resolve().parents[2]]
    for cwd in candidates:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(cwd), stderr=subprocess.DEVNULL,
                text=True, timeout=5,
            ).strip()
            dirty = bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=str(cwd),
                stderr=subprocess.DEVNULL, text=True, timeout=5,
            ).strip())
            return {"git_commit": commit, "git_dirty": dirty}
        except (OSError, subprocess.SubprocessError):
            continue
    return {"git_commit": "UNKNOWN", "git_dirty": None}


def _cpu_model() -> str:
    """Platform API boş döndüğünde Linux cpuinfo'dan okunabilir bir ad üret."""
    model = platform.processor().strip()
    if model:
        return model
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name") and ":" in line:
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "UNKNOWN"


def _ram_total_bytes() -> Optional[int]:
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


def _runtime_metadata() -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_total_bytes": _ram_total_bytes(),
        "device": "cpu",
        "gpu": None,
        "cuda_device_name": None,
        "torch_version": None,
        "cuda_version": None,
        "cudnn_version": None,
    }
    try:
        import torch

        metadata["torch_version"] = getattr(torch, "__version__", None)
        metadata["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        try:
            metadata["cudnn_version"] = torch.backends.cudnn.version()
        except (AttributeError, RuntimeError):
            pass
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            metadata["device"] = f"cuda:{index}"
            metadata["gpu"] = torch.cuda.get_device_name(index)
            metadata["cuda_device_name"] = metadata["gpu"]
    except ImportError:
        pass
    return metadata


def seed_everything(seed: int) -> None:
    """Kullanılabilir RNG'leri aynı seed'e getir; eksik opsiyonel paketleri atla."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, path)


class _Tee:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@dataclass
class ExperimentRun:
    experiment_id: str
    directory: Path
    manifest: Dict[str, Any]
    _started_monotonic: float = field(default_factory=time.perf_counter, repr=False)

    @classmethod
    def create(
        cls,
        root: os.PathLike,
        config: Mapping[str, Any],
        seed: int,
        dataset_hash: str,
        parameters: Optional[Mapping[str, Any]] = None,
        model_path: Optional[os.PathLike] = None,
    ) -> "ExperimentRun":
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        existing = [
            int(path.name.split("-")[1])
            for path in root_path.glob("EXP-[0-9]*")
            if path.is_dir() and path.name.split("-")[1].isdigit()
        ]
        number = max(existing, default=0) + 1
        while True:
            experiment_id = f"EXP-{number:04d}"
            directory = root_path / experiment_id
            try:
                directory.mkdir()
                break
            except FileExistsError:  # başka süreç aynı kimliği atomik olarak aldı
                number += 1

        config_data = dict(config)
        config_path = directory / "config.yaml"
        # JSON, YAML 1.2'nin geçerli bir alt kümesidir. Böylece dosya standart
        # YAML araçlarıyla okunur ve manifest çekirdeği opsiyonel parser'a bağlanmaz.
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config_data, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")

        if model_path is None:
            model_hash = "NONE"
        else:
            model_hash = file_sha256(model_path)
        (directory / "model_hash.txt").write_text(model_hash + "\n", encoding="utf-8")
        (directory / "stdout.log").touch()

        parameter_metadata = dict(parameters or {})
        parameter_count = parameter_metadata.get(
            "parameter_count", parameter_metadata.get("physical_parameters")
        )
        manifest: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": int(seed),
            "dataset_hash": str(dataset_hash),
            "config_hash": canonical_hash(config_data),
            "parameters": parameter_metadata,
            "parameter_count": parameter_count,
            "model_hash": model_hash,
            "timings": {
                "total_seconds": None,
                "training_seconds": None,
                "inference_seconds": None,
            },
            "result": "RUNNING",
            **_git_metadata(),
            **_runtime_metadata(),
        }
        _atomic_json(directory / "manifest.json", manifest)
        return cls(experiment_id, directory, manifest)

    @contextlib.contextmanager
    def capture_stdout(self) -> Iterator[None]:
        """stdout'u hem terminale hem koşunun ``stdout.log`` dosyasına yaz."""
        with (self.directory / "stdout.log").open("a", encoding="utf-8") as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout, log)):
                yield

    def complete(self, results: Mapping[str, Any]) -> None:
        result_data = dict(results)
        _atomic_json(self.directory / "results.json", result_data)
        reported_timings = result_data.get("timings", {})
        if not isinstance(reported_timings, Mapping):
            reported_timings = {}
        self.manifest["timings"] = {
            "total_seconds": round(time.perf_counter() - self._started_monotonic, 6),
            "training_seconds": reported_timings.get("training_seconds"),
            "inference_seconds": reported_timings.get("inference_seconds"),
        }
        self.manifest["result"] = "COMPLETED"
        self.manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(self.directory / "manifest.json", self.manifest)

    def fail(self, error: BaseException) -> None:
        results = {"error_type": type(error).__name__, "error": str(error)}
        _atomic_json(self.directory / "results.json", results)
        self.manifest["timings"]["total_seconds"] = round(
            time.perf_counter() - self._started_monotonic, 6
        )
        self.manifest["result"] = "FAILED"
        self.manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(self.directory / "manifest.json", self.manifest)


@dataclass
class SeedSweepReport:
    seeds: List[int]
    experiment_ids: List[str]
    dataset_hash: str
    config_hash: str
    parameters: Dict[str, Any]
    manifests: List[Dict[str, Any]]
    results: List[Dict[str, Any]]
    aggregate: Dict[str, Dict[str, float]]
    deterministic_results: bool
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _numeric_leaves(value: Mapping[str, Any], prefix: str = "") -> Dict[str, float]:
    leaves: Dict[str, float] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            leaves.update(_numeric_leaves(item, path))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            leaves[path] = float(item)
    return leaves


def run_seed_sweep(
    callback: Callable[[int], Mapping[str, Any]],
    seeds: Sequence[int],
    root: os.PathLike,
    config: Mapping[str, Any],
    dataset_hash: str,
    parameters: Optional[Mapping[str, Any]] = None,
    model_path: Optional[os.PathLike] = None,
) -> SeedSweepReport:
    """Callback'i her seed için ayrı, tam manifestli deney olarak çalıştır."""
    normalized_seeds = [int(seed) for seed in seeds]
    if not normalized_seeds:
        raise ValueError("En az bir seed gerekli")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("Seed değerleri benzersiz olmalı")

    runs: List[ExperimentRun] = []
    results: List[Dict[str, Any]] = []
    for seed in normalized_seeds:
        seed_everything(seed)
        run = ExperimentRun.create(
            root=root, config=config, seed=seed, dataset_hash=dataset_hash,
            parameters=parameters, model_path=model_path,
        )
        runs.append(run)
        try:
            with run.capture_stdout():
                print(f"[{run.experiment_id}] seed={seed}")
                result = dict(callback(seed))
            run.complete(result)
            results.append(result)
        except BaseException as error:
            run.fail(error)
            raise

    flattened = [_numeric_leaves(result) for result in results]
    common_keys = set(flattened[0]).intersection(*(set(row) for row in flattened[1:]))
    aggregate: Dict[str, Dict[str, float]] = {}
    for key in sorted(common_keys):
        values = [row[key] for row in flattened]
        aggregate[key] = {
            # Tiny scientific losses (ör. 1e-11 NMSE) sıfıra yuvarlanmasın.
            "mean": round(statistics.fmean(values), 16),
            "std": round(statistics.pstdev(values), 16),
            "min": min(values),
            "max": max(values),
        }
    deterministic = all(result == results[0] for result in results[1:])
    return SeedSweepReport(
        seeds=normalized_seeds,
        experiment_ids=[run.experiment_id for run in runs],
        dataset_hash=str(dataset_hash),
        config_hash=canonical_hash(dict(config)),
        parameters=dict(parameters or {}),
        manifests=[dict(run.manifest) for run in runs],
        results=[dict(result) for result in results],
        aggregate=aggregate,
        deterministic_results=deterministic,
        note=(
            "Seedler arası özdeşlik yalnız tekrarlanabilirlik kontrolüdür; "
            "istatistiksel model kalitesi kanıtı değildir."
        ),
    )


__all__ = [
    "ExperimentRun",
    "SeedSweepReport",
    "canonical_hash",
    "file_sha256",
    "run_seed_sweep",
    "seed_everything",
]
