# -*- coding: utf-8 -*-
"""Kronecker vs eşit parametreli rank-1 baseline testleri."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from experiments.experience_loop.run_kronecker_vs_dense import (  # noqa: E402
    run_kronecker_vs_dense_benchmark,
)
from hga.evaluation import (  # noqa: E402
    kronecker_capacity_contract,
    run_kronecker_dense_trial,
)


def test_kapasite_sozlesmesi_n4_parametre_degil():
    capacity = kronecker_capacity_contract(16)
    assert capacity["physical_parameter_budget_each"] == 2 * 16**2
    assert capacity["full_operator_entries_n4"] == 16**4
    assert capacity["operator_entries_are_parameters"] is False
    assert capacity["kronecker_max_effective_operator_rank"] == 16**2
    assert capacity["rank1_bottleneck_max_effective_operator_rank"] == 1


def test_kapasite_sozlesmesi_gecersiz_n_reddi():
    with pytest.raises(ValueError, match="n >= 2"):
        kronecker_capacity_contract(1)


def test_module_import_torch_yokken_sys_exit_yapmaz():
    code = "import experiments.experience_loop.run_kronecker_vs_dense; print('ok')"
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=KOK, capture_output=True, text=True, check=True,
    )
    assert completed.stdout.strip() == "ok"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch kurulu değil")
def test_kronecker_vs_dense_param_efficiency_and_counter_task():
    report = run_kronecker_dense_trial(
        n=8, steps=80, batch_size=16, test_samples=64, seed=42,
    )
    capacity = report["capacity"]
    assert capacity["physical_parameter_budget_each"] == 2 * 8**2
    assert capacity["full_operator_entries_n4"] == 8**4
    assert capacity["operator_entries_are_parameters"] is False
    assert report["fairness"]["same_physical_parameter_budget"]

    kron_task = report["tasks"]["kronecker_teacher"]
    rank1_task = report["tasks"]["rank1_teacher"]
    assert kron_task["winner_by_test_normalized_mse"] == "kronecker"
    assert rank1_task["winner_by_test_normalized_mse"] == "rank1_bottleneck"

    for task in report["tasks"].values():
        models = task["models"]
        assert models["kronecker"]["physical_parameters"] == 128
        assert models["rank1_bottleneck"]["physical_parameters"] == 128
        assert models["kronecker"]["optimization_stable"]
        assert models["rank1_bottleneck"]["optimization_stable"]
        assert models["kronecker"]["effective_operator_rank"] <= 64
        assert models["rank1_bottleneck"]["effective_operator_rank"] <= 1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch kurulu değil")
def test_legacy_wrapper_compatibility():
    result = run_kronecker_vs_dense_benchmark(n=8, adim_sayisi=20, seed=3)
    assert result["params_kron"] == result["params_dense"] == 128
    assert result["virtual_ops_kron"] == 8**4
    assert result["report"]["capacity"]["operator_entries_are_parameters"] is False


def test_kuratorlu_bes_seed_kronecker_raporu():
    path = Path(KOK) / "raporlar" / "kronecker_5seed_summary.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["seeds"] == [1, 2, 3, 4, 5]
    assert len(report["manifests"]) == len(report["results"]) == 5
    assert all(not manifest["git_dirty"] for manifest in report["manifests"])
    assert all(manifest["torch_version"].startswith("2.3.1")
               for manifest in report["manifests"])
    kron_nmse = report["aggregate"][
        "tasks.kronecker_teacher.models.kronecker.test.normalized_mse"
    ]["mean"]
    rank1_on_kron = report["aggregate"][
        "tasks.kronecker_teacher.models.rank1_bottleneck.test.normalized_mse"
    ]["mean"]
    rank1_nmse = report["aggregate"][
        "tasks.rank1_teacher.models.rank1_bottleneck.test.normalized_mse"
    ]["mean"]
    kron_on_rank1 = report["aggregate"][
        "tasks.rank1_teacher.models.kronecker.test.normalized_mse"
    ]["mean"]
    assert 0.0 < kron_nmse < 1e-9
    assert rank1_on_kron > 0.9
    assert 0.0 < rank1_nmse < 1e-4
    assert kron_on_rank1 > 0.9


# Torch KURULU olsa bile "torch yok" yolunu gerçekten koşturmak için alt
# süreçte torch'u görünmez yapan meta path finder. Böylece bu sözleşme testi
# ortama bağlı olarak atlanmaz (skip yerine gerçek doğrulama).
#
# Önemli: gerçek "kurulu değil" ortamında `importlib.util.find_spec("torch")`
# ImportError ATMAZ, None döner; `import torch` ise ModuleNotFoundError verir.
# Simülasyon bu semantiği birebir taklit etmelidir, yoksa test gerçekte
# olmayan bir davranışı doğrular.
_TORCH_ENGELLE = """
import importlib.machinery
import sys

# Torch'u diskte "yokmuş" gibi göstermenin doğru yolu: onu bulan PathFinder'ı
# sarmalayıp None döndürmek. Bu tek müdahale her iki semantiği de otomatik
# olarak doğru verir:
#   importlib.util.find_spec("torch") -> None
#   import torch                      -> ModuleNotFoundError
_gercek_find_spec = importlib.machinery.PathFinder.find_spec


def _gizleyen_find_spec(name, path=None, target=None):
    if name == "torch" or name.startswith("torch."):
        return None
    return _gercek_find_spec(name, path, target)


importlib.machinery.PathFinder.find_spec = staticmethod(_gizleyen_find_spec)
for _ad in [m for m in sys.modules if m == "torch" or m.startswith("torch.")]:
    del sys.modules[_ad]
"""


def test_torchsuz_ortam_simulasyonu_gercekten_torchsuz():
    """Yardımcı engelleyicinin kendisi çalışıyor mu? (testin testi)"""
    kod = _TORCH_ENGELLE + """
import importlib.util

assert importlib.util.find_spec("torch") is None, "find_spec None dönmeli"
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    print("YOK")
else:
    print("BULUNDU")
"""
    completed = subprocess.run(
        [sys.executable, "-c", kod], cwd=KOK, capture_output=True, text=True,
        check=True,
    )
    assert completed.stdout.strip() == "YOK"


def test_torch_yokken_fonksiyon_acik_import_hatasi_verir():
    """run_kronecker_dense_trial torch yokken sessiz kalmaz, ImportError atar."""
    kod = _TORCH_ENGELLE + """
from hga.evaluation import run_kronecker_dense_trial

try:
    run_kronecker_dense_trial(n=4, steps=1)
except ImportError as hata:
    assert "PyTorch" in str(hata), str(hata)
    print("IMPORT_ERROR_OK")
else:
    print("HATA_YOK")
"""
    completed = subprocess.run(
        [sys.executable, "-c", kod], cwd=KOK, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("IMPORT_ERROR_OK")


def test_torch_yokken_cli_acik_hata_verir_ve_manifest_yazmaz(tmp_path):
    """CLI torch yokken çıkış kodu != 0 döner ve yarım EXP dizini bırakmaz."""
    kod = _TORCH_ENGELLE + f"""
import sys
sys.argv = ["hga", "kronecker-benchmark", "--experiment-root", {str(tmp_path)!r}]
from hga.__main__ import main
main(sys.argv[1:])
"""
    completed = subprocess.run(
        [sys.executable, "-c", kod], cwd=KOK, capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert "PyTorch gerekli" in completed.stderr
    assert not list(tmp_path.glob("EXP-*"))
