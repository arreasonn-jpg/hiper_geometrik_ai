# -*- coding: utf-8 -*-
"""Kronecker vs eşit parametreli rank-1 baseline testleri."""
import importlib.util
import os
import subprocess
import sys

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


def test_torch_yokken_fonksiyon_ve_cli_acik_hata_verir(tmp_path):
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("PyTorch mevcut")
    with pytest.raises(ImportError, match="PyTorch"):
        run_kronecker_dense_trial(n=4, steps=1)
    completed = subprocess.run(
        [sys.executable, "-m", "hga", "kronecker-benchmark",
         "--experiment-root", str(tmp_path)],
        cwd=KOK, capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert "PyTorch gerekli" in completed.stderr
    assert not list(tmp_path.glob("EXP-*"))
