# -*- coding: utf-8 -*-
"""Kronecker Bilinear vs Dense Linear Comparison Test."""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from experiments.experience_loop.run_kronecker_vs_dense import run_kronecker_vs_dense_benchmark


def test_kronecker_vs_dense_param_efficiency():
    result = run_kronecker_vs_dense_benchmark(n=16, adim_sayisi=50)
    assert result["params_kron"] == result["params_dense"]
    assert result["virtual_ops_kron"] == 16**4
    # Kronecker loss should be significantly lower than param-matched dense on bilinear structure
    assert result["test_loss_kron"] < result["test_loss_dense"]
