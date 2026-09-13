# -*- coding: utf-8 -*-
"""Kronecker vs eşit bütçeli rank-1 baseline deney giriş noktası.

Kanonik uygulama ``hga.evaluation.kronecker`` içindedir. Bu dosya eski import
ve komut satırı yolunu koruyan ince wrapper'dır; PyTorch yokken import sırasında
``sys.exit`` çağırmaz.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.evaluation.kronecker import run_kronecker_dense_trial  # noqa: E402


def run_kronecker_vs_dense_benchmark(
    n: int = 32,
    adim_sayisi: int = 150,
    seed: int = 42,
) -> Dict[str, Any]:
    """Eski API'yi koru ve v2 bilimsel raporunu ``report`` alanında döndür."""
    report = run_kronecker_dense_trial(n=n, steps=adim_sayisi, seed=seed)
    task = report["tasks"]["kronecker_teacher"]["models"]
    kron = task["kronecker"]
    dense = task["rank1_bottleneck"]
    result = {
        "n": n,
        "params_kron": kron["physical_parameters"],
        "params_dense": dense["physical_parameters"],
        "virtual_ops_kron": report["capacity"]["full_operator_entries_n4"],
        "train_loss_kron": kron["final_step_loss"],
        "train_loss_dense": dense["final_step_loss"],
        "test_loss_kron": kron["test"]["mse"],
        "test_loss_dense": dense["test"]["mse"],
        "time_kron_ms": kron["training_seconds"] * 1000,
        "time_dense_ms": dense["training_seconds"] * 1000,
        "report": report,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        run_kronecker_vs_dense_benchmark(args.n, args.steps, args.seed)
    except ImportError as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
