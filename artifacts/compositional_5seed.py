# -*- coding: utf-8 -*-
"""
Compositional gorevde 5 seed × 3 mimari.

Onceki tek-seed sonuc: flat_kronecker 8x az paramla dense ile esit.
Bu sonucu istatistiksel olarak dogrula.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from hga.evaluation.hierarchical_compositional import (
    compare_on_hierarchical,
)


def main():
    seeds = [1, 2, 3, 4, 5]
    all_runs: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"\n═══ Seed {seed} ═══")
        r = compare_on_hierarchical(
            n_base=16, num_classes=4,
            n_train=2000, n_test=500,
            epochs=100, seed=seed,
        )
        all_runs.append(r)

    # Ozet
    print("\n═══ Ozet (5 seed) ═══")
    by_name: Dict[str, Dict[str, List[float]]] = {}
    for r in all_runs:
        for m in r["results"]:
            n = m["name"]
            by_name.setdefault(n, {"acc": [], "params": [], "time": []})
            by_name[n]["acc"].append(m["test_acc"])
            by_name[n]["params"].append(m["params"])
            by_name[n]["time"].append(m["wall_time_sec"])

    print(f"{'model':26s} {'params':>10s} {'test_acc':>18s} {'time_s':>8s}")
    print("-" * 65)
    for name, d in by_name.items():
        acc_mean = statistics.mean(d["acc"])
        acc_std = statistics.stdev(d["acc"]) if len(d["acc"]) > 1 else 0
        print(f"{name:26s} {int(statistics.mean(d['params'])):>10,} "
              f"{acc_mean:>10.4f} ± {acc_std:.4f} "
              f"{statistics.mean(d['time']):>8.2f}")

    Path("artifacts/compositional_5seed_results.json").write_text(
        json.dumps(all_runs, indent=2), encoding="utf-8")
    print("\nOK — sonuclar kaydedildi")


if __name__ == "__main__":
    main()
