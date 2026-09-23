# -*- coding: utf-8 -*-
"""
Flat Kronecker cokusunu onlemek icin LayerNorm + residual varyantlari.

Onceki sonuc (hierarchical_param_matched.py):
  flat_K168 (168 katman) -> 0.257 test acc (coktu)

Hipotez: normalizasyon eksik -> bilgi yok oluyor.
Cozum: LayerNorm + residual baglanti.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from artifacts.hierarchical_param_matched import (
    BilinearLayer, DenseMLP, HierarchicalKroneckerNet,
    make_task, count_params, TrainResult, train_model,
)


class FlatKroneckerNorm(nn.Module):
    """Flat + LayerNorm + residual."""

    def __init__(self, n: int, depth: int, num_classes: int):
        super().__init__()
        self.n = n
        self.layers = nn.ModuleList([BilinearLayer(n, n) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm((n, n)) for _ in range(depth)])
        self.head = nn.Linear(n * n, num_classes)

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            h = F.silu(layer(x))
            h = norm(h)
            x = x + h  # residual
        return self.head(x.flatten(1))


class FlatKroneckerNormNoRes(nn.Module):
    """Flat + LayerNorm, residual YOK (kontrol grubu)."""

    def __init__(self, n: int, depth: int, num_classes: int):
        super().__init__()
        self.layers = nn.ModuleList([BilinearLayer(n, n) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm((n, n)) for _ in range(depth)])
        self.head = nn.Linear(n * n, num_classes)

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = F.silu(layer(x))
            x = norm(x)
        return self.head(x.flatten(1))


class HierarchicalKroneckerNorm(nn.Module):
    """Hierarchical + LayerNorm."""

    def __init__(self, n_base: int, depth: int, expansion: int, num_classes: int):
        super().__init__()
        sizes = [n_base * (expansion ** i) for i in range(depth + 1)]
        self.layers = nn.ModuleList([
            BilinearLayer(sizes[i], sizes[i + 1]) for i in range(depth)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm((s, s)) for s in sizes[1:]])
        self.head = nn.Linear(sizes[-1] ** 2, num_classes)

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = F.silu(layer(x))
            x = norm(x)
        return self.head(x.flatten(1))


def run_experiment(seed: int = 42, n_base: int = 8,
                   epochs: int = 300, num_classes: int = 4,
                   n_train: int = 2000, n_test: int = 500) -> Dict[str, Any]:
    """Flat cokusu LayerNorm+residual ile duzeliyor mu?"""
    task = make_task(n_base, num_classes, n_train, n_test, seed=seed)
    results: List[TrainResult] = []

    K = 16  # derin zincir — onceki deneyde flat_K168 cokmustu

    # 1) Baseline: duz flat (cokuyor)
    results.append(train_model(
        FlatKroneckerNorm(n_base, K, num_classes) if False else
        __import__("artifacts.hierarchical_param_matched", fromlist=["FlatKroneckerNet"]).FlatKroneckerNet(n_base, K, num_classes),
        task, epochs=epochs, seed=seed, name=f"flat_K{K}_baseline"))

    # 2) + LayerNorm (residual yok)
    results.append(train_model(
        FlatKroneckerNormNoRes(n_base, K, num_classes),
        task, epochs=epochs, seed=seed, name=f"flat_K{K}_norm"))

    # 3) + LayerNorm + residual
    results.append(train_model(
        FlatKroneckerNorm(n_base, K, num_classes),
        task, epochs=epochs, seed=seed, name=f"flat_K{K}_norm_res"))

    # 4) Hierarchical (referans)
    results.append(train_model(
        HierarchicalKroneckerNet(n_base, 3, 2, num_classes),
        task, epochs=epochs, seed=seed, name="hier_K3_exp2"))

    # 5) Hierarchical + LayerNorm
    results.append(train_model(
        HierarchicalKroneckerNorm(n_base, 3, 2, num_classes),
        task, epochs=epochs, seed=seed, name="hier_K3_exp2_norm"))

    # 6) Dense (ust sinir)
    results.append(train_model(
        DenseMLP(n_base, hidden=128, num_classes=num_classes),
        task, epochs=epochs, seed=seed, name="dense_K3"))

    return {"seed": seed, "n_base": n_base, "K": K,
            "results": [asdict(r) for r in results]}


if __name__ == "__main__":
    all_runs = []
    for seed in (1, 2, 3, 4, 5):
        r = run_experiment(seed=seed)
        all_runs.append(r)
        print(f"\n═══ Seed {seed} ═══")
        for m in r["results"]:
            print(f"  {m['name']:25s} params={m['params']:>8,} "
                  f"train={m['train_acc']:.4f} test={m['test_acc']:.4f}")

    # Ozet
    import statistics
    print("\n═══ Ozet (5 seed) ═══")
    by_name: Dict[str, Dict[str, List[float]]] = {}
    for r in all_runs:
        for m in r["results"]:
            by_name.setdefault(m["name"], {"acc": [], "params": []})
            by_name[m["name"]]["acc"].append(m["test_acc"])
            by_name[m["name"]]["params"].append(m["params"])
    for name, d in by_name.items():
        print(f"  {name:25s} params={int(statistics.mean(d['params'])):>8,} "
              f"test={statistics.mean(d['acc']):.4f} ± {statistics.stdev(d['acc']):.4f}")

    Path("artifacts/hierarchical_layernorm_results.json").write_text(
        json.dumps(all_runs, indent=2), encoding="utf-8")
    print("\nOK — sonuclar kaydedildi")
