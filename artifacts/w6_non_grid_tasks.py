# -*- coding: utf-8 -*-
"""
W6: Grid olmayan gorevlerde Kronecker testi.

Sirkuler mantik riskini kir:
  - Grid yapili gorevlerde Kronecker iyi (onceki bulgu)
  - Grid OLMAYAN gorevlerde Kronecker nasil?

Gorevler:
  1. graph_task    — Dugum ozellikleri + kenar yapisi (Kronecker uygun degil)
  2. sequence_task — Sirali (1D) dizi, son N'in ortalamasi
  3. random_factor — Rastgele faktorize matris (kismen uygun)
  4. grid_baseline — Grid yapili kontrol (Kronecker beklenir)

Her gorev icin: Dense vs Flat Kronecker (20 seed).
"""
import json
import statistics
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from artifacts.task_structure_metrics import (
    autocorrelation_2d, block_structure_score,
)
from artifacts.w5_proper_6tasks import (
    DenseMLP, FlatKroneckerNet, count_params, train,
)


# ─── Grid Olmayan Görevler ──────────────────────────────────────────

def task_sequence(n=16, n_train=2000, n_test=500, seed=42):
    """1D sirali gorev: 256-uzunlukta dizi, son 16 elemanin ortalamasi
    hangi ceyrekte? (Grid yapisi YOK, 16x16'ya reshape ediliyor ama
    dizi dogal sirali — 2D yapi anlamsiz.)"""
    rng = np.random.RandomState(seed)
    L = n * n  # 256

    def gen(N):
        X = rng.randn(N, L).astype(np.float32)
        # Son 64 elemani 4 parcaya bol, hangisinin ortalamasi en yuksek?
        tail = X[:, -64:].reshape(N, 4, 16)
        means = tail.mean(axis=2)
        y = means.argmax(1).astype(np.int64)
        return X.reshape(N, n, n), y

    X_tr, y_tr = gen(n_train)
    X_te, y_te = gen(n_test)
    return {
        "x_train": torch.from_numpy(X_tr),
        "y_train": torch.from_numpy(y_tr),
        "x_test":  torch.from_numpy(X_te),
        "y_test":  torch.from_numpy(y_te),
        "num_classes": 4,
    }


def task_graph(n=16, n_train=2000, n_test=500, seed=42):
    """Graf gorevi: 16 dugum, her dugum 16 ozellik (toplam 256).
    Sinif = en yuksek dereceli dugumun indeksi mod 4 (graf topolojisi).
    Grid yapisi YOK."""
    rng = np.random.RandomState(seed)

    def gen(N):
        X = rng.randn(N, n, n).astype(np.float32)  # (N, 16, 16) = dugum x ozellik
        # Her ornek icin rastgele graf: her dugumun ortalama ozelligi
        # Sinif: hangi ceyrek (4 dugumluk gruplar) en yuksek ortalamaya sahip?
        quarter_means = X.reshape(N, 4, 4, n).mean(axis=(2, 3))
        y = quarter_means.argmax(1).astype(np.int64)
        return X, y

    X_tr, y_tr = gen(n_train)
    X_te, y_te = gen(n_test)
    return {
        "x_train": torch.from_numpy(X_tr),
        "y_train": torch.from_numpy(y_tr),
        "x_test":  torch.from_numpy(X_te),
        "y_test":  torch.from_numpy(y_te),
        "num_classes": 4,
    }


def task_grid_baseline(n=16, n_train=2000, n_test=500, seed=42,
                       noise: float = 0.3):
    """Grid yapili kontrol gorevi (onceki deneyle ayni)."""
    rng = np.random.RandomState(seed)
    bs = 4
    def gen(N):
        powers = rng.rand(N, 4, 4).astype(np.float32)
        X = np.repeat(np.repeat(powers, bs, axis=1), bs, axis=2)
        if noise > 0:
            X = X + noise * rng.randn(*X.shape).astype(np.float32)
        q = np.zeros((N, 2, 2), dtype=np.float32)
        for i in range(2):
            for j in range(2):
                q[:, i, j] = powers[:, i*2:(i+1)*2, j*2:(j+1)*2].sum(axis=(1, 2))
        y = q.reshape(N, 4).argmax(1).astype(np.int64)
        return X, y
    X_tr, y_tr = gen(n_train)
    X_te, y_te = gen(n_test)
    return {
        "x_train": torch.from_numpy(X_tr),
        "y_train": torch.from_numpy(y_tr),
        "x_test":  torch.from_numpy(X_te),
        "y_test":  torch.from_numpy(y_te),
        "num_classes": 4,
    }


def task_random_factor(n=16, n_train=2000, n_test=500, seed=42):
    """Rastgele faktorize matris: Y = A @ X @ B (lineer faktor).
    Kronecker tam bu yapiyi temsil eder, ama faktorler RASTGELE."""
    rng = np.random.RandomState(seed)
    A = rng.randn(n, n).astype(np.float32) / np.sqrt(n)
    B = rng.randn(n, n).astype(np.float32) / np.sqrt(n)

    def gen(N):
        X = rng.randn(N, n, n).astype(np.float32)
        Y = np.einsum("ij,njk,kl->nil", A, X, B)
        # Sinif: Y'nin ceyrek toplamlari argmax
        q = Y.reshape(N, 2, n // 2, 2, n // 2).sum(axis=(2, 4))
        y = q.reshape(N, 4).argmax(1).astype(np.int64)
        return X, y

    X_tr, y_tr = gen(n_train)
    X_te, y_te = gen(n_test)
    return {
        "x_train": torch.from_numpy(X_tr),
        "y_train": torch.from_numpy(y_tr),
        "x_test":  torch.from_numpy(X_te),
        "y_test":  torch.from_numpy(y_te),
        "num_classes": 4,
    }


# ─── Ana Deney ───────────────────────────────────────────────────────

def run_task(task_fn, name: str, seeds, epochs: int = 60,
             n_train: int = 2000, n_test: int = 500) -> Dict[str, Any]:
    """Tek gorev icin 20 seed × 2 model."""
    rows = []
    for s in seeds:
        task = task_fn(n_train=n_train, n_test=n_test, seed=s)
        n = task["x_train"].shape[1]
        nc = task["num_classes"]

        dense = DenseMLP(n, hidden=n * n // 4, num_classes=nc)
        dense_acc = train(dense, task, epochs=epochs, seed=s)
        dp = count_params(dense)

        flat = FlatKroneckerNet(n, depth=1, num_classes=nc)
        flat_acc = train(flat, task, epochs=epochs, seed=s)
        fp = count_params(flat)

        rows.append({
            "seed": s, "dense_acc": dense_acc, "flat_acc": flat_acc,
            "winner": "flat" if flat_acc >= dense_acc else "dense",
        })

    dm = statistics.mean([r["dense_acc"] for r in rows])
    ds = statistics.stdev([r["dense_acc"] for r in rows])
    fm = statistics.mean([r["flat_acc"] for r in rows])
    fs = statistics.stdev([r["flat_acc"] for r in rows])
    fw = sum(1 for r in rows if r["winner"] == "flat")

    # Yapi metrikleri (kucuk ornek)
    t2 = task_fn(n_train=200, n_test=50, seed=1)
    X = t2["x_train"].numpy()
    side = X.shape[1]
    ac = autocorrelation_2d(X)
    bs = block_structure_score(X, block_size=max(2, side // 4))

    return {
        "name": name, "n": n, "num_classes": nc,
        "autocorr": ac, "block_score": bs,
        "dense_mean": dm, "dense_std": ds, "dense_params": dp,
        "flat_mean": fm, "flat_std": fs, "flat_params": fp,
        "flat_wins": fw, "n_seeds": len(seeds),
        "winner_mean": "flat" if fm >= dm else "dense",
    }


if __name__ == "__main__":
    SEEDS = list(range(1, 51))
    TASKS = [
        ("sequence",           task_sequence,       100),
        ("graph",              task_graph,           60),
        ("random_factor",      task_random_factor,   60),
        ("grid_baseline",      task_grid_baseline,  100),
    ]

    print("═══ W6: Grid vs Non-Grid Gorevler ═══\n")
    results = []
    for name, fn, epochs in TASKS:
        r = run_task(fn, name, SEEDS, epochs=epochs)
        results.append(r)
        print(f"{name:20s}  autocorr={r['autocorr']:+.4f}  "
              f"block={r['block_score']:+.4f}")
        print(f"  Dense({r['dense_params']:>7,})={r['dense_mean']:.4f}"
              f"±{r['dense_std']:.4f}  "
              f"Flat({r['flat_params']:>6,})={r['flat_mean']:.4f}"
              f"±{r['flat_std']:.4f}  "
              f"-> {r['winner_mean']} ({r['flat_wins']}/{r['n_seeds']})")
        print()

    # Ozet: grid vs non-grid
    print("═══ Ozet ═══")
    grid = next(r for r in results if r["name"] == "grid_baseline")
    print(f"GRID (baseline):        Dense={grid['dense_mean']:.4f}  "
          f"Flat={grid['flat_mean']:.4f}  "
          f"Flat wins: {grid['flat_wins']}/{grid['n_seeds']}")
    for r in results:
        if r["name"] == "grid_baseline":
            continue
        diff = r["flat_mean"] - r["dense_mean"]
        print(f"NON-GRID ({r['name']:15s}):  Dense={r['dense_mean']:.4f}  "
              f"Flat={r['flat_mean']:.4f}  "
              f"Flat wins: {r['flat_wins']}/{r['n_seeds']}  (delta={diff:+.4f})")

    Path("artifacts/w6_non_grid_50seed_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print("\nOK — sonuclar kaydedildi")
