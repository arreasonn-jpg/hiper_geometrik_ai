# -*- coding: utf-8 -*-
"""
W5 celsiki cozum: gurultu seviyesi vs Kronecker zaferi.

3 kosul: noise in {0.0, 0.3, 0.6}
20 seed, ayni gorev (block compositional), ayni modeller.
"""
import json
import statistics
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from artifacts.w5_proper_6tasks import (
    DenseMLP, FlatKroneckerNet, count_params, train,
)


def task_block_noise(n=16, n_train=2000, n_test=500, seed=42,
                     noise: float = 0.0):
    """Block compositional + ayarlanabilir gurultu."""
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


def run_condition(noise: float, seeds, epochs=100,
                  n_train=2000, n_test=500):
    rows = []
    for s in seeds:
        task = task_block_noise(n_train=n_train, n_test=n_test,
                                 seed=s, noise=noise)
        n = 16; nc = 4
        dense = DenseMLP(n, hidden=n * n // 4, num_classes=nc)
        dense_acc = train(dense, task, epochs=epochs, seed=s)
        dense_p = count_params(dense)

        flat = FlatKroneckerNet(n, depth=1, num_classes=nc)
        flat_acc = train(flat, task, epochs=epochs, seed=s)
        flat_p = count_params(flat)

        rows.append({
            "seed": s, "dense_acc": dense_acc, "flat_acc": flat_acc,
            "winner": "flat" if flat_acc >= dense_acc else "dense",
        })
    return {
        "noise": noise,
        "dense_mean": statistics.mean([r["dense_acc"] for r in rows]),
        "dense_std": statistics.stdev([r["dense_acc"] for r in rows]),
        "flat_mean": statistics.mean([r["flat_acc"] for r in rows]),
        "flat_std": statistics.stdev([r["flat_acc"] for r in rows]),
        "flat_wins": sum(1 for r in rows if r["winner"] == "flat"),
        "n_seeds": len(seeds),
        "dense_params": dense_p, "flat_params": flat_p,
    }


if __name__ == "__main__":
    SEEDS = list(range(1, 21))
    print("═══ Gurultu Ablasyonu — 3 kosul, 20 seed ═══\n")
    results = []
    for noise in (0.0, 0.15, 0.3, 0.45, 0.6):
        r = run_condition(noise, SEEDS)
        results.append(r)
        print(f"noise={noise:.1f}  Dense={r['dense_mean']:.4f}±{r['dense_std']:.4f}  "
              f"Flat={r['flat_mean']:.4f}±{r['flat_std']:.4f}  "
              f"Flat kazandi: {r['flat_wins']}/{r['n_seeds']}")

    # Trend
    print("\n═══ Trend ═══")
    noises = [r["noise"] for r in results]
    win_rates = [r["flat_wins"] / r["n_seeds"] for r in results]
    from scipy.stats import pearsonr
    if len(set(win_rates)) > 1:
        r_tr, p_tr = pearsonr(noises, win_rates)
        print(f"Gurultu vs Flat kazanma orani: r={r_tr:+.3f}, p={p_tr:.4f}")
    else:
        print("Kazanma orani sabit — trend yok")

    Path("artifacts/w5_noise_ablation_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print("\nOK — sonuclar kaydedildi")
