# -*- coding: utf-8 -*-
"""
W5 duzgun: 6 GERCEKTEN farkli gorev + 20 seed × 2 mimari.

Onceki hata: simple_compositional ve image_compositional ayni
ureticiden geliyordu. Simdi 6 farkli gorev tanimliyoruz.

Gorevler (artacak otokorelasyon sirasina gore):
  1. random_linear        — rastgele W, ~0 otokorelasyon
  2. permuted_mnist       — piksel permutasyonu, ~0
  3. noise_mnist          — salt-pepper, dusuk
  4. simple_block         — 4x4 sabit bloklar, orta-yuksek
  5. grid_correlated      — blok + uzamsal korelasyon, yuksek
  6. fashion_mnist        — gercek goruntu, yuksek
"""
import gzip
import json
import struct
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from artifacts.task_structure_metrics import (
    autocorrelation_2d, block_structure_score,
)


DATA_MNIST = Path("data/mnist")
DATA_FASHION = Path("data/fashion_mnist")


def load_idx(images_path, labels_path):
    with gzip.open(images_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
    with gzip.open(labels_path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return images, labels


def load_n(data_dir: Path, n: int, seed: int, which: str = "train"):
    img = data_dir / f"{'train' if which == 'train' else 't10k'}-images.gz"
    lbl = data_dir / f"{'train' if which == 'train' else 't10k'}-labels.gz"
    if not img.exists():
        img = data_dir / f"{'train' if which == 'train' else 'test'}-images.gz"
        lbl = data_dir / f"{'train' if which == 'train' else 'test'}-labels.gz"
    X, y = load_idx(img, lbl)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    return X[idx].astype(np.float32) / 255.0, y[idx].astype(np.int64)


# ─── 6 Görev Üreticisi ───────────────────────────────────────────────

def task_random_linear(n_train=2000, n_test=500, seed=42):
    """Rastgele lineer ogretmen: y = argmax(W @ x)."""
    rng = np.random.RandomState(seed)
    n_features = 64
    W = rng.randn(10, n_features) / np.sqrt(n_features)
    def gen(nn_):
        X = rng.randn(nn_, n_features).astype(np.float32)
        y = (W @ X.T).argmax(0).astype(np.int64)
        return X, y
    X_tr, y_tr = gen(n_train)
    X_te, y_te = gen(n_test)
    side = int(np.sqrt(n_features))
    return {
        "x_train": torch.from_numpy(X_tr.reshape(-1, side, side)),
        "y_train": torch.from_numpy(y_tr),
        "x_test":  torch.from_numpy(X_te.reshape(-1, side, side)),
        "y_test":  torch.from_numpy(y_te),
        "num_classes": 10,
    }


def task_permuted_mnist(n_train=2000, n_test=500, seed=42):
    rng = np.random.RandomState(seed)
    X_tr, y_tr = load_n(DATA_MNIST, n_train, seed, "train")
    X_te, y_te = load_n(DATA_MNIST, n_test, seed + 1, "test")
    perm = rng.permutation(28 * 28)
    X_tr = X_tr.reshape(len(X_tr), -1)[:, perm].reshape(-1, 28, 28)
    X_te = X_te.reshape(len(X_te), -1)[:, perm].reshape(-1, 28, 28)
    return {
        "x_train": torch.from_numpy(X_tr.copy()),
        "y_train": torch.from_numpy(y_tr),
        "x_test":  torch.from_numpy(X_te.copy()),
        "y_test":  torch.from_numpy(y_te),
        "num_classes": 10,
    }


def task_noise_mnist(n_train=2000, n_test=500, seed=42):
    rng = np.random.RandomState(seed)
    X_tr, y_tr = load_n(DATA_MNIST, n_train, seed, "train")
    X_te, y_te = load_n(DATA_MNIST, n_test, seed + 1, "test")
    def noise(X):
        m = rng.rand(*X.shape) < 0.3
        s = rng.rand(*X.shape) < 0.5
        Z = X.copy()
        Z[m & s] = 1.0; Z[m & ~s] = 0.0
        return Z
    return {
        "x_train": torch.from_numpy(noise(X_tr)),
        "y_train": torch.from_numpy(y_tr),
        "x_test":  torch.from_numpy(noise(X_te)),
        "y_test":  torch.from_numpy(y_te),
        "num_classes": 10,
    }


def task_simple_block(n=16, n_train=2000, n_test=500, seed=42):
    """4x4 sabit bloklar, her blokta sinyal gucu, ceyrek argmax."""
    rng = np.random.RandomState(seed)
    bs = 4
    def gen(N):
        powers = rng.rand(N, 4, 4).astype(np.float32)
        # 4x4 bloklar → 16x16 goruntu
        X = np.repeat(np.repeat(powers, bs, axis=1), bs, axis=2)
        # Sinif: hangi ceyrek en guclu
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


def task_grid_correlated(n=16, n_train=2000, n_test=500, seed=42):
    """Blok + uzamsal korelasyon + gurultu."""
    rng = np.random.RandomState(seed)
    bs = 4
    def gen(N):
        # Temel blok gucleri + komsu korelasyon
        powers = rng.rand(N, 4, 4).astype(np.float32)
        # Yumusatma: komsu bloklarla karistir
        sm = np.zeros_like(powers)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                sm += np.roll(np.roll(powers, di, axis=1), dj, axis=2)
        powers = sm / 9.0
        X = np.repeat(np.repeat(powers, bs, axis=1), bs, axis=2)
        X = X + 0.3 * rng.randn(*X.shape).astype(np.float32)
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


def task_fashion_mnist(n_train=2000, n_test=500, seed=42):
    X_tr, y_tr = load_n(DATA_FASHION, n_train, seed, "train")
    X_te, y_te = load_n(DATA_FASHION, n_test, seed + 1, "test")
    return {
        "x_train": torch.from_numpy(X_tr),
        "y_train": torch.from_numpy(y_tr),
        "x_test":  torch.from_numpy(X_te),
        "y_test":  torch.from_numpy(y_te),
        "num_classes": 10,
    }


# ─── Modeller (basit, ortak) ─────────────────────────────────────────

class BilinearLayer(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.A = nn.Parameter(torch.empty(n_out, n_in))
        self.B = nn.Parameter(torch.empty(n_in, n_out))
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)
    def forward(self, x):
        return torch.einsum("oi,bij,jk->bok", self.A, x, self.B)


class DenseMLP(nn.Module):
    def __init__(self, n, hidden, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(n * n, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, num_classes)
    def forward(self, x):
        x = x.flatten(1)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        return self.head(x)


class FlatKroneckerNet(nn.Module):
    def __init__(self, n, depth, num_classes):
        super().__init__()
        self.layers = nn.ModuleList([BilinearLayer(n, n) for _ in range(depth)])
        self.head = nn.Linear(n * n, num_classes)
    def forward(self, x):
        for layer in self.layers:
            x = F.silu(layer(x))
        return self.head(x.flatten(1))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train(model, task, epochs=100, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X_tr, y_tr = task["x_train"], task["y_train"]
    X_te, y_te = task["x_test"], task["y_test"]
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(X_tr), y_tr)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        te_acc = (model(X_te).argmax(1) == y_te).float().mean().item()
    return te_acc


def run_seed(task_fn, seed, n_train, n_test, epochs, hidden_mult=4):
    """Tek seed icin: task uret, iki modeli egit, kazanan dondur."""
    task = task_fn(n_train=n_train, n_test=n_test, seed=seed)
    n = task["x_train"].shape[1]  # 8, 16, veya 28
    nc = task["num_classes"]

    # Dense
    dense = DenseMLP(n, hidden=n * n // hidden_mult, num_classes=nc)
    dense_acc = train(dense, task, epochs=epochs, seed=seed)
    dense_p = count_params(dense)

    # Flat Kronecker K=1
    flat = FlatKroneckerNet(n, depth=1, num_classes=nc)
    flat_acc = train(flat, task, epochs=epochs, seed=seed)
    flat_p = count_params(flat)

    return {
        "seed": seed,
        "dense_acc": dense_acc, "dense_params": dense_p,
        "flat_acc": flat_acc, "flat_params": flat_p,
        "winner": "flat" if flat_acc >= dense_acc else "dense",
    }


# ─── Ana ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scipy.stats import pearsonr
    import statistics

    TASKS = {
        "random_linear":     (task_random_linear,     100),
        "permuted_mnist":    (task_permuted_mnist,     60),
        "noise_mnist":       (task_noise_mnist,        60),
        "simple_block":      (task_simple_block,      100),
        "grid_correlated":   (task_grid_correlated,   100),
        "fashion_mnist":     (task_fashion_mnist,      60),
    }

    SEEDS = list(range(1, 21))  # 20 seed
    N_TRAIN = 2000
    N_TEST = 500

    summary = {}
    for tname, (fn, epochs) in TASKS.items():
        print(f"\n═══ {tname} ({epochs} epoch, {len(SEEDS)} seed) ═══")
        rows = []
        for s in SEEDS:
            r = run_seed(fn, s, N_TRAIN, N_TEST, epochs)
            rows.append(r)
        dense_mean = statistics.mean([r["dense_acc"] for r in rows])
        flat_mean = statistics.mean([r["flat_acc"] for r in rows])
        dense_std = statistics.stdev([r["dense_acc"] for r in rows])
        flat_std = statistics.stdev([r["flat_acc"] for r in rows])
        flat_wins = sum(1 for r in rows if r["winner"] == "flat")
        dp = rows[0]["dense_params"]
        fp = rows[0]["flat_params"]

        # Otokorelasyon
        task = fn(n_train=200, n_test=50, seed=1)
        X = task["x_train"].numpy()
        side = X.shape[1]
        ac = autocorrelation_2d(X)
        bs = block_structure_score(X, block_size=max(2, side // 4))

        summary[tname] = {
            "autocorr": ac, "block_score": bs,
            "dense_mean": dense_mean, "dense_std": dense_std,
            "flat_mean": flat_mean, "flat_std": flat_std,
            "dense_params": dp, "flat_params": fp,
            "flat_wins": flat_wins, "n_seeds": len(SEEDS),
            "winner_mean": "flat" if flat_mean >= dense_mean else "dense",
        }
        print(f"  autocorr={ac:+.4f}  block={bs:+.4f}")
        print(f"  Dense({dp:,})={dense_mean:.4f}±{dense_std:.4f}  "
              f"Flat({fp:,})={flat_mean:.4f}±{flat_std:.4f}")
        print(f"  Flat kazandi: {flat_wins}/{len(SEEDS)} seed")

    # Korelasyon
    print("\n═══ Korelasyon Analizi ═══")
    autocorrs = [s["autocorr"] for s in summary.values()]
    blocks = [s["block_score"] for s in summary.values()]
    # Kazanma: flat mi?
    flat_wins_binary = [1 if s["winner_mean"] == "flat" else 0
                        for s in summary.values()]
    # Kazanma orani (seed yuzdesi)
    flat_win_rates = [s["flat_wins"] / s["n_seeds"] for s in summary.values()]

    print("\nOtokorelasyon vs Kazanan (binary):")
    if len(set(flat_wins_binary)) > 1:
        r1, p1 = pearsonr(autocorrs, flat_wins_binary)
        print(f"  r={r1:+.3f}, p={p1:.4f}")
    else:
        r1, p1 = 0.0, 1.0
        print("  Tum gorevler ayni kazanan")

    print("\nOtokorelasyon vs Flat kazanma orani (continuous):")
    r2, p2 = pearsonr(autocorrs, flat_win_rates)
    print(f"  r={r2:+.3f}, p={p2:.4f}")

    print("\nBlok skoru vs Flat kazanma orani:")
    r3, p3 = pearsonr(blocks, flat_win_rates)
    print(f"  r={r3:+.3f}, p={p3:.4f}")

    Path("artifacts/w5_proper_results.json").write_text(
        json.dumps({"summary": summary, "correlation": {
            "autocorr_vs_binary_r": float(r1), "autocorr_vs_binary_p": float(p1),
            "autocorr_vs_rate_r": float(r2), "autocorr_vs_rate_p": float(p2),
            "block_vs_rate_r": float(r3), "block_vs_rate_p": float(p3),
        }}, indent=2), encoding="utf-8")
    print("\nOK — sonuclar kaydedildi")
