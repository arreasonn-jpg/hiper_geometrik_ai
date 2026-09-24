# -*- coding: utf-8 -*-
"""
MNIST uzerinde Kronecker vs Dense MLP testi.

Soru: Sentetik grid'de gozlenen Kronecker verimliligi
gercek goruntu verisinde de gecerli mi?

Modeller:
  - Dense MLP: 784 -> hidden -> 10
  - Flat Kronecker: (28,28) grid -> K katman -> head
  - Hierarchical: cok buyuk (87K+ param), simdilik atla
"""
from __future__ import annotations

import gzip
import json
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = Path("data/mnist")


def load_mnist(which: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    """MNIST idx formatini oku."""
    if which == "train":
        img_path = DATA_DIR / "train-images.gz"
        lbl_path = DATA_DIR / "train-labels.gz"
    else:
        img_path = DATA_DIR / "test-images.gz"
        lbl_path = DATA_DIR / "test-labels.gz"

    with gzip.open(img_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic: {magic}"
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)

    with gzip.open(lbl_path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic: {magic}"
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images, labels


def make_task(n_train: int = 2000, n_test: int = 500,
              seed: int = 42) -> Dict[str, torch.Tensor]:
    """MNIST'ten alt-ornek al, normalize et."""
    rng = np.random.RandomState(seed)
    X_tr, y_tr = load_mnist("train")
    X_te, y_te = load_mnist("test")

    idx_tr = rng.choice(len(X_tr), n_train, replace=False)
    idx_te = rng.choice(len(X_te), n_test, replace=False)

    X_tr = X_tr[idx_tr].astype(np.float32) / 255.0
    y_tr = y_tr[idx_tr].astype(np.int64)
    X_te = X_te[idx_te].astype(np.float32) / 255.0
    y_te = y_te[idx_te].astype(np.int64)

    return {
        "x_train": torch.from_numpy(X_tr),
        "y_train": torch.from_numpy(y_tr),
        "x_test": torch.from_numpy(X_te),
        "y_test": torch.from_numpy(y_te),
        "num_classes": 10,
    }


# ─── Modeller ────────────────────────────────────────────────────────

class BilinearLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(n_out, n_in))
        self.B = nn.Parameter(torch.empty(n_in, n_out))
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)

    def forward(self, x):
        return torch.einsum("oi,bij,jk->bok", self.A, x, self.B)


class DenseMLP(nn.Module):
    def __init__(self, n: int, hidden: int, num_classes: int):
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
    def __init__(self, n: int, depth: int, num_classes: int):
        super().__init__()
        self.layers = nn.ModuleList([BilinearLayer(n, n) for _ in range(depth)])
        self.head = nn.Linear(n * n, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = F.silu(layer(x))
        return self.head(x.flatten(1))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class TrainResult:
    name: str
    params: int
    train_acc: float
    test_acc: float
    wall_time_sec: float


def train_model(model: nn.Module, task: Dict[str, torch.Tensor],
                epochs: int = 30, lr: float = 1e-3,
                seed: int = 42, name: str = "model") -> TrainResult:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X_tr, y_tr = task["x_train"], task["y_train"]
    X_te, y_te = task["x_test"], task["y_test"]
    params = count_params(model)
    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(X_tr), y_tr)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        train_acc = (model(X_tr).argmax(1) == y_tr).float().mean().item()
        test_acc = (model(X_te).argmax(1) == y_te).float().mean().item()
    return TrainResult(name=name, params=params, train_acc=train_acc,
                       test_acc=test_acc,
                       wall_time_sec=time.perf_counter() - t0)


# ─── Ana Deney ───────────────────────────────────────────────────────

def run_one_seed(seed: int, n_train: int = 2000, n_test: int = 500,
                 epochs: int = 30) -> List[Dict[str, Any]]:
    """MNIST 28x28 icin 3 model: dense, flat Kronecker (farkli derinlikler)."""
    task = make_task(n_train=n_train, n_test=n_test, seed=seed)
    n = 28
    results = []

    # 1) Dense MLP — referans
    for hidden in (32, 128):
        model = DenseMLP(n, hidden=hidden, num_classes=10)
        r = train_model(model, task, epochs=epochs, seed=seed,
                        name=f"dense_h{hidden}")
        results.append(asdict(r))

    # 2) Flat Kronecker — farkli derinlikler
    for depth in (1, 2, 3):
        model = FlatKroneckerNet(n, depth=depth, num_classes=10)
        r = train_model(model, task, epochs=epochs, seed=seed,
                        name=f"flat_K{depth}")
        results.append(asdict(r))

    return results


if __name__ == "__main__":
    all_runs = []
    for seed in range(1, 6):
        print(f"\n═══ Seed {seed} ═══")
        r = run_one_seed(seed)
        all_runs.append({"seed": seed, "results": r})
        for m in r:
            print(f"  {m['name']:14s} params={m['params']:>8,} "
                  f"train={m['train_acc']:.4f} test={m['test_acc']:.4f} "
                  f"({m['wall_time_sec']:.1f}s)")

    # Ozet
    import statistics
    print("\n═══ Ozet (5 seed) ═══")
    by_name: Dict[str, Dict[str, List[float]]] = {}
    for run in all_runs:
        for m in run["results"]:
            n = m["name"]
            by_name.setdefault(n, {"acc": [], "params": []})
            by_name[n]["acc"].append(m["test_acc"])
            by_name[n]["params"].append(m["params"])
    for name, d in by_name.items():
        print(f"  {name:14s} params={int(statistics.mean(d['params'])):>8,} "
              f"test={statistics.mean(d['acc']):.4f} ± {statistics.stdev(d['acc']):.4f}")

    Path("artifacts/mnist_kronecker_results.json").write_text(
        json.dumps(all_runs, indent=2), encoding="utf-8")
    print("\nOK — sonuclar kaydedildi")
