# -*- coding: utf-8 -*-
"""Hiyerarşik Kronecker — Gerçek Eğitim Karşılaştırması.

Üç model, aynı sentetik görevde eğitilir:
1. DenseMLP:           flatten(n²) → hidden → classes
2. FlatKroneckerNet:   K katman (n,n)→(n,n) + head
3. HierarchicalNet:    K katman büyüyen boyut + head

Görev: sabit lineer fonksiyonun argmax'ı (sentetik, deterministik).
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Katmanlar ──────────────────────────────────────────────────────

class BilinearLayer(nn.Module):
    """Y = SiLU(A @ X @ B), A: (n_out, n_in), B: (n_in, n_out)."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.A = nn.Parameter(torch.empty(n_out, n_in))
        self.B = nn.Parameter(torch.empty(n_in, n_out))
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)

    def forward(self, x):
        # x: (batch, n_in, n_in) → (batch, n_out, n_out)
        return torch.einsum("oi,bij,jk->bok", self.A, x, self.B)


# ─── Modeller ───────────────────────────────────────────────────────

class DenseMLP(nn.Module):
    """Baseline: düz MLP."""

    def __init__(self, n: int, hidden: int, depth: int, num_classes: int):
        super().__init__()
        layers = []
        in_dim = n * n
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        layers.append(nn.Linear(hidden, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))


class FlatKroneckerNet(nn.Module):
    """Aynı boyutlu K katman."""

    def __init__(self, n: int, depth: int, num_classes: int):
        super().__init__()
        self.layers = nn.ModuleList([BilinearLayer(n, n) for _ in range(depth)])
        self.head = nn.Linear(n * n, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = F.silu(layer(x))
        return self.head(x.flatten(1))


class HierarchicalKroneckerNet(nn.Module):
    """Büyüyen boyutlu K katman."""

    def __init__(self, n_base: int, depth: int, expansion: int, num_classes: int):
        super().__init__()
        sizes = [n_base * (expansion ** i) for i in range(depth + 1)]
        self.layers = nn.ModuleList([
            BilinearLayer(sizes[i], sizes[i + 1]) for i in range(depth)
        ])
        self.head = nn.Linear(sizes[-1] ** 2, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = F.silu(layer(x))
        return self.head(x.flatten(1))


# ─── Sentetik Görev ─────────────────────────────────────────────────

def make_task(n: int, num_classes: int, n_train: int, n_test: int,
              seed: int = 42) -> Dict[str, torch.Tensor]:
    """Sabit lineer öğretmen: class = argmax(W @ flatten(x))."""
    torch.manual_seed(seed)
    W = torch.randn(num_classes, n * n) / math.sqrt(n * n)

    def gen(count):
        x = torch.randn(count, n, n)
        logits = (x.flatten(1) @ W.T)
        y = logits.argmax(1)
        return x, y

    x_tr, y_tr = gen(n_train)
    x_te, y_te = gen(n_test)
    return {"x_train": x_tr, "y_train": y_tr,
            "x_test": x_te, "y_test": y_te,
            "W": W}


# ─── Eğitim ─────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    name: str
    params: int
    train_acc: float
    test_acc: float
    final_loss: float
    wall_time_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def train_model(model: nn.Module, task: Dict[str, torch.Tensor],
                epochs: int = 200, lr: float = 3e-3, batch: int = 64,
                seed: int = 42, name: str = "model") -> TrainResult:
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    x_tr, y_tr = task["x_train"], task["y_train"]
    x_te, y_te = task["x_test"], task["y_test"]
    n = x_tr.shape[0]

    started = time.perf_counter()
    final_loss = float("nan")
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            optimizer.zero_grad()
            out = model(x_tr[idx])
            loss = loss_fn(out, y_tr[idx])
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())

    with torch.no_grad():
        train_acc = float((model(x_tr).argmax(1) == y_tr).float().mean().item())
        test_acc = float((model(x_te).argmax(1) == y_te).float().mean().item())

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return TrainResult(
        name=name, params=params,
        train_acc=round(train_acc, 4),
        test_acc=round(test_acc, 4),
        final_loss=round(final_loss, 4),
        wall_time_sec=round(time.perf_counter() - started, 2),
    )


# ─── Ana Karşılaştırma ──────────────────────────────────────────────

def compare_models(n_base: int = 4, depth: int = 3, expansion: int = 2,
                   num_classes: int = 8, n_train: int = 2000,
                   n_test: int = 500, epochs: int = 200,
                   seed: int = 42) -> Dict[str, Any]:
    """Üç modeli AYNI GİRDİ boyutunda eğit ve karşılaştır.

    Adil karşılaştırma: tüm modeller (n_base, n_base) girdisi alır.
    - Dense MLP:     flatten(n_base²) → hidden → classes
    - Flat Kron:     (n_base, n_base) → K katman → head
    - Hierarchical:  (n_base, n_base) → K katman (büyüyen) → head
    """
    final_size = n_base * (expansion ** depth)
    task = make_task(n=n_base, num_classes=num_classes,
                     n_train=n_train, n_test=n_test, seed=seed)

    # Hierarchical: (n_base, n_base) → ... → (final_size, final_size) → head
    hier = HierarchicalKroneckerNet(n_base, depth, expansion, num_classes)

    # Flat: (n_base, n_base) → K katman → (n_base, n_base) → head
    flat = FlatKroneckerNet(n_base, depth, num_classes)

    # Dense MLP: flatten(n_base²) → gizli boyut hierarchical ile eşit
    dense = DenseMLP(n=n_base, hidden=128,
                     depth=2, num_classes=num_classes)

    results: List[TrainResult] = []
    for name, model in [("dense_mlp", dense),
                        ("flat_kronecker", flat),
                        ("hierarchical_kronecker", hier)]:
        r = train_model(model, task, epochs=epochs, seed=seed, name=name)
        results.append(r)
        print(f"  {name:26s} params={r.params:>10,}  "
              f"train={r.train_acc:.4f}  test={r.test_acc:.4f}  "
              f"loss={r.final_loss:.4f}  süre={r.wall_time_sec:.1f}s")

    return {
        "n_base": n_base, "depth": depth, "expansion": expansion,
        "final_size": final_size, "num_classes": num_classes,
        "n_train": n_train, "n_test": n_test, "epochs": epochs,
        "results": [r.to_dict() for r in results],
    }


if __name__ == "__main__":
    print("=" * 80)
    print("Hiyerarşik Kronecker — Eğitim Karşılaştırması")
    print("=" * 80)
    print()

    for n_base, expansion in [(4, 2), (4, 3), (4, 4)]:
        print(f"\n### n_base={n_base}, depth=3, expansion={expansion}\n")
        report = compare_models(n_base=n_base, depth=3, expansion=expansion,
                                num_classes=8, n_train=2000,
                                n_test=500, epochs=200)
        print()
