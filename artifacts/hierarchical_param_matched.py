# -*- coding: utf-8 -*-
"""
Parametre-eslesmeli hiyerarsik Kronecker ablation.

Soru: ayni parametre butcesinde hangi mimari kazaniyor?
  - flat_kronecker:        K katman (n,n) -> (n,n)
  - hierarchical_kronecker: K katman (n, n*e^i)
  - dense_mlp:             flatten -> hidden -> classes

Bu deney, "hierarchical daha iyi" iddiasinin parametre sayisindan mi
yoksa mimariden mi geldigini test eder.
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


class BilinearLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.mercek_A = nn.Parameter(torch.empty(n_out, n_in))
        self.mercek_B = nn.Parameter(torch.empty(n_in, n_out))
        nn.init.xavier_uniform_(self.mercek_A)
        nn.init.xavier_uniform_(self.mercek_B)

    def forward(self, x):
        return torch.einsum("oi,bij,jk->bok", self.mercek_A, x, self.mercek_B)


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


class HierarchicalKroneckerNet(nn.Module):
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


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─── Görev ──────────────────────────────────────────────────────────

def make_task(n: int, num_classes: int, n_train: int, n_test: int,
              seed: int = 42) -> Dict[str, torch.Tensor]:
    """Sabit doğrusal öğretmen: class = argmax(W @ flatten(x))."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(num_classes, n * n, generator=g) / (n * n) ** 0.5

    def gen(nn_: int):
        X = torch.randn(nn_, n, n, generator=g)
        y = (W @ X.flatten(1).T).argmax(0)
        return X, y

    X_tr, y_tr = gen(n_train)
    X_te, y_te = gen(n_test)
    return {"x_train": X_tr, "y_train": y_tr,
            "x_test": X_te, "y_test": y_te,
            "num_classes": num_classes}


@dataclass
class TrainResult:
    name: str
    params: int
    train_acc: float
    test_acc: float
    final_loss: float
    wall_time_sec: float


def train_model(model: nn.Module, task: Dict[str, torch.Tensor],
                epochs: int = 200, lr: float = 1e-3,
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
        logits = model(X_tr)
        loss = F.cross_entropy(logits, y_tr)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        train_acc = (model(X_tr).argmax(1) == y_tr).float().mean().item()
        test_acc = (model(X_te).argmax(1) == y_te).float().mean().item()
        final_loss = F.cross_entropy(model(X_tr), y_tr).item()
    wall = time.perf_counter() - t0

    return TrainResult(name=name, params=params, train_acc=train_acc,
                       test_acc=test_acc, final_loss=final_loss,
                       wall_time_sec=wall)


# ─── Bütçe Arama ────────────────────────────────────────────────────

@dataclass
class Config:
    kind: str            # "flat" | "hier" | "dense"
    n: int
    depth: int
    expansion: int       # sadece hier için
    hidden: int          # sadece dense için
    params: int


def build_model(cfg: Config, num_classes: int) -> nn.Module:
    if cfg.kind == "flat":
        return FlatKroneckerNet(cfg.n, cfg.depth, num_classes)
    if cfg.kind == "hier":
        return HierarchicalKroneckerNet(cfg.n, cfg.depth, cfg.expansion, num_classes)
    if cfg.kind == "dense":
        return DenseMLP(cfg.n, cfg.hidden, num_classes)
    raise ValueError(f"bilinmeyen: {cfg.kind}")


def enumerate_configs(num_classes: int = 4) -> List[Config]:
    out: List[Config] = []
    # Flat: n, depth
    for n in (8, 12, 16, 24, 32):
        for depth in (1, 2, 3, 4, 6):
            m = FlatKroneckerNet(n, depth, num_classes)
            out.append(Config("flat", n, depth, 1, 0, count_params(m)))
    # Hier: n_base, depth, expansion
    for n in (4, 6, 8, 12, 16):
        for depth in (2, 3, 4):
            for exp in (2, 3):
                try:
                    m = HierarchicalKroneckerNet(n, depth, exp, num_classes)
                except Exception:
                    continue
                out.append(Config("hier", n, depth, exp, 0, count_params(m)))
    # Dense: n (input size), hidden
    for n in (8, 12, 16, 24, 32):
        for hidden in (16, 32, 64, 128, 256):
            m = DenseMLP(n, hidden, num_classes)
            out.append(Config("dense", n, 0, 0, hidden, count_params(m)))
    return out


def find_matched(configs: List[Config], target: int,
                 same_input: int | None = None) -> Dict[str, Config]:
    """Her mimari tipi için target'a en yakın config'i döndür."""
    by_kind: Dict[str, List[Config]] = {"flat": [], "hier": [], "dense": []}
    for c in configs:
        if same_input is not None and c.n != same_input:
            continue
        by_kind[c.kind].append(c)
    out: Dict[str, Config] = {}
    for kind, items in by_kind.items():
        if not items:
            continue
        out[kind] = min(items, key=lambda c: abs(c.params - target))
    return out


# ─── İki koşullu deney ──────────────────────────────────────────────

def run_experiment(seed: int = 42, n_base: int = 8,
                   epochs: int = 300, num_classes: int = 4,
                   n_train: int = 2000, n_test: int = 500) -> Dict[str, Any]:
    """A) Depth-eşleşmiş (aynı K)
       B) Param-eşleşmiş (hedef param bütçesi)
    """
    task = make_task(n_base, num_classes, n_train, n_test, seed=seed)
    results: Dict[str, List[TrainResult]] = {"A_depth_matched": [], "B_param_matched": []}

    # ─── A) Depth-eşleşmiş ──
    K = 3
    models_A = [
        ("flat_K3", FlatKroneckerNet(n_base, K, num_classes)),
        ("hier_K3_exp2", HierarchicalKroneckerNet(n_base, K, 2, num_classes)),
        ("dense_K3", DenseMLP(n_base, hidden=128, num_classes=num_classes)),
    ]
    for name, model in models_A:
        r = train_model(model, task, epochs=epochs, seed=seed, name=f"A::{name}")
        results["A_depth_matched"].append(r)

    # ─── B) Param-eşleşmiş (hedef = hier_K3_exp2 paramı) ──
    hier_params = count_params(HierarchicalKroneckerNet(n_base, K, 2, num_classes))
    # Flat'i daha derin yap
    flat_depth_for_match = max(1, (hier_params - n_base * n_base * num_classes) //
                               (2 * n_base * n_base))
    models_B = [
        (f"flat_K{flat_depth_for_match}", FlatKroneckerNet(n_base, flat_depth_for_match, num_classes)),
        ("hier_K3_exp2", HierarchicalKroneckerNet(n_base, K, 2, num_classes)),
        ("dense_matched", DenseMLP(n_base, hidden=max(16, hier_params // (4 * n_base * n_base)), num_classes=num_classes)),
    ]
    for name, model in models_B:
        r = train_model(model, task, epochs=epochs, seed=seed, name=f"B::{name}")
        results["B_param_matched"].append(r)

    return {"seed": seed, "n_base": n_base,
            "hier_target_params": hier_params,
            "A": [asdict(r) for r in results["A_depth_matched"]],
            "B": [asdict(r) for r in results["B_param_matched"]]}


if __name__ == "__main__":
    all_runs = []
    for seed in (1, 2, 3, 4, 5):
        r = run_experiment(seed=seed)
        all_runs.append(r)
        print(f"\n═══ Seed {seed} ═══")
        for key in ("A", "B"):
            print(f"--- {key} ---")
            for m in r[key]:
                print(f"  {m['name']:25s} params={m['params']:>8,} "
                      f"train={m['train_acc']:.4f} test={m['test_acc']:.4f}")

    # Özet
    print("\n═══ Özet (5 seed ortalama) ═══")
    for key in ("A", "B"):
        print(f"\n--- {key} ---")
        by_name: Dict[str, List[float]] = {}
        by_params: Dict[str, int] = {}
        for r in all_runs:
            for m in r[key]:
                by_name.setdefault(m["name"], []).append(m["test_acc"])
                by_params[m["name"]] = m["params"]
        for name, accs in by_name.items():
            import statistics
            print(f"  {name:25s} params={by_params[name]:>8,} "
                  f"test={statistics.mean(accs):.4f} ± {statistics.stdev(accs):.4f}")

    Path("artifacts/hierarchical_param_matched_results.json").write_text(
        json.dumps(all_runs, indent=2), encoding="utf-8")
    print("\nOK — sonuclar kaydedildi")
