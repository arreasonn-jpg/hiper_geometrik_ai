# -*- coding: utf-8 -*-
"""
Goruntu compositional gorevde Kronecker mimari testi.

Gorev: 2D grid goruntusunde blok yapisi + uzamsal korelasyon.
Hipotez: Kronecker, goruntu gibi yapisal verilerde de verimli.

Modeller:
  - Dense MLP (flatten)
  - Flat Kronecker (2D grid -> 2D grid)
  - Hierarchical Kronecker
"""
from __future__ import annotations

import json
import statistics
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


# ─── Görüntü Compositional Görev ────────────────────────────────────

def make_image_task(n: int = 16, num_classes: int = 4,
                    n_train: int = 2000, n_test: int = 500,
                    seed: int = 42) -> Dict[str, torch.Tensor]:
    """2D grid görüntü compositional.

    Yapı:
      - n x n grid (n=16)
      - 4x4 blok ızgarası (4 satır x 4 sütun blok)
      - Her blokta "sinyal" değeri (0-1 arası)
      - Uzamsal korelasyon: komşu bloklar benzer
      - Sınıf: 4 çeyrekten hangisi en güçlü?

    Bu, gerçek görüntü gibi:
      - Uzamsal yapı (grid)
      - Lokal korelasyon (komşuluk)
      - Global özet (çeyrek argmax)
    """
    g = torch.Generator().manual_seed(seed)
    block_size = 4  # n // 4
    n_blocks = n // block_size  # 4

    def gen_sample():
        # 4x4 blok sinyal gücü
        block_power = torch.rand(n_blocks, n_blocks, generator=g)
        # Sınıf = hangi çeyrek en güçlü (2x2 çeyrek)
        quarters = torch.zeros(2, 2)
        for qr in range(2):
            for qc in range(2):
                block = block_power[
                    qr * 2:(qr + 1) * 2,
                    qc * 2:(qc + 1) * 2,
                ]
                quarters[qr, qc] = block.sum()
        label = quarters.flatten().argmax().item()

        # Blokları piksele çevir (her blok 4x4 sabit değer + gürültü)
        img = torch.zeros(n, n)
        for br in range(n_blocks):
            for bc in range(n_blocks):
                val = block_power[br, bc]
                img[
                    br * block_size:(br + 1) * block_size,
                    bc * block_size:(bc + 1) * block_size,
                ] = val
        # Gürültü
        img = img + 0.3 * torch.randn(n, n, generator=g)
        return img, label

    train_pairs = [gen_sample() for _ in range(n_train)]
    test_pairs = [gen_sample() for _ in range(n_test)]
    X_tr = torch.stack([p[0] for p in train_pairs])
    y_tr = torch.tensor([p[1] for p in train_pairs])
    X_te = torch.stack([p[0] for p in test_pairs])
    y_te = torch.tensor([p[1] for p in test_pairs])

    return {"x_train": X_tr, "y_train": y_tr,
            "x_test": X_te, "y_test": y_te,
            "num_classes": num_classes}


@dataclass
class TrainResult:
    name: str
    params: int
    train_acc: float
    test_acc: float
    wall_time_sec: float


def train_model(model: nn.Module, task: Dict[str, torch.Tensor],
                epochs: int = 100, lr: float = 1e-3,
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


def run_one_seed(seed: int, n_base: int = 16, epochs: int = 100):
    task = make_image_task(n=n_base, num_classes=4, seed=seed)
    results = []
    for name, model in [
        ("dense_mlp", DenseMLP(n_base, hidden=n_base * 4, num_classes=4)),
        ("flat_kronecker", FlatKroneckerNet(n_base, 3, num_classes=4)),
        ("hierarchical_kronecker", HierarchicalKroneckerNet(n_base, 3, 2, num_classes=4)),
    ]:
        r = train_model(model, task, epochs=epochs, seed=seed, name=name)
        results.append(asdict(r))
    return results


if __name__ == "__main__":
    all_runs = []
    for seed in range(1, 21):
        print(f"\n═══ Seed {seed} ═══")
        r = run_one_seed(seed)
        all_runs.append({"seed": seed, "results": r})
        for m in r:
            print(f"  {m['name']:26s} params={m['params']:>8,} "
                  f"train={m['train_acc']:.4f} test={m['test_acc']:.4f}")

    # Ozet
    print("\n═══ Ozet (5 seed) ═══")
    by_name: Dict[str, Dict[str, List[float]]] = {}
    for run in all_runs:
        for m in run["results"]:
            n = m["name"]
            by_name.setdefault(n, {"acc": [], "params": []})
            by_name[n]["acc"].append(m["test_acc"])
            by_name[n]["params"].append(m["params"])
    for name, d in by_name.items():
        print(f"  {name:26s} params={int(statistics.mean(d['params'])):>8,} "
              f"test={statistics.mean(d['acc']):.4f} ± {statistics.stdev(d['acc']):.4f}")

    # ─── Istatistiksel testler ──
    import math
    print("\n═══ Istatistiksel Testler ═══")
    flat = by_name["flat_kronecker"]["acc"]
    dense = by_name["dense_mlp"]["acc"]
    hier = by_name["hierarchical_kronecker"]["acc"]

    def paired_t(a, b):
        diffs = [ai - bi for ai, bi in zip(a, b)]
        md = statistics.mean(diffs)
        sd = statistics.stdev(diffs) if len(diffs) > 1 else 0
        se = sd / math.sqrt(len(diffs))
        t = md / se if se > 0 else 0
        p_val = math.erfc(abs(t) / math.sqrt(2))
        return {"mean_diff": md, "t": t, "p": p_val, "sd_diff": sd}

    def cohens_d(a, b):
        diffs = [ai - bi for ai, bi in zip(a, b)]
        md = statistics.mean(diffs)
        sd = statistics.stdev(diffs) if len(diffs) > 1 else 0
        return md / sd if sd > 0 else 0

    for label, a, b in [("flat vs dense", flat, dense),
                        ("flat vs hier", flat, hier),
                        ("dense vs hier", dense, hier)]:
        r = paired_t(a, b)
        d = cohens_d(a, b)
        print(f"\n  {label}:")
        print(f"    mean_diff = {r['mean_diff']:+.4f}")
        print(f"    t = {r['t']:+.3f}  (p ≈ {r['p']:.4f})")
        print(f"    Cohen's d = {d:+.3f}")

    Path("artifacts/image_compositional_results.json").write_text(
        json.dumps(all_runs, indent=2), encoding="utf-8")
    print("\nOK — sonuclar kaydedildi")
