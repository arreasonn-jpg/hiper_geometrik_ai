# -*- coding: utf-8 -*-
"""Hiyerarşik/kompozisyonel görev — hiyerarşik mimarinin doğal alanı.

Girdi: (n, n) matris, n = 4^k
Yapı: katmanlı örüntü — alt-bölgeler üst-bölgelere beslenir.

Bu, düz MLP'nin ölçek bilgisini kaybettiği, hiyerarşik yapının doğal
olduğu bir görevdir. Hipotez: hiyerarşik Kronecker burada Dense'i geçer.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Model katmanları (öncekiyle aynı) ─────────────────────────────

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


# ─── Hiyerarşik görev üreteci ──────────────────────────────────────

def make_hierarchical_task(n: int = 16, n_train: int = 4000, n_test: int = 1000,
                           noise: float = 0.3, seed: int = 42
                           ) -> Dict[str, torch.Tensor]:
    """XOR'suz hiyerarşik görev — öğrenilebilir versiyon.

    n = 16 → 16 blok (4x4)
    Yapı:
      Level 1: her 4x4 blokta "sinyal gücü" (sürekli, türevlenebilir)
      Level 2: 4 çeyrek → her çeyrek 4 bloğun ortalaması
      Level 3: Sınıf = hangi çeyrek en güçlü? (argmax → 4 sınıf)

    Fark: XOR yok, toplama + argmax var. Öğrenilebilir.
    """
    torch.manual_seed(seed)
    block_size = 4
    num_blocks_per_axis = n // block_size
    assert num_blocks_per_axis ** 2 == 16, f"n=16 için tasarlandı (gelen: {n})"

    def gen_sample():
        # 16 blokta "güç" değeri (0-1 arası, sürekli)
        block_power = torch.rand(16)  # uniform [0, 1]
        # Grid'e yerleştir
        blocks = torch.zeros(16, block_size, block_size)
        for i in range(16):
            # Blok içinde sinyal + gürültü
            blocks[i] = block_power[i] + noise * torch.randn(block_size, block_size)
        # Grid'e diz
        x = blocks.reshape(num_blocks_per_axis, num_blocks_per_axis,
                          block_size, block_size)
        x = x.permute(0, 2, 1, 3).reshape(n, n)

        # Hiyerarşik etiket: 4 çeyrek → her çeyrek 4 bloğun ortalaması
        bp = block_power.reshape(4, 4)  # 4 çeyrek, her biri 2x2 blok
        # Çeyrek güçleri
        q_powers = torch.tensor([
            bp[0:2, 0:2].mean(),
            bp[0:2, 2:4].mean(),
            bp[2:4, 0:2].mean(),
            bp[2:4, 2:4].mean(),
        ])
        label = q_powers.argmax().long()
        return x, label

    train = [gen_sample() for _ in range(n_train)]
    test = [gen_sample() for _ in range(n_test)]
    return {
        "x_train": torch.stack([t[0] for t in train]),
        "y_train": torch.stack([t[1] for t in train]),
        "x_test": torch.stack([t[0] for t in test]),
        "y_test": torch.stack([t[1] for t in test]),
        "num_classes": 4,
    }


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
                epochs: int = 150, lr: float = 3e-3, batch: int = 64,
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


# ─── Ana karşılaştırma ──────────────────────────────────────────────

def compare_on_hierarchical(n_base: int = 16, num_classes: int = 4,
                            n_train: int = 4000, n_test: int = 1000,
                            epochs: int = 150, seed: int = 42) -> Dict[str, Any]:
    """Hiyerarşik görevde üç modeli karşılaştır."""
    task = make_hierarchical_task(n=n_base, n_train=n_train,
                                  n_test=n_test, seed=seed)

    # Dense: adil bütçe — gizli boyut n_base² kadar (parametre sayısı dengeli)
    dense = DenseMLP(n=n_base, hidden=n_base * 4, depth=2, num_classes=num_classes)
    flat = FlatKroneckerNet(n_base, depth=3, num_classes=num_classes)
    hier = HierarchicalKroneckerNet(n_base, depth=3, expansion=2,
                                    num_classes=num_classes)

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
        "n_base": n_base, "num_classes": num_classes,
        "n_train": n_train, "n_test": n_test, "epochs": epochs,
        "results": [r.to_dict() for r in results],
    }


if __name__ == "__main__":
    print("=" * 80)
    print("Hiyerarşik/kompozisyonel görev — hiyerarşik mimarinin doğal alanı")
    print("=" * 80)
    print()
    for n_base in (16, 32):
        print(f"\n### n_base={n_base}\n")
        report = compare_on_hierarchical(n_base=n_base, epochs=150)
        print()
