# -*- coding: utf-8 -*-
"""Hiyerarşik Kronecker v2 — ölçeklenebilir + seyrek.

Önceki sürüm torch.kron ile operatör matrisi kuruyordu → O(n^4) bellek.
Bu sürüm matematiksel özdeşliği kullanır:
    σ(B.T ⊗ A) = σ(B.T) ⊗ σ(A)
Yani tekil değerler dış çarpımla hesaplanır → O(n²) bellek.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List


def _torch():
    try:
        import torch
    except ImportError as error:
        raise ImportError("PyTorch gerekli") from error
    return torch


def _spectrum_from_svd(torch, A, B):
    """σ(A ⊗ B) = σ(A) ⊗ σ(B) özdeşliğiyle spektrum (kron matrisi kurmaz)."""
    sigma_A = torch.linalg.svdvals(A.detach().double())  # (n_out,)
    sigma_B = torch.linalg.svdvals(B.detach().double())  # (n_in,)

    # Dış çarpım: tüm çiftler
    kron_sigma = (sigma_A[:, None] * sigma_B[None, :]).flatten()
    # Azalan sırala
    kron_sigma = torch.sort(kron_sigma, descending=True).values

    total = float(kron_sigma.sum().item())
    sq_total = float((kron_sigma ** 2).sum().item())
    largest = float(kron_sigma[0].item()) if kron_sigma.numel() else 0.0
    # Sayısal rank için eşik: büyük değerin çok altındakiler sıfır sayılır
    threshold = largest * max(kron_sigma.numel(), 1) * 2.22e-16
    rank = int((kron_sigma > threshold).sum().item())

    if sq_total > 0:
        energy = (kron_sigma ** 2) / sq_total
        energy = energy[energy > 0]
        entropy = float(-(energy * energy.log()).sum().item())
        eff_dim = math.exp(entropy)
    else:
        eff_dim = 0.0

    return {
        "rank": rank,
        "effective_dimension": round(eff_dim, 4),
        "largest_singular_value": round(largest, 8),
    }


@dataclass
class HierarchyRow:
    layer_index: int
    shape_in: int
    shape_out: int
    params_in_layer: int
    cumulative_params: int
    rank: int
    effective_dimension: float
    flops_per_sample: int


@dataclass
class HierarchyReport:
    mode: str
    n_base: int
    depth: int
    expansion: int
    rank_budget: int  # 0 = full rank
    rows: List[HierarchyRow]
    total_params: int
    total_flops: int
    final_rank: int
    final_effective_dimension: float
    wall_time_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "n_base": self.n_base,
            "depth": self.depth,
            "expansion": self.expansion,
            "rank_budget": self.rank_budget,
            "rows": [asdict(r) for r in self.rows],
            "total_params": self.total_params,
            "total_flops": self.total_flops,
            "final_rank": self.final_rank,
            "final_effective_dimension": self.final_effective_dimension,
            "wall_time_sec": self.wall_time_sec,
        }

    def markdown(self) -> str:
        lines = [
            f"## {self.mode.upper()} · n_base={self.n_base} · depth={self.depth} · "
            f"rank_budget={self.rank_budget or 'full'}",
            "",
            "| Katman | Giriş | Çıkış | Parametre | Toplam | Rank | Etkin Boyut | FLOPs/örnek |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in self.rows:
            lines.append(
                f"| {r.layer_index} | {r.shape_in} | {r.shape_out} | "
                f"{r.params_in_layer:,} | {r.cumulative_params:,} | "
                f"{r.rank} | {r.effective_dimension} | {r.flops_per_sample:,} |"
            )
        lines.extend([
            "",
            f"**Toplam parametre:** {self.total_params:,}",
            f"**Toplam FLOPs/örnek:** {self.total_flops:,}",
            f"**Son rank:** {self.final_rank}",
            f"**Son etkin boyut:** {self.final_effective_dimension}",
            f"**Süre:** {self.wall_time_sec:.2f}s",
        ])
        return "\n".join(lines)


def _make_factors(torch, n_in: int, n_out: int, rank_budget: int):
    """A: (n_out, n_in) ve B: (n_in, n_out) oluştur.

    rank_budget > 0 ise düşük rank kullan: A = U_A @ V_A^T
    Bu, 'seyrek bağlantı' kavramının lineer cebirsel karşılığıdır.
    """
    if rank_budget <= 0 or rank_budget >= min(n_in, n_out):
        A = torch.empty(n_out, n_in)
        B = torch.empty(n_in, n_out)
        torch.nn.init.xavier_uniform_(A)
        torch.nn.init.xavier_uniform_(B)
        return A, B

    r = min(rank_budget, n_in, n_out)
    U_A = torch.empty(n_out, r); V_A = torch.empty(n_in, r)
    U_B = torch.empty(n_in, r); V_B = torch.empty(n_out, r)
    torch.nn.init.xavier_uniform_(U_A); torch.nn.init.xavier_uniform_(V_A)
    torch.nn.init.xavier_uniform_(U_B); torch.nn.init.xavier_uniform_(V_B)
    A = U_A @ V_A.T
    B = U_B @ V_B.T
    return A, B


def run_hierarchy(mode: str, n_base: int = 16, depth: int = 3,
                  expansion: int = 2, rank_budget: int = 0,
                  seed: int = 1) -> HierarchyReport:
    """Üç mod:
    - 'flat':         her katman (n, n) → (n, n)
    - 'hierarchical': her katman (n, n) → (n·exp, n·exp)
    - 'sparse':       hierarchical + düşük rank (rank_budget kullanılır)
    """
    torch = _torch()
    torch.manual_seed(int(seed))
    started = time.perf_counter()

    if mode not in {"flat", "hierarchical", "sparse"}:
        raise ValueError("mode: flat | hierarchical | sparse")

    if mode == "flat":
        sizes = [n_base] * (depth + 1)
    else:  # hierarchical, sparse
        sizes = [n_base * (expansion ** i) for i in range(depth + 1)]

    rows: List[HierarchyRow] = []
    cumulative_params = 0
    total_flops = 0

    for i in range(depth):
        n_in = sizes[i]
        n_out = sizes[i + 1]

        if mode == "sparse":
            A, B = _make_factors(torch, n_in, n_out, rank_budget)
            params = (A.numel() + B.numel())
            # Düşük rank FLOPs: A @ X @ B
            flops = n_out * n_in * n_in + n_out * n_out * n_in
        else:
            A = torch.empty(n_out, n_in); B = torch.empty(n_in, n_out)
            torch.nn.init.xavier_uniform_(A); torch.nn.init.xavier_uniform_(B)
            params = A.numel() + B.numel()
            flops = n_out * n_in * n_in + n_out * n_out * n_in

        cumulative_params += params
        total_flops += flops

        spec = _spectrum_from_svd(torch, A, B)

        rows.append(HierarchyRow(
            layer_index=i,
            shape_in=n_in,
            shape_out=n_out,
            params_in_layer=params,
            cumulative_params=cumulative_params,
            rank=spec["rank"],
            effective_dimension=spec["effective_dimension"],
            flops_per_sample=flops,
        ))

    return HierarchyReport(
        mode=mode,
        n_base=n_base,
        depth=depth,
        expansion=expansion,
        rank_budget=rank_budget,
        rows=rows,
        total_params=cumulative_params,
        total_flops=total_flops,
        final_rank=rows[-1].rank,
        final_effective_dimension=rows[-1].effective_dimension,
        wall_time_sec=round(time.perf_counter() - started, 3),
    )


if __name__ == "__main__":
    print("=" * 80)
    print("Hiyerarşik Kronecker v2 — Ölçek + Seyreklik Testi")
    print("=" * 80)

    # Test 1: Ölçek
    print("\n### TEST 1: Ölçek Genişlemesi (flat vs hierarchical)\n")
    for n_base in (16, 32, 64, 128):
        flat = run_hierarchy("flat", n_base=n_base, depth=3)
        hier = run_hierarchy("hierarchical", n_base=n_base, depth=3)
        print(f"n_base={n_base}:")
        print(f"  FLAT:         params={flat.total_params:>12,}  rank={flat.final_rank:>8}  eff_dim={flat.final_effective_dimension:>10.2f}")
        print(f"  HIERARCHICAL: params={hier.total_params:>12,}  rank={hier.final_rank:>8}  eff_dim={hier.final_effective_dimension:>10.2f}")
        print(f"  Kazanç: {hier.final_effective_dimension / flat.final_effective_dimension:.2f}× etkin boyut")
        print()

    # Test 2: Seyreklik (düşük rank)
    print("\n### TEST 2: Seyreklik (rank_budget etkisi)\n")
    n_base = 32
    for rb in (0, 16, 32, 64):
        sparse = run_hierarchy("sparse", n_base=n_base, depth=3, rank_budget=rb)
        etiket = "full" if rb == 0 else f"rank={rb}"
        print(f"  {etiket:>10} | params={sparse.total_params:>10,} | "
              f"rank={sparse.final_rank:>6} | eff_dim={sparse.final_effective_dimension:>8.2f} | "
              f"süre={sparse.wall_time_sec:.2f}s")
