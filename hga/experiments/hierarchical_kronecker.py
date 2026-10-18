# -*- coding: utf-8 -*-
"""Hiyerarşik Kronecker PoC.

Vizyon: her katman bir öncekinden büyür (n -> n² -> n⁴ ...).
Mevcut kod: aynı boyutlu düz zincir (n -> n -> n ...).

Bu modül, hiyerarşik genişlemenin ifade gücünü artırıp artırmadığını ölçer.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


def _torch():
    try:
        import torch
    except ImportError as error:
        raise ImportError("PyTorch gerekli (pip install -e .)") from error
    return torch


def _spectrum(torch, matrix):
    """Tekil değer spektrumundan rank ve etkin boyut."""
    singular = torch.linalg.svdvals(matrix.detach().double())
    sq_total = float((singular ** 2).sum().item())
    largest = float(singular[0].item()) if singular.numel() else 0.0
    threshold = largest * max(matrix.shape) * 2.22e-16
    rank = int((singular > threshold).sum().item())
    if sq_total > 0:
        energy = (singular ** 2) / sq_total
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
    depth: int
    layer_index: int
    shape_in: int
    shape_out: int
    params_in_layer: int
    cumulative_params: int
    rank: int
    effective_dimension: float


@dataclass
class HierarchyReport:
    n_base: int
    depth: int
    expansion: int
    mode: str  # "flat" veya "hierarchical"
    rows: List[HierarchyRow]
    final_effective_dimension: float
    final_rank: int
    total_params: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_base": self.n_base,
            "depth": self.depth,
            "expansion": self.expansion,
            "mode": self.mode,
            "rows": [asdict(r) for r in self.rows],
            "final_effective_dimension": self.final_effective_dimension,
            "final_rank": self.final_rank,
            "total_params": self.total_params,
        }

    def markdown(self) -> str:
        lines = [
            f"## {self.mode.upper()} — n_base={self.n_base}, depth={self.depth}, expansion={self.expansion}",
            "",
            "| Katman | Giriş | Çıkış | Parametre | Toplam Param | Rank | Etkin Boyut |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in self.rows:
            lines.append(
                f"| {r.layer_index} | {r.shape_in} | {r.shape_out} | "
                f"{r.params_in_layer:,} | {r.cumulative_params:,} | "
                f"{r.rank} | {r.effective_dimension} |"
            )
        lines.extend([
            "",
            f"**Toplam parametre:** {self.total_params:,}",
            f"**Son rank:** {self.final_rank}",
            f"**Son etkin boyut:** {self.final_effective_dimension}",
        ])
        return "\n".join(lines)


def run_hierarchy(mode: str, n_base: int = 16, depth: int = 3,
                  expansion: int = 2, seed: int = 1,
                  samples: int = 128) -> HierarchyReport:
    """İki modu karşılaştır: 'flat' (mevcut) ve 'hierarchical' (vizyon).

    flat:         her katman (n, n) -> (n, n)
    hierarchical: her katman (n, n) -> (n*expansion, n*expansion)
    """
    torch = _torch()
    torch.manual_seed(int(seed))

    if mode not in {"flat", "hierarchical"}:
        raise ValueError("mode: 'flat' veya 'hierarchical'")

    rows: List[HierarchyRow] = []
    cumulative = 0

    # Katman boyutlarını hesapla
    if mode == "flat":
        sizes = [n_base] * (depth + 1)
    else:
        sizes = [n_base * (expansion ** i) for i in range(depth + 1)]

    # Girdi tensörü (n_base, n_base)
    x = torch.randn(samples, n_base, n_base)

    for i in range(depth):
        n_in = sizes[i]
        n_out = sizes[i + 1]
        # Bilinear katman: Y = A @ X @ B, A: (n_out, n_in), B: (n_in, n_out)
        A = torch.empty(n_out, n_in)
        B = torch.empty(n_in, n_out)
        torch.nn.init.xavier_uniform_(A)
        torch.nn.init.xavier_uniform_(B)

        params = A.numel() + B.numel()
        cumulative += params

        # İleri geçiş: X (samples, n_in, n_in) -> Y (samples, n_out, n_out)
        # A @ X -> (samples, n_out, n_in), sonra @ B -> (samples, n_out, n_out)
        y = torch.einsum("oi,bij,jk->bok", A, x, B)
        # Non-lineerlik
        y = torch.nn.functional.silu(y)

        # Operatörü ölç: A'yı vec edip Kronecker matrisi kurmak yerine,
        # bu katmanın girdi-çıktı Jacobian'ını yaklaşık ölç.
        # Basit yaklaşım: A'nın ve B'nin rank'larını çarp.
        rank_a = int(torch.linalg.matrix_rank(A).item())
        rank_b = int(torch.linalg.matrix_rank(B).item())
        operator_rank = rank_a * rank_b
        max_rank = n_in * n_out  # (n_out x n_in) @ (n_out x n_out) uzayı
        operator_rank = min(operator_rank, max_rank)

        # Etkin boyut: operatör matrisinin (A ⊗ B) tekil değerleri
        kron = torch.kron(B.T.contiguous(), A.contiguous())
        spec = _spectrum(torch, kron)

        rows.append(HierarchyRow(
            depth=i + 1,
            layer_index=i,
            shape_in=n_in,
            shape_out=n_out,
            params_in_layer=params,
            cumulative_params=cumulative,
            rank=spec["rank"],
            effective_dimension=spec["effective_dimension"],
        ))

        x = y

    return HierarchyReport(
        n_base=n_base,
        depth=depth,
        expansion=expansion,
        mode=mode,
        rows=rows,
        final_effective_dimension=rows[-1].effective_dimension,
        final_rank=rows[-1].rank,
        total_params=cumulative,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("Hiyerarşik Kronecker PoC — Vizyon vs Mevcut")
    print("=" * 70)
    print()
    for mode in ("flat", "hierarchical"):
        report = run_hierarchy(mode, n_base=16, depth=3, expansion=2)
        print(report.markdown())
        print()
