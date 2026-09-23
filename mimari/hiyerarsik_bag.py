# -*- coding: utf-8 -*-
"""Hiyerarşik Küresel Bağ — U-Net tarzı piramit.

Mevcut KureselZincir sabit boyutlu (n → n → n...).
Bu sürüm piramit: n → n·e → n·e² → ... → n·e → n.

Skip connection'lar U-Net prensibine göre: up fazındaki her katman,
down fazındaki karşılığıyla toplanır.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GenelBilinearKatmani(nn.Module):
    """Genelleştirilmiş bilinear katman: (B, n_in, n_in) → (B, n_out, n_out)."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.mercek_A = nn.Parameter(torch.empty(n_out, n_in))
        self.mercek_B = nn.Parameter(torch.empty(n_in, n_out))
        nn.init.xavier_uniform_(self.mercek_A)
        nn.init.xavier_uniform_(self.mercek_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_in, n_in) → (B, n_out, n_out)
        return torch.einsum("oi,bij,jk->bok", self.mercek_A, x, self.mercek_B)

    def kapasite(self) -> dict:
        return {
            "n_in": self.n_in,
            "n_out": self.n_out,
            "gercek_parametre": 2 * self.n_in * self.n_out,
            "sanal_operator_girdisi": self.n_out * self.n_in * self.n_in * self.n_out,
        }


class HiyerarsikZincir(nn.Module):
    """U-Net tarzı piramit zincir.

    katman_sayisi çift olmalı: K/2 up, K/2 down.
    expansion: her up katmanında boyut çarpanı (örn. 2).
    """

    def __init__(self, n_base: int, katman_sayisi: int = 4, expansion: int = 2,
                 dropout: float = 0.0, aktivasyon: str = "silu"):
        super().__init__()
        if katman_sayisi < 2 or katman_sayisi % 2 != 0:
            raise ValueError("katman_sayisi çift ve en az 2 olmalı")
        self.n_base = n_base
        self.katman_sayisi = katman_sayisi
        self.expansion = expansion
        self.aktivasyon_adi = aktivasyon

        derinlik = katman_sayisi // 2

        # Boyut şeması: n, n·e, n·e², ..., n·e^d  (up)
        #             n·e^d, ..., n·e, n           (down)
        sizes_up = [n_base * (expansion ** i) for i in range(derinlik + 1)]
        sizes_down = list(reversed(sizes_up))

        # Up katmanları: sizes_up[i] → sizes_up[i+1]
        self.up_layers = nn.ModuleList([
            GenelBilinearKatmani(sizes_up[i], sizes_up[i + 1])
            for i in range(derinlik)
        ])
        self.up_norms = nn.ModuleList([
            nn.LayerNorm(sizes_up[i]) for i in range(derinlik)
        ])

        # Down katmanları: sizes_down[i] → sizes_down[i+1]
        self.down_layers = nn.ModuleList([
            GenelBilinearKatmani(sizes_down[i], sizes_down[i + 1])
            for i in range(derinlik)
        ])
        self.down_norms = nn.ModuleList([
            nn.LayerNorm(sizes_down[i]) for i in range(derinlik)
        ])

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _akt(x, ad):
        ad = (ad or "silu").lower()
        if ad == "silu":
            return F.silu(x)
        if ad == "gelu":
            return F.gelu(x)
        if ad == "tanh":
            return torch.tanh(x)
        if ad in ("identity", "none", "yok"):
            return x
        raise ValueError(f"bilinmeyen aktivasyon: {ad}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_base, n_base)
        skip_stack = [x]

        # Up fazı
        h = x
        for layer, norm in zip(self.up_layers, self.up_norms):
            h = layer(self._akt(norm(h), self.aktivasyon_adi))
            h = self.dropout(h)
            skip_stack.append(h)

        # skip_stack: [x, h1, h2, ..., h_d]
        # h_d = peak (B, n·e^d, n·e^d)

        # Down fazı (skip'ler ters sırayla)
        for i, (layer, norm) in enumerate(zip(self.down_layers, self.down_norms)):
            h = layer(self._akt(norm(h), self.aktivasyon_adi))
            h = self.dropout(h)
            # Skip: peak'ten aşağı inerken karşılık gelen up çıktısıyla topla
            # i=0 → skip ile skip_stack[-2] topla
            skip = skip_stack[-(i + 2)]
            h = h + skip

        return h

    def kapasite(self) -> dict:
        return {
            "n_base": self.n_base,
            "katman_sayisi": self.katman_sayisi,
            "expansion": self.expansion,
            "gercek_parametre": sum(p.numel() for p in self.parameters()),
            "sanal_operator_girdisi": sum(
                layer.kapasite()["sanal_operator_girdisi"]
                for layer in list(self.up_layers) + list(self.down_layers)
            ),
        }
