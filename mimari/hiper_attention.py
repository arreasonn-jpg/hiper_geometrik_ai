# -*- coding: utf-8 -*-
"""
Hiper-Geometrik Attention — Nedensel Çok Kafalı Dikkat (SDPA / Flash Attention)
================================================================================

Önceki sürümde bu dosya ÖLÜ KODTU: model `nn.MultiheadAttention` kullanıyordu,
bu katman hiç çağrılmıyor ve `is_causal=False` sabitlenmişti. Artık:
  - model bu katmani GERÇEKTEN kullanıyor (bkz. `kuresel_model.py`),
  - `is_causal=True` varsayılandır → bir token yalnızca GEÇMİŞ tokenlara
    bakabilir (otoregresif dil modeli için zorunlu — rapor 8.4.5).

Korunan optimizasyonlar:
  1. Fused QKV projeksiyonu (3 ayrı katman yerine tek büyük matris)
  2. Pre-LN düzeni (katman öncesi normalizasyon → eğitim kararlılığı)
  3. ReZero (alpha=0 başlangıç → katman başta kimlik fonksiyonu gibi davranır,
     gradyan alta akmaya devam eder)
  4. F.scaled_dot_product_attention — donanım destekliyorsa bellek dostu
     Flash Attention çekirdeğini kullanır (PyTorch 2.0+)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HiperGeometrikAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, dropout: float = 0.1,
                 is_causal: bool = True):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.is_causal = is_causal

        assert self.head_dim * num_heads == emb_dim, "emb_dim, num_heads'e tam bölünmelidir!"

        # 1. OPTİMİZASYON: Fused QKV projeksiyonu
        self.qkv_proj = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=False)

        # 2. OPTİMİZASYON: Pre-LN düzeni
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

        # 3. OPTİMİZASYON: ReZero — öğrenilebilir skaler, 0 ile başlatılır
        self.alpha = nn.Parameter(torch.zeros(1))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Sequence_Length, Embedding_Dim)
        B, S, D = x.shape
        residual = x

        # Pre-LN
        x_norm = self.norm(x)

        # Fused QKV → (B, S, 3*D) → parçala → (B, num_heads, S, head_dim)
        qkv = self.qkv_proj(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # 4. OPTİMİZASYON: SDPA (Flash Attention destekli) + NEDENSEL maske
        context = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=self.is_causal,
        )

        # Kafaları birleştir → (B, S, D)
        context = context.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(context)

        # ReZero: x + alpha * Attention(x)
        return residual + self.alpha * out
