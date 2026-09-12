# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hiper_attention import HiperGeometrikAttention
from .kuresel_bag import OptimizeEdilmisKureselBag
from .decoder import FraktalDecoder


class HiperGeometrikAI(nn.Module):
    """Kelime düzeyi, sabit pencere tabanlı küçük bir sonraki-token tahmin modeli.

    Dürüst mimari özeti (README ile uyumlu):
    1. ``kelime_gomme``  — kelime kimliklerini ``emb_dim`` boyutlu gömmelere çevirir,
       öğrenilen konum kodlaması eklenir.
    2. ``attention``     — ``HiperGeometrikAttention``: fused-QKV, Pre-LN, ReZero ve
       ``F.scaled_dot_product_attention`` (donanım destekliyorsa Flash Attention).
    3. ``u_kure``/``v_kure`` — pencereyi düzleştirip iki ayrı ``n`` boyutlu izdüşüme
       taşır; her izdüşüm L2-normalize edilir ve diğerinin kapısıyla (sigmoid)
       ölçeklenir.
    4. ``kuresel_bag``   — iki izdüşümü iki ``n x n`` 'mercek' matrisinden geçirip
       toplar (tek tensör).
    5. ``decoder``       — ``n`` boyutundan sözlük boyutuna lineer katman (logits).

    Parametre sayısı ~13,6 milyondur (n=1000, sözlük=8000); bu bir n-gram/pencere
    tabanlı dil modelidir, 'katrilyon sinaps' içeren bir mimari değildir.
    """

    def __init__(self, n=1000, baglam_penceresi=8, sozluk_boyutu=8000, emb_dim=64, num_heads=4):
        super().__init__()
        self.n = n
        self.baglam_penceresi = baglam_penceresi

        self.kelime_gomme = nn.Embedding(sozluk_boyutu, emb_dim, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.randn(1, baglam_penceresi, emb_dim) * 0.02)

        # Fused-QKV + Pre-LN + ReZero + SDPA (Flash Attention destekli) dikkat katmanı
        self.attention = HiperGeometrikAttention(emb_dim, num_heads, dropout=0.1)

        self.u_kure = nn.Linear(baglam_penceresi * emb_dim, n)
        self.v_kure = nn.Linear(baglam_penceresi * emb_dim, n)

        self.u_gate = nn.Linear(n, n)
        self.v_gate = nn.Linear(n, n)

        self.kuresel_bag = OptimizeEdilmisKureselBag(n)
        self.norm_bag = nn.LayerNorm(n)

        self.decoder = FraktalDecoder(n, sozluk_boyutu)

    def forward(self, x):
        """x: (B, baglam_penceresi) kelime kimlikleri → (B, sozluk_boyutu) logits."""
        seq_len = x.size(1)
        emb = self.kelime_gomme(x) + self.pos_encoder[:, :seq_len, :]

        # Dikkat katmanı kendi Pre-LN + rezidüel bağlantısını içerir
        emb = self.attention(emb)

        duz = emb.reshape(emb.size(0), -1)

        u = torch.relu(self.u_kure(duz))
        v = torch.relu(self.v_kure(duz))

        u = F.normalize(u, p=2, dim=-1)
        v = F.normalize(v, p=2, dim=-1)

        u_attn = u * torch.sigmoid(self.v_gate(v))
        v_attn = v * torch.sigmoid(self.u_gate(u))

        # İki izdüşümün BİRLEŞİMİ: kuresel_bag artık tek tensör döndürür
        bag = self.kuresel_bag(u_attn, v_attn)

        bag = self.norm_bag(bag)
        logits = self.decoder(bag)
        return logits
