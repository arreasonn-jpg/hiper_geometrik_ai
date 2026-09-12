# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HiperGeometrikAttention(nn.Module):
    """Fused-QKV + Pre-LN + ReZero + Scaled Dot Product Attention katmanı.

    İyileştirmeler:
    1. Fused QKV projeksiyonu (3 ayrı Linear yerine tek matris).
    2. Pre-LN düzeni (katman öncesi LayerNorm → eğitimde daha kararlı).
    3. ReZero: ``alpha`` sıfırla başlar; katman başta birim (identity) fonksiyonu
       gibi davranır ama gradyan alta akmaya devam eder.
    4. ``F.scaled_dot_product_attention`` (PyTorch 2.0+): donanım destekliyorsa
       Flash Attention çekirdeğini kullanır.

    ``is_causal=True`` yapılırsa dikkat maskesi nedensel (causal) olur. Bu proje
    her adımda sabit bir pencereden tek bir sonraki-token tahmini ürettiği için
    varsayılan ``False`` bırakılmıştır; katmanı gerçek bir decoder/GPT gövdesine
    taşırsanız mutlaka ``True`` kullanın.
    """

    def __init__(self, emb_dim: int, num_heads: int, dropout: float = 0.1, is_causal: bool = False):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.is_causal = is_causal

        assert self.head_dim * num_heads == emb_dim, "emb_dim, num_heads'e tam bölünmelidir!"

        # 1. OPTİMİZASYON: Fused QKV Projeksiyonu (3 ayrı katman yerine tek büyük matris)
        self.qkv_proj = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=False)

        # 2. OPTİMİZASYON: Pre-LN Düzeni (Eğitim kararlılığı için katman öncesi normalizasyon)
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

        # 3. OPTİMİZASYON: ReZero Yaklaşımı (Gradyanı kesmeyen öğrenilebilir skaler başlangıç)
        self.alpha = nn.Parameter(torch.zeros(1))

        self._reset_parameters()

    def _reset_parameters(self):
        # QKV ve Çıkış projeksiyonları için standart Xavier/Glorot başlatması
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Sequence_Length, Embedding_Dim)
        B, S, D = x.shape
        residual = x

        # 4. OPTİMİZASYON: Pre-LN Uygulaması
        x_norm = self.norm(x)

        # Fused QKV hesaplaması -> (B, S, 3*D)
        qkv = self.qkv_proj(x_norm)

        # Q, K, V parçalarına ayırma (Chunking) -> Her biri (B, S, D)
        q, k, v = qkv.chunk(3, dim=-1)

        # Çoklu kafa (Multi-head) için tensörleri yeniden şekillendirme
        # Son boyut düzeni Flash Attention için: (B, num_heads, S, head_dim) olmalıdır
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # 5. OPTİMİZASYON: PyTorch 2.0+ Scaled Dot Product (Flash Attention) Entegrasyonu
        context = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=self.is_causal,
        )

        # Kafaları birleştirme (Merge heads) -> (B, S, D)
        context = context.transpose(1, 2).contiguous().view(B, S, D)

        # Çıkış projeksiyonu
        out = self.out_proj(context)

        # ReZero mekanizması ile çıktı ekleme: Başlangıçta alpha=0 olduğu için katman
        # kimlik fonksiyonu (identity) gibi davranır ama alpha üzerinden alta gradyan akar!
        return residual + self.alpha * out
