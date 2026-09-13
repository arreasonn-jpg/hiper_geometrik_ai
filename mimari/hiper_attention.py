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

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Dikkat ileri geçişi.

        ``attention_mask`` opsiyoneldir ve ``(B, S)`` biçiminde 1/True = gerçek
        token, 0/False = PAD anlamına gelir. Verildiğinde PAD anahtarları
        maskelenir; PAD sorguların çıktısı da sıfırlanır. Bu, değişken
        uzunluklu batch'lerde modelin PAD token'larından öğrenmesini engeller.
        """
        # x: (Batch, Sequence_Length, Embedding_Dim)
        B, S, D = x.shape
        residual = x

        pad_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (B, S):
                raise ValueError(f"attention_mask şekli {(B, S)} olmalı, gelen: {tuple(attention_mask.shape)}")
            pad_mask = attention_mask.to(device=x.device, dtype=torch.bool)
            # PAD pozisyonları residual/LayerNorm yoluna pozisyon kodu sızdırmasın.
            x = x * pad_mask.unsqueeze(-1).to(dtype=x.dtype)
            residual = x

        # Pre-LN
        x_norm = self.norm(x)

        # Fused QKV → (B, S, 3*D) → parçala → (B, num_heads, S, head_dim)
        qkv = self.qkv_proj(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # 4. OPTİMİZASYON: SDPA (Flash Attention destekli) + NEDENSEL/PAD maske
        # PyTorch SDPA'da bool maskede True = dikkat edilebilir konumdur.
        attn_mask = None
        is_causal = self.is_causal
        if pad_mask is not None:
            key_mask = pad_mask.view(B, 1, 1, S)
            if self.is_causal:
                causal = torch.ones(S, S, device=x.device, dtype=torch.bool).tril()
                attn_mask = key_mask & causal.view(1, 1, S, S)
                is_causal = False  # attn_mask ile nedenselliği zaten birleştirdik
            else:
                attn_mask = key_mask.expand(B, 1, S, S)
            # Tamamen PAD olan sorgu satırlarında SDPA softmax(NaN) üretmesin.
            # Çıktı aşağıda tekrar PAD maskesiyle sıfırlanacağı için güvenli.
            bos_satir = ~attn_mask.any(dim=-1, keepdim=True)
            if bos_satir.any():
                guvenli = torch.zeros_like(attn_mask)
                guvenli[..., 0] = True
                attn_mask = torch.where(bos_satir, guvenli, attn_mask)

        context = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

        # Kafaları birleştir → (B, S, D)
        context = context.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(context)

        # ReZero: x + alpha * Attention(x)
        out = residual + self.alpha * out
        if pad_mask is not None:
            out = out * pad_mask.unsqueeze(-1).to(dtype=out.dtype)
        cikti: torch.Tensor = out
        return cikti

    @torch.no_grad()
    def attention_haritasi(self, x: torch.Tensor,
                           attention_mask: torch.Tensor | None = None,
                           kafa_ortalama: bool = True) -> torch.Tensor:
        """Gözlemlenebilirlik için dikkat ısı haritası döndür.

        Dönüş varsayılan olarak ``(B, S, S)`` kafa ortalamasıdır. ``kafa_ortalama=False``
        verilirse ``(B, H, S, S)`` ham kafa ağırlıkları döner. Bu yol eğitimde
        kullanılmaz; küçük/orta uzunluklu analizler içindir.
        """
        was_training = self.training
        self.eval()
        B, S, D = x.shape
        pad_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (B, S):
                raise ValueError(f"attention_mask şekli {(B, S)} olmalı, gelen: {tuple(attention_mask.shape)}")
            pad_mask = attention_mask.to(device=x.device, dtype=torch.bool)
            x = x * pad_mask.unsqueeze(-1).to(dtype=x.dtype)

        x_norm = self.norm(x)
        qkv = self.qkv_proj(x_norm)
        q, k, _v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        skor = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.ones(S, S, device=x.device, dtype=torch.bool)
        if self.is_causal:
            mask = mask.tril()
        mask = mask.view(1, 1, S, S)
        if pad_mask is not None:
            mask = mask & pad_mask.view(B, 1, 1, S)
        bos_satir = ~mask.any(dim=-1, keepdim=True)
        if bos_satir.any():
            guvenli = torch.zeros_like(mask)
            guvenli[..., 0] = True
            mask = torch.where(bos_satir, guvenli, mask)
        skor = skor.masked_fill(~mask, torch.finfo(skor.dtype).min)
        agirlik = torch.softmax(skor, dim=-1)
        if pad_mask is not None:
            agirlik = agirlik * pad_mask.view(B, 1, S, 1).to(dtype=agirlik.dtype)
        if was_training:
            self.train()
        return agirlik.mean(dim=1) if kafa_ortalama else agirlik

    def forward_cacheli(self, x: torch.Tensor, cache: dict | None = None,
                        attention_mask: torch.Tensor | None = None):
        """İncremental inference için KV cache'li attention.

        Bu API attention katmanı seviyesindedir: ``x`` yalnız yeni token(lar)ın
        embedding'idir ``(B, T, D)``. ``cache`` önceki çağrıdan dönen
        ``{"k": ..., "v": ..., "mask": ...}`` sözlüğüdür. Dönüş:
        ``(out, yeni_cache)``.

        Not: HGA'nın geometrik encoder'ı tüm bağlamı düzleştirdiği için tam
        modelde uzun-context hızlandırması ayrıca bir sliding-window cache
        orkestrasyonu gerektirir. Bu yöntem KV cache çekirdeğini doğrular ve
        o entegrasyon için sabit API sağlar.
        """
        B, T, D = x.shape
        residual = x
        pad_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (B, T):
                raise ValueError(f"attention_mask şekli {(B, T)} olmalı, gelen: {tuple(attention_mask.shape)}")
            pad_mask = attention_mask.to(device=x.device, dtype=torch.bool)
            x = x * pad_mask.unsqueeze(-1).to(dtype=x.dtype)
            residual = x

        x_norm = self.norm(x)
        qkv = self.qkv_proj(x_norm)
        q, k_new, v_new = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_new = k_new.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_new = v_new.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if cache is None:
            k, v = k_new, v_new
            gecmis = 0
            cache_mask = pad_mask
        else:
            k_old, v_old = cache.get("k"), cache.get("v")
            if k_old is None or v_old is None:
                raise ValueError("cache içinde 'k' ve 'v' olmalı")
            if k_old.shape[:2] != (B, self.num_heads) or v_old.shape[:2] != (B, self.num_heads):
                raise ValueError("cache batch/head boyutu mevcut girdiyle uyuşmuyor")
            gecmis = int(k_old.size(2))
            k = torch.cat([k_old, k_new], dim=2)
            v = torch.cat([v_old, v_new], dim=2)
            eski_mask = cache.get("mask")
            if eski_mask is not None:
                eski_mask = eski_mask.to(device=x.device, dtype=torch.bool)
            if pad_mask is None:
                cache_mask = eski_mask
            elif eski_mask is None:
                cache_mask = torch.cat([
                    torch.ones(B, gecmis, device=x.device, dtype=torch.bool),
                    pad_mask], dim=1)
            else:
                cache_mask = torch.cat([eski_mask, pad_mask], dim=1)

        toplam = gecmis + T
        # Query q_i, geçmişin tamamına ve kendi bloğunda <= i konumlarına bakabilir.
        q_pos = torch.arange(T, device=x.device).view(T, 1)
        k_pos = torch.arange(toplam, device=x.device).view(1, toplam)
        mask = k_pos <= (gecmis + q_pos)
        mask = mask.view(1, 1, T, toplam).expand(B, 1, T, toplam)
        if cache_mask is not None:
            mask = mask & cache_mask.view(B, 1, 1, toplam)
        bos_satir = ~mask.any(dim=-1, keepdim=True)
        if bos_satir.any():
            guvenli = torch.zeros_like(mask)
            guvenli[..., 0] = True
            mask = torch.where(bos_satir, guvenli, mask)

        context = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        out = residual + self.alpha * self.out_proj(context)
        if pad_mask is not None:
            out = out * pad_mask.unsqueeze(-1).to(dtype=out.dtype)
        yeni_cache = {"k": k.detach(), "v": v.detach()}
        if cache_mask is not None:
            yeni_cache["mask"] = cache_mask.detach()
        return out, yeni_cache
