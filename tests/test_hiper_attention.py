# -*- coding: utf-8 -*-
"""HiperGeometrikAttention: şekil, ReZero başlangıcı ve nedensellik testleri."""
import torch

from mimari.hiper_attention import HiperGeometrikAttention


def test_cikti_sekli_korunur():
    attn = HiperGeometrikAttention(emb_dim=16, num_heads=4)
    x = torch.randn(3, 8, 16)
    assert attn(x).shape == x.shape


def test_rezero_baslangicinda_kimlik_fonksiyonu():
    """alpha=0 ile başladığı için katman başta tam birim (identity) fonksiyondur."""
    attn = HiperGeometrikAttention(emb_dim=16, num_heads=4)
    x = torch.randn(2, 8, 16)
    assert torch.allclose(attn(x), x), "alpha=0 iken çıkış girdiye eşit olmalı"


def test_is_causal_parametresi_kabul_edilir():
    attn = HiperGeometrikAttention(emb_dim=16, num_heads=4, is_causal=True)
    x = torch.randn(2, 8, 16)
    assert attn(x).shape == x.shape
    assert attn.is_causal is True


def test_gradyan_akar():
    attn = HiperGeometrikAttention(emb_dim=16, num_heads=4)
    x = torch.randn(2, 8, 16)
    attn(x).sum().backward()
    assert attn.qkv_proj.weight.grad is not None
    assert attn.out_proj.weight.grad is not None
    assert attn.alpha.grad is not None
