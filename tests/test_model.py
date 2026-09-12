# -*- coding: utf-8 -*-
"""HiperGeometrikAI: forward şekli ve tam gradyan akışı testleri.

REGRESYON: Eski model kuresel_bag'in yalnızca u çıktısını kullandığı için
mercek_B (ve v yolu) hiç eğitilmiyordu. Bu test TÜM parametrelerin gradyan
aldığını doğrular.
"""
import torch

from mimari.kuresel_model import HiperGeometrikAI
from ortak import model_olustur


def kucuk_model(**kw):
    varsayilan = dict(n=32, baglam_penceresi=4, sozluk_boyutu=50, emb_dim=16, num_heads=4)
    varsayilan.update(kw)
    return HiperGeometrikAI(**varsayilan)


def test_forward_logit_sekli():
    model = kucuk_model()
    x = torch.randint(0, 50, (2, 4))
    logits = model(x)
    assert logits.shape == (2, 50)


def test_tum_parametreler_gradyan_aliyor():
    model = kucuk_model()
    x = torch.randint(0, 50, (4, 4))
    y = torch.randint(0, 50, (4,))
    model(x).shape  # forward çalışıyor
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()

    olen = [ad for ad, p in model.named_parameters() if p.grad is None]
    assert not olen, f"Gradyan almayan (ölü) parametreler: {olen}"


def test_dikkat_katmani_hiper_geometrik():
    """Model nn.MultiheadAttention değil HiperGeometrikAttention kullanmalı."""
    from mimari.hiper_attention import HiperGeometrikAttention
    model = kucuk_model()
    assert isinstance(model.attention, HiperGeometrikAttention)


def test_model_olustur_n_parametresini_iletir():
    """REGRESYON: eski model_olustur kopyaları 'n' adını hiç eşleştiremiyordu."""
    m64 = model_olustur(sozluk_boyutu=50, n=64, baglam_penceresi=4, emb_dim=16)
    m128 = model_olustur(sozluk_boyutu=50, n=128, baglam_penceresi=4, emb_dim=16)
    assert m64.n == 64
    assert m64.u_kure.out_features == 64
    assert m128.u_kure.out_features == 128


def test_model_olustur_sozluk_boyutu_iletilir():
    model = model_olustur(sozluk_boyutu=123, n=32, baglam_penceresi=4, emb_dim=16)
    assert model.kelime_gomme.num_embeddings == 123
    assert model.decoder.anlam_cozucu.out_features == 123
