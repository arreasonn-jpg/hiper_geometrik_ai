# -*- coding: utf-8 -*-
"""OptimizeEdilmisKureselBag: birleşim ve gradyan akışı testleri.

REGRESYON: Eski sürümde forward (u_yeni, v_yeni) tuple döndürüyor, model
yalnızca bag[0]'ı kullandığı için mercek_B HİÇ gradyan almıyordu (ölü parametre).
"""
import torch

from mimari.kuresel_bag import OptimizeEdilmisKureselBag


def test_forward_tek_tensor_donduur():
    katman = OptimizeEdilmisKureselBag(n=16)
    u = torch.randn(4, 16)
    v = torch.randn(4, 16)
    cikti = katman(u, v)
    assert isinstance(cikti, torch.Tensor), "forward tuple değil tek tensör döndürmeli"
    assert cikti.shape == (4, 16)


def test_her_iki_mercek_de_gradyan_aliyor():
    katman = OptimizeEdilmisKureselBag(n=16)
    u = torch.randn(4, 16, requires_grad=True)
    v = torch.randn(4, 16, requires_grad=True)
    katman(u, v).sum().backward()
    assert katman.mercek_A.weight.grad is not None
    assert katman.mercek_B.weight.grad is not None, (
        "mercek_B ölü parametre olmamalı: v yolu da loss'a katkı vermelidir"
    )


def test_bilesim_sifir_girdiyle():
    """u ve v sıfırsa aktivasyon sonrası çıktı da sıfıra yakın olmalı."""
    katman = OptimizeEdilmisKureselBag(n=8)
    cikti = katman(torch.zeros(2, 8), torch.zeros(2, 8))
    assert torch.allclose(cikti, torch.zeros_like(cikti))
