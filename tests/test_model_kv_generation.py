# -*- coding: utf-8 -*-
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)
MIMARI = os.path.join(KOK, "mimari")
if MIMARI not in sys.path:
    sys.path.insert(0, MIMARI)


def _torch_yoksa_atla():
    try:
        import torch  # noqa: F401
        return False
    except Exception:
        return True


def test_model_forward_cacheli_pencere_full_ile_esdeger():
    if _torch_yoksa_atla():
        return
    import torch
    from kuresel_model import HiperGeometrikAI

    torch.manual_seed(11)
    model = HiperGeometrikAI(n=8, katman_sayisi=1, baglam_penceresi=4,
                             emb_dim=8, num_heads=2, sozluk_boyutu=32,
                             dropout=0.0, seyrek_tablo_boyutu=0,
                             bilgilendir=False).eval()
    x = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 7]], dtype=torch.long)
    tam = model(x)
    cacheli, cache = model.forward_cacheli_pencere(x)
    assert torch.allclose(tam, cacheli, atol=1e-5)
    assert cache["dikkat"]["k"].shape[2] == 4
    assert cache["prefix_reused"] == 0


def test_model_forward_cacheli_pencere_prefix_reuse():
    if _torch_yoksa_atla():
        return
    import torch
    from kuresel_model import HiperGeometrikAI

    torch.manual_seed(12)
    model = HiperGeometrikAI(n=8, katman_sayisi=1, baglam_penceresi=4,
                             emb_dim=8, num_heads=2, sozluk_boyutu=32,
                             dropout=0.0, seyrek_tablo_boyutu=0,
                             bilgilendir=False).eval()
    x1 = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    x2 = torch.tensor([[1, 2, 3, 5]], dtype=torch.long)
    _, cache = model.forward_cacheli_pencere(x1)
    tam2 = model(x2)
    cacheli2, cache2 = model.forward_cacheli_pencere(x2, cache=cache)
    assert torch.allclose(tam2, cacheli2, atol=1e-5)
    assert cache2["prefix_reused"] == 3


if __name__ == "__main__":
    test_model_forward_cacheli_pencere_full_ile_esdeger()
    test_model_forward_cacheli_pencere_prefix_reuse()
    print("OK")
