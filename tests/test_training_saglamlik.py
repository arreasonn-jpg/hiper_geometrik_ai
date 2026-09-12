# -*- coding: utf-8 -*-
"""
Eğitim/çekirdek sağlamlık testleri (torch varsa çalışır).
"""
import importlib.util
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIMARI = os.path.join(KOK, "mimari")
for _p in [KOK, MIMARI]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _torch_yoksa_atla():
    return importlib.util.find_spec("torch") is None


def test_checkpoint_backward_gradyan_esdegerligi():
    if _torch_yoksa_atla():
        print("  (torch yok — checkpoint testi atlandı)")
        return
    import torch
    from kuresel_bag import KureselZincir

    torch.manual_seed(0)
    z1 = KureselZincir(8, katman_sayisi=2, dropout=0.0, checkpoint_kullan=False)
    z2 = KureselZincir(8, katman_sayisi=2, dropout=0.0, checkpoint_kullan=True)
    z2.load_state_dict(z1.state_dict())
    x1 = torch.randn(2, 8, 8, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1 = z1(x1).sum()
    y2 = z2(x2).sum()
    y1.backward()
    y2.backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-6)
    for (n1, p1), (n2, p2) in zip(z1.named_parameters(), z2.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1.grad, p2.grad, atol=1e-6), n1


def test_attention_padding_mask_sifirlar():
    if _torch_yoksa_atla():
        return
    import torch
    from hiper_attention import HiperGeometrikAttention

    att = HiperGeometrikAttention(emb_dim=8, num_heads=2, dropout=0.0, is_causal=True).eval()
    att.alpha.data.fill_(1.0)
    x = torch.randn(1, 5, 8)
    mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool)
    y = att(x, attention_mask=mask)
    assert torch.allclose(y[:, :2], torch.zeros_like(y[:, :2]), atol=1e-6)
    assert torch.isfinite(y).all()


def test_train_val_bol_baglam_boslugu():
    if _torch_yoksa_atla():
        return
    from egitim.egitici import KureselEgitimMotoru

    ids = list(range(100))
    train, val = KureselEgitimMotoru.train_val_bol(ids, baglam=8, validation_split=0.2)
    assert val is not None
    assert max(train) < min(val) - 1  # arada boşluk var
    assert min(val) - max(train) >= 9  # 8 token gap + ardışık fark


def test_kuresel_egitim_motoru_tiny_checkpoint_log():
    if _torch_yoksa_atla():
        return
    import contextlib
    import io
    import json
    import os
    import tempfile
    import torch
    from egitim.egitici import KureselEgitimMotoru
    from kuresel_model import HiperGeometrikAI

    torch.manual_seed(123)
    model = HiperGeometrikAI(n=8, katman_sayisi=1, baglam_penceresi=4, emb_dim=8,
                             num_heads=2, sozluk_boyutu=32, dropout=0.0,
                             seyrek_tablo_boyutu=0, bilgilendir=False)
    ids = [4 + (i % 20) for i in range(80)]
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = os.path.join(tmp, "logs")
        ckpt_dir = os.path.join(tmp, "ckpt")
        motor = KureselEgitimMotoru(
            model, ogrenme_hizi=1e-3, toplam_cag=2, karisik_hassasiyet=False,
            grad_clip=0.5, warmup_cag=1, lr_min_factor=0.1,
            log_dizini=log_dir, checkpoint_dizini=ckpt_dir,
            son_checkpoint_sayisi=1,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            motor.ngram_egitim_dongusu(ids, cag_sayisi=2, batch_size=8,
                                       validation_split=0.2)
        assert len(motor.cag_gecmisi) == 2
        assert all(r.train_loss == r.train_loss for r in motor.cag_gecmisi)
        assert motor.gradient_gecmisi and all(g.sonlu for g in motor.gradient_gecmisi)
        assert os.path.exists(os.path.join(log_dir, "metrics.csv"))
        jsonl_yol = os.path.join(log_dir, "metrics.jsonl")
        assert os.path.exists(jsonl_yol)
        with open(jsonl_yol, "r", encoding="utf-8") as f:
            satirlar = [json.loads(s) for s in f if s.strip()]
        assert satirlar[-1]["run"] == "temel"
        latest = os.path.join(ckpt_dir, "latest.pt")
        assert os.path.exists(latest)
        assert os.path.exists(os.path.join(ckpt_dir, "best.pt"))
        assert len([p for p in os.listdir(ckpt_dir) if p.startswith("epoch_")]) <= 1
        ckpt = torch.load(latest, map_location="cpu")
        assert ckpt["checkpoint_version"] == "hga-train-v1"
        assert ckpt["model_meta"]["baglam_penceresi"] == 4

        yeni = HiperGeometrikAI(n=8, katman_sayisi=1, baglam_penceresi=4, emb_dim=8,
                                num_heads=2, sozluk_boyutu=32, dropout=0.0,
                                seyrek_tablo_boyutu=0, bilgilendir=False)
        motor2 = KureselEgitimMotoru(yeni, toplam_cag=3, karisik_hassasiyet=False)
        assert motor2.checkpoint_yukle(latest) == 2
        with contextlib.redirect_stdout(io.StringIO()):
            motor2.ngram_egitim_dongusu(ids, cag_sayisi=3, batch_size=8,
                                        validation_split=0.2)
        assert [r.cag for r in motor2.cag_gecmisi] == [3]


def test_attention_kv_cache_full_ile_esdeger():
    if _torch_yoksa_atla():
        return
    import torch
    from hiper_attention import HiperGeometrikAttention

    torch.manual_seed(7)
    att = HiperGeometrikAttention(emb_dim=8, num_heads=2, dropout=0.0, is_causal=True).eval()
    att.alpha.data.fill_(1.0)
    x = torch.randn(2, 5, 8)
    tam = att(x)
    cache = None
    parcalar = []
    for i in range(x.size(1)):
        y, cache = att.forward_cacheli(x[:, i:i + 1], cache=cache)
        parcalar.append(y)
    ink = torch.cat(parcalar, dim=1)
    assert torch.allclose(tam, ink, atol=1e-5)
    assert cache["k"].shape == (2, 2, 5, 4)


if __name__ == "__main__":
    testler = [(ad, fn) for ad, fn in sorted(globals().items())
               if ad.startswith("test_") and callable(fn)]
    basarisiz = 0
    for ad, fn in testler:
        try:
            fn()
            print(f"  ✅ {ad}")
        except Exception as e:  # noqa: BLE001
            basarisiz += 1
            print(f"  ❌ {ad}: {type(e).__name__}: {e}")
    print("\nTÜM TESTLER GEÇTİ" if basarisiz == 0 else f"{basarisiz} test başarısız")
    sys.exit(1 if basarisiz else 0)
