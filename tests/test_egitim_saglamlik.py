# -*- coding: utf-8 -*-
"""AMP/checkpoint sağlamlık helper testleri."""
import importlib.util
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIMARI = os.path.join(KOK, "mimari")
for _p in [KOK, MIMARI]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from egitim.saglamlik import (  # noqa: E402
    amp_dogrula,
    checkpoint_gradyan_dogrula,
    checkpoint_state_manifest,
    checkpoint_uyumluluk_raporu,
)


def _torch_yoksa_atla():
    return importlib.util.find_spec("torch") is None


def test_amp_dogrula_smoke():
    if _torch_yoksa_atla():
        print("  (torch yok — AMP sağlamlık testi atlandı)")
        return
    import torch
    from kuresel_model import HiperGeometrikAI

    model = HiperGeometrikAI(n=8, katman_sayisi=1, baglam_penceresi=4, emb_dim=8,
                             num_heads=2, sozluk_boyutu=32, dropout=0.0,
                             seyrek_tablo_boyutu=0, bilgilendir=False)
    x = torch.randint(4, 32, (2, 4))
    y = torch.randint(4, 32, (2,))
    r = amp_dogrula(model, x, y)
    assert r["ok"] is True
    assert r["loss_sonlu"] is True


def test_checkpoint_uyumluluk_raporu():
    if _torch_yoksa_atla():
        return
    import os
    import tempfile

    import torch
    from kuresel_model import HiperGeometrikAI

    torch.manual_seed(0)
    m1 = HiperGeometrikAI(n=8, katman_sayisi=1, baglam_penceresi=4, emb_dim=8,
                          num_heads=2, sozluk_boyutu=32, dropout=0.0,
                          seyrek_tablo_boyutu=0, bilgilendir=False)
    manifest = checkpoint_state_manifest(m1.state_dict())
    assert "kelime_gomme.weight" in manifest
    assert manifest["kelime_gomme.weight"]["shape"] == [32, 8]

    with tempfile.TemporaryDirectory() as tmp:
        yol = os.path.join(tmp, "ckpt.pt")
        torch.save({"checkpoint_version": "test-v1",
                    "model_state_dict": m1.state_dict()}, yol)
        ok = checkpoint_uyumluluk_raporu(m1, yol)
        assert ok["ok"] is True
        assert ok["checkpoint_version"] == "test-v1"
        assert ok["model_meta"] is None

        m2 = HiperGeometrikAI(n=8, katman_sayisi=1, baglam_penceresi=4, emb_dim=8,
                              num_heads=2, sozluk_boyutu=40, dropout=0.0,
                              seyrek_tablo_boyutu=0, bilgilendir=False)
        kotu = checkpoint_uyumluluk_raporu(m2, yol, strict=False)
        assert kotu["ok"] is False
        assert "kelime_gomme.weight" in kotu["sekil_uyumsuz"]


def test_checkpoint_gradyan_dogrula_smoke():
    if _torch_yoksa_atla():
        return
    import torch
    from kuresel_bag import KureselZincir

    torch.manual_seed(0)
    x = torch.randn(2, 8, 8)
    r = checkpoint_gradyan_dogrula(
        lambda ckpt: KureselZincir(8, katman_sayisi=2, dropout=0.0, checkpoint_kullan=ckpt), x)
    assert r["ok"] is True


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
