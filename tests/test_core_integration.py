# -*- coding: utf-8 -*-
"""Tokenizer → Embedding → Attention → Geometric → Sparse Memory → Decoder entegrasyonu."""
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


def test_tokenizer_model_uclu_zincir_ileri_geri():
    if _torch_yoksa_atla():
        print("  (torch yok — core entegrasyon testi atlandı)")
        return
    import torch
    from bpe_tokenizer import BPETokenizer
    from kuresel_model import HiperGeometrikAI

    tok = BPETokenizer(baglam_penceresi=8, max_vocab_size=512, min_freq=1)
    tok.fit_on_text(("Ali ataya bindi. Ayşe arabaya bindi. ") * 10, verbose=False)
    vocab = max(tok.sozluk_boyutu, 64)
    model = HiperGeometrikAI(n=16, katman_sayisi=2, baglam_penceresi=8, emb_dim=16,
                             num_heads=4, sozluk_boyutu=vocab, dropout=0.0,
                             seyrek_tablo_boyutu=128, seyrek_boyut=8,
                             bilgilendir=False)
    x = torch.tensor([tok.encode("Ali ataya bindi", pad=True)], dtype=torch.long)
    logits = model(x)
    assert logits.shape == (1, vocab)
    hedef = torch.tensor([tok.EOS_ID], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits, hedef)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_aynı_seed_aynı_cikti():
    if _torch_yoksa_atla():
        return
    import torch
    from kuresel_model import HiperGeometrikAI

    cfg = dict(n=16, katman_sayisi=2, baglam_penceresi=4, emb_dim=8,
               num_heads=2, sozluk_boyutu=64, dropout=0.0,
               seyrek_tablo_boyutu=64, seyrek_boyut=4, bilgilendir=False)
    torch.manual_seed(123)
    m1 = HiperGeometrikAI(**cfg).eval()
    torch.manual_seed(123)
    m2 = HiperGeometrikAI(**cfg).eval()
    x = torch.tensor([[1, 2, 3, 4], [0, 0, 5, 6]], dtype=torch.long)
    y1, y2 = m1(x), m2(x)
    assert torch.allclose(y1, y2, atol=1e-6)


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
