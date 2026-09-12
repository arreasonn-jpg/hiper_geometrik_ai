# -*- coding: utf-8 -*-
"""Talimat fine-tuning helper testleri (torch gerektirmez)."""
import json
import os
import sys
import tempfile

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIMARI = os.path.join(KOK, "mimari")
EGITIM = os.path.join(KOK, "egitim")
for _p in [KOK, MIMARI, EGITIM]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bpe_tokenizer import BPETokenizer  # noqa: E402
from talimat_egitici import _csv_logla, talimat_ornekleri_olustur, train_val_bol  # noqa: E402


def _tok():
    metin = "soru merhaba cevap selam son soru nasılsın cevap iyiyim teşekkürler son"
    tok = BPETokenizer(baglam_penceresi=4, max_vocab_size=500, min_freq=1)
    tok.fit_on_text(metin, verbose=False)
    return tok


def test_talimat_ornekleri_sadece_cevap_sonrasi():
    tok = _tok()
    X, Y = talimat_ornekleri_olustur(tok, [{"soru": "merhaba", "cevap": "selam"}], baglam=4)
    assert len(X) == len(Y) > 0
    assert all(len(x) == 4 for x in X)
    decoded_targets = tok.decode(Y)
    assert "selam" in decoded_targets or "son" in decoded_targets


def test_talimat_train_val_bol():
    X = [[i] for i in range(10)]
    Y = list(range(10))
    Xtr, Ytr, Xv, Yv = train_val_bol(X, Y, validation_split=0.2)
    assert len(Xtr) == len(Ytr) == 8
    assert len(Xv) == len(Yv) == 2
    assert Xv == [[8], [9]]


def test_talimat_jsonl_log():
    with tempfile.TemporaryDirectory() as tmp:
        _csv_logla(tmp, {"epoch": 1, "train_loss": 1.2, "lr": 0.001})
        assert os.path.exists(os.path.join(tmp, "talimat_metrics.csv"))
        with open(os.path.join(tmp, "talimat_metrics.jsonl"), "r", encoding="utf-8") as f:
            satir = json.loads(f.readline())
        assert satir["run"] == "talimat"
        assert satir["epoch"] == 1


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
