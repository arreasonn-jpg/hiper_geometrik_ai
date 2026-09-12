# -*- coding: utf-8 -*-
"""Türkçe mini benchmark harness testleri."""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIMARI = os.path.join(KOK, "mimari")
for _p in [KOK, MIMARI]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bpe_tokenizer import BPETokenizer  # noqa: E402
from model_config import VARSAYILAN_MODEL_CONFIG  # noqa: E402
from hga.evaluation import mini_turkce_corpus, tokenizer_kapsami  # noqa: E402


def test_mini_turkce_corpus_karakter_kapsami():
    metin = " ".join(mini_turkce_corpus())
    for c in "şçğıöüİ":
        assert c in metin


def test_tokenizer_kapsami_unk_dusuk():
    corpus = mini_turkce_corpus()
    tok = BPETokenizer(max_vocab_size=VARSAYILAN_MODEL_CONFIG.sozluk_boyutu,
                       min_freq=1)
    tok.fit_on_text("\n".join(corpus), verbose=False)
    r = tokenizer_kapsami(tok, corpus)
    assert r["token_sayisi"] > 0
    assert r["unk_orani"] == 0.0
    assert all(r["turkce_karakter_kapsami"].values())


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
