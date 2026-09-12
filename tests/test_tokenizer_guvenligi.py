# -*- coding: utf-8 -*-
"""
BPE tokenizer güvenlik/doğrulama testleri.
Çalıştırma:
    python tests/test_tokenizer_guvenligi.py
    pytest tests/test_tokenizer_guvenligi.py -q
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIMARI = os.path.join(KOK, "mimari")
for _p in [KOK, MIMARI]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bpe_tokenizer import BPETokenizer  # noqa: E402
from model_config import VARSAYILAN_MODEL_CONFIG  # noqa: E402


def _tok():
    metin = ("Şeker çığ gibi yağdı. Iğdır ve İstanbul Türkçe karakterleri korur. "
             "Çağrı, öğüt, ölçü, ümit ve ışık örnekleri. ") * 8
    tok = BPETokenizer(baglam_penceresi=8,
                       max_vocab_size=VARSAYILAN_MODEL_CONFIG.sozluk_boyutu,
                       min_freq=1)
    tok.fit_on_text(metin, verbose=False)
    return tok


def test_turkce_karakterler_ve_buyuk_i():
    tok = _tok()
    temiz = tok._metni_temizle("İSTANBUL IĞDIR ŞÇĞIÖÜ")
    assert temiz == ["istanbul", "ığdır", "şçğıöü"]
    assert "\u0307" not in "".join(temiz)  # birleşik nokta sızıntısı yok
    geri = tok.decode(tok.encode("Şeker çığ ölçü ışık İstanbul Iğdır"))
    for parca in ["şeker", "çığ", "ölçü", "ışık", "istanbul", "ığdır"]:
        assert parca in geri, geri


def test_special_token_ve_vocab_tutarliligi():
    tok = _tok()
    assert tok.ozel_tokenlari_dogrula(strict=True)["ok"] is True
    rapor = tok.vocab_tutarliligi(expected_size=tok.sozluk_boyutu,
                                  embedding_boyutu=tok.sozluk_boyutu,
                                  strict=True)
    assert rapor["ok"] is True
    bozuk = tok.vocab_tutarliligi(expected_size=tok.sozluk_boyutu + 1)
    assert bozuk["ok"] is False


def test_byte_level_fallback_bilinmeyen_karakter():
    tok = _tok()
    assert tok.byte_fallback_var is True
    metin = "robot 🤖"
    ids = tok.encode(metin)
    assert tok.UNK_ID not in ids
    assert tok.decode(ids) == metin


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
