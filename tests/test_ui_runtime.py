# -*- coding: utf-8 -*-
import os
import sys
from types import SimpleNamespace

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.ui_runtime import CEVAP_ETIKETI, SORU_ETIKETI, model_vocab_boyutu


class Tok:
    def __init__(self, n):
        self.sozluk = {str(i): i for i in range(n)}


def test_ui_runtime_torchsuz_import_edilir():
    assert SORU_ETIKETI == "soru"
    assert CEVAP_ETIKETI == "cevap"


def test_model_vocab_kucuk_sozlukte_confige_doner():
    cfg = SimpleNamespace(sozluk_boyutu=8000)
    assert model_vocab_boyutu(Tok(3), cfg) == 8000
    assert model_vocab_boyutu(Tok(128), cfg) == 128


if __name__ == "__main__":
    test_ui_runtime_torchsuz_import_edilir()
    test_model_vocab_kucuk_sozlukte_confige_doner()
    print("OK")
