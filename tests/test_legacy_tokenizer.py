# -*- coding: utf-8 -*-
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIMARI = os.path.join(KOK, "mimari")
if MIMARI not in sys.path:
    sys.path.insert(0, MIMARI)

from model_config import VARSAYILAN_MODEL_CONFIG
from tokenizer import GeometrikTokenizer


def test_legacy_tokenizer_config_varsayilan_ve_turkce():
    tok = GeometrikTokenizer()
    assert tok.max_vocab_size == VARSAYILAN_MODEL_CONFIG.sozluk_boyutu
    tok.fit("İstanbul Iğdır ışık işler")
    assert "istanbul" in tok.sozluk
    assert "ığdır" in tok.sozluk
    assert "i̇stanbul" not in tok.sozluk  # birleşik nokta sızıntısı yok
    assert tok.vocab_tutarliligi(strict=True) is True


def test_legacy_tokenizer_embedding_uyumsuzlugu():
    tok = GeometrikTokenizer(max_vocab_size=10).fit("a b c d e f")
    assert tok.vocab_tutarliligi(embedding_boyutu=len(tok.sozluk)) is True
    assert tok.vocab_tutarliligi(embedding_boyutu=3) is False


if __name__ == "__main__":
    test_legacy_tokenizer_config_varsayilan_ve_turkce()
    test_legacy_tokenizer_embedding_uyumsuzlugu()
    print("OK")
