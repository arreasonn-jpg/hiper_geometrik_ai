# -*- coding: utf-8 -*-
"""
[STATÜ: LEGACY / DEPRECATED] — Karakter/Kelime Bazlı Eski Tokenizer
===================================================================
(Roadmap P0-003, P0-004 — Tek standart: mimari.bpe_tokenizer.BPETokenizer)

DİKKAT: HGA mimarisinin tek standart tokenizasyon motoru `mimari/bpe_tokenizer.py`
(BPETokenizer) sınıfıdır. Bu dosya sadece eski checkpoint ve test uyumluluğu için
korunmaktadır.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, Optional

try:
    from model_config import VARSAYILAN_MODEL_CONFIG
except Exception:  # paket import'u
    try:
        from .model_config import VARSAYILAN_MODEL_CONFIG
    except Exception:  # çok eski ortam fallback'i
        VARSAYILAN_MODEL_CONFIG = None  # type: ignore


def _varsayilan_vocab() -> int:
    return int(getattr(VARSAYILAN_MODEL_CONFIG, "sozluk_boyutu", 8000))


class GeometrikTokenizer:
    PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
    OZEL = {"<PAD>": PAD_ID, "<UNK>": UNK_ID, "<BOS>": BOS_ID, "<EOS>": EOS_ID}

    def __init__(self, max_vocab_size: Optional[int] = None):
        self.max_vocab_size = int(max_vocab_size if max_vocab_size is not None else _varsayilan_vocab())
        self.sozluk: Dict[str, int] = dict(self.OZEL)
        self.id_to_kelime: Dict[int, str] = {v: k for k, v in self.sozluk.items()}

    @staticmethod
    def _turkce_kucult(metin: str) -> str:
        metin = str(metin).replace("İ", "i").replace("I", "ı")
        return unicodedata.normalize("NFC", metin.lower())

    def _kelimeler(self, metin: str):
        return re.findall(r"\b\w+\b", self._turkce_kucult(metin), flags=re.UNICODE)

    def fit(self, metin):
        kelimeler = self._kelimeler(metin)
        en_cok = Counter(kelimeler).most_common(max(0, self.max_vocab_size - len(self.OZEL)))
        self.sozluk = dict(self.OZEL)
        self.id_to_kelime = {v: k for k, v in self.sozluk.items()}
        for idx, (k, _) in enumerate(en_cok, start=len(self.OZEL)):
            self.sozluk[k] = idx
            self.id_to_kelime[idx] = k
        return self

    def sozluk_olustur(self, metin):
        return self.fit(metin)

    def kaydet(self, yol="sozluk.json"):
        with open(yol, "w", encoding="utf-8") as f:
            json.dump({"sozluk": self.sozluk}, f, ensure_ascii=False)

    def yukle(self, yol="sozluk.json"):
        with open(yol, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.sozluk = {k: int(v) for k, v in data["sozluk"].items()}
        self.id_to_kelime = {int(v): k for k, v in self.sozluk.items()}
        return self

    def text_to_ids(self, metin):
        return [self.sozluk.get(w, self.UNK_ID) for w in self._kelimeler(metin)]

    def encode(self, metin):
        return self.text_to_ids(metin)

    def decode(self, ids: Iterable[int]):
        return " ".join(self.id_to_kelime.get(int(i), "") for i in ids if int(i) > self.EOS_ID)

    def ids_to_text(self, ids):
        return self.decode(ids)

    def ozel_tokenlari_dogrula(self) -> bool:
        return all(self.sozluk.get(tok) == idx and self.id_to_kelime.get(idx) == tok
                   for tok, idx in self.OZEL.items())

    def vocab_tutarliligi(self, embedding_boyutu: Optional[int] = None,
                          strict: bool = False) -> bool:
        ids = list(self.sozluk.values())
        ok = (self.ozel_tokenlari_dogrula() and len(ids) == len(set(ids)) and
              set(ids) == set(range(len(ids))))
        if embedding_boyutu is not None:
            ok = ok and int(embedding_boyutu) >= len(self.sozluk)
        if strict and not ok:
            raise ValueError("GeometrikTokenizer vocab/special-token tutarsız")
        return bool(ok)


__all__ = ["GeometrikTokenizer"]
