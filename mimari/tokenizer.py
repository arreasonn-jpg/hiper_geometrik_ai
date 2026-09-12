# -*- coding: utf-8 -*-
import re, json, os
from collections import Counter

class GeometrikTokenizer:
    def __init__(self, max_vocab_size=8000):
        self.max_vocab_size = max_vocab_size
        self.PAD_ID, self.UNK_ID, self.BOS_ID, self.EOS_ID = 0, 1, 2, 3
        self.sozluk = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.id_to_kelime = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"}

    def fit(self, metin):
        kelimeler = re.findall(r"\b\w+\b", metin.lower())
        en_cok = Counter(kelimeler).most_common(self.max_vocab_size - 4)
        self.sozluk = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.id_to_kelime = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"}
        for idx, (k, _) in enumerate(en_cok, start=4):
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
        return [self.sozluk.get(w, self.UNK_ID) for w in re.findall(r"\b\w+\b", metin.lower())]

    def encode(self, metin):
        return self.text_to_ids(metin)

    def decode(self, ids):
        return " ".join(self.id_to_kelime.get(i, "") for i in ids if i > 3)

    def ids_to_text(self, ids):
        return self.decode(ids)