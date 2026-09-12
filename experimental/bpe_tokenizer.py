"""
Hiper-Geometrik AI — Subword Reconstructor BPE Tokenizer
"""
from __future__ import annotations
import os, re, json, time
from collections import Counter
from typing import List, Dict, Tuple
import torch

class BPETokenizer:
    PAD_ID = 0; UNK_ID = 1; BOS_ID = 2; EOS_ID = 3
    OZEL = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    def __init__(self, baglam_penceresi: int = 16, max_vocab_size: int = 8000, min_freq: int = 2):
        self.baglam_penceresi = int(baglam_penceresi)
        self.max_vocab_size = int(max_vocab_size)
        self.min_freq = int(min_freq)
        self.kelime_to_id: Dict[str, int] = {}
        self.id_to_kelime: Dict[int, str] = {}
        self.subword_set: set = set()
        self.sozluk_boyutu: int = 0
        self._egitildi = False
        self._cache: Dict[str, List[str]] = {}

    def _metni_temizle(self, metin: str) -> List[str]:
        if not metin: return []
        metin = str(metin).lower()
        metin = re.sub(r"([.,!?;:\"'()\[\]{}…])", r" \1 ", metin)
        metin = re.sub(r"\s+", " ", metin).strip()
        return [k for k in metin.split(" ") if k]

    def fit_on_text(self, metin: str, verbose: bool = True) -> "BPETokenizer":
        t0 = time.time()
        kelimeler = self._metni_temizle(metin) or ["merhaba", "dünya"]
        wfreq = Counter(kelimeler)
        top_words = dict(wfreq.most_common(12000))

        k_sembolleri = {tuple(list(w) + ["</w>"]): f for w, f in top_words.items()}
        char_set = {s for syms in k_sembolleri for s in syms}

        merges = []
        for step in range(5000):
            ciftler = Counter()
            for s, f in k_sembolleri.items():
                if len(s) < 2: continue
                for i in range(len(s) - 1):
                    ciftler[(s[i], s[i + 1])] += f
            if not ciftler: break
            en_iyi, en_skor = ciftler.most_common(1)[0]
            if en_skor < self.min_freq: break

            a, b = en_iyi; birlesik = a + b; yeni = {}
            for s, f in k_sembolleri.items():
                s_list = list(s); i = 0; out = []
                while i < len(s_list):
                    if i < len(s_list) - 1 and s_list[i] == a and s_list[i + 1] == b:
                        out.append(birlesik); i += 2
                    else:
                        out.append(s_list[i]); i += 1
                yeni[tuple(out)] = yeni.get(tuple(out), 0) + f
            k_sembolleri = yeni
            merges.append(birlesik)

        self.kelime_to_id = {t: i for i, t in enumerate(self.OZEL)}
        next_id = len(self.OZEL)

        all_subwords = sorted(char_set | set(merges))
        for tok in all_subwords:
            if next_id >= self.max_vocab_size: break
            if tok not in self.kelime_to_id:
                self.kelime_to_id[tok] = next_id
                next_id += 1

        self.id_to_kelime = {v: k for k, v in self.kelime_to_id.items()}
        self.sozluk_boyutu = len(self.kelime_to_id)
        self.subword_set = set(self.kelime_to_id.keys())
        self._cache.clear()
        self._egitildi = True

        if verbose:
            print(f"[BPE PERFECT] ✅ Vocab ({self.sozluk_boyutu}) {time.time()-t0:.2f} saniyede oluşturuldu!")
        return self

    def _fast_parcala(self, kelime: str) -> List[str]:
        if kelime in self._cache: return self._cache[kelime]
        if len(kelime) == 1 or not kelime.isalnum():
            self._cache[kelime] = [kelime]; return [kelime]

        res = []
        w = kelime + "</w>"
        i = 0
        n = len(w)

        while i < n:
            matched = False
            for l in range(min(n - i, 24), 0, -1):
                sub = w[i : i + l]
                if sub in self.subword_set:
                    res.append(sub)
                    i += l
                    matched = True
                    break
            if not matched:
                res.append(w[i])
                i += 1

        if len(self._cache) < 100000:
            self._cache[kelime] = res
        return res

    def egitim_token_listesi(self, metin: str) -> List[str]:
        kelimeler = self._metni_temizle(metin)
        return [p for w in kelimeler for p in self._fast_parcala(w)]

    def text_to_ids(self, metin, pad: bool = True) -> torch.Tensor:
        klm = self._metni_temizle(metin) if isinstance(metin, str) else list(metin)
        pieces = [p for w in klm for p in self._fast_parcala(w)]
        ids = [self.kelime_to_id.get(p, self.UNK_ID) for p in pieces]
        if not pad: return torch.tensor(ids or [self.PAD_ID], dtype=torch.long)
        if len(ids) >= self.baglam_penceresi: ids = ids[-self.baglam_penceresi:]
        else: ids = [self.PAD_ID] * (self.baglam_penceresi - len(ids)) + ids
        return torch.tensor(ids, dtype=torch.long)

    def id_to_text(self, idx: int) -> str:
        tok = self.id_to_kelime.get(int(idx), "")
        if tok in self.OZEL: return ""
        return tok.replace("</w>", " ") if tok.endswith("</w>") else tok

    def kaydet(self, yol: str):
        data = {"baglam_penceresi": self.baglam_penceresi, "max_vocab_size": self.max_vocab_size,
                "kelime_to_id": self.kelime_to_id}
        with open(yol, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False)

    def yukle(self, yol: str) -> "BPETokenizer":
        with open(yol, "r", encoding="utf-8") as f: data = json.load(f)
        self.baglam_penceresi = int(data.get("baglam_penceresi", self.baglam_penceresi))
        self.max_vocab_size = int(data.get("max_vocab_size", self.max_vocab_size))
        self.kelime_to_id = {k: int(v) for k, v in data.get("kelime_to_id", {}).items()}
        self.id_to_kelime = {v: k for k, v in self.kelime_to_id.items()}
        self.sozluk_boyutu = len(self.kelime_to_id)
        self.subword_set = set(self.kelime_to_id.keys())
        self._egitildi = True
        return self