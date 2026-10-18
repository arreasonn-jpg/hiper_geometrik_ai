# -*- coding: utf-8 -*-
"""HGA Türkçe LM Ablasyonları.

Hangi bileşen HGA'yı zorluyor?
- Attention
- Kronecker Zinciri
- Seyrek Bellek
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Proje kökünü ve mimari dizinini sys.path'e ekle
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mimari"))

from hga.experiments.turkish_lm import (
    PROFILES,
    _build_windows,
    _train_and_eval,
    build_lm_models,
    prepare_lm_corpus,
)
from mimari.bpe_tokenizer import BPETokenizer
from mimari.kuresel_model import HiperGeometrikAI


class IdentityAttention(nn.Module):
    def forward(self, x, attention_mask=None):
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x

class AblatedHGA(HiperGeometrikAI):
    def __init__(self, *args, no_attn=False, no_kron=False, **kwargs):
        super().__init__(*args, **kwargs)
        if no_attn:
            self.dikkat = IdentityAttention()
        if no_kron:
            self.kuresel_bag = nn.Identity()

def main():
    # Smoke profili kullan (hız için), 1 seed yeterli (mekanizma tespiti için)
    config = dict(PROFILES["smoke"])
    config["steps"] = 400 # Varsayılan smoke steps
    config["batch_size"] = 128
    seed = 42

    print("Veri hazırlanıyor...")
    corpus = prepare_lm_corpus(max_docs=config["max_docs"], corpus="twt")
    train_metin = "\n".join(corpus.split_texts("train"))

    tokenizer = BPETokenizer(baglam_penceresi=int(config["context"]),
                             max_vocab_size=int(config["max_vocab"]),
                             min_freq=2)
    tokenizer.fit_on_text(train_metin, verbose=False)
    vocab_size = max(int(tokenizer.sozluk_boyutu), 8)

    split_ids = {s: [tokenizer.encode(d, pad=False) for d in corpus.split_texts(s)] for s in ("train", "dev", "test")}
    train_x, train_y = _build_windows(torch, split_ids["train"], int(config["context"]))
    evals = {s: _build_windows(torch, split_ids[s], int(config["context"])) for s in ("dev", "test")}

    torch.manual_seed(seed)
    schedule = torch.randint(0, int(train_x.shape[0]), (int(config["steps"]), int(config["batch_size"])))

    # Model kurucuları
    kurucular, _ = build_lm_models(vocab_size, config)

    # Ablasyon modelleri (HGA için)
    def make_hga(no_attn=False, no_kron=False, no_sparse=False):
        kw = {
            "n": int(config["hga_n"]), "baglam_penceresi": int(config["context"]),
            "sozluk_boyutu": vocab_size, "emb_dim": int(config["emb_dim"]),
            "num_heads": int(config["heads"]), "katman_sayisi": int(config["hga_layers"]),
            "dropout": 0.0, "bilgilendir": False,
            "seyrek_tablo_boyutu": 0 if no_sparse else 4096, # smoke için küçük tablo
            "seyrek_boyut": 32
        }
        return AblatedHGA(no_attn=no_attn, no_kron=no_kron, **kw)

    modeller = {
        "dense": kurucular["dense"],
        "transformer": kurucular["transformer"],
        "hga_full": lambda: make_hga(),
        "hga_no_attn": lambda: make_hga(no_attn=True),
        "hga_no_kron": lambda: make_hga(no_kron=True),
        "hga_no_sparse": lambda: make_hga(no_sparse=True),
    }

    results = {}
    for name, ctor in modeller.items():
        print(f"Eğitiliyor: {name} ...")
        torch.manual_seed(seed) # Her model için aynı başlangıç
        model = ctor()
        t0 = time.time()
        res = _train_and_eval(torch, nn, model, train_x, train_y, evals, schedule, config)
        t1 = time.time()
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results[name] = {
            "ppl": res["test"]["perplexity"],
            "top1": res["test"]["top1_accuracy"],
            "params": params,
            "sure": round(t1-t0, 1)
        }
        print(f"  -> PPL: {results[name]['ppl']:.2f} | Top1: {results[name]['top1']:.3f} | Params: {params:,}")

    out = Path("artifacts/w7_runs/p3_lm_ablasyon.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSonuçlar kaydedildi: {out}")

if __name__ == "__main__":
    main()
