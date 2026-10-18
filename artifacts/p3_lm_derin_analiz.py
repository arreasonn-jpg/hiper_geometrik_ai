"""HGA Türkçe LM: Derin Ablasyon ve Hiperparametre Analizi.

2000 step'te HGA neden yavaş?
- Attention mı, Kronecker mi?
- LR mi, K (derinlik) mi, n (boyut) mi?
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mimari"))

from hga.experiments.turkish_lm import PROFILES, _build_windows, build_lm_models, prepare_lm_corpus
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

def evaluate(model, evals, config):
    model.eval()
    res = {}
    for ad, (x, y) in evals.items():
        toplam_log, sayi = 0.0, 0
        with torch.no_grad():
            for i in range(0, int(x.shape[0]), int(config["eval_batch_size"])):
                logits = model(x[i:i+int(config["eval_batch_size"])])
                y_batch = y[i:i+int(config["eval_batch_size"])]
                log_p = torch.log_softmax(logits, dim=-1)
                toplam_log += float(-log_p.gather(1, y_batch.unsqueeze(1)).sum())
                sayi += int(y_batch.shape[0])
        ce = toplam_log / max(1, sayi)
        res[ad] = {"ppl": round(math.exp(min(50.0, ce)), 2)}
    return res

def main():
    config = dict(PROFILES["smoke"])
    config["steps"] = 2000
    seed = 42

    print("Veri hazırlanıyor...")
    corpus = prepare_lm_corpus(max_docs=config["max_docs"], corpus="twt")
    train_metin = "\n".join(corpus.split_texts("train"))
    tokenizer = BPETokenizer(baglam_penceresi=int(config["context"]),
                             max_vocab_size=int(config["max_vocab"]), min_freq=2)
    tokenizer.fit_on_text(train_metin, verbose=False)
    vocab_size = max(int(tokenizer.sozluk_boyutu), 8)

    split_ids = {s: [tokenizer.encode(d, pad=False) for d in corpus.split_texts(s)]
                 for s in ("train", "dev", "test")}
    train_x, train_y = _build_windows(torch, split_ids["train"], int(config["context"]))
    evals = {s: _build_windows(torch, split_ids[s], int(config["context"]))
             for s in ("dev", "test")}

    kurucular, _ = build_lm_models(vocab_size, config)

    def hga_ctor(no_attn=False, no_kron=False, lr=None, n=None, k=None):
        kw = {
            "n": n or int(config["hga_n"]),
            "baglam_penceresi": int(config["context"]),
            "sozluk_boyutu": vocab_size,
            "emb_dim": int(config["emb_dim"]),
            "num_heads": int(config["heads"]),
            "katman_sayisi": k or int(config["hga_layers"]),
            "dropout": 0.0, "bilgilendir": False,
            "seyrek_tablo_boyutu": 4096, "seyrek_boyut": 32
        }
        m = AblatedHGA(no_attn=no_attn, no_kron=no_kron, **kw)
        lr_val = lr or float(config["learning_rate"])
        opt = torch.optim.AdamW(m.parameters(), lr=lr_val)
        return m, opt

    def dense_ctor():
        m = kurucular["dense"]()
        return m, torch.optim.AdamW(m.parameters(), lr=float(config["learning_rate"]))

    variants = {
        "dense_baseline": dense_ctor,
        "hga_full_lr003": lambda: hga_ctor(),
        "hga_no_attn": lambda: hga_ctor(no_attn=True),
        "hga_no_kron": lambda: hga_ctor(no_kron=True),
        "hga_lr_high": lambda: hga_ctor(lr=0.01),
        "hga_lr_low": lambda: hga_ctor(lr=0.0001),
        "hga_n_32": lambda: hga_ctor(n=32),
        "hga_n_8": lambda: hga_ctor(n=8),
        "hga_k_2": lambda: hga_ctor(k=2),
    }

    results = {}
    for name, ctor in variants.items():
        print(f"Eğitiliyor: {name} ...")
        torch.manual_seed(seed)
        model, opt = ctor()

        # Schedule
        schedule = torch.randint(0, int(train_x.shape[0]), (config["steps"], config["batch_size"]))
        loss_fn = nn.CrossEntropyLoss()

        t0 = time.time()
        model.train()
        for adim in range(config["steps"]):
            idx = schedule[adim]
            opt.zero_grad()
            loss = loss_fn(model(train_x[idx]), train_y[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            opt.step()

        t1 = time.time()
        dev_res = evaluate(model, evals, config)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results[name] = {
            "dev_ppl": dev_res["dev"]["ppl"],
            "test_ppl": dev_res["test"]["ppl"],
            "params": params,
            "sure": round(t1-t0, 1)
        }
        print(f"  -> Test PPL: {results[name]['test_ppl']:.2f} | Parametre: {params:,} | Süre: {results[name]['sure']}s")

    out = Path("artifacts/w7_runs/p3_lm_derin_analiz.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSonuçlar kaydedildi: {out}")

if __name__ == "__main__":
    main()
