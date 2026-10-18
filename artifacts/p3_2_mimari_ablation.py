"""P3-2: HGA Derin Mimari Ablasyonu.

HGA'da hangi bileşen (pozisyon, embedding, encoder, decoder)
modeli yavaşlatıyor?
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


class AblatedHGA(HiperGeometrikAI):
    def __init__(self, *args, no_pos=False, frozen_emb=False, linear_decoder=False, no_encoder=False, **kwargs):
        super().__init__(*args, **kwargs)

        if no_pos:
            self.pos_encoder.data.zero_()
            self.pos_encoder.requires_grad = False

        if frozen_emb:
            self.kelime_gomme.weight.requires_grad = False

        if linear_decoder:
            self.decoder = nn.Sequential(nn.Flatten(1), nn.Linear(self.n * self.n, self.sozluk_boyutu))

        if no_encoder:
            # GeometrikVeriEncoder yerine basit bir projeksiyon
            S_emb = self.baglam_penceresi * int(kwargs.get('emb_dim', 128))
            self.encoder = nn.Sequential(
                nn.Linear(S_emb, self.n * self.n),
                nn.Unflatten(1, (self.n, self.n))
            )

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
    config["steps"] = 1000 # Hız için 1000 step
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

    def hga_ctor(**ablation_kwargs):
        kw = {
            "n": int(config["hga_n"]), "baglam_penceresi": int(config["context"]),
            "sozluk_boyutu": vocab_size, "emb_dim": int(config["emb_dim"]),
            "num_heads": int(config["heads"]), "katman_sayisi": int(config["hga_layers"]),
            "dropout": 0.0, "bilgilendir": False,
            "seyrek_tablo_boyutu": 4096, "seyrek_boyut": 32
        }
        return AblatedHGA(**kw, **ablation_kwargs)

    variants = {
        "hga_full": lambda: hga_ctor(),
        "hga_no_pos": lambda: hga_ctor(no_pos=True),
        "hga_frozen_emb": lambda: hga_ctor(frozen_emb=True),
        "hga_linear_decoder": lambda: hga_ctor(linear_decoder=True),
        "hga_no_encoder": lambda: hga_ctor(no_encoder=True),
    }

    results = {}
    for name, ctor in variants.items():
        print(f"Eğitiliyor: {name} ...")
        torch.manual_seed(seed)
        model = ctor()
        opt = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
        loss_fn = nn.CrossEntropyLoss()
        schedule = torch.randint(0, int(train_x.shape[0]), (config["steps"], config["batch_size"]))

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

    out = Path("artifacts/w7_runs/p3_2_mimari_ablation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSonuçlar kaydedildi: {out}")

if __name__ == "__main__":
    main()
