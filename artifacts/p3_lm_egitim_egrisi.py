"""HGA'nın eğitim eğrisini izle: 4000 step'te ne oluyor?

Her 200 step'te dev PPL ölç, dense / transformer / hga karşılaştır.
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


def evaluate(model, evals, config):
    model.eval()
    sonuc = {}
    for ad, (x, y) in evals.items():
        toplam_log, sayi = 0.0, 0
        with torch.no_grad():
            for ofset in range(0, int(x.shape[0]), int(config["eval_batch_size"])):
                logits = model(x[ofset:ofset + int(config["eval_batch_size"])])
                hedefler = y[ofset:ofset + int(config["eval_batch_size"])]
                log_p = torch.log_softmax(logits, dim=-1)
                toplam_log += float(-log_p.gather(1, hedefler.unsqueeze(1)).sum())
                sayi += int(hedefler.shape[0])
        ce = toplam_log / max(1, sayi)
        sonuc[ad] = {"ppl": round(math.exp(min(50.0, ce)), 2)}
    return sonuc


def main():
    # Smoke korpus, ama full profile'ın adım sayısına yakın
    config = dict(PROFILES["smoke"])
    config["steps"] = 2000        # 400 yerine 2000 (ana hipotez testi)
    config["batch_size"] = 128
    eval_every = 200
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

    results = {}
    for name in ("dense", "transformer", "hga"):
        torch.manual_seed(seed)
        model = kurucular[name]()
        opt = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
        loss_fn = nn.CrossEntropyLoss()
        schedule = torch.randint(0, int(train_x.shape[0]),
                                 (config["steps"], config["batch_size"]))
        egri = []
        model.train()
        t0 = time.time()
        for adim in range(config["steps"]):
            idx = schedule[adim]
            opt.zero_grad()
            loss = loss_fn(model(train_x[idx]), train_y[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            opt.step()
            if (adim + 1) % eval_every == 0:
                ev = evaluate(model, evals, config)
                egri.append({"step": adim + 1, "dev_ppl": ev["dev"]["ppl"]})
                print(f"  {name:12s} step {adim+1:5d} dev_ppl={ev['dev']['ppl']:.2f}")
        results[name] = {"egri": egri, "sure": round(time.time() - t0, 1)}
        print(f"  {name} bitti ({results[name]['sure']}s)")

    out = Path("artifacts/w7_runs/p3_lm_egitim_egrisi.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSonuç: {out}")


if __name__ == "__main__":
    main()
