# -*- coding: utf-8 -*-
"""Multiclass paradigm: gorev-bagimsizlik testi.

Task: (subject, relation, object) -> class in {0..K-1}
- Symbolic: known features icin kural, gizli ozelliklerde cekimser
- Neural: MLP classifier (tam coverage)
- Hybrid: veto + fallback

Amac: gain = cov_sym * (sym_acc_ans - neu_acc) formulunu multiclass'ta test et.
"""
import random
import statistics as st
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


@dataclass
class Ornek:
    subject_id: str
    relation_id: str
    object_id: str
    label: int
    symbolic_decidable: bool


def gorev_uret(entity_count=120, relation_count=3, train_size=1200,
               test_size=400, hidden_ratio=0.30, K=4, seed=42):
    rng = random.Random(seed)
    entities = [f"E_{i:04d}" for i in range(entity_count)]
    relations = [f"R_{i:02d}" for i in range(relation_count)]

    true_feat = {e: rng.randrange(K) for e in entities}
    hidden = set(rng.sample(entities, int(entity_count * hidden_ratio)))

    def label(s, o):
        return (true_feat[s] + true_feat[o]) % K

    def gen(n):
        out = []
        while len(out) < n:
            s = rng.choice(entities)
            o = rng.choice(entities)
            if s == o:
                continue
            r = rng.choice(relations)
            lbl = label(s, o)
            dec = s not in hidden and o not in hidden
            out.append(Ornek(s, r, o, lbl, dec))
        return out

    return {
        "true_feat": true_feat,
        "hidden": hidden,
        "relations": relations,
        "train": gen(train_size),
        "test": gen(test_size),
        "K": K,
    }


def symbolic_pred(gorev, examples):
    """Known features icin exact kural; gizli olanlarda None."""
    out = []
    for e in examples:
        if e.symbolic_decidable:
            s = gorev["true_feat"][e.subject_id]
            o = gorev["true_feat"][e.object_id]
            out.append((s + o) % gorev["K"])
        else:
            out.append(None)
    return out


class NeuralModel(nn.Module):
    def __init__(self, n_entities, n_relations, K, dim=32):
        super().__init__()
        self.emb_e = nn.Embedding(n_entities, dim)
        self.emb_r = nn.Embedding(n_relations, dim)
        self.body = nn.Sequential(
            nn.Linear(dim * 3, 64), nn.ReLU(), nn.Linear(64, K)
        )

    def forward(self, s, r, o):
        x = torch.cat([self.emb_e(s), self.emb_r(r), self.emb_e(o)], dim=1)
        return self.body(x)


def neural_train(gorev, seed=0, epochs=200, lr=3e-3, batch=64):
    torch.manual_seed(seed)
    entity_list = sorted(gorev["true_feat"])
    e2i = {e: i for i, e in enumerate(entity_list)}
    r2i = {r: i for i, r in enumerate(gorev["relations"])}

    def tensors(examples):
        s = torch.tensor([e2i[e.subject_id] for e in examples])
        r = torch.tensor([r2i[e.relation_id] for e in examples])
        o = torch.tensor([e2i[e.object_id] for e in examples])
        y = torch.tensor([e.label for e in examples])
        return s, r, o, y

    s_tr, r_tr, o_tr, y_tr = tensors(gorev["train"])
    s_te, r_te, o_te, y_te = tensors(gorev["test"])

    model = NeuralModel(len(entity_list), len(gorev["relations"]), gorev["K"])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n = len(gorev["train"])
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(s_tr[idx], r_tr[idx], o_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(s_te, r_te, o_te).argmax(1).tolist()
    return preds


def main():
    print("="*70)
    print("MULTICLASS PARADIGM — GOREV-BAGIMSIZLIK TESTI")
    print("="*70)

    results = []
    for seed in [1, 2, 3, 4, 5]:
        gorev = gorev_uret(seed=seed)
        K = gorev["K"]

        # Symbolic
        sym_preds = symbolic_pred(gorev, gorev["test"])
        sym_n_ans = sum(1 for p in sym_preds if p is not None)
        sym_correct = sum(1 for p, e in zip(sym_preds, gorev["test"])
                          if p is not None and p == e.label)
        sym_cov = sym_n_ans / len(gorev["test"])
        sym_acc_ans = sym_correct / max(sym_n_ans, 1)

        # Neural
        neu_preds = neural_train(gorev, seed=seed)
        neu_acc = sum(1 for p, e in zip(neu_preds, gorev["test"])
                      if p == e.label) / len(gorev["test"])

        # Hybrid: veto + fallback
        hyb_preds = [
            sym if sym is not None else neu
            for sym, neu in zip(sym_preds, neu_preds)
        ]
        hyb_acc = sum(1 for p, e in zip(hyb_preds, gorev["test"])
                      if p == e.label) / len(gorev["test"])

        observed = hyb_acc - neu_acc
        predicted = sym_cov * (sym_acc_ans - neu_acc)
        error = abs(observed - predicted)

        results.append({
            "seed": seed,
            "K": K,
            "sym_cov": sym_cov,
            "sym_acc_ans": sym_acc_ans,
            "neu_acc": neu_acc,
            "hyb_acc": hyb_acc,
            "observed": observed,
            "predicted": predicted,
            "error": error,
        })

        print(f"seed={seed}: cov={sym_cov:.4f} sym_acc={sym_acc_ans:.4f} "
              f"neu={neu_acc:.4f} hyb={hyb_acc:.4f} "
              f"obs={observed:+.4f} pred={predicted:+.4f} err={error:.4f}")

    print()
    print("="*70)
    print("OZET")
    print("="*70)
    print(f"{'Seed':>6} {'K':>4} {'cov':>8} {'sym_acc':>9} {'neu':>8} "
          f"{'obs':>9} {'pred':>9} {'err':>8}")
    print("-"*70)
    for r in results:
        print(f"{r['seed']:>6} {r['K']:>4} {r['sym_cov']:>8.4f} "
              f"{r['sym_acc_ans']:>9.4f} {r['neu_acc']:>8.4f} "
              f"{r['observed']:>+9.4f} {r['predicted']:>+9.4f} {r['error']:>8.4f}")

    mean_err = st.mean(r["error"] for r in results)
    print()
    print(f"Ortalama hata: {mean_err:.5f}")
    print(f"Tum hatalar < 0.01: {all(r['error'] < 0.01 for r in results)}")
    print()
    print("Sonuc: Multiclass gorevde formul " +
          ("GECERLI" if mean_err < 0.01 else "GECERSIZ"))


if __name__ == "__main__":
    main()
