# -*- coding: utf-8 -*-
"""Multi-label paradigm: her ornek K bagimsiz bit.

Bit-level accuracy uzerinde formul testi:
    bit_acc = (1/K) * sum_k acc_k
Formul her bit icin gecerli, ortalamasi da gecerli olmali.
"""
import random
import statistics as st
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Ornek:
    subject_id: str
    relation_id: str
    object_id: str
    bits: tuple  # K bool
    symbolic_decidable: bool


def gorev_uret(entity_count=120, relation_count=3, K=4,
               train_size=1200, test_size=400,
               hidden_ratio=0.30, seed=42):
    rng = random.Random(seed)
    entities = [f"E_{i:04d}" for i in range(entity_count)]
    relations = [f"R_{i:02d}" for i in range(relation_count)]

    # Her entity icin K bit
    entity_bits = {e: tuple(rng.randint(0, 1) for _ in range(K)) for e in entities}
    hidden = set(rng.sample(entities, int(entity_count * hidden_ratio)))

    def bits(s, o):
        # XOR: her bit bagimsiz
        return tuple(entity_bits[s][k] ^ entity_bits[o][k] for k in range(K))

    def gen(n):
        out = []
        while len(out) < n:
            s = rng.choice(entities)
            o = rng.choice(entities)
            if s == o:
                continue
            r = rng.choice(relations)
            b = bits(s, o)
            dec = s not in hidden and o not in hidden
            out.append(Ornek(s, r, o, b, dec))
        return out

    return {
        "entity_bits": entity_bits,
        "hidden": hidden,
        "relations": relations,
        "K": K,
        "train": gen(train_size),
        "test": gen(test_size),
    }


def symbolic_pred(gorev, examples):
    """Known: exact bits; unknown: None."""
    out = []
    for e in examples:
        if e.symbolic_decidable:
            s = gorev["entity_bits"][e.subject_id]
            o = gorev["entity_bits"][e.object_id]
            out.append(tuple(s[k] ^ o[k] for k in range(gorev["K"])))
        else:
            out.append(None)
    return out


class MLModel(nn.Module):
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
    entity_list = sorted(gorev["entity_bits"])
    e2i = {e: i for i, e in enumerate(entity_list)}
    r2i = {r: i for i, r in enumerate(gorev["relations"])}
    K = gorev["K"]

    def tensors(examples):
        s = torch.tensor([e2i[e.subject_id] for e in examples])
        r = torch.tensor([r2i[e.relation_id] for e in examples])
        o = torch.tensor([e2i[e.object_id] for e in examples])
        y = torch.tensor([[int(b) for b in e.bits] for e in examples], dtype=torch.float)
        return s, r, o, y

    s_tr, r_tr, o_tr, y_tr = tensors(gorev["train"])
    s_te, r_te, o_te, y_te = tensors(gorev["test"])

    model = MLModel(len(entity_list), len(gorev["relations"]), K)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

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
        logits = model(s_te, r_te, o_te)
        preds = (torch.sigmoid(logits) >= 0.5).int().tolist()
    return preds  # her ornek: K bit listesi


def bit_accuracy(pred_bits_list, examples, K):
    """Bit-level accuracy."""
    total = 0
    correct = 0
    for pred, e in zip(pred_bits_list, examples):
        for k in range(K):
            total += 1
            if int(pred[k]) == int(e.bits[k]):
                correct += 1
    return correct / total


def main():
    print("="*75)
    print("MULTI-LABEL PARADIGM — BIT-LEVEL FORMUL TESTI")
    print("="*75)

    results = []
    for seed in [1, 2, 3, 4, 5]:
        gorev = gorev_uret(seed=seed)
        K = gorev["K"]

        # Symbolic: bits veya None
        sym_preds = symbolic_pred(gorev, gorev["test"])
        sym_known = [i for i, p in enumerate(sym_preds) if p is not None]
        sym_cov = len(sym_known) / len(gorev["test"])

        if sym_known:
            sym_correct = sum(
                1 for i in sym_known for k in range(K)
                if sym_preds[i][k] == gorev["test"][i].bits[k]
            )
            sym_acc_ans = sym_correct / (len(sym_known) * K)
        else:
            sym_acc_ans = 0.0

        # Neural: bit-level
        neu_preds = neural_train(gorev, seed=seed)
        neu_bit_acc = bit_accuracy(neu_preds, gorev["test"], K)

        # Hybrid: veto + fallback
        hyb_preds = []
        for sym, neu in zip(sym_preds, neu_preds):
            if sym is not None:
                hyb_preds.append(sym)
            else:
                hyb_preds.append(neu)
        hyb_bit_acc = bit_accuracy(hyb_preds, gorev["test"], K)

        observed = hyb_bit_acc - neu_bit_acc
        predicted = sym_cov * (sym_acc_ans - neu_bit_acc)
        error = abs(observed - predicted)

        results.append({
            "seed": seed, "K": K,
            "cov": sym_cov, "sym_acc": sym_acc_ans,
            "neu_bit": neu_bit_acc, "hyb_bit": hyb_bit_acc,
            "obs": observed, "pred": predicted, "err": error,
        })

        print(f"seed={seed}: cov={sym_cov:.4f} sym_bit={sym_acc_ans:.4f} "
              f"neu_bit={neu_bit_acc:.4f} hyb_bit={hyb_bit_acc:.4f}")
        print(f"  obs={observed:+.4f} pred={predicted:+.4f} err={error:.4f}")

    print()
    print("="*75)
    print("OZET")
    print("="*75)
    print(f"{'Seed':>5} {'cov':>8} {'sym_bit':>9} {'neu_bit':>9} "
          f"{'obs':>9} {'pred':>9} {'err':>8}")
    print("-"*75)
    for r in results:
        print(f"{r['seed']:>5} {r['cov']:>8.4f} {r['sym_acc']:>9.4f} "
              f"{r['neu_bit']:>9.4f} {r['obs']:>+9.4f} "
              f"{r['pred']:>+9.4f} {r['err']:>8.4f}")

    mean_err = st.mean(r["err"] for r in results)
    print()
    print(f"Ortalama hata: {mean_err:.5f}")
    print(f"Tum hatalar < 0.01: {all(r['err'] < 0.01 for r in results)}")
    print(f"Tum hatalar < 0.02: {all(r['err'] < 0.02 for r in results)}")


if __name__ == "__main__":
    main()
