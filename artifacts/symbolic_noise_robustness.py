# -*- coding: utf-8 -*-
"""Sembolik gurultu altinda formul dayanikliligi.

Sembolik kola kontrollu gurultu ekle. Formulun ne zaman kirildigini test et.

Gurultu seviyeleri: 0%, 5%, 10%, 20%, 30%, 50%
"""
import random
import statistics as st
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Ornek:
    subject: str
    relation: str
    object_: str
    label: int
    symbolic_decidable: bool


def gorev_uret(n_entities=120, n_relations=3, K=4,
               train_size=1200, test_size=400,
               hidden_ratio=0.30, seed=42):
    rng = random.Random(seed)
    entities = [f"E_{i:04d}" for i in range(n_entities)]
    relations = [f"R_{i:02d}" for i in range(n_relations)]
    feat = {e: rng.randrange(K) for e in entities}
    hidden = set(rng.sample(entities, int(n_entities * hidden_ratio)))

    def label(s, o):
        return (feat[s] + feat[o]) % K

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
        "feat": feat, "hidden": hidden, "relations": relations,
        "K": K, "train": gen(train_size), "test": gen(test_size),
    }


def symbolic_pred_noisy(gorev, examples, noise_rate, seed):
    """Sembolik tahmin + kontrollu gurultu."""
    rng = random.Random(seed + 999)
    out = []
    for e in examples:
        if e.symbolic_decidable:
            correct = (gorev["feat"][e.subject] + gorev["feat"][e.object_]) % gorev["K"]
            # Gurultu: dogru cevabi yanlis bir cevapla degistir
            if rng.random() < noise_rate:
                wrong = (correct + rng.randint(1, gorev["K"] - 1)) % gorev["K"]
                out.append(wrong)
            else:
                out.append(correct)
        else:
            out.append(None)
    return out


class MLP(nn.Module):
    def __init__(self, n_e, n_r, K, dim=32):
        super().__init__()
        self.e = nn.Embedding(n_e, dim)
        self.r = nn.Embedding(n_r, dim)
        self.body = nn.Sequential(
            nn.Linear(dim * 3, 64), nn.ReLU(), nn.Linear(64, K)
        )

    def forward(self, s, r, o):
        return self.body(torch.cat([self.e(s), self.r(r), self.e(o)], dim=1))


def neural_train(gorev, seed, epochs=100):
    torch.manual_seed(seed)
    ents = sorted(gorev["feat"])
    e2i = {e: i for i, e in enumerate(ents)}
    r2i = {r: i for i, r in enumerate(gorev["relations"])}

    def tensors(ex):
        s = torch.tensor([e2i[e.subject] for e in ex])
        r = torch.tensor([r2i[e.relation] for e in ex])
        o = torch.tensor([e2i[e.object_] for e in ex])
        y = torch.tensor([e.label for e in ex])
        return s, r, o, y

    s_tr, r_tr, o_tr, y_tr = tensors(gorev["train"])
    s_te, r_te, o_te, _ = tensors(gorev["test"])

    model = MLP(len(ents), len(gorev["relations"]), gorev["K"])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()

    n = len(gorev["train"])
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = perm[i:i+64]
            opt.zero_grad()
            loss = loss_fn(model(s_tr[idx], r_tr[idx], o_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(s_te, r_te, o_te).argmax(1).tolist()
    return preds


def main():
    print("="*75)
    print("SEMBOLIK GURULTU DAYANIKLILIGI")
    print("="*75)
    print()
    print(f"{'gurultu':>8} {'cov':>8} {'sym_acc':>9} {'neu':>8} "
          f"{'obs':>9} {'pred':>9} {'err':>8}")
    print("-"*75)

    noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
    seeds = [1, 2, 3]

    for noise in noise_levels:
        errs = []
        covs = []
        sym_accs = []
        obs_list = []
        pred_list = []

        for seed in seeds:
            gorev = gorev_uret(seed=seed)
            K = gorev["K"]

            sym = symbolic_pred_noisy(gorev, gorev["test"], noise, seed)
            sym_ans_idx = [i for i, p in enumerate(sym) if p is not None]
            cov = len(sym_ans_idx) / len(gorev["test"])
            sym_acc = sum(1 for i in sym_ans_idx
                          if sym[i] == gorev["test"][i].label) / max(len(sym_ans_idx), 1)

            neu = neural_train(gorev, seed=seed)
            neu_acc = sum(1 for p, e in zip(neu, gorev["test"])
                          if p == e.label) / len(gorev["test"])

            hyb = [s if s is not None else n for s, n in zip(sym, neu)]
            hyb_acc = sum(1 for p, e in zip(hyb, gorev["test"])
                          if p == e.label) / len(gorev["test"])

            obs = hyb_acc - neu_acc
            pred = cov * (sym_acc - neu_acc)
            err = abs(obs - pred)

            covs.append(cov); sym_accs.append(sym_acc)
            obs_list.append(obs); pred_list.append(pred); errs.append(err)

        print(f"{noise*100:>7.0f}% {st.mean(covs):>8.4f} {st.mean(sym_accs):>9.4f} "
              f"{'':>8} {st.mean(obs_list):>+9.4f} {st.mean(pred_list):>+9.4f} "
              f"{st.mean(errs):>8.4f}")

    print()
    print("Yorum:")
    print("  - Gurultu arttikca sym_acc duser")
    print("  - Formul hatasi (err) nasil degisiyor?")
    print("  - Hangi noktada formul kirilir?")


if __name__ == "__main__":
    main()
