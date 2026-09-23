# -*- coding: utf-8 -*-
"""
Synthetic NER formula test.

Gorev: cumlede gecen varliklarin tipini etiketle (PER/LOC/ORG).
Symbolic kol: gazetteer'de olan varliklari etiketler, yoksa cekimser kalir.
Neural kol: kucuk MLP, sifirdan egitilir.
Hibrit: symbolic once, cekimser kalirsa neural.

Formul: kazanc = kapsam_sem x (F1_sem - F1_neu)
"""
from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np

SEEDS = [1, 2, 3, 4, 5]
TYPES = ["PER", "LOC", "ORG"]
N_KNOWN = 400         # gazetteer'de olan varliklar
N_HIDDEN = 400        # gazetteer'de olmayan
N_SENT_PER_ENTITY = 5 # her varlik icin cumle ornegi
HIDDEN_RATIOS = [0.10, 0.30, 0.50, 0.70]
DIM = 32

def make_vocab(seed):
    rng = random.Random(seed)
    known = {}
    hidden = {}
    for i in range(N_KNOWN):
        known[f"K{i}"] = rng.choice(TYPES)
    for i in range(N_HIDDEN):
        hidden[f"H{i}"] = rng.choice(TYPES)
    return known, hidden

def make_context_vec(rng):
    v = np.zeros(DIM, dtype=np.float32)
    for _ in range(3):
        v[rng.randrange(DIM)] = 1.0
    return v

def make_data(seed, hidden_ratio):
    rng = random.Random(seed * 1000 + int(hidden_ratio * 100))
    known, hidden = make_vocab(seed)
    n_hid_used = int(len(known) * hidden_ratio / (1 - hidden_ratio))
    n_hid_used = min(n_hid_used, len(hidden))
    hidden_items = list(hidden.items())[:n_hid_used]

    X, y, is_known = [], [], []
    for ent, typ in list(known.items()) + hidden_items:
        for _ in range(N_SENT_PER_ENTITY):
            X.append(make_context_vec(rng))
            y.append(TYPES.index(typ))
            is_known.append(ent.startswith("K"))
    X = np.stack(X); y = np.array(y); is_known = np.array(is_known)
    # shuffle
    idx = rng.sample(range(len(y)), len(y))
    return X[idx], y[idx], is_known[idx]

def train_neural(X, y, seed, epochs=300, lr=0.05):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.1, (DIM, 64)).astype(np.float32)
    b1 = np.zeros(64, dtype=np.float32)
    W2 = rng.normal(0, 0.1, (64, 3)).astype(np.float32)
    b2 = np.zeros(3, dtype=np.float32)

    def softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z); return e / e.sum(1, keepdims=True)

    for _ in range(epochs):
        h = np.maximum(0, X @ W1 + b1)
        logits = h @ W2 + b2
        p = softmax(logits)
        yh = np.eye(3, dtype=np.float32)[y]
        dlog = (p - yh) / len(y)
        dW2 = h.T @ dlog; db2 = dlog.sum(0)
        dh = dlog @ W2.T; dh[h <= 0] = 0
        dW1 = X.T @ dh; db1 = dh.sum(0)
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2
    return W1, b1, W2, b2

def predict_neural(X, params):
    W1, b1, W2, b2 = params
    h = np.maximum(0, X @ W1 + b1)
    return (h @ W2 + b2).argmax(1)

def f1_macro(y_true, y_pred, n_classes=3):
    f1s = []
    for c in range(n_classes):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        if tp + fp + fn == 0:
            f1s.append(0.0); continue
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
    return float(np.mean(f1s))

def run(seed, hidden_ratio):
    X, y, is_known = make_data(seed, hidden_ratio)
    n = len(y)
    n_tr = int(0.8 * n)
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_te, y_te, k_te = X[n_tr:], y[n_tr:], is_known[n_tr:]

    params = train_neural(X_tr, y_tr, seed)

    # Neural baseline
    pred_n = predict_neural(X_te, params)
    f1_neu = f1_macro(y_te, pred_n)

    # Symbolic: answered on known entities, abstains on hidden
    pred_s = np.full(len(y_te), -1, dtype=int)
    answered = k_te
    pred_s[answered] = y_te[answered]  # perfect on known
    cov = answered.mean()
    f1_s = f1_macro(y_te[answered], pred_s[answered]) if answered.sum() > 0 else 0.0

    # Hybrid
    pred_h = pred_n.copy()
    pred_h[answered] = y_te[answered]
    f1_h = f1_macro(y_te, pred_h)

    # Formula
    gain_obs = f1_h - f1_neu
    gain_pred = cov * (f1_s - f1_neu)
    err = abs(gain_obs - gain_pred)

    return {
        "seed": seed, "hidden_ratio": hidden_ratio, "n_test": int(len(y_te)),
        "cov": round(cov, 4), "f1_s": round(f1_s, 4), "f1_n": round(f1_neu, 4),
        "f1_h": round(f1_h, 4),
        "gain_obs": round(gain_obs, 4), "gain_pred": round(gain_pred, 4),
        "eps": round(err, 4),
    }

if __name__ == "__main__":
    results = []
    for seed in SEEDS:
        for hr in HIDDEN_RATIOS:
            results.append(run(seed, hr))

    eps = [r["eps"] for r in results]
    print(f"{'seed':>4} {'hid':>5} {'cov':>6} {'F1_s':>6} {'F1_n':>6} {'F1_h':>6} "
          f"{'g_obs':>7} {'g_pred':>7} {'eps':>6}")
    for r in results:
        print(f"{r['seed']:>4} {r['hidden_ratio']:>5.2f} {r['cov']:>6.3f} "
              f"{r['f1_s']:>6.3f} {r['f1_n']:>6.3f} {r['f1_h']:>6.3f} "
              f"{r['gain_obs']:>+7.4f} {r['gain_pred']:>+7.4f} {r['eps']:>6.4f}")
    print()
    print(f"Mean eps:  {np.mean(eps):.4f}")
    print(f"Max eps:   {np.max(eps):.4f}")
    print(f"N < 0.01:  {sum(1 for e in eps if e < 0.01)}/{len(eps)}")
    print(f"N < 0.025: {sum(1 for e in eps if e < 0.025)}/{len(eps)}")

    Path("artifacts/ner_formula_results.json").write_text(
        json.dumps({"results": results,
                    "summary": {"mean_eps": float(np.mean(eps)),
                                "max_eps": float(np.max(eps)),
                                "n_under_001": sum(1 for e in eps if e < 0.01),
                                "n_total": len(eps)}},
                   indent=2), encoding="utf-8")
