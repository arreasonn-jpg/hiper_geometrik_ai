# -*- coding: utf-8 -*-
"""
NER formula test — TOKEN ACCURACY (linear metric).

F1 lineer degil, bu yuzden formulun ozdesligi bozulur.
Token accuracy lineer oldugu icin formül tam olarak gecerli olmali.
"""
import json
from pathlib import Path
import numpy as np
from artifacts.ner_formula_test import (
    make_data, train_neural, predict_neural, SEEDS, HIDDEN_RATIOS
)

def acc(y_true, y_pred):
    return float((y_true == y_pred).mean())

def acc_mask(y_true, y_pred, mask):
    if mask.sum() == 0:
        return 0.0
    return float((y_true[mask] == y_pred[mask]).mean())

def run(seed, hr):
    X, y, is_known = make_data(seed, hr)
    n = len(y); n_tr = int(0.8 * n)
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_te, y_te, k_te = X[n_tr:], y[n_tr:], is_known[n_tr:]

    params = train_neural(X_tr, y_tr, seed)
    pred_n = predict_neural(X_te, params)

    # Linear accuracy
    acc_n_all = acc(y_te, pred_n)
    abstain_mask = ~k_te
    acc_n_perp = acc_mask(y_te, pred_n, abstain_mask)

    cov = k_te.mean()
    acc_s = acc_mask(y_te, y_te, k_te) if k_te.sum() else 0.0  # symbolic perfect on answered

    # Hybrid
    pred_h = pred_n.copy(); pred_h[k_te] = y_te[k_te]
    acc_h = acc(y_te, pred_h)

    # Formula
    gain_obs = acc_h - acc_n_all
    gain_pred_simple = cov * (acc_s - acc_n_all)
    gain_pred_exact = cov * (acc_s - acc_n_all) + (1-cov) * (acc_n_perp - acc_n_all)

    eps_simple = abs(gain_obs - gain_pred_simple)
    eps_exact  = abs(gain_obs - gain_pred_exact)

    return {
        "seed": seed, "hidden_ratio": hr,
        "cov": round(cov, 4),
        "acc_s": round(acc_s, 4), "acc_n": round(acc_n_all, 4),
        "acc_n_perp": round(acc_n_perp, 4), "acc_h": round(acc_h, 4),
        "gain_obs": round(gain_obs, 4),
        "gain_pred_simple": round(gain_pred_simple, 4),
        "gain_pred_exact": round(gain_pred_exact, 4),
        "eps_simple": round(eps_simple, 4),
        "eps_exact":  round(eps_exact, 4),
    }

if __name__ == "__main__":
    results = [run(s, hr) for s in SEEDS for hr in HIDDEN_RATIOS]

    print(f"{'seed':>4} {'hid':>5} {'cov':>6} {'acc_s':>6} {'acc_n':>6} "
          f"{'acc_n_perp':>10} {'g_obs':>7} {'g_simple':>9} {'g_exact':>8} "
          f"{'eps_s':>7} {'eps_e':>7}")
    print("-" * 90)
    for r in results:
        print(f"{r['seed']:>4} {r['hidden_ratio']:>5.2f} {r['cov']:>6.3f} "
              f"{r['acc_s']:>6.3f} {r['acc_n']:>6.3f} {r['acc_n_perp']:>10.3f} "
              f"{r['gain_obs']:>+7.4f} {r['gain_pred_simple']:>+9.4f} "
              f"{r['gain_pred_exact']:>+8.4f} {r['eps_simple']:>7.4f} "
              f"{r['eps_exact']:>7.4f}")

    eps_s = [r["eps_simple"] for r in results]
    eps_e = [r["eps_exact"]  for r in results]
    print()
    print(f"SIMPLE formula eps:  mean={np.mean(eps_s):.4f}  max={np.max(eps_s):.4f}  "
          f"<0.01: {sum(1 for e in eps_s if e<0.01)}/{len(eps_s)}")
    print(f"EXACT  formula eps:  mean={np.mean(eps_e):.4f}  max={np.max(eps_e):.4f}  "
          f"<0.001: {sum(1 for e in eps_e if e<0.001)}/{len(eps_e)}")

    Path("artifacts/ner_accuracy_results.json").write_text(
        json.dumps({"results": results,
                    "summary": {
                        "simple_mean_eps": float(np.mean(eps_s)),
                        "simple_max_eps": float(np.max(eps_s)),
                        "simple_n_under_001": sum(1 for e in eps_s if e < 0.01),
                        "exact_mean_eps": float(np.mean(eps_e)),
                        "exact_max_eps": float(np.max(eps_e)),
                        "exact_n_under_0001": sum(1 for e in eps_e if e < 0.001),
                        "n_total": len(eps_s)}},
                   indent=2), encoding="utf-8")
