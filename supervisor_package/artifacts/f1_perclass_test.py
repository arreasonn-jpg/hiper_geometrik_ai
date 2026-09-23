# -*- coding: utf-8 -*-
"""
Per-class F1 weighted formula testi.

Teori:
    F1_h_macro = (1/K) * sum_c [ w_c * F1_s^c + (1-w_c) * F1_n^c ]
    w_c = D_s^c / (D_s^c + D_n^c)
"""
import json
from pathlib import Path
import numpy as np
from artifacts.ner_formula_test import (
    make_data, train_neural, predict_neural,
    SEEDS, HIDDEN_RATIOS, TYPES,
)

def per_class_tp_fp_fn(y_true, y_pred, c):
    tp = int(((y_pred == c) & (y_true == c)).sum())
    fp = int(((y_pred == c) & (y_true != c)).sum())
    fn = int(((y_pred != c) & (y_true == c)).sum())
    return tp, fp, fn

def f1_class(tp, fp, fn):
    if tp + fp + fn == 0:
        return 0.0
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * p * r / (p + r) if p + r > 0 else 0.0

def run(seed, hr):
    X, y, is_known = make_data(seed, hr)
    n = len(y); n_tr = int(0.8 * n)
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_te, y_te, k_te = X[n_tr:], y[n_tr:], is_known[n_tr:]

    params = train_neural(X_tr, y_tr, seed)
    pred_n = predict_neural(X_te, params)

    ans_mask = k_te
    abs_mask = ~k_te

    # Hybrid
    pred_h = pred_n.copy()
    pred_h[ans_mask] = y_te[ans_mask]

    # Per-class F1
    f1_s_per_c, f1_n_perp_per_c, f1_n_per_c, f1_h_per_c = [], [], [], []
    w_per_c = []
    for c in range(len(TYPES)):
        # Symbolic on answered
        tp_s, fp_s, fn_s = per_class_tp_fp_fn(y_te[ans_mask], y_te[ans_mask], c)
        f1_s_c = f1_class(tp_s, fp_s, fn_s)

        # Neural on abstained
        tp_np, fp_np, fn_np = per_class_tp_fp_fn(y_te[abs_mask], pred_n[abs_mask], c)
        f1_np_c = f1_class(tp_np, fp_np, fn_np)

        # Neural on all
        tp_n, fp_n, fn_n = per_class_tp_fp_fn(y_te, pred_n, c)
        f1_n_c = f1_class(tp_n, fp_n, fn_n)

        # Hybrid on all
        tp_h, fp_h, fn_h = per_class_tp_fp_fn(y_te, pred_h, c)
        f1_h_c = f1_class(tp_h, fp_h, fn_h)

        # Per-class weight
        D_s_c = 2 * tp_s + fp_s + fn_s
        D_n_c = 2 * tp_np + fp_np + fn_np
        w_c = D_s_c / (D_s_c + D_n_c) if (D_s_c + D_n_c) > 0 else 0.0

        f1_s_per_c.append(f1_s_c)
        f1_n_perp_per_c.append(f1_np_c)
        f1_n_per_c.append(f1_n_c)
        f1_h_per_c.append(f1_h_c)
        w_per_c.append(w_c)

    # Macro
    F1_h = float(np.mean(f1_h_per_c))
    F1_n = float(np.mean(f1_n_per_c))
    F1_s_macro = float(np.mean(f1_s_per_c))
    F1_n_perp_macro = float(np.mean(f1_n_perp_per_c))

    gain_obs = F1_h - F1_n

    # Per-class formula
    gain_perclass = sum(
        w_c * (f1_s_c - f1_n_c) + (1 - w_c) * (f1_np_c - f1_n_c)
        for w_c, f1_s_c, f1_n_c, f1_np_c in zip(
            w_per_c, f1_s_per_c, f1_n_per_c, f1_n_perp_per_c)
    ) / len(TYPES)
    eps_perclass = abs(gain_obs - gain_perclass)

    # Old linear
    cov = float(k_te.mean())
    gain_linear = cov * (F1_s_macro - F1_n)
    eps_linear = abs(gain_obs - gain_linear)

    return {
        "seed": seed, "hidden_ratio": hr,
        "cov": round(cov, 4),
        "w_range": f"{min(w_per_c):.2f}-{max(w_per_c):.2f}",
        "F1_h": round(F1_h, 4),
        "F1_n": round(F1_n, 4),
        "gain_obs": round(gain_obs, 4),
        "gain_perclass": round(gain_perclass, 4),
        "gain_linear": round(gain_linear, 4),
        "eps_perclass": round(eps_perclass, 6),
        "eps_linear": round(eps_linear, 4),
    }

if __name__ == "__main__":
    results = [run(s, hr) for s in SEEDS for hr in HIDDEN_RATIOS]

    print(f"{'seed':>4} {'hid':>5} {'cov':>6} {'w_range':>12} "
          f"{'F1_h':>6} {'F1_n':>6} {'eps_lin':>8} {'eps_pc':>9}")
    print("-" * 75)
    for r in results:
        print(f"{r['seed']:>4} {r['hidden_ratio']:>5.2f} {r['cov']:>6.3f} "
              f"{r['w_range']:>12} {r['F1_h']:>6.3f} {r['F1_n']:>6.3f} "
              f"{r['eps_linear']:>8.4f} {r['eps_perclass']:>9.6f}")

    eps_l = [r["eps_linear"] for r in results]
    eps_p = [r["eps_perclass"] for r in results]
    print()
    print(f"LINEER:        mean={np.mean(eps_l):.4f}  max={np.max(eps_l):.4f}")
    print(f"PER-CLASS:     mean={np.mean(eps_p):.6f}  max={np.max(eps_p):.6f}")

    Path("artifacts/f1_perclass_results.json").write_text(
        json.dumps({"results": results, "summary": {
            "linear_mean_eps": float(np.mean(eps_l)),
            "perclass_mean_eps": float(np.mean(eps_p)),
            "perclass_max_eps": float(np.max(eps_p)),
        }}, indent=2), encoding="utf-8")
