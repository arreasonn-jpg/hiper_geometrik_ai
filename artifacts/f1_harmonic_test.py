# -*- coding: utf-8 -*-
"""
F1 Harmonik Formulu — sayisal dogrulama.

Teori (bkz. F1_HARMONIC_FORMULA.md):
    F1_h = W * F1_s + (1-W) * F1_n_perp
    W = D_s / (D_s + D_n)
    D_k = 2*TP_k + FP_k + FN_k  (k in {s, n})

Bu ozdeslik TAM olmalidir (makine hassasiyetinde).
"""
import json
from pathlib import Path
import numpy as np
from artifacts.ner_formula_test import (
    make_data, train_neural, predict_neural,
    SEEDS, HIDDEN_RATIOS, TYPES,
)

def confusion(y_true, y_pred, class_idx):
    tp = ((y_pred == class_idx) & (y_true == class_idx)).sum()
    fp = ((y_pred == class_idx) & (y_true != class_idx)).sum()
    fn = ((y_pred != class_idx) & (y_true == class_idx)).sum()
    return int(tp), int(fp), int(fn)

def f1_macro_full(y_true, y_pred):
    f1s = []
    for c in range(len(TYPES)):
        tp, fp, fn = confusion(y_true, y_pred, c)
        if tp + fp + fn == 0:
            f1s.append(0.0); continue
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
    return float(np.mean(f1s))

def macro_D(y_true, y_pred):
    """D = sum_c (2*TP + FP + FN) — pozitif kutlesi."""
    total = 0
    for c in range(len(TYPES)):
        tp, fp, fn = confusion(y_true, y_pred, c)
        total += 2 * tp + fp + fn
    return int(total)

def run(seed, hr):
    X, y, is_known = make_data(seed, hr)
    n = len(y); n_tr = int(0.8 * n)
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_te, y_te, k_te = X[n_tr:], y[n_tr:], is_known[n_tr:]

    params = train_neural(X_tr, y_tr, seed)
    pred_n = predict_neural(X_te, params)

    # Bolumler
    ans_mask = k_te
    abs_mask = ~k_te

    # F1_s: sembolik = mukemmel (y_te), cevaplanan alt kume
    pred_s = np.full(len(y_te), -1, dtype=int)
    pred_s[ans_mask] = y_te[ans_mask]
    F1_s = f1_macro_full(y_te[ans_mask], pred_s[ans_mask]) if ans_mask.sum() else 0.0

    # F1_n_perp: neural, cekimser alt kume
    F1_n_perp = f1_macro_full(y_te[abs_mask], pred_n[abs_mask]) if abs_mask.sum() else 0.0

    # F1_h: hibrit
    pred_h = pred_n.copy()
    pred_h[ans_mask] = y_te[ans_mask]
    F1_h = f1_macro_full(y_te, pred_h)

    # F1_n: neural, tum
    F1_n = f1_macro_full(y_te, pred_n)

    # D_s, D_n
    D_s = macro_D(y_te[ans_mask], pred_s[ans_mask]) if ans_mask.sum() else 0
    D_n = macro_D(y_te[abs_mask], pred_n[abs_mask]) if abs_mask.sum() else 0

    W = D_s / (D_s + D_n) if (D_s + D_n) > 0 else 0.0

    # Formul tahminleri
    F1_h_pred = W * F1_s + (1 - W) * F1_n_perp
    gain_obs = F1_h - F1_n
    gain_pred = W * (F1_s - F1_n) + (1 - W) * (F1_n_perp - F1_n)
    eps_harmonic = abs(gain_obs - gain_pred)

    # Eski (lineer) formul tahmini
    cov = k_te.mean()
    gain_lin = cov * (F1_s - F1_n)
    eps_linear = abs(gain_obs - gain_lin)

    return {
        "seed": seed, "hidden_ratio": hr,
        "cov": round(float(cov), 4),
        "W": round(float(W), 4),
        "F1_s": round(F1_s, 4),
        "F1_n": round(F1_n, 4),
        "F1_n_perp": round(F1_n_perp, 4),
        "F1_h": round(F1_h, 4),
        "gain_obs": round(gain_obs, 4),
        "gain_harmonic": round(gain_pred, 4),
        "gain_linear": round(gain_lin, 4),
        "eps_harmonic": round(eps_harmonic, 6),
        "eps_linear": round(eps_linear, 4),
    }

if __name__ == "__main__":
    results = [run(s, hr) for s in SEEDS for hr in HIDDEN_RATIOS]

    print(f"{'seed':>4} {'hid':>5} {'cov':>6} {'W':>6} {'F1_s':>6} {'F1_n':>6} "
          f"{'F1_n^p':>7} {'F1_h':>6} {'eps_lin':>8} {'eps_harm':>9}")
    print("-" * 80)
    for r in results:
        print(f"{r['seed']:>4} {r['hidden_ratio']:>5.2f} {r['cov']:>6.3f} "
              f"{r['W']:>6.3f} {r['F1_s']:>6.3f} {r['F1_n']:>6.3f} "
              f"{r['F1_n_perp']:>7.3f} {r['F1_h']:>6.3f} "
              f"{r['eps_linear']:>8.4f} {r['eps_harmonic']:>9.6f}")

    eps_l = [r["eps_linear"] for r in results]
    eps_h = [r["eps_harmonic"] for r in results]
    print()
    print(f"LINEER formula:    mean={np.mean(eps_l):.4f}  max={np.max(eps_l):.4f}")
    print(f"HARMONIK formula:  mean={np.mean(eps_h):.6f}  max={np.max(eps_h):.6f}")
    print()
    print(f"Iyilestirme: {np.mean(eps_l)/max(np.mean(eps_h), 1e-9):.0f}x")

    Path("artifacts/f1_harmonic_results.json").write_text(
        json.dumps({"results": results, "summary": {
            "linear_mean_eps": float(np.mean(eps_l)),
            "linear_max_eps": float(np.max(eps_l)),
            "harmonic_mean_eps": float(np.mean(eps_h)),
            "harmonic_max_eps": float(np.max(eps_h)),
        }}, indent=2), encoding="utf-8")
