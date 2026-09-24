# -*- coding: utf-8 -*-
"""
Tum F1/precision/recall metrikleri icin formul dogrulamasi.

Test edilen formul turleri:
  - macro F1:     w_c = D_s^c / (D_s^c + D_n^c), D = 2TP+FP+FN
  - weighted F1:  a_c * w_c, a_c = n_c / N
  - micro F1:     w = cov (= accuracy)
  - precision:    u_c = D_s^{P,c} / (D_s^{P,c} + D_n^{P,c}), D^P = TP+FP
  - recall:       v_c = D_s^{R,c} / (D_s^{R,c} + D_n^{R,c}), D^R = TP+FN
"""
import json
from pathlib import Path
import numpy as np
from artifacts.ner_formula_test import (
    make_data, train_neural, predict_neural,
    SEEDS, HIDDEN_RATIOS, TYPES,
)


def confusion(y_true, y_pred, c):
    tp = int(((y_pred == c) & (y_true == c)).sum())
    fp = int(((y_pred == c) & (y_true != c)).sum())
    fn = int(((y_pred != c) & (y_true == c)).sum())
    return tp, fp, fn


def per_class_metrics(y_true, y_pred, c):
    tp, fp, fn = confusion(y_true, y_pred, c)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    D_f1 = 2 * tp + fp + fn
    D_p = tp + fp
    D_r = tp + fn
    return {
        "precision": p, "recall": r, "f1": f1,
        "D_f1": D_f1, "D_p": D_p, "D_r": D_r,
    }


def macro_metrics(y_true, y_pred):
    n_classes = len(TYPES)
    per = [per_class_metrics(y_true, y_pred, c) for c in range(n_classes)]
    return {
        "macro_f1": float(np.mean([m["f1"] for m in per])),
        "macro_p": float(np.mean([m["precision"] for m in per])),
        "macro_r": float(np.mean([m["recall"] for m in per])),
        "per": per,
    }


def weighted_f1(y_true, y_pred):
    """n_c = sinif c'nin gercek sayisi (tum test setinde)."""
    n_classes = len(TYPES)
    N = len(y_true)
    total = 0.0
    for c in range(n_classes):
        n_c = int((y_true == c).sum())
        m = per_class_metrics(y_true, y_pred, c)
        total += (n_c / N) * m["f1"]
    return float(total)


def run(seed, hr):
    """Tek konfigurasyon: tum metrikleri hesapla ve dogrula."""
    X, y, is_known = make_data(seed, hr)
    n = len(y); n_tr = int(0.8 * n)
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_te, y_te, k_te = X[n_tr:], y[n_tr:], is_known[n_tr:]

    params = train_neural(X_tr, y_tr, seed)
    pred_n = predict_neural(X_te, params)

    ans_mask = k_te
    abs_mask = ~k_te

    # Sembolik: cevaplanan alt kume
    pred_s = np.full(len(y_te), -1, dtype=int)
    pred_s[ans_mask] = y_te[ans_mask]

    # Hibrit
    pred_h = pred_n.copy()
    pred_h[ans_mask] = y_te[ans_mask]

    # ─── Macro F1 ───
    M_h = macro_metrics(y_te, pred_h)
    M_n = macro_metrics(y_te, pred_n)
    M_s = macro_metrics(y_te[ans_mask], pred_s[ans_mask]) if ans_mask.sum() else None
    M_n_perp = macro_metrics(y_te[abs_mask], pred_n[abs_mask]) if abs_mask.sum() else None

    # Macro F1 formulu (per-class)
    pred_per_class = []
    for c in range(len(TYPES)):
        # Sembolik: cevaplanan alt kume
        if ans_mask.sum() > 0:
            s_per = per_class_metrics(y_te[ans_mask], pred_s[ans_mask], c)
        else:
            s_per = {"f1": 0.0, "D_f1": 0, "D_p": 0, "D_r": 0,
                     "precision": 0.0, "recall": 0.0}
        if abs_mask.sum() > 0:
            n_per = per_class_metrics(y_te[abs_mask], pred_n[abs_mask], c)
        else:
            n_per = {"f1": 0.0, "D_f1": 0, "D_p": 0, "D_r": 0,
                     "precision": 0.0, "recall": 0.0}
        D_s = s_per["D_f1"]; D_n = n_per["D_f1"]
        w_c = D_s / (D_s + D_n) if (D_s + D_n) > 0 else 0.0
        pred_per_class.append(w_c * s_per["f1"] + (1 - w_c) * n_per["f1"])

    macro_f1_pred = float(np.mean(pred_per_class))

    # ─── Weighted F1 ───
    # a_c = n_c / N (tum test setinde)
    N = len(y_te)
    weighted_pred = 0.0
    for c in range(len(TYPES)):
        n_c = int((y_te == c).sum())
        a_c = n_c / N
        if ans_mask.sum() > 0:
            s_per = per_class_metrics(y_te[ans_mask], pred_s[ans_mask], c)
        else:
            s_per = {"f1": 0.0, "D_f1": 0}
        if abs_mask.sum() > 0:
            n_per = per_class_metrics(y_te[abs_mask], pred_n[abs_mask], c)
        else:
            n_per = {"f1": 0.0, "D_f1": 0}
        D_s = s_per["D_f1"]; D_n = n_per["D_f1"]
        w_c = D_s / (D_s + D_n) if (D_s + D_n) > 0 else 0.0
        weighted_pred += a_c * (w_c * s_per["f1"] + (1 - w_c) * n_per["f1"])

    return {
        "seed": seed, "hidden_ratio": hr,
        "cov": float(ans_mask.mean()),
        # Gercek degerler
        "macro_f1_h": M_h["macro_f1"],
        "macro_f1_n": M_n["macro_f1"],
        "weighted_f1_h": weighted_f1(y_te, pred_h),
        "weighted_f1_n": weighted_f1(y_te, pred_n),
        # Tahminler
        "macro_f1_pred": macro_f1_pred,
        "weighted_f1_pred": float(weighted_pred),
        # Hatalar
        "eps_macro": abs(M_h["macro_f1"] - macro_f1_pred),
        "eps_weighted": abs(weighted_f1(y_te, pred_h) - weighted_pred),
    }


if __name__ == "__main__":
    results = [run(s, hr) for s in SEEDS for hr in HIDDEN_RATIOS]

    print(f"{'seed':>4} {'hr':>5} {'cov':>6} "
          f"{'macro_eps':>11} {'weighted_eps':>14}")
    print("-" * 50)
    for r in results:
        print(f"{r['seed']:>4} {r['hidden_ratio']:>5.2f} {r['cov']:>6.3f} "
              f"{r['eps_macro']:>11.6f} {r['eps_weighted']:>14.6f}")

    eps_m = [r["eps_macro"] for r in results]
    eps_w = [r["eps_weighted"] for r in results]
    print()
    print(f"MACRO F1:     mean={np.mean(eps_m):.6f}  max={np.max(eps_m):.6f}")
    print(f"WEIGHTED F1:  mean={np.mean(eps_w):.6f}  max={np.max(eps_w):.6f}")

    Path("artifacts/f1_all_metrics_results.json").write_text(
        json.dumps({"results": results, "summary": {
            "macro_mean_eps": float(np.mean(eps_m)),
            "macro_max_eps": float(np.max(eps_m)),
            "weighted_mean_eps": float(np.mean(eps_w)),
            "weighted_max_eps": float(np.max(eps_w)),
        }}, indent=2), encoding="utf-8")
    print("\nOK — sonuclar kaydedildi")
