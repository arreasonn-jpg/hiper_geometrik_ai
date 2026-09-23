# -*- coding: utf-8 -*-
"""
NER icin KESIN hata ozdesligi testi.

Basit formul: kazanc ~ cov * (F1_s - F1_n)
Kesin formul: eps = (1-cov) * |F1_n_perp - F1_n|
  burada F1_n_perp = cekimser alt kumedeki neural F1
"""
import json
from pathlib import Path
import numpy as np
from artifacts.ner_formula_test import (
    make_data, train_neural, predict_neural, f1_macro, SEEDS, HIDDEN_RATIOS
)

def f1_on_subset(y_true, y_pred, mask, n_classes=3):
    if mask.sum() == 0:
        return 0.0
    return f1_macro(y_true[mask], y_pred[mask], n_classes)

def run(seed, hidden_ratio):
    X, y, is_known = make_data(seed, hidden_ratio)
    n = len(y); n_tr = int(0.8 * n)
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_te, y_te, k_te = X[n_tr:], y[n_tr:], is_known[n_tr:]

    params = train_neural(X_tr, y_tr, seed)
    pred_n = predict_neural(X_te, params)

    f1_n_all = f1_macro(y_te, pred_n)
    abstain_mask = ~k_te
    f1_n_perp = f1_on_subset(y_te, pred_n, abstain_mask)
    cov = k_te.mean()

    # kesin formul
    eps_pred = (1 - cov) * abs(f1_n_perp - f1_n_all)

    # gozlenen basit formul hatasi
    pred_s = np.full(len(y_te), -1, dtype=int)
    pred_s[k_te] = y_te[k_te]
    f1_s = f1_macro(y_te[k_te], pred_s[k_te]) if k_te.sum() else 0.0

    pred_h = pred_n.copy(); pred_h[k_te] = y_te[k_te]
    f1_h = f1_macro(y_te, pred_h)
    gain_obs = f1_h - f1_n_all
    gain_pred = cov * (f1_s - f1_n_all)
    eps_obs = abs(gain_obs - gain_pred)

    return {
        "seed": seed, "hidden_ratio": hidden_ratio,
        "cov": round(cov, 4),
        "f1_n_all": round(f1_n_all, 4),
        "f1_n_perp": round(f1_n_perp, 4),
        "one_minus_cov": round(1 - cov, 4),
        "abs_diff": round(abs(f1_n_perp - f1_n_all), 4),
        "eps_pred": round(eps_pred, 4),
        "eps_obs": round(eps_obs, 4),
        "ratio_obs_over_pred": round(eps_obs / eps_pred, 4) if eps_pred > 1e-6 else None,
    }

if __name__ == "__main__":
    results = [run(s, hr) for s in SEEDS for hr in HIDDEN_RATIOS]

    print(f"{'seed':>4} {'hid':>5} {'1-cov':>6} {'|d|':>6} "
          f"{'eps_pred':>9} {'eps_obs':>8} {'ratio':>6}")
    print("-" * 55)
    for r in results:
        ratio = r["ratio_obs_over_pred"]
        ratio_s = f"{ratio:.3f}" if ratio is not None else "  --"
        print(f"{r['seed']:>4} {r['hidden_ratio']:>5.2f} "
              f"{r['one_minus_cov']:>6.3f} {r['abs_diff']:>6.3f} "
              f"{r['eps_pred']:>9.4f} {r['eps_obs']:>8.4f} {ratio_s:>6}")

    # Oran: kesin formul gozlenen hatayi acikliyor mu?
    ratios = [r["ratio_obs_over_pred"] for r in results
              if r["ratio_obs_over_pred"] is not None]
    print()
    print(f"Mean ratio (obs/pred): {np.mean(ratios):.3f}")
    print(f"Std  ratio:            {np.std(ratios):.3f}")
    print(f"Min  ratio:            {np.min(ratios):.3f}")
    print(f"Max  ratio:            {np.max(ratios):.3f}")
    print()
    print("Yorum: ratio ~ 1.0 => kesin formul hatayi acikliyor")

    Path("artifacts/ner_exact_error_results.json").write_text(
        json.dumps({"results": results,
                    "summary": {"mean_ratio": float(np.mean(ratios)),
                                "std_ratio": float(np.std(ratios))}},
                   indent=2), encoding="utf-8")
