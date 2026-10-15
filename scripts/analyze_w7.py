"""W7 mekanizma analizi: hangi metrik noise ile en güçlü korele?"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

RNG_SEED = 20261015
N_BOOT = 10_000
N_PERM = 10_000


def _pearson(x, y):
    r = stats.pearsonr(np.asarray(x), np.asarray(y))
    return float(r.statistic), float(r.pvalue)


def bootstrap_ci(x, y, n_boot=N_BOOT, seed=RNG_SEED):
    xa, ya = np.asarray(x), np.asarray(y)
    rng = np.random.default_rng(seed)
    n = len(xa)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            vals.append(stats.pearsonr(xa[idx], ya[idx]).statistic)
        except Exception:
            pass
    vals = np.array([v for v in vals if np.isfinite(v)])
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _metrics_from_row(r: dict) -> dict[str, float]:
    """Bir noise satırından tüm karşılaştırılabilir metrikleri çıkar."""
    return {
        "noise": r["noise"],
        "flat_wins": r["flat_wins"] / r["n_seeds"],
        "dense_gap": r["dense"]["train_test_gap_mean"],
        "flat_gap": r["flat"]["train_test_gap_mean"],
        "gap_delta": (r["dense"]["train_test_gap_mean"]
                      - r["flat"]["train_test_gap_mean"]),
        "dense_rank": r["dense"]["eff_rank_mean"],
        "flat_rank": r["flat"]["eff_rank_mean"],
        "rank_delta": (r["dense"]["eff_rank_mean"]
                       - r["flat"]["eff_rank_mean"]),
        "dense_acc": r["dense"]["test_acc_mean"],
        "flat_acc": r["flat"]["test_acc_mean"],
        "acc_delta": (r["flat"]["test_acc_mean"]
                      - r["dense"]["test_acc_mean"]),
    }


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1
               else "artifacts/w7_runs/w7_full_50seed.json")
    rapor = json.loads(src.read_text(encoding="utf-8"))
    satirlar = [_metrics_from_row(r) for r in rapor["satirlar"]]

    nv = [s["noise"] for s in satirlar]
    win = [s["flat_wins"] for s in satirlar]

    # Her metrik için noise ile Pearson r + CI + perm
    metrikler = ["dense_gap", "flat_gap", "gap_delta",
                 "dense_rank", "flat_rank", "rank_delta",
                 "dense_acc", "flat_acc", "acc_delta"]
    bulgular = {}
    for m in metrikler:
        ys = [s[m] for s in satirlar]
        r, p = _pearson(nv, ys)
        lo, hi = bootstrap_ci(nv, ys)
        bulgular[m] = {"r": r, "p": p, "ci95": [lo, hi]}

    # Win-rate ile en güçlü korele olan metrik
    en_guclu = max(bulgular.items(),
                   key=lambda kv: abs(kv[1]["r"]) if np.isfinite(kv[1]["r"]) else 0)

    cikti = {
        "protokol": "w7_mechanism_analysis_v1",
        "kaynak": str(src),
        "n_noise": rapor["n_noise"],
        "n_seeds": rapor["n_seeds"],
        "flat_win_rate": win,
        "noise": nv,
        "korelasyonlar": bulgular,
        "winrate_vs_metrics": {
            m: _pearson([s[m] for s in satirlar], win)
            for m in metrikler
        },
        "en_guclu_ilgi": {"metrik": en_guclu[0], **en_guclu[1]},
    }

    out = src.with_name(src.stem + "_analysis.json")
    out.write_text(json.dumps(cikti, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    print("═══ W7 Mekanizma Analizi ═══\n")
    print(f"Kaynak: {src}")
    print(f"n_noise={rapor['n_noise']}, n_seeds={rapor['n_seeds']}\n")
    print("Noise ile korelasyon:")
    for m, b in sorted(bulgular.items(),
                       key=lambda kv: -abs(kv[1]["r"])
                       if np.isfinite(kv[1]["r"]) else 0):
        ci = f"[{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]"
        print(f"  {m:15s}: r={b['r']:+.3f}  p={b['p']:.4f}  CI={ci}")
    print(f"\nWin-rate ile en güçlü: {en_guclu[0]} (r={en_guclu[1]['r']:+.3f})")
    print(f"\nSonuç: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
