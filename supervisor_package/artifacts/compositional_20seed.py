# -*- coding: utf-8 -*-
"""
Compositional gorevde 20 seed istatistiksel guclendirme.

Soru: flat Kronecker'in dense MLP'ye karsi 8x verimliligi
istatistiksel olarak anlamli mi?

Testler:
  - Paired t-test (dense vs flat, ayni seed)
  - Cohen's d (effect size)
  - Bootstrap CI (%95)
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List

from hga.evaluation.hierarchical_compositional import compare_on_hierarchical


def paired_t_test(a: List[float], b: List[float]) -> Dict[str, float]:
    """a ve b eslesmis ornekler. t-stat ve yaklasik p-degeri."""
    n = len(a)
    diffs = [ai - bi for ai, bi in zip(a, b)]
    mean_d = statistics.mean(diffs)
    sd_d = statistics.stdev(diffs) if n > 1 else 0.0
    if sd_d == 0:
        return {"t": 0.0, "p_approx": 1.0, "mean_diff": mean_d}
    se = sd_d / math.sqrt(n)
    t = mean_d / se
    # Yaklasik iki kuyrukli p (normal approx; n>=20 icin iyi)
    # erf kullanmadan basit yaklasim
    p_approx = math.erfc(abs(t) / math.sqrt(2))
    return {"t": t, "p_approx": p_approx, "mean_diff": mean_d,
            "sd_diff": sd_d, "se": se}


def cohens_d(a: List[float], b: List[float]) -> float:
    """Paired Cohen's d."""
    diffs = [ai - bi for ai, bi in zip(a, b)]
    mean_d = statistics.mean(diffs)
    sd_d = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    return mean_d / sd_d if sd_d > 0 else 0.0


def bootstrap_ci(diffs: List[float], n_boot: int = 5000,
                 alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    import random
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return {"ci_lo": lo, "ci_hi": hi, "mean": statistics.mean(diffs)}


def main():
    seeds = list(range(1, 21))
    by_model: Dict[str, Dict[str, List[float]]] = {}

    for seed in seeds:
        r = compare_on_hierarchical(
            n_base=16, num_classes=4,
            n_train=2000, n_test=500,
            epochs=100, seed=seed,
        )
        for m in r["results"]:
            name = m["name"]
            by_model.setdefault(name, {"acc": [], "params": []})
            by_model[name]["acc"].append(m["test_acc"])
            by_model[name]["params"].append(m["params"])
        print(f"  seed={seed:2d}  " + "  ".join(
            f"{m['name'][:6]}={m['test_acc']:.4f}" for m in r["results"]))

    # ─── Ozet ──
    print("\n═══ Ozet (20 seed) ═══")
    for name, d in by_model.items():
        acc = d["acc"]
        print(f"  {name:26s} params={int(statistics.mean(d['params'])):>8,} "
              f"test={statistics.mean(acc):.4f} ± {statistics.stdev(acc):.4f}")

    # ─── Istatistiksel testler ──
    print("\n═══ Istatistiksel Testler ═══")
    flat = by_model["flat_kronecker"]["acc"]
    dense = by_model["dense_mlp"]["acc"]
    hier = by_model["hierarchical_kronecker"]["acc"]

    for label, a, b in [
        ("flat vs dense", flat, dense),
        ("flat vs hier", flat, hier),
        ("dense vs hier", dense, hier),
    ]:
        t = paired_t_test(a, b)
        d = cohens_d(a, b)
        ci = bootstrap_ci([ai - bi for ai, bi in zip(a, b)])
        print(f"\n  {label}:")
        print(f"    mean_diff = {t['mean_diff']:+.4f}")
        print(f"    paired t  = {t['t']:+.3f}  (p ≈ {t['p_approx']:.4f})")
        print(f"    Cohen's d = {d:+.3f}")
        print(f"    %95 CI    = [{ci['ci_lo']:+.4f}, {ci['ci_hi']:+.4f}]")

    Path("artifacts/compositional_20seed_results.json").write_text(
        json.dumps({
            "seeds": seeds,
            "by_model": {k: v for k, v in by_model.items()},
        }, indent=2), encoding="utf-8")
    print("\nOK — sonuclar kaydedildi")


if __name__ == "__main__":
    main()
