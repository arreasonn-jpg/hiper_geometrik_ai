# -*- coding: utf-8 -*-
"""Sentetik insan degerlendirici simulasyonu.

Gercek insan yerine sentetik rater'lar uretir; analiz hattinin
dogru calistigini test eder.

3 senaryo:
- Yuksek anlasma (dusuk noise): alpha > 0.8
- Orta anlasma: 0.667 < alpha < 0.8
- Dusuk anlasma (yuksek noise): alpha < 0.667
"""
import random
import statistics as st
from typing import Dict, List

from hga.evaluation.human_evaluation import (
    ARMS, DIMENSIONS, analyze_ratings, krippendorff_alpha,
)


def simulate_raters(
    true_scores: Dict[str, List[float]],
    n_raters: int = 10,
    noise: float = 1.0,
    rater_bias_std: float = 0.0,
    seed: int = 42,
) -> Dict[str, Dict[int, List[float]]]:
    """Her rater icin bias + noise ekle, 1-5 araligina kirp."""
    rng = random.Random(seed)
    ratings: Dict[str, Dict[int, List[float]]] = {}
    for dim, scores in true_scores.items():
        scale = DIMENSIONS[dim]["scale"]
        lo, hi = scale[0], scale[-1]
        per_unit: Dict[int, List[float]] = {i: [] for i in range(len(scores))}
        for rater in range(n_raters):
            bias = rng.gauss(0, rater_bias_std) if rater_bias_std > 0 else 0.0
            for i, s in enumerate(scores):
                val = s + bias + rng.gauss(0, noise)
                val = max(lo, min(hi, round(val)))
                per_unit[i].append(float(val))
        ratings[dim] = per_unit
    return ratings


def main():
    print("="*70)
    print("SENTETIK INSAN DEGERLENDIRICI SIMULASYONU")
    print("="*70)
    print()

    # Sentetik gorev: 50 item x 4 kol
    n_items = 50
    true_scores: Dict[str, List[float]] = {}
    for dim, meta in DIMENSIONS.items():
        scale = meta["scale"]
        lo, hi = scale[0], scale[-1]
        base = {"symbolic": hi * 0.9, "hga": hi * 0.7,
                "dense": hi * 0.4, "transformer": hi * 0.5}
        scores = []
        for i in range(n_items):
            arm = list(ARMS)[i % len(ARMS)]
            scores.append(base[arm])
        true_scores[dim] = scores

    print(f"Item: {n_items}, Kol: {len(ARMS)}, Boyut: {len(DIMENSIONS)}")
    print()

    scenarios = [
        ("YUKSEK ANLASMA", 0.3, 0.0),
        ("ORTA ANLASMA", 1.0, 0.5),
        ("DUSUK ANLASMA", 1.8, 1.0),
    ]

    for name, noise, bias_std in scenarios:
        print("="*70)
        print(f"SENARYO: {name}  (noise={noise}, bias_std={bias_std})")
        print("="*70)

        ratings = simulate_raters(
            true_scores, n_raters=10,
            noise=noise, rater_bias_std=bias_std, seed=42,
        )
        result = analyze_ratings(ratings)

        for dim in DIMENSIONS:
            alpha = result[dim]["alpha"]
            if alpha is None:
                print(f"  {dim:<25} alpha=TANIMSIZ")
                continue
            mark = "OK" if alpha >= 0.8 else ("GECICI" if alpha >= 0.667 else "ZAYIF")
            print(f"  {dim:<25} alpha={alpha:>7.4f}  [{mark}]")

        s = result["_summary"]
        print()
        print(f"  Kabul edilebilir: {s['acceptable_dimensions']}/{s['dimensions']}")
        print(f"  Tumu kabul: {s['all_acceptable']}")
        print()


if __name__ == "__main__":
    main()
