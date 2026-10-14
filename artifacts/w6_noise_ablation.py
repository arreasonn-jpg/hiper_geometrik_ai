# -*- coding: utf-8 -*-
"""W6: Noise seviyesi vs Kronecker zaferi — istatistiksel güçlendirme.

Preprint 3'ün "n=5" sınırlılığını gidermek için:
- 13 noise seviyesi (W5: 5)
- 50 seed (W5 ile aynı)
- Bootstrap %95 CI (10.000 yeniden örneklem)
- Permütasyon testi (10.000 permütasyon)
- Holm-Bonferroni çoklu karşılaştırma düzeltmesi

Deterministik: bootstrap/permütasyon RNG'leri sabit tohumlu.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats

from artifacts.w5_noise_ablation import run_condition

VARSAYILAN_NOISE = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                    0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
VARSAYILAN_SEEDLER = tuple(range(1, 51))  # 50 seed

# Deterministik bootstrap/permütasyon için sabit tohum
RNG_SEED = 20261014
N_BOOTSTRAP = 10_000
N_PERMUTASYON = 10_000
ALFA = 0.05


def bootstrap_pearson_ci(
    x: Sequence[float], y: Sequence[float],
    n_boot: int = N_BOOTSTRAP, seed: int = RNG_SEED,
) -> tuple[float, float, float]:
    """Pearson r için bootstrap %95 CI döndürür (r, lo, hi)."""
    xa, ya = np.asarray(x), np.asarray(y)
    r_obs = float(stats.pearsonr(xa, ya).statistic)
    rng = np.random.default_rng(seed)
    ornekler = np.empty(n_boot)
    n = len(xa)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # Aynı indekslerle seçilen alt örneklemde Pearson r
        try:
            ornekler[i] = stats.pearsonr(xa[idx], ya[idx]).statistic
        except Exception:
            ornekler[i] = np.nan
    gecerli = ornekler[~np.isnan(ornekler)]
    # Defansif: tüm bootstrap NaN ise (örn. sabit win_rate), CI tanımsız.
    if gecerli.size == 0:
        return r_obs, float('nan'), float('nan')
    lo, hi = np.percentile(gecerli, [2.5, 97.5])
    return r_obs, float(lo), float(hi)


def permutasyon_testi(
    x: Sequence[float], y: Sequence[float],
    n_perm: int = N_PERMUTASYON, seed: int = RNG_SEED,
) -> tuple[float, float]:
    """İki yanlı permütasyon testi: (r_obs, p_deger)."""
    xa, ya = np.asarray(x), np.asarray(y)
    r_obs = float(stats.pearsonr(xa, ya).statistic)
    rng = np.random.default_rng(seed + 1)
    sayac = 0
    for _ in range(n_perm):
        yp = rng.permutation(ya)
        r_perm = float(stats.pearsonr(xa, yp).statistic)
        if abs(r_perm) >= abs(r_obs):
            sayac += 1
    # +1 düzeltmesi (Phipson & Smyth 2010)
    p = (sayac + 1) / (n_perm + 1)
    return r_obs, float(p)


def holm_bonferroni(p_degerleri: dict[str, float], alfa: float = ALFA) -> dict[str, dict]:
    """Holm-Bonferroni düzeltmesi: her test için p_düzeltilmiş ve karar."""
    sirali = sorted(p_degerleri.items(), key=lambda kv: kv[1])
    m = len(sirali)
    sonuc: dict[str, dict] = {}
    onceki_karar = True
    for i, (ad, p) in enumerate(sirali):
        esik = alfa / (m - i)
        if onceki_karar and p < esik:
            karar = "REDDET"
        else:
            karar = "KABUL"
            onceki_karar = False
        p_duzeltilmis = min(1.0, p * (m - i))
        sonuc[ad] = {
            "p_ham": p, "esik_holm": esik,
            "p_duzeltilmis": p_duzeltilmis, "karar": karar,
        }
    return sonuc


def _pearson(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Güvenli Pearson: r, p (sabit veri için inf/nan korumalı)."""
    if len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan"), float("nan")
    res = stats.pearsonr(np.asarray(x), np.asarray(y))
    return float(res.statistic), float(res.pvalue)


def kosum(
    noise_seviyeleri: Sequence[float],
    seedler: Sequence[int],
    epochs: int = 100,
    n_train: int = 2000,
    n_test: int = 500,
) -> dict[str, Any]:
    """13 noise × 50 seed'i koştur; her seviyede r ve CI üret."""
    satirlar: list[dict] = []
    for noise in noise_seviyeleri:
        r = run_condition(
            noise, list(seedler), epochs=epochs,
            n_train=n_train, n_test=n_test,
        )
        satirlar.append(r)
        print(
            f"noise={noise:.2f}  dense={r['dense_mean']:.4f}±{r['dense_std']:.4f}  "
            f"flat={r['flat_mean']:.4f}±{r['flat_std']:.4f}  "
            f"flat_kazandi={r['flat_wins']}/{r['n_seeds']}"
        )

    noises = [s["noise"] for s in satirlar]
    win_rates = [s["flat_wins"] / s["n_seeds"] for s in satirlar]
    r_obs, p_ham = _pearson(noises, win_rates)
    _, lo, hi = bootstrap_pearson_ci(noises, win_rates)
    r_perm, p_perm = permutasyon_testi(noises, win_rates)

    # H1: negatif korelasyon (W5'te gözlenen)
    r_h1, p_h1 = -0.38, 0.46  # W5'ten sabit (raporlama amaçlı)

    holm = holm_bonferroni({"H2_noise": p_perm, "H1_structure": p_h1})

    return {
        "protokol": "w6_noise_ablation_v1",
        "n_noise": len(noise_seviyeleri),
        "n_seeds": len(seedler),
        "epochs": epochs,
        "n_train": n_train,
        "n_test": n_test,
        "rng_seed": RNG_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "n_permutasyon": N_PERMUTASYON,
        "alfa": ALFA,
        "satirlar": satirlar,
        "ozet": {
            "pearson_r": r_obs,
            "pearson_p_ham": p_ham,
            "bootstrap_ci95": [lo, hi],
            "permutasyon_r": r_perm,
            "permutasyon_p": p_perm,
            "H1_r_sabit": r_h1,
            "H1_p_sabit": p_h1,
            "holm": holm,
        },
    }


def _argparse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="W6 noise ablasyonu (n>=12).")
    ap.add_argument("--noise-levels", default=",".join(str(v) for v in VARSAYILAN_NOISE),
                    help="Virgüllü noise listesi")
    ap.add_argument("--seeds", default="1-50",
                    help="'1-50' veya '1,2,3' biçiminde")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--out", default="artifacts/w6_noise_ablation_results.json")
    return ap.parse_args()


def _seeds_parse(s: str) -> list[int]:
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> int:
    ns = _argparse()
    noise = tuple(float(x) for x in ns.noise_levels.split(",") if x.strip())
    seedler = _seeds_parse(ns.seeds)
    print(f"═══ W6 Noise Ablasyonu — {len(noise)} seviye × {len(seedler)} seed ═══\n")
    rapor = kosum(noise, seedler, epochs=ns.epochs,
                  n_train=ns.n_train, n_test=ns.n_test)
    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rapor, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n═══ ÖZET ═══")
    oz = rapor["ozet"]
    print(f"Pearson r (n={rapor['n_noise']}): {oz['pearson_r']:+.3f}")
    print(f"Bootstrap %95 CI: [{oz['bootstrap_ci95'][0]:+.3f}, {oz['bootstrap_ci95'][1]:+.3f}]")
    print(f"Permütasyon p: {oz['permutasyon_p']:.4f}")
    print(f"Holm kararı: {oz['holm']['H2_noise']['karar']} "
          f"(p_ham={oz['holm']['H2_noise']['p_ham']:.4f}, "
          f"p_düzeltilmiş={oz['holm']['H2_noise']['p_duzeltilmis']:.4f})")
    print(f"\nSonuç: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
