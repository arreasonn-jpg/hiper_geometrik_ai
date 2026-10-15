# -*- coding: utf-8 -*-
"""W7: Kronecker implicit-regularisation MEKANIZMASI.

W6: noise ↑ → Kronecker win-rate ↑ (r=+0.956, n=13).
W7: Bu korelasyonun MEKANIZMASI nedir? Hangi olculebilir ozellik
(train/test gap, effective rank, spectral entropy, weight norm,
gradient norm) noise ile birlikte degisiyor?

Protokol:
- Ayni gorev ve seed'ler (W5/W6 uyumlu)
- Genisletilmis train(): train/test loss, grad norm
- Egitim sonrasi: effective rank, participation ratio, spectral entropy
- Deterministik RNG
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from artifacts.w5_noise_ablation import task_block_noise
from artifacts.w5_proper_6tasks import (
    DenseMLP,
    FlatKroneckerNet,
)

RNG_SEED = 20261015


# ── Metrik yardımcıları ────────────────────────────────────────────
def _matrices_of(model) -> list[torch.Tensor]:
    """Modeldeki 2D ağırlık matrislerini topla (Linear ve Bilinear A/B)."""
    mats = []
    for name, p in model.named_parameters():
        if p.dim() == 2:
            mats.append(p.detach())
    return mats


def effective_rank_metrics(model) -> dict[str, float]:
    """Tüm 2D ağırlıkların ortalama effective rank / entropy / norm."""
    mats = _matrices_of(model)
    if not mats:
        return {"eff_rank": 0.0, "participation_ratio": 0.0,
                "spectral_entropy": 0.0, "weight_norm": 0.0}
    ranks, prs, ents, norms = [], [], [], []
    for m in mats:
        s = torch.linalg.svdvals(m.float())
        s = s[s > 0]
        if s.numel() == 0:
            continue
        p = s / s.sum()
        ent = -(p * p.log()).sum().item()
        # Effective rank = exp(entropy)
        er = float(np.exp(ent))
        # Participation ratio = (Σσ)² / Σσ²
        pr = (s.sum() ** 2 / (s ** 2).sum()).item()
        ranks.append(er)
        prs.append(pr)
        ents.append(ent)
        norms.append(m.norm().item())
    if not ranks:
        return {"eff_rank": 0.0, "participation_ratio": 0.0,
                "spectral_entropy": 0.0, "weight_norm": 0.0}
    return {
        "eff_rank": float(np.mean(ranks)),
        "participation_ratio": float(np.mean(prs)),
        "spectral_entropy": float(np.mean(ents)),
        "weight_norm": float(np.mean(norms)),
    }


# ── Genişletilmiş eğitim: train/test loss + grad norm ──────────────
def train_with_metrics(
    model, task, epochs: int = 100, lr: float = 1e-3, seed: int = 42,
) -> dict[str, float]:
    """W5 train()'in genişletilmiş hâli: son train/test loss + grad norm."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X_tr, y_tr = task["x_train"], task["y_train"]
    X_te, y_te = task["x_test"], task["y_test"]
    model.train()
    son_grad_norm = 0.0
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X_tr)
        loss = F.cross_entropy(logits, y_tr)
        loss.backward()
        # Son epoch'un grad norm'u
        gn = sum(p.grad.norm().item() ** 2 for p in model.parameters()
                 if p.grad is not None) ** 0.5
        son_grad_norm = float(gn)
        opt.step()

    model.eval()
    with torch.no_grad():
        train_logits = model(X_tr)
        test_logits = model(X_te)
        train_loss = float(F.cross_entropy(train_logits, y_tr).item())
        test_loss = float(F.cross_entropy(test_logits, y_te).item())
        train_acc = float((train_logits.argmax(1) == y_tr).float().mean().item())
        test_acc = float((test_logits.argmax(1) == y_te).float().mean().item())

    return {
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_test_gap": test_loss - train_loss,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "grad_norm": son_grad_norm,
    }


# ── Tek noise seviyesi: tüm seed'ler ───────────────────────────────
def run_condition_w7(
    noise: float, seeds: Sequence[int],
    epochs: int = 100, n_train: int = 2000, n_test: int = 500,
) -> dict[str, Any]:
    """Tek noise seviyesinde tüm seed'leri koş, iki modelin
    acc/loss/gap/rank/norm metriklerini topla."""
    rows: list[dict] = []
    for s in seeds:
        task = task_block_noise(
            n_train=n_train, n_test=n_test, seed=s, noise=noise)
        n, nc = 16, 4
        dense = DenseMLP(n, hidden=n * n // 4, num_classes=nc)
        flat = FlatKroneckerNet(n, depth=1, num_classes=nc)

        dense_m = train_with_metrics(dense, task, epochs=epochs, seed=s)
        flat_m = train_with_metrics(flat, task, epochs=epochs, seed=s)
        dense_r = effective_rank_metrics(dense)
        flat_r = effective_rank_metrics(flat)

        rows.append({
            "seed": s,
            "dense": {**dense_m, **dense_r},
            "flat": {**flat_m, **flat_r},
            "winner": "flat" if flat_m["test_acc"] >= dense_m["test_acc"] else "dense",
        })

    def _mean(key_outer: str, key_inner: str) -> float:
        return float(np.mean([r[key_outer][key_inner] for r in rows]))

    def _std(key_outer: str, key_inner: str) -> float:
        return float(np.std([r[key_outer][key_inner] for r in rows]))

    return {
        "noise": noise,
        "n_seeds": len(seeds),
        "flat_wins": sum(1 for r in rows if r["winner"] == "flat"),
        "dense": {
            "test_acc_mean": _mean("dense", "test_acc"),
            "test_acc_std": _std("dense", "test_acc"),
            "train_test_gap_mean": _mean("dense", "train_test_gap"),
            "grad_norm_mean": _mean("dense", "grad_norm"),
            "eff_rank_mean": _mean("dense", "eff_rank"),
            "spectral_entropy_mean": _mean("dense", "spectral_entropy"),
            "weight_norm_mean": _mean("dense", "weight_norm"),
        },
        "flat": {
            "test_acc_mean": _mean("flat", "test_acc"),
            "test_acc_std": _std("flat", "test_acc"),
            "train_test_gap_mean": _mean("flat", "train_test_gap"),
            "grad_norm_mean": _mean("flat", "grad_norm"),
            "eff_rank_mean": _mean("flat", "eff_rank"),
            "spectral_entropy_mean": _mean("flat", "spectral_entropy"),
            "weight_norm_mean": _mean("flat", "weight_norm"),
        },
        "per_seed": rows,
    }


# ── Main ───────────────────────────────────────────────────────────
VARSAYILAN_NOISE = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                    0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
VARSAYILAN_SEEDLER = tuple(range(1, 51))


def _seeds_parse(s: str) -> list[int]:
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="W7: Kronecker mekanizma analizi (rank/gap/norm).")
    ap.add_argument("--noise-levels", default=",".join(map(str, VARSAYILAN_NOISE)))
    ap.add_argument("--seeds", default="1-50")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--out", default="artifacts/w7_runs/w7_mechanism.json")
    ns = ap.parse_args()

    noise = tuple(float(x) for x in ns.noise_levels.split(",") if x.strip())
    seedler = _seeds_parse(ns.seeds)
    print(f"═══ W7 Kronecker Mekanizma — {len(noise)} noise × {len(seedler)} seed ═══\n")

    satirlar = []
    for nv in noise:
        r = run_condition_w7(nv, seedler, epochs=ns.epochs,
                             n_train=ns.n_train, n_test=ns.n_test)
        satirlar.append(r)
        print(
            f"noise={nv:.2f}  "
            f"dense(acc={r['dense']['test_acc_mean']:.3f}, "
            f"gap={r['dense']['train_test_gap_mean']:.3f}, "
            f"erank={r['dense']['eff_rank_mean']:.1f})  "
            f"flat(acc={r['flat']['test_acc_mean']:.3f}, "
            f"gap={r['flat']['train_test_gap_mean']:.3f}, "
            f"erank={r['flat']['eff_rank_mean']:.1f})  "
            f"flat_wins={r['flat_wins']}/{r['n_seeds']}")

    rapor = {
        "protokol": "w7_kronecker_mechanism_v1",
        "n_noise": len(noise),
        "n_seeds": len(seedler),
        "epochs": ns.epochs,
        "n_train": ns.n_train,
        "n_test": ns.n_test,
        "rng_seed": RNG_SEED,
        "satirlar": satirlar,
    }
    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rapor, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nSonuç: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
