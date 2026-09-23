# -*- coding: utf-8 -*-
"""TWT'de hibrit symbolic-neural karsilastirmasi.

Uc kol:
- symbolic: SelectiveArcSchemaVerifier (TWT dahili)
- neural:   HGA (mevcut twt_baselines'den)
- hybrid:   veto + fallback (symbolic != None ise o, degilse neural)

Iki esik rejimi test edilir:
- repo:  positive=0.6, negative=0.4  (konuskan)
- tight: positive=0.9, negative=0.1  (cekingen)
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5



def run_twt_hybrid(
    seeds: List[int] = [1, 2, 3, 4, 5],
    thresholds: List[Tuple[float, float, str]] = [
        (0.6, 0.4, "repo"),
        (0.9, 0.1, "tight"),
    ],
) -> Dict[str, Any]:
    import sys
    sys.path.insert(0, ".")
    import torch
    import torch.nn as nn

    from hga.evaluation.real_turkish import (
        SelectiveArcSchemaVerifier,
        prepare_real_turkish_task,
    )
    from hga.evaluation.twt_baselines import (
        ARCHITECTURE_CONFIG,
        ArcFeatureVocabulary,
        build_twt_models,
    )

    task = prepare_real_turkish_task()
    print(f"TWT: train={len(task.train)}, test={len(task.test)}")

    # Test setinde pozitif oran
    pos = sum(c.expected_valid for c in task.test)
    print(f"Test positive: {pos}/{len(task.test)} ({100*pos/len(task.test):.1f}%)")
    print()

    results: Dict[str, Any] = {"thresholds": {}, "config": {
        "seeds": seeds,
        "train_size": len(task.train),
        "test_size": len(task.test),
    }}

    for pos_t, neg_t, name in thresholds:
        print(f"{'='*60}")
        print(f"REJIM: {name} (positive={pos_t}, negative={neg_t})")
        print(f"{'='*60}")

        # Symbolic egit
        verifier = SelectiveArcSchemaVerifier(
            positive_threshold=pos_t,
            negative_threshold=neg_t,
            alpha=1.0,
        )
        verifier.fit(task.train)

        # Symbolic tahmin
        sym_preds = [verifier.predict(c) for c in task.test]
        sym_cov = sum(1 for p in sym_preds if p is not None) / len(sym_preds)
        sym_correct = sum(1 for p, c in zip(sym_preds, task.test)
                          if p is not None and p == c.expected_valid)
        sym_acc_answered = sym_correct / max(sum(1 for p in sym_preds if p is not None), 1)
        # Accuracy treating None as wrong
        sym_acc_full = sum(1 for p, c in zip(sym_preds, task.test)
                           if p is not None and p == c.expected_valid) / len(task.test)

        print(f"Symbolic: coverage={sym_cov:.4f}, acc_answered={sym_acc_answered:.4f}, "
              f"acc_full={sym_acc_full:.4f}")

        # Neural: HGA (mevcut twt_baselines'den)
        # Basitlestirilmis: bir seed icin HGA egit
        vocab = ArcFeatureVocabulary.fit(task.train)
        train_rows, _ = vocab.encode(task.train)
        test_rows, _ = vocab.encode(task.test)
        train_x = torch.tensor(train_rows, dtype=torch.long)
        test_x = torch.tensor(test_rows, dtype=torch.long)
        train_y = torch.tensor([int(c.expected_valid) for c in task.train], dtype=torch.long)

        neural_accs = []
        hybrid_accs = []
        hybrid_covs = []

        for seed in seeds:
            torch.manual_seed(seed)
            cfg = dict(ARCHITECTURE_CONFIG)
            constructors = build_twt_models(vocab.size, config=cfg)
            model = constructors["hga"]()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
            loss_fn = nn.CrossEntropyLoss()

            # Batch schedule
            gen = torch.Generator().manual_seed(seed)
            n = len(task.train)
            schedule = [torch.randint(0, n, (512,), generator=gen) for _ in range(32)]

            model.train()
            for idx in schedule:
                optimizer.zero_grad()
                loss = loss_fn(model(train_x[idx]), train_y[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            # Neural tahmin
            model.eval()
            with torch.no_grad():
                logits = model(test_x)
                neural_preds = logits.argmax(1).tolist()

            neural_acc = sum(1 for p, c in zip(neural_preds, task.test)
                             if bool(p) == c.expected_valid) / len(task.test)
            neural_accs.append(neural_acc)

            # Hybrid: symbolic veto + neural fallback
            hybrid_preds = []
            for s_pred, n_pred, c in zip(sym_preds, neural_preds, task.test):
                if s_pred is not None:
                    hybrid_preds.append(s_pred)
                else:
                    hybrid_preds.append(bool(n_pred))

            hybrid_acc = sum(1 for p, c in zip(hybrid_preds, task.test)
                             if p == c.expected_valid) / len(task.test)
            hybrid_accs.append(hybrid_acc)

            # Hybrid coverage = symbolic coverage (neural her zaman konusuyor)
            hybrid_cov = sum(1 for s in sym_preds if s is not None) / len(sym_preds)
            hybrid_covs.append(hybrid_cov)

        neural_mean = _mean(neural_accs)
        neural_std = _std(neural_accs) if len(neural_accs) > 1 else 0
        hybrid_mean = _mean(hybrid_accs)
        hybrid_std = _std(hybrid_accs) if len(hybrid_accs) > 1 else 0

        print(f"Neural:  acc={neural_mean:.4f} +/- {neural_std:.4f}")
        print(f"Hybrid:  acc={hybrid_mean:.4f} +/- {hybrid_std:.4f}")
        print(f"Gain (hybrid - neural): {hybrid_mean - neural_mean:+.4f}")
        print()

        results["thresholds"][name] = {
            "positive_threshold": pos_t,
            "negative_threshold": neg_t,
            "symbolic_coverage": round(sym_cov, 4),
            "symbolic_acc_answered": round(sym_acc_answered, 4),
            "symbolic_acc_full": round(sym_acc_full, 4),
            "neural_mean": round(neural_mean, 4),
            "neural_std": round(neural_std, 4),
            "hybrid_mean": round(hybrid_mean, 4),
            "hybrid_std": round(hybrid_std, 4),
            "gain": round(hybrid_mean - neural_mean, 4),
        }

    return results


if __name__ == "__main__":
    import json
    from pathlib import Path
    r = run_twt_hybrid()
    Path("artifacts/twt_hybrid_results.json").write_text(
        json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("OK - artifacts/twt_hybrid_results.json")
