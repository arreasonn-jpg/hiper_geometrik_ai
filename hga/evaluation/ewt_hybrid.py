# -*- coding: utf-8 -*-
"""EWT (Ingilizce) hibrit symbolic-neural testi.

TWT'de bulunan hibrit zaferi EWT'de de dogrula.
- symbolic: TWT'nin SelectiveArcSchemaVerifier'i (duck typing)
- neural:   Basit dense MLP (embedding + MLP)
- hybrid:   veto + fallback
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


def run_ewt_hybrid(
    seeds: List[int] = [1, 2, 3, 4, 5],
    thresholds: List[Tuple[float, float, str]] = [
        (0.50, 0.50, "very_loose"),
        (0.60, 0.40, "repo"),
        (0.80, 0.20, "strict"),
        (0.90, 0.10, "tight"),
    ],
    task_preparer=None,
    lang_name: str = "EWT",
) -> Dict[str, Any]:
    import sys
    sys.path.insert(0, ".")
    import torch
    import torch.nn as nn

    from hga.evaluation.english_ewt import prepare_english_ewt_task
    from hga.evaluation.real_turkish import SelectiveArcSchemaVerifier

    if task_preparer is None:
        task_preparer = prepare_english_ewt_task
    task = task_preparer()
    print(f"{lang_name}: train={len(task.train)}, test={len(task.test)}")
    pos = sum(c.expected_valid for c in task.test)
    print(f"Test positive: {pos}/{len(task.test)} ({100*pos/len(task.test):.1f}%)")

    # Kategorik vocabulary
    def _get_vocab(values):
        return {v: i for i, v in enumerate(sorted(set(values)))}

    dep_vocab = _get_vocab([c.dependent_upos for c in task.train])
    rel_vocab = _get_vocab([c.relation for c in task.train])
    head_vocab = _get_vocab([c.head_upos for c in task.train])

    def _feats(cands):
        dep = torch.tensor([dep_vocab.get(c.dependent_upos, 0) for c in cands], dtype=torch.long)
        rel = torch.tensor([rel_vocab.get(c.relation, 0) for c in cands], dtype=torch.long)
        head = torch.tensor([head_vocab.get(c.head_upos, 0) for c in cands], dtype=torch.long)
        return dep, rel, head

    train_dep, train_rel, train_head = _feats(task.train)
    test_dep, test_rel, test_head = _feats(task.test)
    train_y = torch.tensor([int(c.expected_valid) for c in task.train], dtype=torch.long)

    n_dep = len(dep_vocab)
    n_rel = len(rel_vocab)
    n_head = len(head_vocab)

    class EWTDense(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb_dep = nn.Embedding(n_dep + 1, 16)
            self.emb_rel = nn.Embedding(n_rel + 1, 16)
            self.emb_head = nn.Embedding(n_head + 1, 16)
            self.body = nn.Sequential(
                nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 2),
            )
        def forward(self, dep, rel, head):
            x = torch.cat([self.emb_dep(dep), self.emb_rel(rel), self.emb_head(head)], dim=1)
            return self.body(x)

    results: Dict[str, Any] = {"thresholds": {}, "config": {
        "seeds": seeds,
        "train_size": len(task.train),
        "test_size": len(task.test),
    }}

    for pos_t, neg_t, name in thresholds:
        print()
        print(f"{'='*60}")
        print(f"REJIM: {name} (positive={pos_t}, negative={neg_t})")
        print(f"{'='*60}")

        verifier = SelectiveArcSchemaVerifier(
            positive_threshold=pos_t, negative_threshold=neg_t, alpha=1.0,
        )
        verifier.fit(task.train)

        sym_preds = [verifier.predict(c) for c in task.test]
        sym_cov = sum(1 for p in sym_preds if p is not None) / len(sym_preds)
        sym_correct = sum(1 for p, c in zip(sym_preds, task.test)
                          if p is not None and p == c.expected_valid)
        sym_n_answered = max(sum(1 for p in sym_preds if p is not None), 1)
        sym_acc_answered = sym_correct / sym_n_answered
        sym_acc_full = sym_correct / len(task.test)

        print(f"Symbolic: coverage={sym_cov:.4f}, acc_answered={sym_acc_answered:.4f}, acc_full={sym_acc_full:.4f}")

        neural_accs = []
        hybrid_accs = []

        for seed in seeds:
            torch.manual_seed(seed)
            model = EWTDense()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
            loss_fn = nn.CrossEntropyLoss()

            gen = torch.Generator().manual_seed(seed)
            n = len(task.train)
            schedule = [torch.randint(0, n, (64,), generator=gen) for _ in range(32)]

            model.train()
            for idx in schedule:
                optimizer.zero_grad()
                loss = loss_fn(
                    model(train_dep[idx], train_rel[idx], train_head[idx]),
                    train_y[idx],
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                logits = model(test_dep, test_rel, test_head)
                neural_preds = logits.argmax(1).tolist()

            neural_acc = sum(1 for p, c in zip(neural_preds, task.test)
                             if bool(p) == c.expected_valid) / len(task.test)
            neural_accs.append(neural_acc)

            hybrid_preds = []
            for s_pred, n_pred in zip(sym_preds, neural_preds):
                if s_pred is not None:
                    hybrid_preds.append(s_pred)
                else:
                    hybrid_preds.append(bool(n_pred))

            hybrid_acc = sum(1 for p, c in zip(hybrid_preds, task.test)
                             if p == c.expected_valid) / len(task.test)
            hybrid_accs.append(hybrid_acc)

        neural_mean = _mean(neural_accs)
        neural_std = _std(neural_accs)
        hybrid_mean = _mean(hybrid_accs)
        hybrid_std = _std(hybrid_accs)

        print(f"Neural:  acc={neural_mean:.4f} +/- {neural_std:.4f}")
        print(f"Hybrid:  acc={hybrid_mean:.4f} +/- {hybrid_std:.4f}")
        print(f"Gain (hybrid - neural): {hybrid_mean - neural_mean:+.4f}")

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
    r = run_ewt_hybrid()
    Path("artifacts/ewt_hybrid_results.json").write_text(
        json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print()
    print("OK - artifacts/ewt_hybrid_results.json")
