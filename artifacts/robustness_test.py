"""Hafta 1 Gün 3 — Robustness testleri."""
import json
import statistics as st
import sys
sys.path.insert(0, '.')

from hga.evaluation.paradigma import run_paradigm_ablation

SEEDS = [1, 2, 3]

def run_one(**kwargs):
    """3 seed ortalaması."""
    results = [run_paradigm_ablation(seed=s, epochs=60, **kwargs) for s in SEEDS]
    def avg(arm, metric):
        return st.mean([r.arms[arm][metric] for r in results])
    return {
        "symbolic_acc": avg("symbolic", "accuracy"),
        "neural_acc": avg("neural", "accuracy"),
        "hybrid_acc": avg("hybrid", "accuracy"),
        "hybrid_unseen": avg("hybrid", "accuracy_unseen"),
        "neural_unseen": avg("neural", "accuracy_unseen"),
        "gain": avg("hybrid", "accuracy") - avg("neural", "accuracy"),
    }

# ── Test A: Entity count ────────────────────────────────────────────
print("="*70)
print("TEST A: Entity Count")
print("="*70)
print(f"{'entity':>8} | {'sym':>8} | {'neu':>8} | {'hyb':>8} | {'gain':>8}")
print("-"*55)
results_a = []
for n in [60, 120, 240]:
    r = run_one(entity_count=n)
    r["entity_count"] = n
    results_a.append(r)
    print(f"{n:>8} | {r['symbolic_acc']:>8.4f} | {r['neural_acc']:>8.4f} | "
          f"{r['hybrid_acc']:>8.4f} | {r['gain']:>+8.4f}")

# ── Test B: Relation count ──────────────────────────────────────────
print()
print("="*70)
print("TEST B: Relation Count")
print("="*70)
print(f"{'rel':>8} | {'sym':>8} | {'neu':>8} | {'hyb':>8} | {'gain':>8}")
print("-"*55)
results_b = []
for n in [4, 8, 16]:
    r = run_one(relation_count=n)
    r["relation_count"] = n
    results_b.append(r)
    print(f"{n:>8} | {r['symbolic_acc']:>8.4f} | {r['neural_acc']:>8.4f} | "
          f"{r['hybrid_acc']:>8.4f} | {r['gain']:>+8.4f}")

# ── Test C: Unseen ratio ────────────────────────────────────────────
print()
print("="*70)
print("TEST C: Unseen Entity Ratio")
print("="*70)
print(f"{'unseen':>8} | {'neu_uns':>8} | {'hyb_uns':>8} | {'gain_uns':>10}")
print("-"*55)
results_c = []
for n in [0.10, 0.20, 0.40]:
    r = run_one(unseen_entity_ratio=n)
    r["unseen_ratio"] = n
    r["unseen_gain"] = r["hybrid_unseen"] - r["neural_unseen"]
    results_c.append(r)
    print(f"{n:>8.2f} | {r['neural_unseen']:>8.4f} | {r['hybrid_unseen']:>8.4f} | "
          f"{r['unseen_gain']:>+10.4f}")

# Kaydet
out = {"A_entity": results_a, "B_relation": results_b, "C_unseen": results_c}
with open("artifacts/robustness_test.json", "w") as f:
    json.dump(out, f, indent=2)

print()
print("✅ Kaydedildi: artifacts/robustness_test.json")
