# -*- coding: utf-8 -*-
"""Paradigma 10-seed istatistiksel analiz (doğru JSON yapısı)."""
import json
import statistics as st
from pathlib import Path
from scipy import stats

d = json.loads(Path("artifacts/paradigma_10seed.json").read_text())
reports = d["reports"]

# Seed bazlı accuracies
sym_accs = [r["arms"]["symbolic"]["accuracy"] for r in reports]
neu_accs = [r["arms"]["neural"]["accuracy"] for r in reports]
hyb_accs = [r["arms"]["hybrid"]["accuracy"] for r in reports]

# Seed bazlı unseen accuracies
sym_unseen = [r["arms"]["symbolic"]["accuracy_unseen"] for r in reports]
neu_unseen = [r["arms"]["neural"]["accuracy_unseen"] for r in reports]
hyb_unseen = [r["arms"]["hybrid"]["accuracy_unseen"] for r in reports]

print("="*70)
print("PARADIGMA 10-SEED İSTATİSTİKSEL ANALİZ")
print("="*70)
print()

print("### Seed bazlı ACCURACY")
print(f"{'seed':>6} | {'symbolic':>10} | {'neural':>10} | {'hybrid':>10}")
print("-" * 50)
for i, (s, n, h) in enumerate(zip(sym_accs, neu_accs, hyb_accs), 1):
    print(f"{i:>6} | {s:>10.4f} | {n:>10.4f} | {h:>10.4f}")
print("-" * 50)
print(f"{'ORT':>6} | {st.mean(sym_accs):>10.4f} | {st.mean(neu_accs):>10.4f} | {st.mean(hyb_accs):>10.4f}")
print(f"{'STD':>6} | {st.stdev(sym_accs):>10.4f} | {st.stdev(neu_accs):>10.4f} | {st.stdev(hyb_accs):>10.4f}")
print()

print("### Seed bazlı ACCURACY_UNSEEN (cold-start)")
print(f"{'seed':>6} | {'symbolic':>10} | {'neural':>10} | {'hybrid':>10}")
print("-" * 50)
for i, (s, n, h) in enumerate(zip(sym_unseen, neu_unseen, hyb_unseen), 1):
    print(f"{i:>6} | {s:>10.4f} | {n:>10.4f} | {h:>10.4f}")
print("-" * 50)
print(f"{'ORT':>6} | {st.mean(sym_unseen):>10.4f} | {st.mean(neu_unseen):>10.4f} | {st.mean(hyb_unseen):>10.4f}")
print()

print("="*70)
print("PAIRED T-TESTLERİ")
print("="*70)
print()

def paired_test(name, a, b):
    diffs = [x - y for x, y in zip(a, b)]
    mean_d = st.mean(diffs)
    std_d = st.stdev(diffs)
    n = len(diffs)
    t = mean_d / (std_d / (n ** 0.5))
    p = 2 * (1 - stats.t.cdf(abs(t), df=n-1))
    d_cohen = mean_d / std_d
    se = std_d / (n ** 0.5)
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_low = mean_d - t_crit * se
    ci_high = mean_d + t_crit * se

    print(f"## {name}")
    print(f"  mean Δ = {mean_d:+.5f}")
    print(f"  %95 CI = [{ci_low:+.5f}, {ci_high:+.5f}]")
    print(f"  Cohen's d = {d_cohen:.3f}")
    print(f"  t = {t:.3f}, df = {n-1}, p = {p:.6f}")
    if p < 0.001:
        print(f"  ✅ p < 0.001 — HIGHLY SIGNIFICANT")
    elif p < 0.01:
        print(f"  ✅ p < 0.01 — SIGNIFICANT")
    elif p < 0.05:
        print(f"  ✅ p < 0.05 — SIGNIFICANT")
    else:
        print(f"  ❌ p > 0.05 — NOT significant")
    print()

# Accuracy tests
paired_test("hybrid - neural (accuracy)", hyb_accs, neu_accs)
paired_test("hybrid - symbolic (accuracy)", hyb_accs, sym_accs)
paired_test("neural - symbolic (accuracy)", neu_accs, sym_accs)

# Unseen tests (en kritik)
paired_test("hybrid - neural (accuracy_unseen)", hyb_unseen, neu_unseen)
paired_test("hybrid - symbolic (accuracy_unseen)", hyb_unseen, sym_unseen)
