import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
agg = report["aggregate"]

print("="*60)
print("30-SEED ANALİZ")
print("="*60)
print()

# F1 paired diffs
flat_f1 = [r["f1"] for r in agg["flat_hga"]["raw"]]
rich_f1 = [agg["hiyerarsik_rich"]["raw"][i]["f1"] for i in range(len(flat_f1))]
eq_f1 = [agg["hiyerarsik_eq"]["raw"][i]["f1"] for i in range(len(flat_f1))]

import statistics as st
diffs = [r - f for r, f in zip(rich_f1, flat_f1)]
mean_d = st.mean(diffs)
std_d = st.stdev(diffs)
n = len(diffs)

print(f"Flat F1:  {st.mean(flat_f1):.4f} ± {st.stdev(flat_f1):.4f}")
print(f"Rich F1:  {st.mean(rich_f1):.4f} ± {st.stdev(rich_f1):.4f}")
print(f"Eq F1:    {st.mean(eq_f1):.4f} ± {st.stdev(eq_f1):.4f}")
print()
print(f"Paired (rich - flat):")
print(f"  mean Δ = {mean_d:+.5f}")
print(f"  std Δ  = {std_d:.5f}")
print(f"  Cohen's d = {mean_d/std_d:.3f}")
print()

# t-testi
t = mean_d / (std_d / (n ** 0.5))
from scipy import stats
p = 2 * (1 - stats.t.cdf(abs(t), df=n-1))
print(f"  t-stat = {t:.3f}")
print(f"  p-değeri = {p:.4f}")
print()

if p < 0.05:
    print("🎉 ANLAMLI: p < 0.05")
    print("   → Pozitif sonuç yayına hazır!")
elif p < 0.10:
    print("⚠️  MARJINAL: 0.05 < p < 0.10")
    print("   → Daha fazla seed veya daha büyük etki gerekli")
else:
    print("❌ ANLAMSIZ: p > 0.10")
    print("   → Dürüst negatif sonuç yayınlanır")
