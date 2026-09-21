"""Ablasyon grafiği — yayın için."""
import json
import matplotlib.pyplot as plt

with open("artifacts/ablation_hidden_ratio.json") as f:
    data = json.load(f)

ratios = [d["hidden_ratio"] for d in data]
sym = [d["symbolic_acc"] for d in data]
neu = [d["neural_acc"] for d in data]
hyb = [d["hybrid_acc"] for d in data]

plt.figure(figsize=(10, 6))
plt.plot(ratios, sym, "o-", label="Symbolic", color="#e74c3c", linewidth=2, markersize=10)
plt.plot(ratios, neu, "s-", label="Neural", color="#3498db", linewidth=2, markersize=10)
plt.plot(ratios, hyb, "^-", label="Hybrid", color="#2ecc71", linewidth=2.5, markersize=12)

# Shade the "gain" area
plt.fill_between(ratios, neu, hyb, alpha=0.2, color="#2ecc71", label="Hybrid gain")

plt.xlabel("Hidden Feature Ratio", fontsize=13)
plt.ylabel("Accuracy", fontsize=13)
plt.title("Hybrid Mechanism: Monotonic Gain over Neural Baseline", fontsize=14)
plt.legend(fontsize=12, loc="lower left")
plt.grid(True, alpha=0.3)
plt.ylim(0.0, 1.05)
plt.xticks(ratios)

plt.tight_layout()
plt.savefig("artifacts/ablation_hidden_ratio.png", dpi=150)
print("✅ Saved: artifacts/ablation_hidden_ratio.png")
