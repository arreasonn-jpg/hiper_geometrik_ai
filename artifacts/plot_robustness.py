"""Robustness grafikleri — yayın için 3 panel."""
import json
import matplotlib.pyplot as plt

with open("artifacts/robustness_test.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: Entity
a = data["A_entity"]
ax = axes[0]
x = [d["entity_count"] for d in a]
ax.plot(x, [d["neural_acc"] for d in a], "s-", label="Neural", color="#3498db", lw=2)
ax.plot(x, [d["hybrid_acc"] for d in a], "^-", label="Hybrid", color="#2ecc71", lw=2.5)
ax.set_xlabel("Entity Count", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("A) Entity Scaling", fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend()

# Panel B: Relation
b = data["B_relation"]
ax = axes[1]
x = [d["relation_count"] for d in b]
ax.plot(x, [d["neural_acc"] for d in b], "s-", label="Neural", color="#3498db", lw=2)
ax.plot(x, [d["hybrid_acc"] for d in b], "^-", label="Hybrid", color="#2ecc71", lw=2.5)
ax.set_xlabel("Relation Count", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("B) Relation Scaling", fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend()

# Panel C: Unseen
c = data["C_unseen"]
ax = axes[2]
x = [d["unseen_ratio"] for d in c]
ax.plot(x, [d["neural_unseen"] for d in c], "s-", label="Neural (unseen)", color="#3498db", lw=2)
ax.plot(x, [d["hybrid_unseen"] for d in c], "^-", label="Hybrid (unseen)", color="#2ecc71", lw=2.5)
ax.set_xlabel("Unseen Entity Ratio", fontsize=12)
ax.set_ylabel("Accuracy (unseen)", fontsize=12)
ax.set_title("C) Cold-Start", fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("artifacts/robustness_plot.png", dpi=150)
print("✅ Saved: artifacts/robustness_plot.png")
