"""Ablasyon: hidden_feature_ratio değişince ne olur?"""
import sys
sys.path.insert(0, '.')

from hga.evaluation.paradigma import run_paradigm_ablation
import json

ratios = [0.0, 0.15, 0.30, 0.50, 0.70]
results = []

for ratio in ratios:
    # Her ratio için 3 seed
    seed_results = []
    for seed in [1, 2, 3]:
        r = run_paradigm_ablation(
            hidden_feature_ratio=ratio,
            seed=seed,
            epochs=60,
        )
        seed_results.append({
            "seed": seed,
            "symbolic_acc": r.arms["symbolic"]["accuracy"],
            "symbolic_cov": r.arms["symbolic"]["coverage"],
            "neural_acc": r.arms["neural"]["accuracy"],
            "neural_unseen": r.arms["neural"]["accuracy_unseen"],
            "hybrid_acc": r.arms["hybrid"]["accuracy"],
            "hybrid_unseen": r.arms["hybrid"]["accuracy_unseen"],
        })
    
    # Ortalamalar
    import statistics as st
    avg = {
        "hidden_ratio": ratio,
        "symbolic_acc": st.mean([s["symbolic_acc"] for s in seed_results]),
        "symbolic_cov": st.mean([s["symbolic_cov"] for s in seed_results]),
        "neural_acc": st.mean([s["neural_acc"] for s in seed_results]),
        "neural_unseen": st.mean([s["neural_unseen"] for s in seed_results]),
        "hybrid_acc": st.mean([s["hybrid_acc"] for s in seed_results]),
        "hybrid_unseen": st.mean([s["hybrid_unseen"] for s in seed_results]),
    }
    results.append(avg)
    print(f"hidden_ratio={ratio:.2f}: "
          f"sym_acc={avg['symbolic_acc']:.4f} (cov={avg['symbolic_cov']:.3f}) | "
          f"neu_acc={avg['neural_acc']:.4f} | "
          f"hyb_acc={avg['hybrid_acc']:.4f}")

# JSON kaydet
with open("artifacts/ablation_hidden_ratio.json", "w") as f:
    json.dump(results, f, indent=2)

print()
print("=" * 70)
print("YORUM")
print("=" * 70)
print()
print("Beklenti: hidden_ratio ↑ → symbolic coverage ↓ → hybrid % neural'a yaklaşır")
print("Eğer hidden_ratio=0.7'de hybrid hala neural'ı geçiyorsa → veto mekanizması güçlü")
