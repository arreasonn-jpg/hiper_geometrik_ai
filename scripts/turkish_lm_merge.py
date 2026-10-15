"""4 batch JSON'unu tek 20-seed rapora birleştir (özetleri yeniden hesapla)."""
import json
import sys
from pathlib import Path


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _recompute_summary(birlesik):
    """Per-arm per_seed listelerinden summary alanlarını yeniden üret."""
    arms = birlesik["arms"]
    # Per-arm ortalama PPL
    ppl_by_arm = {
        arm: _mean([k["test"]["perplexity"] for k in d["per_seed"]])
        for arm, d in arms.items()
    }
    # bigram / unigram arms yoksa base'den koru
    base_perp = birlesik.get("perplexity", {})
    best_arm = min(ppl_by_arm, key=ppl_by_arm.get)
    yeni = dict(base_perp)
    yeni["best_arm"] = best_arm
    yeni["best_neural_test_ppl_mean"] = ppl_by_arm[best_arm]
    for arm in ("hga", "dense", "transformer"):
        if arm in ppl_by_arm:
            yeni[f"{arm}_test_ppl_mean"] = round(ppl_by_arm[arm], 8)
    # top1 majority (base'den)
    birlesik["perplexity"] = yeni

    # checks.human_ratings_collected varsa dokunma; multi_seed kapısı
    if "checks" in birlesik:
        birlesik["checks"]["multi_seed_reported"] = len(birlesik["seeds"]) >= 2

    return birlesik


def merge(batch_paths, out_path):
    batches = [json.loads(Path(p).read_text(encoding="utf-8")) for p in batch_paths]
    base = batches[0]
    tum_seedler = []
    for b in batches:
        tum_seedler.extend(b.get("seeds", []))

    birlesik = dict(base)
    birlesik["seeds"] = sorted(set(tum_seedler))
    birlesik["n_seeds"] = len(birlesik["seeds"])
    birlesik["kaynak_batchlar"] = [str(p) for p in batch_paths]

    birlesik["arms"] = {}
    for arm in base["arms"]:
        per_seed = []
        for b in batches:
            if arm in b.get("arms", {}):
                per_seed.extend(b["arms"][arm].get("per_seed", []))
        birlesik["arms"][arm] = {"per_seed": per_seed}

    birlesik = _recompute_summary(birlesik)
    Path(out_path).write_text(
        json.dumps(birlesik, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK: {len(birlesik['seeds'])} seed -> {out_path}")


if __name__ == "__main__":
    merge(sys.argv[1:-1], sys.argv[-1])
