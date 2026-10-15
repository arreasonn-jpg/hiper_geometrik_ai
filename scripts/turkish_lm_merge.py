"""4 batch JSON'unu tek 20-seed rapora birleştir."""
import json
import sys
from pathlib import Path

def merge(batch_paths, out_path):
    batches = [json.loads(Path(p).read_text(encoding="utf-8")) for p in batch_paths]
    base = batches[0]
    # Tüm seed ve per_seed kayıtlarını birleştir
    tum_seedler = []
    for b in batches:
        tum_seedler.extend(b.get("seeds", []))

    birlesik = dict(base)
    birlesik["seeds"] = sorted(set(tum_seedler))
    birlesik["n_seeds"] = len(birlesik["seeds"])
    birlesik["kaynak_batchlar"] = [str(p) for p in batch_paths]

    # Kol bazında per_seed birleştir
    birlesik["arms"] = {}
    for arm in base["arms"]:
        per_seed = []
        for b in batches:
            if arm in b.get("arms", {}):
                per_seed.extend(b["arms"][arm].get("per_seed", []))
        birlesik["arms"][arm] = {"per_seed": per_seed}

    Path(out_path).write_text(
        json.dumps(birlesik, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK: {len(birlesik['seeds'])} seed -> {out_path}")

if __name__ == "__main__":
    merge(sys.argv[1:-1], sys.argv[-1])
