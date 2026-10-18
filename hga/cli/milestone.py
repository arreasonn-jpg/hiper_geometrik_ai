"""HGA CLI komutu (Faz 2 refactor)."""
from __future__ import annotations

import json
import os


def _milestone(cycles, batch, initial_facts, operands_max, negatives_per_fact,
               memory_slots, seeds, experiment_root, out=None, markdown=None,
               ledger=None, checkpoints=None):
    """K₀→Kₙ milestone tablosu: versiyonlanmış + defterli kapalı döngü."""
    from hga.evaluation import canonical_hash, run_seed_sweep
    from hga.experience import run_milestone_experiment

    seed_values = [int(v.strip()) for v in seeds.split(",") if v.strip()]
    if checkpoints:
        nokta_listesi = [int(v.strip()) for v in checkpoints.split(",") if v.strip()]
    else:
        nokta_listesi = [c for c in (0, 10, 50, 100) if c <= int(cycles)]
    config = {
        "benchmark": "milestone-versioned-ledgered-closed-loop-v1",
        "cycles": int(cycles), "batch_size": int(batch),
        "initial_facts": int(initial_facts), "operands_max": int(operands_max),
        "negatives_per_fact": int(negatives_per_fact),
        "memory_slots": int(memory_slots), "checkpoints": nokta_listesi,
    }
    dataset_hash = canonical_hash({
        "generator": "arithmetic-frontier-holdout-v2",
        "operands_max": int(operands_max),
        "negatives_per_fact": int(negatives_per_fact),
    })

    toplanan = []

    def run_one(seed):
        rapor = run_milestone_experiment(
            cycles=int(cycles), batch_size=int(batch),
            initial_facts=int(initial_facts), operands_max=int(operands_max),
            negatives_per_fact=int(negatives_per_fact), seed=seed,
            memory_slots=int(memory_slots), checkpoints=nokta_listesi,
            ledger_path=ledger,
        )
        print(rapor.markdown())
        print(f"knowledge_chain_valid={rapor.knowledge_chain_valid} "
              f"ledger_chain_valid={rapor.ledger_chain_valid} "
              f"final={rapor.final_knowledge_version} "
              f"ledger_entries={rapor.ledger_total_entries}")
        if rapor.rollback_drill:
            d = rapor.rollback_drill
            print(f"rollback: {d['healthy_version']} → {d['corrupted_version']} "
                  f"(yanlış={d['incorrect_facts_when_corrupted']}) → "
                  f"{d['restored_version']} (yanlış={d['incorrect_facts_after_rollback']}) "
                  f"geçmiş_korundu={d['history_preserved']}")
        print(f"C_E={rapor.capacity['c_e_total']} C_V={rapor.capacity['c_v_total']} "
              f"C_V/C_E={rapor.capacity['c_v_over_c_e']}")
        toplanan.append(rapor)
        return rapor.to_dict()

    report = run_seed_sweep(
        run_one, seeds=seed_values, root=experiment_root, config=config,
        dataset_hash=dataset_hash,
        parameters={
            "ground_truth": "independent-arithmetic-environment",
            "closed_loop_writes_only_verified": True,
            "knowledge_versioning_enabled": True,
            "immutable_ledger_enabled": True,
            "rollback_drill_is_failure_injection": True,
            "neural_training_used": False,
        },
    ).to_dict()
    print("Milestone çoklu-seed özeti:")
    print(f"  experiments: {', '.join(report['experiment_ids'])}")
    for key in ("ledger_total_entries", "capacity.c_e_total", "capacity.c_v_total",
                "capacity.c_v_over_c_e"):
        values = report["aggregate"].get(key)
        if values:
            print(f"  {key:<28}: {values['mean']} ± {values['std']}")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"  report: {out}")
    if markdown:
        from hga.experience.milestone import milestone_markdown
        os.makedirs(os.path.dirname(os.path.abspath(markdown)) or ".", exist_ok=True)
        with open(markdown, "w", encoding="utf-8") as handle:
            handle.write(milestone_markdown(toplanan, report))
        print(f"  markdown: {markdown}")
