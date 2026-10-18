"""HGA CLI komutu (Faz 2 refactor)."""
from __future__ import annotations

import json
import os

from hga.cli.common import _yaz_rapor


def _ogrenme_olcek(profile="smoke", seeds=None, out=None, markdown=None):
    """P1: Self-learning ölçeklendirme (100 → 1000 → 3000 cycle)."""
    from hga.experiments.self_learning_scaling import (
        run_self_learning_scaling,
        self_learning_scaling_markdown,
    )
    tohumlar = [int(v) for v in (seeds or "1").split(",") if v.strip()]
    rapor = run_self_learning_scaling(profile=profile, seeds=tohumlar)
    md = self_learning_scaling_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)

def _ogrenme_transfer(seeds=None, source_examples=120,
                       target_support_examples=40,
                       target_unseen_examples=40,
                       out=None, markdown=None):
    """P1: Cross-domain self-learning transfer negatif-kontrol benchmarkı."""
    from hga.experiments.self_learning_transfer import (
        run_self_learning_transfer_benchmark,
        self_learning_transfer_markdown,
    )
    tohumlar = [int(v) for v in (seeds or "1,2,3").split(",") if v.strip()]
    rapor = run_self_learning_transfer_benchmark(
        seeds=tohumlar,
        source_examples=int(source_examples),
        target_support_examples=int(target_support_examples),
        target_unseen_examples=int(target_unseen_examples),
    )
    md = self_learning_transfer_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)

def _self_learning_benchmark(cycles, batch, initial_facts, operands_max,
                             negatives_per_fact, memory_slots, seeds,
                             experiment_root, out=None):
    from hga.experience import (
        run_multi_environment_self_learning,
        run_self_learning_experiment,
        run_self_training_collapse_test,
        run_verifier_fault_injection,
    )
    from hga.experiments import canonical_hash, run_seed_sweep
    from hga.memory import run_memory_capacity_sweep

    seed_values = [int(value.strip()) for value in seeds.split(",") if value.strip()]
    config = {
        "benchmark": "self-learning-collapse-robustness-v3",
        "cycles": int(cycles), "batch_size": int(batch),
        "initial_facts": int(initial_facts), "operands_max": int(operands_max),
        "negatives_per_fact": int(negatives_per_fact),
        "memory_slots": int(memory_slots),
        "verifier_fault_rates": {"false_acceptance": 0.25, "false_rejection": 0.25},
        "memory_recall_target": 0.95,
        "multi_environment": {
            "environments": ["arithmetic", "logic", "consistency"],
            "cycles": min(int(cycles), 10),
            "batch_per_environment": min(int(batch), 8),
            "memory_capacity": max(int(memory_slots), 256),
        },
    }
    dataset_hash = canonical_hash({
        "generator": "arithmetic-frontier-holdout-v2",
        "operands_max": int(operands_max),
        "negatives_per_fact": int(negatives_per_fact),
        "test_holdout_fraction": 0.10,
        "multi_environment_generators": [
            "arithmetic-v1", "modus-ponens-v1", "property-consistency-v1"
        ],
    })

    def run_one(seed):
        expansion = run_self_learning_experiment(
            cycles=cycles, batch_size=batch, initial_facts=initial_facts,
            operands_max=operands_max, negatives_per_fact=negatives_per_fact,
            seed=seed, memory_slots=memory_slots,
        )
        multi = run_multi_environment_self_learning(
            cycles=min(int(cycles), 10),
            batch_per_environment=min(int(batch), 8),
            seed=seed,
            memory_capacity=max(int(memory_slots), 256),
        )
        collapse = run_self_training_collapse_test(
            cycles=max(2, cycles), batch_size=batch,
            initial_facts=min(initial_facts, 10), operands_max=min(operands_max, 15),
            negatives_per_fact=negatives_per_fact, seed=seed,
            memory_slots=min(memory_slots, 4096),
        )
        robustness = run_verifier_fault_injection(seed=seed)
        base_slots = max(1, int(memory_slots))
        slot_counts = sorted({
            max(1, base_slots // 16), max(1, base_slots // 8),
            max(1, base_slots // 4), max(1, base_slots // 2),
            base_slots, base_slots * 2, base_slots * 4, base_slots * 8,
        })
        capacity = run_memory_capacity_sweep(
            context_count=max(256, min(int(cycles) * int(batch), 10_000)),
            slot_counts=slot_counts, table_counts=(1, 2),
            recall_target=0.95, seed=seed,
        )
        print(
            f"closed: K0={expansion.initial_knowledge_size} "
            f"K{cycles}={expansion.final_knowledge_size} "
            f"yield={expansion.experience_yield:.6f} "
            f"FAR={expansion.far_before_verifier:.6f}→{expansion.far:.6f}"
        )
        print(
            f"multi-env: K0={multi.shared_store_initial_facts} "
            f"Kn={multi.shared_store_final_facts} "
            f"memory_recall={multi.shared_memory_retrieval_accuracy:.6f} "
            f"cross_accept={multi.cross_verifier_acceptances}"
        )
        print(
            f"collapse: detected={collapse.collapse_detected} "
            f"repetition={collapse.repetition_rate:.6f} "
            f"contamination={collapse.incorrect_model_facts} FAR={collapse.far:.6f}"
        )
        print(
            f"verifier-fault: precision={robustness.precision:.6f} "
            f"recall={robustness.recall:.6f} FAR={robustness.far:.6f} "
            f"FRR={robustness.frr:.6f} uncertain={robustness.uncertain} "
            f"conflict={robustness.conflict}"
        )
        print(
            "memory-threshold: "
            f"{capacity.minimum_slots_meeting_target}"
        )
        return {
            "closed_verified": expansion.to_dict(),
            "multi_environment": multi.to_dict(),
            "collapse_probe": collapse.to_dict(),
            "verifier_robustness": robustness.to_dict(),
            "memory_capacity": capacity.to_dict(),
        }

    report = run_seed_sweep(
        run_one, seeds=seed_values, root=experiment_root, config=config,
        dataset_hash=dataset_hash,
        parameters={
            "ground_truth": "independent-arithmetic-logic-consistency-environments",
            "environment_count": 3,
            "shared_active_memory": "DYNAMIC_KV",
            "closed_loop_writes_only_verified": True,
            "collapse_is_failure_injection": True,
            "verifier_robustness_is_failure_injection": True,
            "generation_memory_test_isolation_required": True,
            "neural_training_used": False,
        },
    ).to_dict()
    print("Self-learning + collapse benchmark özeti:")
    print(f"  experiments: {', '.join(report['experiment_ids'])}")
    for key in (
        "closed_verified.experience_yield", "closed_verified.far_before_verifier",
        "closed_verified.far", "closed_verified.frr", "closed_verified.final_knowledge_size",
        "closed_verified.incorrect_knowledge",
        "multi_environment.shared_store_final_facts",
        "multi_environment.shared_memory_retrieval_accuracy",
        "multi_environment.cross_verifier_acceptances",
        "collapse_probe.repetition_rate", "collapse_probe.far",
        "collapse_probe.incorrect_model_facts", "collapse_probe.memory_collisions",
        "verifier_robustness.precision", "verifier_robustness.recall",
        "verifier_robustness.f1", "verifier_robustness.far", "verifier_robustness.frr",
        "verifier_robustness.uncertain", "verifier_robustness.conflict",
    ):
        values = report["aggregate"].get(key)
        if values:
            print(f"  {key:<48}: {values['mean']:.6f} ± {values['std']:.6f}")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"  report: {out}")
