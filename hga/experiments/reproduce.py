"""One-command CKPT-001 reproducibility bundle.

The bundle is deliberately a *defined foundation suite*, not an unbounded claim
that every historical exploratory command in the repository has been rerun. It
executes the manifest-backed research suite, English architecture controls, the
HGA scaling probe, and the theory/diagnostic grid under one seed contract.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .english_ewt import (
    english_hga_scaling_markdown,
    prepare_english_ewt_task,
    run_english_ewt_baselines,
    run_english_hga_scaling_probe,
)
from .experiment import ExperimentRun, canonical_hash, file_sha256, seed_everything
from .foundation_analysis import foundation_analysis_markdown, run_foundation_analysis
from .research import research_report_markdown, run_research_benchmark

PROTOCOL = "hga-reproduce-all-foundation-v1"


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _report_signature(
    research: Mapping[str, Any], english: Mapping[str, Any],
    scaling: Mapping[str, Any], foundation: Mapping[str, Any],
) -> str:
    """Hash result values while deliberately excluding timestamps and durations."""
    return canonical_hash({
        "research": {
            "dataset_hash": research["dataset_hash"],
            "config_hash": research["config_hash"],
            "outcome": research["outcome"],
            "sections": {
                name: {
                    "status": section["status"],
                    "score_mean": section["score_mean"],
                    "checks_all_seeds": section["checks_all_seeds"],
                }
                for name, section in research["sections"].items()
            },
        },
        "english": {
            "dataset_hash": english["dataset_hash"],
            "candidate_hashes": english["candidate_hashes"],
            "parameter_counts": english["parameter_counts"],
            "aggregate": english["aggregate"],
            "checks": english["checks"],
        },
        "scaling": {
            "dataset_hash": scaling["dataset_hash"],
            "candidate_hashes": scaling["candidate_hashes"],
            "rows": [{
                "scale": row["scale"], "physical_parameters": row["physical_parameters"],
                "f1": row["f1"], "cross_entropy": row["cross_entropy"],
            } for row in scaling["rows"]],
            "status": scaling["status"],
        },
        "foundation": {
            "checks": foundation["checks"],
            "rank_grid": foundation["rank_grid"],
            "gradient_grid": foundation["gradient_grid"],
            "sparse_memory": foundation["sparse_memory"],
        },
    })


def reproduce_all_markdown(summary: Mapping[str, Any]) -> str:
    """Render the top-level reproducibility-bundle receipt."""
    lines = [
        "# HGA CKPT-001 reproduction receipt", "",
        f"- Protocol: `{summary['protocol']}`",
        f"- Status: **{summary['status']}**",
        f"- Seeds: `{summary['seeds']}` · profile: `{summary['profile']}`",
        f"- Deterministic value fingerprint: `{summary['deterministic_value_fingerprint']}`",
        f"- Experiment: `{summary['experiment_id']}`", "",
        "## Checks", "",
    ]
    lines.extend(
        f"- {'PASS' if value else 'FAIL'} — `{name}`"
        for name, value in summary["checks"].items()
    )
    lines.extend(["", "## Files", ""])
    for name, digest in summary["artifact_sha256"].items():
        lines.append(f"- `{name}` — `{digest}`")
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {item}" for item in summary["limitations"])
    return "\n".join(lines) + "\n"


def run_reproduce_all(
    *, root: str | Path = "artifacts/reproduce-all", seeds: Sequence[int] = (1, 2, 3, 4, 5),
    profile: str = "smoke",
) -> Dict[str, Any]:
    """Create a self-contained, manifest-stamped foundation reproduction bundle.

    Args:
        root: Directory in which a new atomic ``EXP-NNNN`` bundle is created.
        seeds: Unique seeds shared by every stochastic component. At least two
            are required by the scaling probe; five are required for the full
            CKPT-001 statistical contract.
        profile: ``smoke`` for the documented CPU foundation run or ``full``
            for longer training schedules.
    """
    normalized_seeds = [int(seed) for seed in seeds]
    if profile not in {"smoke", "full"}:
        raise ValueError("profile must be 'smoke' or 'full'")
    if len(normalized_seeds) < 2 or len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("reproduce-all requires at least two unique seeds")
    # Set controls before ExperimentRun.create so the top-level manifest records
    # the active Torch/cuDNN state, not merely the requested environment vars.
    seed_everything(normalized_seeds[0])
    task = prepare_english_ewt_task()
    config = {
        "protocol": PROTOCOL,
        "profile": profile,
        "seeds": normalized_seeds,
        "components": ["research_suite", "english_ewt", "hga_scaling", "foundation_analysis"],
        "deterministic_environment": {
            "python_hash_seed_required_at_launch": True,
            "cublas_workspace_config_required_for_cuda": ":4096:8",
        },
    }
    run = ExperimentRun.create(
        root=root, config=config, seed=normalized_seeds[0], dataset_hash=task.dataset_hash,
        parameters={"seed_count": len(normalized_seeds), "profile": profile},
    )
    output = run.directory
    try:
        with run.capture_stdout():
            print(f"[{run.experiment_id}] CKPT-001 reproduction bundle starting")
            seed_everything(normalized_seeds[0])
            research = run_research_benchmark(
                root=output / "research_experiments", seeds=normalized_seeds, profile=profile
            ).to_dict()
            english = run_english_ewt_baselines(seeds=normalized_seeds, profile=profile).to_dict()
            scaling = run_english_hga_scaling_probe(seeds=normalized_seeds, profile=profile)
            foundation = run_foundation_analysis(seed=normalized_seeds[0]).to_dict()

        reports: Dict[str, tuple[Mapping[str, Any], str]] = {
            "research_report.json": (research, research_report_markdown(research)),
            "english_ewt.json": (english, _english_markdown_from_dict(english)),
            "hga_scaling.json": (scaling, english_hga_scaling_markdown(scaling)),
            "foundation_analysis.json": (foundation, foundation_analysis_markdown(foundation)),
        }
        artifact_hashes: Dict[str, str] = {}
        for filename, (data, markdown) in reports.items():
            json_path = output / filename
            markdown_path = json_path.with_suffix(".md")
            _write_json(json_path, data)
            markdown_path.write_text(markdown, encoding="utf-8")
            artifact_hashes[filename] = file_sha256(json_path)
            artifact_hashes[markdown_path.name] = file_sha256(markdown_path)
        five_seed_contract = len(normalized_seeds) >= 5
        checks = {
            "research_suite_completed": research["outcome"] == "COMPLETED",
            "english_source_and_architecture_checks": all(
                bool(value) for name, value in english["checks"].items()
                if name != "five_or_more_seeds"
            ),
            "english_five_seed_contract": five_seed_contract and bool(
                english["checks"]["five_or_more_seeds"]
            ),
            "scaling_guardrail_status": scaling["status"] == "EXPLORATORY_NOT_A_SCALING_LAW",
            "foundation_diagnostics_completed": all(bool(value) for value in foundation["checks"].values()),
            "shared_english_dataset_hash": (
                research["sections"]["english_nlp"]["metrics_reference_seed"]["english_ewt"]["dataset_hash"]
                == english["dataset_hash"] == scaling["dataset_hash"]
            ),
        }
        summary: Dict[str, Any] = {
            "protocol": PROTOCOL,
            "experiment_id": run.experiment_id,
            "status": "COMPLETED" if all(checks.values()) else "COMPLETED_WITH_LIMITATIONS",
            "profile": profile,
            "seeds": normalized_seeds,
            "dataset_hash": task.dataset_hash,
            "deterministic_value_fingerprint": _report_signature(research, english, scaling, foundation),
            "checks": checks,
            "artifact_sha256": artifact_hashes,
            "limitations": [
                "Wall-clock durations, hardware inventory, timestamps, and EXP identifiers are expected to differ across machines and are excluded from the value fingerprint.",
                "Byte-identical floating-point results across CPU/GPU architectures are not promised; compare pinned source/config hashes and seed-level values.",
                "The scaling component remains explicitly exploratory and is not evidence of a scaling law.",
                "A successful local bundle is not evidence that a public GHCR image has been published or that independent review occurred.",
            ],
        }
        receipt = output / "REPRODUCE.md"
        receipt.write_text(reproduce_all_markdown(summary), encoding="utf-8")
        artifact_hashes[receipt.name] = file_sha256(receipt)
        _write_json(output / "reproduce_summary.json", summary)
        artifact_hashes["reproduce_summary.json"] = file_sha256(output / "reproduce_summary.json")
        run.complete(summary)
        return {**summary, "directory": str(output), "manifest": run.manifest}
    except BaseException as error:
        run.fail(error)
        raise


def _english_markdown_from_dict(report: Mapping[str, Any]) -> str:
    """Render an EWT report dict without reconstructing its internal dataclass."""
    lines = [
        "# English UD EWT Architecture Baselines", "",
        f"- Protocol: `{report['protocol']}` · profile: `{report['profile']}` · seeds: `{report['seeds']}`",
        f"- Dataset hash: `{report['dataset_hash']}`", "",
        "| Model | Params | Test accuracy (mean ± std) | Test F1 (mean ± std) | FAR | FRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["aggregate"].items():
        lines.append(
            f"| {name} | {report['parameter_counts'][name]:,} | "
            f"{row['accuracy']['mean']:.4f} ± {row['accuracy']['std']:.4f} | "
            f"{row['f1']['mean']:.4f} ± {row['f1']['std']:.4f} | "
            f"{row['false_acceptance_rate']['mean']:.4f} | {row['false_rejection_rate']['mean']:.4f} |"
        )
    lines.extend(["", "## Integrity checks", ""])
    lines.extend(f"- {'PASS' if value else 'FAIL'} — `{name}`" for name, value in report["checks"].items())
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


__all__ = ["PROTOCOL", "reproduce_all_markdown", "run_reproduce_all"]
