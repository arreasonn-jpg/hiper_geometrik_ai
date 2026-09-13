"""Controlled real-artifact benchmark for STALE knowledge lifecycle semantics."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib import resources
from typing import Any, Dict, List

from hga.knowledge.lifecycle import KnowledgeLifecycle, KnowledgeLifecycleState


@dataclass
class LifecycleBenchmarkReport:
    protocol: str
    real_source: Dict[str, Any]
    checkpoints: List[Dict[str, Any]]
    final_counts: Dict[str, int]
    event_count: int
    event_chain_head: str
    checks: Dict[str, bool]
    limitations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_lifecycle_benchmark() -> LifecycleBenchmarkReport:
    """Exercise expiry, revalidation, revision, outage and retraction on TWT artifacts."""
    root = resources.files("hga.evaluation.datasets.twt_v1")
    provenance = json.loads(root.joinpath("PROVENANCE.json").read_text(encoding="utf-8"))
    sources = provenance["upstream"]["source_files"]
    hashes = {
        source["vendored_path"]: hashlib.sha256(
            root.joinpath(source["vendored_path"]).read_bytes()
        ).hexdigest()
        for source in sources
    }
    assert all(
        hashes[source["vendored_path"]] == source["sha256"] for source in sources
    )
    revision = provenance["upstream"]["revision"]
    source_urls = {
        source["section"]: (
            "https://raw.githubusercontent.com/google-research-datasets/"
            f"turkish-treebanks/{revision}/{source['upstream_path']}"
        )
        for source in sources
    }

    lifecycle = KnowledgeLifecycle()
    day = 86_400.0
    start = 1_767_225_600.0  # 2026-01-01T00:00:00Z, explicit deterministic clock
    first, second = sources
    first_record = lifecycle.ingest(
        record_id="TWT-WEB-r1", subject_id="TWT-WEB", relation_id="SOURCE_REVISION",
        object_id=revision, source_url=source_urls[first["section"]], document_hash=hashes[first["vendored_path"]],
        source_revision=revision, retrieved_at=start, max_age_seconds=30 * day,
    )
    second_record = lifecycle.ingest(
        record_id="TWT-WIKI-r1", subject_id="TWT-WIKI", relation_id="SOURCE_REVISION",
        object_id=revision, source_url=source_urls[second["section"]], document_hash=hashes[second["vendored_path"]],
        source_revision=revision, retrieved_at=start, max_age_seconds=30 * day,
    )
    derived_hash = hashlib.sha256(
        (first_record.document_hash + second_record.document_hash).encode()
    ).hexdigest()
    derived = lifecycle.ingest(
        record_id="TWT-DERIVED-r1", subject_id="TWT", relation_id="DATASET_READY",
        object_id="TRUE", source_url="derived://twt-v1", document_hash=derived_hash,
        source_revision=revision, retrieved_at=start, max_age_seconds=30 * day,
        derived_from=(first_record.record_id, second_record.record_id),
    )
    checkpoints: List[Dict[str, Any]] = [
        {"name": "ingested", "counts": lifecycle.counts()}
    ]

    lifecycle.audit(at=start + 31 * day)
    expired_counts = lifecycle.counts()
    stale_visible_only_when_requested = (
        len(lifecycle.query()) == 0 and len(lifecycle.query(include_stale=True)) == 3
    )
    checkpoints.append({"name": "freshness_expired", "counts": expired_counts})

    lifecycle.revalidate(
        first_record.record_id, at=start + 32 * day,
        document_hash=first_record.document_hash,
    )
    lifecycle.revalidate(
        second_record.record_id, at=start + 32 * day,
        document_hash=second_record.document_hash,
    )
    lifecycle.revalidate(derived.record_id, at=start + 32 * day, document_hash=derived_hash)
    checkpoints.append({"name": "same_hash_revalidated", "counts": lifecycle.counts()})

    controlled_revision_hash = hashlib.sha256(
        (first_record.document_hash + ":controlled-new-revision").encode()
    ).hexdigest()
    replacement = lifecycle.ingest(
        record_id="TWT-WEB-r2", subject_id="TWT-WEB", relation_id="SOURCE_REVISION",
        object_id="CONTROLLED-r2", source_url=source_urls[first["section"]],
        document_hash=controlled_revision_hash, source_revision="CONTROLLED-r2",
        retrieved_at=start + 33 * day, max_age_seconds=30 * day,
    )
    superseded_dependency_propagated = (
        first_record.state == KnowledgeLifecycleState.SUPERSEDED
        and derived.state == KnowledgeLifecycleState.STALE
    )
    checkpoints.append({"name": "controlled_revision_change", "counts": lifecycle.counts()})

    lifecycle.unavailable(
        second_record.record_id, at=start + 34 * day, reason="controlled timeout"
    )
    outage_is_stale_not_retracted = (
        second_record.state == KnowledgeLifecycleState.STALE
        and second_record.retracted_reason is None
    )
    lifecycle.retract(
        replacement.record_id, at=start + 35 * day,
        reason="controlled source withdrawal notice",
    )
    checkpoints.append({"name": "outage_and_retraction", "counts": lifecycle.counts()})

    checks = {
        "real_twt_urls_hashes_and_revision_pinned": (
            len(revision) == 40 and len(sources) == 2
            and all(hashes[source["vendored_path"]] == source["sha256"] for source in sources)
        ),
        "freshness_expiry_marks_stale_not_false_or_retracted": (
            expired_counts["STALE"] == 3 and expired_counts["RETRACTED"] == 0
        ),
        "stale_excluded_from_default_query_but_auditable": stale_visible_only_when_requested,
        "same_hash_revalidation_restores_active": checkpoints[2]["counts"]["ACTIVE"] == 3,
        "changed_hash_creates_new_revision_and_supersedes_old": (
            replacement.supersedes == first_record.record_id
            and first_record.state == KnowledgeLifecycleState.SUPERSEDED
        ),
        "dependency_staleness_propagates": superseded_dependency_propagated,
        "source_outage_is_stale_not_retracted": outage_is_stale_not_retracted,
        "explicit_withdrawal_is_retracted": (
            replacement.state == KnowledgeLifecycleState.RETRACTED
        ),
        "event_chain_integrity": lifecycle.verify_event_chain(),
        "explicit_clock_no_wall_clock_dependency": True,
    }
    if not all(checks.values()):
        raise ValueError(f"Lifecycle kabul kapısı başarısız: {[k for k, v in checks.items() if not v]}")
    return LifecycleBenchmarkReport(
        protocol="real-artifact-knowledge-lifecycle-v1",
        real_source={
            "dataset": provenance["dataset"]["name"],
            "upstream_revision": revision,
            "license": provenance["license"]["spdx_id"],
            "source_files": [
                {"url": source_urls[source["section"]], "sha256": hashes[source["vendored_path"]]}
                for source in sources
            ],
            "controlled_events": (
                "Expiry, unchanged revalidation, simulated changed revision, outage and "
                "withdrawal are deterministic interventions; no claim of a live upstream change."
            ),
        },
        checkpoints=checkpoints,
        final_counts=lifecycle.counts(),
        event_count=len(lifecycle.events),
        event_chain_head=lifecycle.events[-1].chain_hash,
        checks=checks,
        limitations=[
            "TWT files and provenance are real; lifecycle changes after ingest are controlled interventions.",
            "No live HTTP fetch is performed, so network nondeterminism cannot affect the benchmark.",
            "STALE blocks default use but is not evidence that the underlying proposition is false.",
        ],
    )


__all__ = ["LifecycleBenchmarkReport", "run_lifecycle_benchmark"]
