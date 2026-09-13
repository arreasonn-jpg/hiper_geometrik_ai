"""Scientific STALE and real-artifact knowledge lifecycle tests."""
import hashlib

import pytest

from hga.evaluation.lifecycle import run_lifecycle_benchmark
from hga.knowledge.lifecycle import KnowledgeLifecycle, KnowledgeLifecycleState


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _ingest(lifecycle, record_id="R1", source_url="https://example.test/source", **kwargs):
    return lifecycle.ingest(
        record_id=record_id, subject_id="S", relation_id="R", object_id="O",
        source_url=source_url, document_hash=_hash(record_id), source_revision=record_id,
        retrieved_at=0.0, max_age_seconds=10.0, **kwargs,
    )


def test_stale_false_retracted_ve_superseded_birbirinden_ayridir():
    lifecycle = KnowledgeLifecycle()
    first = _ingest(lifecycle)
    lifecycle.audit(at=11.0)
    assert first.state == KnowledgeLifecycleState.STALE
    assert first.retracted_reason is None
    assert lifecycle.query() == []
    assert lifecycle.query(include_stale=True) == [first]

    lifecycle.revalidate("R1", at=12.0, document_hash=first.document_hash)
    second = lifecycle.ingest(
        record_id="R2", subject_id="S", relation_id="R", object_id="O2",
        source_url=first.source_url, document_hash=_hash("changed"), source_revision="r2",
        retrieved_at=13.0, max_age_seconds=10.0,
    )
    assert first.state == KnowledgeLifecycleState.SUPERSEDED
    assert second.supersedes == first.record_id
    lifecycle.retract(second.record_id, at=14.0, reason="withdrawn")
    assert second.state == KnowledgeLifecycleState.RETRACTED
    assert lifecycle.verify_event_chain()


def test_dependency_staleness_transitive_yayilir():
    lifecycle = KnowledgeLifecycle()
    parent = _ingest(lifecycle, "P")
    child = _ingest(lifecycle, "C", "derived://c", derived_from=(parent.record_id,))
    grandchild = _ingest(lifecycle, "G", "derived://g", derived_from=(child.record_id,))
    lifecycle.unavailable(parent.record_id, at=1.0, reason="timeout")
    assert parent.state == child.state == grandchild.state == KnowledgeLifecycleState.STALE
    assert "nonactive_dependencies" in child.stale_reason


def test_changed_hash_revalidation_ve_izlenemez_kayit_reddedilir():
    lifecycle = KnowledgeLifecycle()
    record = _ingest(lifecycle)
    with pytest.raises(ValueError):
        lifecycle.revalidate(record.record_id, at=1.0, document_hash=_hash("new"))
    with pytest.raises(ValueError):
        lifecycle.ingest(
            record_id="bad", subject_id="S", relation_id="R", object_id="O",
            source_url="", document_hash="bad", source_revision="x",
            retrieved_at=0.0, max_age_seconds=0.0,
        )


def test_real_twt_artifact_lifecycle_tum_kapilari_gecer():
    report = run_lifecycle_benchmark()
    assert report.protocol == "real-artifact-knowledge-lifecycle-v1"
    assert report.real_source["license"] == "Apache-2.0"
    assert len(report.real_source["upstream_revision"]) == 40
    assert all(report.checks.values())
    assert report.event_count >= 10
    assert len(report.event_chain_head) == 64
    assert {checkpoint["name"] for checkpoint in report.checkpoints} == {
        "ingested", "freshness_expired", "same_hash_revalidated",
        "controlled_revision_change", "outage_and_retraction",
    }
