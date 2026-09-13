"""Multi-environment kapalı self-learning bilimsel sözleşmesi."""
from hga.experience import run_multi_environment_self_learning


def test_uc_environment_shared_store_ve_dynamic_memoryde_izole_buyur():
    report = run_multi_environment_self_learning(
        cycles=5, batch_per_environment=8, seed=1, memory_capacity=256
    )

    assert report.protocol == "multi-environment-closed-verified-self-learning-v1"
    assert report.environments == ["arithmetic", "logic", "consistency"]
    assert report.shared_store_initial_facts == 9
    assert report.shared_store_final_facts == 62
    assert report.shared_memory_policy == "DYNAMIC_KV"
    assert report.shared_memory_records == 53
    assert report.shared_memory_retrieval_accuracy == 1.0
    assert report.cross_verifier_attempts == report.cross_verifier_uncertain == 6
    assert report.cross_verifier_acceptances == 0
    assert report.cross_environment_fact_contamination == 0
    assert all(report.checks.values())


def test_environment_metrikleri_far_frr_holdout_ve_growthu_ayri_tutar():
    report = run_multi_environment_self_learning(
        cycles=5, batch_per_environment=8, seed=1, memory_capacity=256
    )
    expected = {
        "arithmetic": (13, 27, 27, 16),
        "logic": (20, 20, 20, 23),
        "consistency": (20, 20, 0, 23),
    }
    for name, metric in report.per_environment.items():
        verified, rejected, false_before, final = expected[name]
        assert metric.generated == 40
        assert metric.verified == verified
        assert metric.rejected == rejected
        assert metric.false_acceptance_before_verifier == false_before
        assert metric.false_acceptance == metric.false_rejection == 0
        assert metric.incorrect_durable_knowledge == 0
        assert metric.final_knowledge == final
        assert metric.durable_new_knowledge == metric.verified
        assert metric.generation_holdout_overlap == 0
        assert metric.memory_holdout_overlap == 0
        assert metric.knowledge_holdout_overlap == 0
        assert metric.memory_retrieval_accuracy == 1.0


def test_multi_environment_seed_deterministik_ama_schedule_seed_duyarli():
    first = run_multi_environment_self_learning(
        cycles=4, batch_per_environment=8, seed=3
    )
    second = run_multi_environment_self_learning(
        cycles=4, batch_per_environment=8, seed=3
    )
    other = run_multi_environment_self_learning(
        cycles=4, batch_per_environment=8, seed=4
    )

    assert first.to_dict() == second.to_dict()
    assert first.schedule_hash != other.schedule_hash
    assert all(other.checks.values())


def test_multi_environment_capacity_yetersizse_eviction_basarisi_uydurulmaz():
    # Verified kayıt sayısından küçük shared memory exact-retrieval kapısını
    # geçemez; protokol sessizce başarı raporlamamalıdır.
    try:
        run_multi_environment_self_learning(
            cycles=5, batch_per_environment=8, seed=1, memory_capacity=8
        )
    except ValueError as error:
        assert "shared_memory_exact_retrieval" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Yetersiz memory capacity kabul kapısını düşürmeliydi")


def test_multi_environment_markdown_uc_ortami_ayri_gosterir():
    report = run_multi_environment_self_learning(
        cycles=3, batch_per_environment=8, seed=2
    )
    markdown = report.markdown()
    assert "arithmetic" in markdown
    assert "logic" in markdown
    assert "consistency" in markdown
    assert "Holdout leak" in markdown
