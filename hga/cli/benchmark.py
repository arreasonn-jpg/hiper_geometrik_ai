"""HGA CLI komutu (Faz 2 refactor)."""
from __future__ import annotations


def _championship(profile="smoke", seeds=None, out=None, markdown=None,
                  skip_torch=False):
    """Tüm P0 protokollerini koş ve OTOMATİK araştırma karnesi üret."""
    from hga.evaluation.capability_vector import (
        build_scorecard,
        save_scorecard,
        scorecard_markdown,
    )
    from hga.evaluation.capacity import run_capacity_benchmark
    from hga.evaluation.compositional_v2 import run_compositional_v2_benchmark
    from hga.evaluation.depth_diagnosis import diagnose_depth_collapse
    from hga.evaluation.engineering_audit import (
        audit_engineering,
        audit_reproducibility,
    )
    from hga.evaluation.memory_hierarchy import (
        run_memory_hierarchy_benchmark,
    )
    from hga.evaluation.multi_environment import (
        run_multi_environment_benchmark,
    )
    from hga.evaluation.priority_ablation import run_priority_weight_ablation
    from hga.evaluation.priority_optimization import optimize_priority_weights
    from hga.evaluation.reasoning_depth import measure_reasoning_depth
    from hga.evaluation.seed_statistics import run_core_seed_statistics
    from hga.evaluation.self_learning_scaling import run_self_learning_scaling
    from hga.evaluation.semantic_extraction import (
        run_semantic_extraction_v2_benchmark,
    )
    from hga.evaluation.verifier_ensemble import run_verifier_ensemble_benchmark

    tohumlar = [int(v) for v in (seeds or "1,2,3").split(",") if v.strip()]
    derinlik_profili = "smoke" if profile == "smoke" else "standard"

    print("HGA CHAMPIONSHIP BENCHMARK — tüm P0 protokolleri koşuluyor…\n")
    raporlar = {}
    raporlar["priority_ablation"] = run_priority_weight_ablation(
        k=10, seeds=tohumlar).to_dict()
    print("  ✓ Priority(E) nedensel zincir ablasyonu")
    raporlar["reasoning_depth"] = measure_reasoning_depth(
        profile=derinlik_profili).to_dict()
    print("  ✓ C_R / C_RD çıkarım derinliği")
    raporlar["semantic_extraction"] = run_semantic_extraction_v2_benchmark().to_dict()
    print("  ✓ Türkçe semantik çıkarım v2 gold")
    raporlar["compositional_v2"] = run_compositional_v2_benchmark().to_dict()
    print("  ✓ C_G v2 ham metin genellemesi")
    kapasite = run_capacity_benchmark(operands_max=7)
    raporlar["capacity"] = kapasite.to_dict()
    print("  ✓ Kapasite çerçevesi (P / C_I^UB / C_M^UB / C_E / C_V)")
    raporlar["memory"] = run_memory_hierarchy_benchmark(
        profile="smoke" if profile == "smoke" else "standard").to_dict()
    print("  ✓ Hiyerarşik bellek (hot/warm/cold/archive)")
    raporlar["multi_environment"] = run_multi_environment_benchmark(
        seeds=tuple(range(1, 21))).to_dict()
    print("  ✓ Çoklu ortam + cross-domain verifier izolasyonu")
    raporlar["verifier_adversarial"] = run_verifier_ensemble_benchmark().to_dict()
    print("  ✓ Çoklu verifier ensemble + adversarial proof suite")
    raporlar["self_learning_scaling"] = run_self_learning_scaling(
        profile="smoke" if profile == "smoke" else "standard",
        seeds=tohumlar[:1]).to_dict()
    raporlar["seed_statistics"] = run_core_seed_statistics(
        profile="smoke" if profile == "smoke" else "core",
        signature_profile=profile,
        include_operator=not skip_torch).to_dict()
    print("  ✓ Çekirdek tohum istatistikleri (CI / etki / permütasyon)")
    raporlar["priority_optimization"] = optimize_priority_weights(
        profile="smoke" if profile == "smoke" else "standard").to_dict()
    print("  ✓ Priority(E) ağırlık optimizasyonu + held-out")
    raporlar["depth_diagnosis"] = diagnose_depth_collapse(
        profile="smoke" if profile == "smoke" else "standard").to_dict()
    print("  ✓ Derinlik çöküşü kök neden teşhisi")
    raporlar["human_evaluation"] = _insan_degerlendirme_raporu().to_dict()
    raporlar["engineering"] = audit_engineering().to_dict()
    print("  ✓ Mühendislik sözleşmesi (CI / paketleme / test)")

    if not skip_torch:
        try:
            from hga.evaluation.operator_baselines import (
                run_operator_baseline_benchmark,
            )
            from hga.evaluation.signature import run_signature_benchmark
            raporlar["operator_baselines"] = run_operator_baseline_benchmark(
                n=8, steps=200 if profile == "smoke" else 300,
                seeds=tohumlar[:3]).to_dict()
            print("  ✓ Operatör baseline ailesi (gerçek Dense dahil)")
            raporlar["signature"] = run_signature_benchmark(
                profile=profile, seeds=tohumlar[:1]).to_dict()
            print("  ✓ HGA Signature Benchmark")
            from hga.evaluation.twt_results import run_twt_results
            raporlar["twt_results"] = run_twt_results(
                seeds=tohumlar, profile="smoke").to_dict()
            print("  ✓ TWT gerçek sonuç tablosu (gerçek Türkçe veri)")
            from hga.evaluation.turkish_lm import run_turkish_lm_benchmark
            raporlar["language_modeling"] = run_turkish_lm_benchmark(
                profile="smoke" if profile == "smoke" else "full",
                seeds=tohumlar[:2] if len(tohumlar) >= 2 else None).to_dict()
            print("  ✓ Gerçek Türkçe LM (TWT, belge-ayrık held-out)")
        except ImportError as hata:
            print(f"  ! PyTorch yok, nöral bölümler ATLANDI: {hata}")
            print("    (atlanan bölüm skor üretmez; karne bunu n/a gösterir)")

    # Yeniden üretilebilirlik denetimi EN SON koşar: diğer raporların
    # tohum sayısını ve veri imzasını girdi olarak alır.
    raporlar["reproducibility"] = audit_reproducibility(raporlar).to_dict()
    print("  ✓ Yeniden üretilebilirlik (manifest / determinizm / tohum)")

    karne = build_scorecard(**raporlar)
    print()
    print(scorecard_markdown(karne))
    yollar = save_scorecard(karne, json_path=out, markdown_path=markdown)
    for tur, yol in yollar.items():
        print(f"{tur}: {yol}")

def _research_benchmark(profile, seeds, experiment_root, sections=None,
                        out=None, markdown=None, html_yol=None, strict=False):
    """Manifestli, çoklu-seed birleşik araştırma benchmark karnesi."""
    from hga.evaluation import run_research_benchmark, save_research_report

    seed_values = [int(value.strip()) for value in seeds.split(",") if value.strip()]
    section_values = None
    if sections is not None:
        section_values = [value.strip() for value in sections.split(",") if value.strip()]
    report = run_research_benchmark(
        root=experiment_root, seeds=seed_values, profile=profile,
        sections=section_values,
    )
    paths = save_research_report(
        report,
        json_path=out or "research_report.json",
        markdown_path=markdown or "research_report.md",
        html_path=html_yol or "research_report.html",
    )
    print("╔══════════════════════════════════════════════════════╗")
    print("║              HGA RESEARCH BENCHMARK                  ║")
    print("╠══════════════════════════════════════════════════════╣")
    for name in report.selected_sections:
        section = report.sections[name]
        score = "n/a" if section["score_mean"] is None else f"{section['score_mean']:.3f}"
        print(f"║ {section['label'][:32]:<32} {section['status'][:10]:<10} {score:>7} ║")
    reproduction_status = (
        "COMPLETED" if report.reproducibility["all_manifests_completed"] else "ERROR"
    )
    print(f"║ {'Reproducibility':<32} {reproduction_status:<10} "
          f"{len(report.seeds):>4} sd ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"Outcome: {report.outcome}")
    print(f"Experiments: {', '.join(report.experiment_ids)}")
    for kind, path in paths.items():
        print(f"{kind}: {path}")
    if strict and report.outcome != "COMPLETED":
        raise SystemExit(
            f"research-benchmark --strict: beklenen COMPLETED, alınan {report.outcome}"
        )
