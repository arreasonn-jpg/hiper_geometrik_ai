# -*- coding: utf-8 -*-
"""
HGA Experience Engine — Komut Satırı Arayüzü
=============================================
Kullanım:
    python -m hga bilgi                  # bilgi tabanı demosu (Ali/Ata/Araba/Gökyüzü)
    python -m hga gercek-veri            # gerçek veri → temsil → deneyim → doğrulama
    python -m hga benchmark              # kontrollü benchmark (metrik tablosu)
    python -m hga golden-benchmark       # elle sabit golden set + FAR/FRR/leakage
    python -m hga memory-benchmark       # collision/interference/retrieval stres testi
    python -m hga kronecker-benchmark    # eşit-parametre Kronecker/rank-1 kıyası
    python -m hga self-learning-benchmark # K₀→Kₙ + collapse failure injection
    python -m hga research-benchmark     # birleşik, 5-seed JSON/MD/HTML araştırma karnesi
    python -m hga milestone              # K₀→Kₙ milestone tablosu (versiyon + defter)
    python -m hga kapasite               # P / C_I^UB / C_M^UB / C_E / C_V kapasite çerçevesi
    python -m hga bilgi-surum            # bilgi sürümleme + rollback demosu
    python -m hga defter                 # immutable experience ledger demosu
    python -m hga memory-interference    # kasıtlı çakışma + sabit/dinamik KV kıyası
    python -m hga paradigma              # neural vs symbolic vs hybrid (Faz 21)
    python -m hga olcekli-golden         # 100/1K/10K golden benchmark (Faz 3/6)
    python -m hga kronecker-rank         # effective rank + zincir çöküşü (Faz 19/20)
    python -m hga epistemik              # KNOWN/UNKNOWN/UNCERTAIN/CONFLICT/FALSE (P0-007)
    python -m hga verim                  # NY / UEY / GY / VID verim ayrıştırması (P1-005)
    python -m hga koken                  # provenance denetimi (Faz 27/28)
    python -m hga oncelik                # Priority(E) ağırlık ablasyonu (Faz 25)
    python -m hga oncelik-zincir         # Priority(E) nedensel zincir ablasyonu (P0-1)
    python -m hga operator-baseline      # Kronecker vs gerçek Dense ailesi (P0-2)
    python -m hga signature              # HGA Signature Benchmark v1 (P0-4)
    python -m hga signature-gate         # Signature release gate (PASS/BLOCKED)
    python -m hga verifier-adversarial   # verifier adversarial suite v2
    python -m hga verifier-ensemble      # üç üyeli verifier ensemble
    python -m hga semantik               # Türkçe semantik çıkarım benchmarkı (P0-5)
    python -m hga genelleme-v2           # ham metinden keşif + C_G v2 (P0-6)
    python -m hga cikarim-derinligi      # C_R / C_RD çıkarım derinliği (P0-7)
    python -m hga twt-sonuc              # TWT gerçek sonuç tablosu + FLOPs (P0-3)
    python -m hga english-ewt            # UD English EWT + Transformer/BERT/GPT-style baseline'ları
    python -m hga english-scaling        # HGA English EWT parameter scaling probe (scaling law değildir)
    python -m hga turkce-lm              # gerçek Türkçe LM: tr_corpus_v1 1.11M kelime (P1)
    python -m hga uzun-baglam            # uzun bağlam LM; 512/1024 için --long-context-profile smoke_1024
    python -m hga bellek-hiyerarsi       # hot/warm/cold/archive bellek (P0-8)
    python -m hga bellek-streaming       # 100M-ready streaming/checkpoint smoke harness
    python -m hga cok-ortam              # 5 ortam + verifier izolasyonu (P1)
    python -m hga ogrenme-transfer       # cross-domain self-learning transfer negatif-kontrol
    python -m hga ogrenme-olcek          # self-learning 100→1000→3000 cycle (P1)
    python -m hga tohum-istatistik       # 20 tohum + CI/etki/permütasyon (P1)
    python -m hga insan-degerlendirme    # protokol + Krippendorff alfa (P1)
    python -m hga insan-import           # insan puanı CSV import + α + kol agregasyonu
    python -m hga korpus-genisletme      # Türkçe korpus genişletme pipeline/release gate
    python -m hga derinlik-teshis        # derinlik çöküşü kök neden
    python -m hga oncelik-optimizasyon   # ağırlık araması + held-out
    python -m hga muhendislik            # CI / paketleme / test sözleşmesi
    python -m hga yeniden-uretilebilirlik # manifest / determinizm / tohum
    python -m hga championship-benchmark # tüm P0+P1 + OTOMATİK araştırma karnesi
    python -m hga dogrulama              # kapalı doğrulama hattı (false accept 24→0)
    python -m hga halusinasyon           # factual consistency / hallucination metriği
    python -m hga sweep                  # n/K/context kapasite taraması
    python -m hga tokenizer              # mini Türkçe tokenizer benchmark
    python -m hga perplexity [dosya]     # mini/tiny Türkçe perplexity smoke
    python -m hga checkpoint-rapor ckpt.pt # checkpoint/model şekil uyumluluğu
    python -m hga benchmark-rapor --out rapor.json --markdown rapor.md
    python -m hga veri-kalite [dosya]    # veri kalite filtresi raporu
    python -m hga veri-canli-smoke       # küçük canlı Türkçe veri hattı smoke
    python -m hga manifest dosya.txt     # SHA-256 veri manifesti
    python -m hga observability          # bellek/deneyim/geometri gözlem demosu/paneli
    python -m hga ozet [bilgi.json]      # bilgi tabanı özeti (opsiyonel yükleme)
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIMARI = os.path.join(KOK, "mimari")
for _p in [KOK, MIMARI]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _bilgi_demo():
    from hga.engine import ExperienceEngine
    e = ExperienceEngine()
    e.store.varlik_ekle("Ali", entity_type="insan", properties={"canli": 1},
                        entity_id="E_001", ozel_isim=True)
    e.store.varlik_ekle("Ata", entity_type="hayvan", properties={"binilebilir": 1},
                        entity_id="E_002")
    e.store.varlik_ekle("Araba", entity_type="tasit", properties={"binilebilir": 1},
                        entity_id="E_003")
    e.store.varlik_ekle("Gökyüzü", entity_type="mekan", properties={"binilebilir": 0},
                        entity_id="E_004")
    e.store.iliski_tanimla("Binmek", relation_id="R_001",
                           subject_types=["insan"],
                           requires_object_props={"binilebilir": 1.0})
    adaylar = e.uret(["R_001"])
    e.degerlendir(adaylar)
    print("Deneyim değerlendirmesi (durum makinesi):")
    for a in adaylar:
        nesne = e.store.entities.getir(a.object_id).token
        print(f"  Ali --Binmek--> {nesne:<8} → {a.state.value}")
    print("Metin üretimi:")
    for a in adaylar:
        if a.state.value == "VALID":
            print(f"  \"{e.metin(a)}\"")


def _graf_demo():
    from hga.engine import ExperienceEngine
    e = ExperienceEngine()
    e.store.varlik_ekle("Ali", entity_type="insan", properties={"canli": 1},
                        entity_id="E_001", ozel_isim=True)
    e.store.varlik_ekle("Ata", entity_type="hayvan", properties={"binilebilir": 1},
                        entity_id="E_002")
    e.store.varlik_ekle("Araba", entity_type="tasit", properties={"binilebilir": 1},
                        entity_id="E_003")
    e.store.iliski_tanimla("Binmek", relation_id="R_001",
                           subject_types=["insan"],
                           requires_object_props={"binilebilir": 1.0})
    adaylar = e.uret(["R_001"])
    e.degerlendir(adaylar)
    print("Experience Graph Özeti (Faz 23):")
    print("  ", e.graph.ozet())
    for a in adaylar:
        print("\n" + e.aciklama(a.experience_id))


def _kesif_demo():
    from hga.engine import ExperienceEngine
    e = ExperienceEngine()
    e.store.varlik_ekle("Ali", entity_type="insan", properties={"canli": 1},
                        entity_id="E_001", ozel_isim=True)
    e.store.varlik_ekle("Ata", entity_type="hayvan", properties={"binilebilir": 1},
                        entity_id="E_002")
    e.store.varlik_ekle("Araba", entity_type="tasit", properties={"binilebilir": 1},
                        entity_id="E_003")
    e.store.iliski_tanimla("Binmek", relation_id="R_001",
                           subject_types=["insan"],
                           requires_object_props={"binilebilir": 1.0})
    e.store.olgu_kaydet("E_001", "R_001", "E_002", score=1.0)
    adaylar = e.uret(["R_001"])
    e.degerlendir(adaylar)
    harita = e.uzay_haritasi(adaylar)
    print("Exploration Uzay Haritası (Faz 24):")
    for k, v in harita.to_dict().items():
        print(f"  {k:<20}: {v}")
    print("\nAktif Öğrenme ile En Bilgilendirici Deneyim Seçimi (Faz 25):")
    secilenler = e.aktif_ogrenme_sec(adaylar, k=2)
    for aday, puan in secilenler:
        nesne = e.store.entities.getir(aday.object_id).token
        print(f"  Ali --Binmek--> {nesne:<8} [InfoGain Opt Puanı: {puan:.4f}]")


def _gercek_veri():
    from hga.engine import ExperienceEngine
    e = ExperienceEngine()
    aktarilan = e.gercek_veri(
        ["Ali ataya bindi.", "Ali arabaya bindi.",
         "Ali gökyüzüne bindi.", "Ali gökyüzüne baktı."],
        iliski_kisitlari={"Binmek": {"subject_types": ["insan"],
                                     "requires_object_props": {"binilebilir": 1.0}}})
    print(f"REAL_DATA aktarılan üçlü: {len(aktarilan)}")
    rid = next(r.relation_id for r in e.store.relations.iliskiler()
               if r.token == "Binmek")
    adaylar = e.uret([rid])
    e.degerlendir(adaylar)
    for a in adaylar:
        nesne = e.store.entities.getir(a.object_id).token
        print(f"  Ali --Binmek--> {nesne:<8} → {a.state.value}")
    print("Bilgi özeti:", e.store.ozet())


def _benchmark():
    from hga.engine import ExperienceEngine
    from hga.experience import AritmetikOrtam, aritmetik_etki_alani
    store = aritmetik_etki_alani()
    e = ExperienceEngine(store=store, dogrulayici=AritmetikOrtam().aday_dogrula)
    raporlar = e.dongu(1)
    r = raporlar[0]
    print("Benchmark (1 adım, aritmetik alan):")
    for k, v in r.to_dict().items():
        print(f"  {k:<18}: {v}")


def _dogrulama():
    from hga.engine import ExperienceEngine
    from hga.experience import AritmetikOrtam, aritmetik_etki_alani
    store = aritmetik_etki_alani()
    e = ExperienceEngine(store=store, dogrulayici=AritmetikOrtam().aday_dogrula)
    adaylar = e.uret()
    e.degerlendir(adaylar)
    print(f"Değerlendirme sonrası: {len(adaylar)} aday, hepsi VALID (MODEL_GENERATED)")
    rapor = e.dogrula(adaylar)
    print("Doğrulama hattı sonucu:")
    for k, v in rapor.to_dict().items():
        print(f"  {k:<18}: {v}")


def _golden_benchmark(out=None, seeds=None, experiment_root="experiments"):
    from hga.evaluation import run_golden_benchmark, run_golden_seed_sweep

    if seeds:
        seed_values = [int(value.strip()) for value in seeds.split(",") if value.strip()]
        sweep = run_golden_seed_sweep(experiment_root, seed_values).to_dict()
        print("Golden çoklu-seed deney özeti:")
        print(f"  experiments : {', '.join(sweep['experiment_ids'])}")
        print(f"  deterministic: {sweep['deterministic_results']}")
        for metric in ("metrics.accuracy", "metrics.precision", "metrics.recall",
                       "metrics.f1", "metrics.far", "metrics.frr"):
            values = sweep["aggregate"].get(metric)
            if values:
                print(f"  {metric:<20}: {values['mean']} ± {values['std']}")
        print(f"  note        : {sweep['note']}")
        rapor = sweep
    else:
        rapor = run_golden_benchmark().to_dict()
        print("HGA Golden Benchmark (elle sabitlenmiş v1):")
        print(f"  dataset_hash : {rapor['dataset_hash']}")
        print(f"  leakage_clean: {rapor['leakage']['clean']}")
        for key in ("total", "correct", "accuracy", "precision", "recall", "f1", "far", "frr"):
            print(f"  {key:<13}: {rapor['metrics'][key]}")
        for row in rapor["predictions"]:
            print(f"  {row['experience_id']:<22} {row['expected']:<10} -> {row['predicted']}")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(rapor, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"  report       : {out}")


def _self_learning_benchmark(cycles, batch, initial_facts, operands_max,
                             negatives_per_fact, memory_slots, seeds,
                             experiment_root, out=None):
    from hga.evaluation import canonical_hash, run_seed_sweep
    from hga.experience import (
        run_multi_environment_self_learning,
        run_self_learning_experiment,
        run_self_training_collapse_test,
        run_verifier_fault_injection,
    )
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


def _kronecker_benchmark(n, steps, batch, seeds, device, experiment_root, out=None):
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        raise SystemExit("kronecker-benchmark için PyTorch gerekli")

    from hga.evaluation import canonical_hash, run_kronecker_dense_trial, run_seed_sweep

    seed_values = [int(value.strip()) for value in seeds.split(",") if value.strip()]
    config = {
        "benchmark": "kronecker-vs-rank1-param-matched-v2",
        "n": int(n), "steps": int(steps), "batch_size": int(batch),
        "test_samples": 128, "learning_rate": 0.01,
        "device": device,
        "tasks": ["kronecker_teacher", "rank1_teacher"],
    }
    dataset_hash = canonical_hash({
        "generator": "paired-synthetic-linear-teachers-v2",
        "n": int(n), "steps": int(steps), "batch_size": int(batch),
        "test_samples": 128,
    })

    def run_one(seed):
        report = run_kronecker_dense_trial(
            n=n, steps=steps, batch_size=batch, test_samples=128,
            learning_rate=0.01, seed=seed, device=device,
        )
        for task, task_report in report["tasks"].items():
            models = task_report["models"]
            print(
                f"{task}: kron_nmse={models['kronecker']['test']['normalized_mse']:.6g} "
                f"rank1_nmse={models['rank1_bottleneck']['test']['normalized_mse']:.6g} "
                f"winner={task_report['winner_by_test_normalized_mse']}"
            )
        return report

    report = run_seed_sweep(
        run_one, seeds=seed_values, root=experiment_root, config=config,
        dataset_hash=dataset_hash,
        parameters={
            "physical_parameter_budget_each": 2 * int(n) * int(n),
            "full_operator_entries_n4": int(n) ** 4,
            "n4_is_parameter_count": False,
        },
    ).to_dict()
    print("Kronecker vs rank-1 eşit-parametre benchmark özeti:")
    print(f"  experiments: {', '.join(report['experiment_ids'])}")
    for task in ("kronecker_teacher", "rank1_teacher"):
        for model in ("kronecker", "rank1_bottleneck"):
            key = f"tasks.{task}.models.{model}.test.normalized_mse"
            values = report["aggregate"].get(key)
            if values:
                print(f"  {task}.{model}.test_nmse: {values['mean']:.6g} ± {values['std']:.6g}")
    print("  Not: n⁴ operatör girdisidir; gerçek eğitilebilir parametre değildir.")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"  report: {out}")


def _memory_benchmark_markdown(report, active_dynamic):
    title = (
        "# Aktif Dynamic KV + Fixed Memory Stress"
        if active_dynamic else "# Sparse Memory Stress"
    )
    lines = [
        title,
        "",
        f"- Seedler: `{report['seeds']}`",
        f"- Dataset hash: `{report['dataset_hash']}`",
        f"- Config hash: `{report['config_hash']}`",
        f"- Git commit: `{report['manifests'][0]['git_commit']}`",
        f"- Temiz manifestler: `{all(not m['git_dirty'] for m in report['manifests'])}`",
        "",
    ]
    if active_dynamic and report["results"]:
        counts = report["results"][0]["benchmark"]["context_counts"]
        lines.extend([
            f"## {len(report['seeds'])}-seed aggregate",
            "",
            "| Context | Fixed history recall | Dynamic history recall | "
            "Dynamic active recall | Eviction | RSS MiB | ctx-pair/s |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for count in counts:
            prefix = f"metrics.n{count}."
            def aggregate(name):
                return report["aggregate"][prefix + name]
            fixed = aggregate("fixed_exact_history_recall")
            history = aggregate("dynamic_exact_history_recall")
            active = aggregate("dynamic_active_sample_recall")
            evictions = aggregate("dynamic_evictions")
            rss = aggregate("resident_set_bytes")
            throughput = aggregate("context_pairs_per_second")
            lines.append(
                f"| {count:,} | {fixed['mean']:.6f} ± {fixed['std']:.6f} | "
                f"{history['mean']:.6f} ± {history['std']:.6f} | "
                f"{active['mean']:.6f} ± {active['std']:.6f} | "
                f"{evictions['mean']:,.0f} ± {evictions['std']:,.0f} | "
                f"{rss['mean'] / (1024 * 1024):.2f} ± "
                f"{rss['std'] / (1024 * 1024):.2f} | "
                f"{throughput['mean']:,.0f} ± {throughput['std']:,.0f} |"
            )
        lines.extend([
            "",
            f"Tüm seed/kapılar: `{all(all(r['benchmark']['checks'].values()) for r in report['results'])}`",
            f"1K→10M kabulü: `{all(r['benchmark']['acceptance_1k_to_10m'] for r in report['results'])}`",
            "",
        ])
        for seed, result, manifest in zip(
            report["seeds"], report["results"], report["manifests"]
        ):
            lines.extend([
                f"## Seed {seed}",
                "",
                f"Süre: `{manifest['timings']['total_seconds']}` saniye",
                "",
                "| Context | Fixed recall | Dynamic history recall | "
                "Dynamic active recall | Eviction | RSS MiB | ctx-pair/s |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ])
            benchmark = result["benchmark"]
            for point in benchmark["points"]:
                lines.append(
                    f"| {point['context_count']:,} | "
                    f"{point['fixed_exact_history_recall']:.6f} | "
                    f"{point['dynamic_exact_history_recall']:.6f} | "
                    f"{point['dynamic_active_sample_recall']:.6f} | "
                    f"{point['dynamic_evictions']:,} | "
                    f"{point['resident_set_bytes'] / (1024 * 1024):.2f} | "
                    f"{point['segment_context_pairs_per_second']:,.0f} |"
                )
            lines.extend([
                "",
                f"Tüm kapılar: `{all(benchmark['checks'].values())}`",
                "",
            ])
    lines.extend(["## Sınırlar", ""])
    if report["results"]:
        lines.extend(
            f"- {note}"
            for note in report["results"][0]["benchmark"].get("notes", [])
        )
    return "\n".join(lines) + "\n"


def _memory_benchmark(
    scales, slots, tables, seeds, experiment_root, out=None, markdown=None,
    active_dynamic=False, audit_samples=256,
):
    from hga.evaluation import canonical_hash, run_seed_sweep
    from hga.memory import run_active_memory_stress, run_memory_stress

    context_counts = [int(value.strip()) for value in scales.split(",") if value.strip()]
    table_counts = [int(value.strip()) for value in tables.split(",") if value.strip()]
    seed_values = [int(value.strip()) for value in seeds.split(",") if value.strip()]
    config = {
        "benchmark": (
            "active-memory-stress-v2" if active_dynamic
            else "sparse-memory-collision-v1"
        ),
        "context_counts": context_counts,
        "slot_count": int(slots),
        "table_counts": ([1] if active_dynamic else table_counts),
        "collision_sample_limit": 1000,
        "audit_samples": int(audit_samples) if active_dynamic else None,
        "single_stream": bool(active_dynamic),
    }
    dataset_hash = canonical_hash({
        "generator": "hga-memory-context-v1",
        "context_counts": context_counts,
    })

    def run_one(seed):
        if active_dynamic:
            active_report = run_active_memory_stress(
                context_counts,
                capacity=int(slots),
                seed=seed,
                audit_samples=int(audit_samples),
            )
            metrics = {}
            for point in active_report.points:
                key = f"n{point.context_count}"
                metrics[key] = {
                    "fixed_exact_history_recall": point.fixed_exact_history_recall,
                    "fixed_collision_rate": round(
                        point.fixed_collisions / point.context_count, 8
                    ),
                    "dynamic_exact_history_recall": point.dynamic_exact_history_recall,
                    "dynamic_active_sample_recall": point.dynamic_active_sample_recall,
                    "dynamic_evictions": point.dynamic_evictions,
                    "dynamic_storage_bytes": point.dynamic_estimated_storage_bytes,
                    "fixed_storage_bytes": point.fixed_estimated_storage_bytes,
                    "resident_set_bytes": point.resident_set_bytes,
                    "context_pairs_per_second": point.segment_context_pairs_per_second,
                }
                print(
                    f"{key}: fixed_recall={point.fixed_exact_history_recall:.6f} "
                    f"dynamic_history={point.dynamic_exact_history_recall:.6f} "
                    f"dynamic_active={point.dynamic_active_sample_recall:.6f} "
                    f"evictions={point.dynamic_evictions}"
                )
            return {"metrics": metrics, "benchmark": active_report.to_dict()}

        report = run_memory_stress(
            context_counts, slot_count=slots, table_counts=table_counts, seed=seed,
        )
        metrics = {}
        for result in report.results:
            key = f"n{result.context_count}_t{result.table_count}"
            metrics[key] = {
                "collision_event_rate": result.collision_event_rate,
                "retrieval_accuracy": result.retrieval_accuracy,
                "interference_rate": result.interference_rate,
                "orphaned_slots": result.orphaned_slots,
                "false_positive_rate": result.false_positive_rate,
                "writes_per_second": result.writes_per_second,
                "reads_per_second": result.reads_per_second,
                "estimated_storage_bytes": result.estimated_storage_bytes,
            }
            print(
                f"{key}: collision={result.collision_event_rate:.6f} "
                f"recall={result.retrieval_accuracy:.6f} "
                f"interference={result.interference_rate:.6f}"
            )
        return {"metrics": metrics, "benchmark": report.to_dict()}

    report = run_seed_sweep(
        run_one, seeds=seed_values, root=experiment_root, config=config,
        dataset_hash=dataset_hash,
        parameters={
            "synthetic": True,
            "streaming_context_generation": True,
            "input_corpus_materialized": False,
            "active_dynamic_kv": bool(active_dynamic),
            "physical_capacity_entries": int(slots),
        },
    ).to_dict()
    print(
        "Aktif Dynamic KV + fixed memory stress özeti:"
        if active_dynamic else "Sparse memory benchmark özeti:"
    )
    print(f"  experiments: {', '.join(report['experiment_ids'])}")
    for metric, values in report["aggregate"].items():
        suffixes = (
            "fixed_exact_history_recall", "dynamic_exact_history_recall",
            "dynamic_active_sample_recall", "collision_event_rate",
            "retrieval_accuracy", "interference_rate",
        )
        if metric.endswith(suffixes):
            print(f"  {metric:<58} {values['mean']:.6f} ± {values['std']:.6f}")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"  report: {out}")
    if markdown:
        os.makedirs(os.path.dirname(os.path.abspath(markdown)) or ".", exist_ok=True)
        with open(markdown, "w", encoding="utf-8") as handle:
            handle.write(_memory_benchmark_markdown(report, active_dynamic))
        print(f"  markdown: {markdown}")


def _ozet(yol):
    from hga.knowledge import KnowledgeStore
    if yol:
        k = KnowledgeStore.yukle(yol)
    else:
        k = KnowledgeStore()
    print("Bilgi tabanı özeti:")
    for k, v in k.ozet().items():
        print(f"  {k:<10}: {v}")


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


def _memory_interference(forced, slots, tables, scales, out=None, markdown=None):
    """Faz 15-17: kasıtlı çakışma, politika kıyası ve sabit/dinamik ölçekleme."""
    from hga.memory import run_fixed_vs_dynamic_scaling, run_policy_comparison

    context_counts = [int(v.strip()) for v in scales.split(",") if v.strip()]
    kiyas = run_policy_comparison(forced_collisions=int(forced),
                                  slot_count=int(slots),
                                  table_count=int(tables))
    olcek = run_fixed_vs_dynamic_scaling(context_counts=context_counts,
                                         slot_count=int(slots),
                                         table_count=int(tables))
    print("Kasıtlı çakışma (A ve B aynı slota zorlandı) — politika kıyası:")
    print(kiyas.markdown())
    print()
    for bulgu in kiyas.findings:
        print(f"  - {bulgu}")
    print("\nSabit tablo vs dinamik KV ölçekleme (Faz 17):")
    print(olcek.markdown())
    print(f"\n  recall'ın 1.0'ın altına ilk düştüğü ölçek: "
          f"{olcek.crossover_context_count}")
    for not_ in olcek.notes:
        print(f"  not: {not_}")

    rapor = {"policy_comparison": kiyas.to_dict(), "scaling": olcek.to_dict()}
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(rapor, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"  report: {out}")
    if markdown:
        os.makedirs(os.path.dirname(os.path.abspath(markdown)) or ".", exist_ok=True)
        with open(markdown, "w", encoding="utf-8") as handle:
            handle.write("# Memory Interference ve Sabit/Dinamik Ölçekleme\n\n")
            handle.write("## Kasıtlı çakışma: politika kıyası\n\n")
            handle.write(kiyas.markdown() + "\n\n")
            for bulgu in kiyas.findings:
                handle.write(f"- {bulgu}\n")
            handle.write("\n## Sabit tablo vs dinamik KV\n\n")
            handle.write(olcek.markdown() + "\n\n")
            for not_ in olcek.notes:
                handle.write(f"- {not_}\n")
        print(f"  markdown: {markdown}")


def _paradigma(seeds, out=None, markdown=None, epochs=60):
    """Faz 21: neural-only vs symbolic-only vs hybrid kontrollü ablasyon."""
    from hga.evaluation.paradigma import run_paradigm_sweep

    tohumlar = [int(v.strip()) for v in seeds.split(",") if v.strip()]
    rapor = run_paradigm_sweep(seeds=tohumlar, epochs=int(epochs))
    print(f"Görev: {rapor.task}")
    print(f"Tohumlar: {rapor.seeds}\n")
    print(rapor.markdown())
    print()
    for bulgu in rapor.findings:
        print(f"  - {bulgu}")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(rapor.to_dict(), handle, ensure_ascii=False, indent=2,
                      sort_keys=True)
        print(f"  report: {out}")
    if markdown:
        os.makedirs(os.path.dirname(os.path.abspath(markdown)) or ".", exist_ok=True)
        with open(markdown, "w", encoding="utf-8") as handle:
            handle.write("# Faz 21 — Neural vs Symbolic vs Hybrid\n\n")
            handle.write(f"Tohumlar: {rapor.seeds}\n\n")
            handle.write(f"Görev: `{rapor.task}`\n\n")
            handle.write(rapor.markdown() + "\n\n")
            for bulgu in rapor.findings:
                handle.write(f"- {bulgu}\n")
        print(f"  markdown: {markdown}")


def _olcekli_golden(sizes, seeds, hard=False, out=None, markdown=None):
    """Faz 3/6: golden benchmark'ı 100/1.000/10.000 ölçeğine çıkar."""
    from hga.evaluation.scaled_golden import run_scaled_golden_sweep

    olcekler = [int(v.strip()) for v in sizes.split(",") if v.strip()]
    tohumlar = [int(v.strip()) for v in seeds.split(",") if v.strip()]
    rapor = run_scaled_golden_sweep(sizes=olcekler, seeds=tohumlar, hard=bool(hard))
    print(f"Ölçekler: {rapor.sizes} | Tohumlar: {rapor.seeds} | "
          f"zor mod: {rapor.hard}")
    print(f"Sızıntı denetimi: {'temiz' if rapor.leakage_clean else 'KİRLİ'}\n")
    print(rapor.markdown())
    print()
    for bulgu in rapor.findings:
        print(f"  - {bulgu}")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(rapor.to_dict(), handle, ensure_ascii=False, indent=2,
                      sort_keys=True)
        print(f"  report: {out}")
    if markdown:
        os.makedirs(os.path.dirname(os.path.abspath(markdown)) or ".", exist_ok=True)
        with open(markdown, "w", encoding="utf-8") as handle:
            handle.write("# Ölçeklendirilmiş Golden Benchmark\n\n")
            handle.write(f"Ölçekler: {rapor.sizes} · Tohumlar: {rapor.seeds} · "
                         f"zor mod: {rapor.hard}\n\n")
            handle.write(rapor.markdown() + "\n\n")
            for bulgu in rapor.findings:
                handle.write(f"- {bulgu}\n")
        print(f"  markdown: {markdown}")


_TEK_TOHUM_SINIRI = "Her rapor tek tohumun tek koşusudur"


def _toplu_sinirlar(sinirlar, tohum_sayisi):
    """Tek koşu için yazılan sınır cümlesini çok tohumlu rapora uyarla.

    run_yield_experiment tek koşu ürettiği için "güven aralığı yok" der. Bu
    cümle çok tohumlu toplu raporda YANLIŞ olur: tablo zaten bootstrap GA
    içerir. Tohum sayısına göre cümleyi değiştiriyoruz.
    """
    if tohum_sayisi <= 1:
        return list(sinirlar)
    uyarlanmis = []
    for sinir in sinirlar:
        if sinir.startswith(_TEK_TOHUM_SINIRI):
            uyarlanmis.append(
                f"Güven aralıkları {tohum_sayisi} tohumun bootstrap dağılımından "
                "hesaplandı; tohum sayısı düşük olduğu için aralıklar geniştir."
            )
        else:
            uyarlanmis.append(sinir)
    return uyarlanmis


def _verim(cycles=20, batch=32, initial_facts=40, operands_max=15,
           negatives_per_fact=3, seeds=None, out=None, markdown=None):
    """P1-005: EY'yi NY / UEY / GY / VID eksenlerine ayır."""
    from hga.evaluation.statistics import summarize_seed_metric
    from hga.experience.verim import run_yield_experiment

    tohumlar = [int(v.strip()) for v in (seeds or "1").split(",") if v.strip()]
    raporlar = [
        run_yield_experiment(
            cycles=int(cycles), batch_size=int(batch),
            initial_facts=int(initial_facts), operands_max=int(operands_max),
            negatives_per_fact=int(negatives_per_fact), seed=tohum,
        )
        for tohum in tohumlar
    ]
    ilk = raporlar[0]

    print("Deneyim Verimi Ayrıştırması (P1-005)\n")
    print(f"  protokol : {ilk.protocol}")
    print(f"  tohumlar : {tohumlar}")
    print(f"  döngü    : {cycles} × batch {batch}\n")

    eksenler = [
        ("EY  (klasik)", "experience_yield", "doğrulanan / üretilen"),
        ("NY  Yenilik", "novelty_yield", "ayrık YENİ olgu / üretilen"),
        ("UEY Kullanışlı", "useful_experience_yield", "doğru + geri çağrılabilir"),
        ("GY  Genelleme", "generalization_yield", "holdout doğruluk artışı"),
        ("VID Bilgi yoğ.", "verified_information_density", "bit / üretilen deneyim"),
    ]
    print(f"  {'metrik':<16}{'ortalama':>10}{'%95 GA':>22}  açıklama")
    for etiket, alan, aciklama in eksenler:
        degerler = [getattr(rapor, alan) for rapor in raporlar]
        ozet = summarize_seed_metric(degerler)
        aralik = f"[{ozet['ci_lower']:.4f}, {ozet['ci_upper']:.4f}]"
        print(f"  {etiket:<16}{ozet['mean']:>10.4f}{aralik:>22}  {aciklama}")

    print("\n  AYRIŞMA (bu metrikler EY'den farklı bir şey söylüyor mu?)")
    for anahtar in ("ey_vs_ny", "ey_vs_uey", "ny_vs_uey"):
        print(f"    {anahtar:<24}: {ilk.divergence[anahtar]:+.4f}")
    print(f"    {'EY yeniliği abartıyor':<24}: "
          f"{'EVET' if ilk.divergence['ey_overstates_novelty'] else 'hayır'}")
    print(f"    {'EY kullanışlılığı abartıyor':<24}: "
          f"{'EVET' if ilk.divergence['ey_overstates_usefulness'] else 'hayır'}")

    print("\n  HAM SAYIMLAR (tohum 1)")
    print(f"    üretilen={ilk.generated}  doğrulanan={ilk.verified}  "
          f"ayrık_yeni={ilk.distinct_new_facts}  kullanışlı={ilk.useful_facts}")
    print(f"    tekrar_üretim={ilk.duplicate_generations}  "
          f"bellek_çakışma={ilk.memory_collisions}  yanlış_olgu={ilk.incorrect_facts}")
    print(f"    holdout: öncesi {ilk.holdout_before['decided']}/{ilk.holdout_before['total']} karar, "
          f"sonrası {ilk.holdout_after['decided']}/{ilk.holdout_after['total']} karar "
          f"(isabet {ilk.holdout_after['accuracy_on_decided']:.4f})")

    print(f"\n  BULGULAR (tohum {tohumlar[0]})")
    for bulgu in ilk.findings:
        print(f"    - {bulgu}")
    print("\n  SINIRLAR")
    for sinir in _toplu_sinirlar(ilk.limitations, len(tohumlar)):
        print(f"    - {sinir}")

    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            ozetler = {}
            for _, alan, _aciklama in eksenler:
                degerler = [getattr(rapor, alan) for rapor in raporlar]
                ozetler[alan] = summarize_seed_metric(degerler)
            json.dump({"seeds": tohumlar,
                       "aggregate": ozetler,
                       "reports": [rapor.to_dict() for rapor in raporlar]},
                      handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"\n  report: {out}")
    if markdown:
        os.makedirs(os.path.dirname(os.path.abspath(markdown)) or ".", exist_ok=True)
        with open(markdown, "w", encoding="utf-8") as handle:
            handle.write("# Deneyim Verimi Ayrıştırması (P1-005)\n\n")
            handle.write(f"Tohumlar: {tohumlar}\n\n")
            handle.write("| Metrik | Ortalama | %95 GA |\n|---|---:|---:|\n")
            for etiket, alan, _ in eksenler:
                degerler = [getattr(rapor, alan) for rapor in raporlar]
                ozet = summarize_seed_metric(degerler)
                handle.write(f"| {etiket} | {ozet['mean']:.4f} | "
                             f"[{ozet['ci_lower']:.4f}, {ozet['ci_upper']:.4f}] |\n")
            handle.write(f"\n## Bulgular (tohum {tohumlar[0]})\n\n")
            for bulgu in ilk.findings:
                handle.write(f"- {bulgu}\n")
            handle.write("\n## Sınırlar\n\n")
            for sinir in _toplu_sinirlar(ilk.limitations, len(tohumlar)):
                handle.write(f"- {sinir}\n")
        print(f"  markdown: {markdown}")


def _cok_adimli(seeds=None, hops=None, distractors=None, slots=4096,
                out=None, markdown=None):
    """P1-004/P1-006: çok adımlı çıkarım + uzun bağlam dayanıklılığı."""
    from hga.evaluation.multi_hop import (
        DEFAULT_DISTRACTORS,
        DEFAULT_HOPS,
        multi_hop_markdown,
        run_multi_hop_benchmark,
    )

    def _liste(metin, varsayilan):
        if not metin:
            return varsayilan
        return tuple(int(v.strip()) for v in str(metin).split(",") if v.strip())

    tohumlar = _liste(seeds, (1, 2, 3))
    hop_listesi = _liste(hops, DEFAULT_HOPS)
    dolgu_listesi = _liste(distractors, DEFAULT_DISTRACTORS)

    rapor = run_multi_hop_benchmark(
        hops=hop_listesi, distractor_levels=dolgu_listesi,
        seeds=tohumlar, slot_sayisi=int(slots),
    )

    print("Çok Adımlı Çıkarım ve Uzun Bağlam (P1-004 / P1-006)\n")
    print(f"  protokol   : {rapor.protocol}")
    print(f"  veri imzası: {rapor.dataset_hash}")
    print(f"  tohumlar   : {rapor.seeds}   bellek slotu: {slots}\n")

    print("  DOĞRULUK IZGARASI (satır = zincir derinliği, sütun = dolgu olgu)")
    etiketler = {
        hop: f"{hop} adım" + (" (geri çağ.)" if hop == 1 else "")
        for hop in rapor.hops
    }
    genislik = max(len(e) for e in etiketler.values()) + 2
    print(" " * (4 + genislik) + "".join(
        f"{d:>10}" for d in rapor.distractor_levels))
    for hop in rapor.hops:
        hucreler = {c.distractors: c for c in rapor.cells if c.hop == hop}
        satir = f"    {etiketler[hop]:<{genislik}}" + "".join(
            f"{hucreler[d].accuracy:>10.4f}" for d in rapor.distractor_levels
        )
        print(satir)

    print("\n  ASIL METRİKLER")
    print(f"    çok adımlı çıkarım (hop>=2) : {rapor.inference_accuracy:.4f}")
    print(f"    tek adımlı geri çağırma     : {rapor.recall_accuracy:.4f}")
    print(f"    en derin güvenilir zincir   : {rapor.deepest_reliable_hop} adım")
    print(f"    bağlam bozulması            : {rapor.context_degradation:+.4f}")

    print("\n  NEGATİF KONTROL (zinciri takip etmeyen sabit cevaplar)")
    for kol in rapor.degenerate_arms:
        print(f"    {kol.arm:<14}{kol.accuracy:>9.4f}")
    print(f"    {'MOTOR':<14}{rapor.overall_accuracy:>9.4f}")

    print("\n  KABUL KAPILARI")
    for ad, sonuc in rapor.checks.items():
        print(f"    {ad:<28}: {'GEÇTİ' if sonuc else 'KALDI'}")

    print("\n  BULGULAR")
    for bulgu in rapor.findings:
        print(f"    - {bulgu}")
    print("\n  SINIRLAR")
    for sinir in rapor.limitations:
        print(f"    - {sinir}")

    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(rapor.to_dict(), handle, ensure_ascii=False,
                      indent=2, sort_keys=True)
        print(f"\n  report: {out}")
    if markdown:
        os.makedirs(os.path.dirname(os.path.abspath(markdown)) or ".", exist_ok=True)
        with open(markdown, "w", encoding="utf-8") as handle:
            handle.write(multi_hop_markdown(rapor))
        print(f"  markdown: {markdown}")


def _epistemik(seeds=None, out=None, markdown=None):
    """P0-007: KNOWN/UNKNOWN/UNCERTAIN/CONFLICT/FALSE epistemik benchmarkı."""
    from hga.evaluation.epistemic import (
        EpistemicDataset,
        run_epistemic_baselines,
        run_epistemic_benchmark,
    )

    veri = EpistemicDataset()
    rapor = run_epistemic_benchmark(veri)
    tohumlar = [int(v.strip()) for v in (seeds or "1").split(",") if v.strip()]

    print("Epistemik Benchmark — bilmediğini biliyor mu? (P0-007)\n")
    print(f"  protokol       : {rapor.protocol}")
    print(f"  veri kümesi    : {rapor.dataset_hash[:16]}…  ({rapor.total_cases} vaka)")
    print(f"  tohumlar       : {tohumlar} (protokol deterministik)\n")

    print("  ASIL METRİKLER")
    print(f"    yanlış güven oranı (↓)   : {rapor.false_confidence_rate:.3f} "
          f"({rapor.false_confidence_cases} vaka)")
    print(f"    bilinmeyen doğruluğu (↑) : {rapor.unknown_accuracy:.3f}")
    print(f"    sessiz kabul oranı (↓)   : {rapor.silent_failure_rate:.3f}")
    print(f"    genel doğruluk           : {rapor.accuracy:.3f}\n")

    print("  SINIF BAZINDA")
    print(f"    {'sınıf':<12}{'n':>4}{'doğru':>7}{'isabet':>9}  gözlenen durumlar")
    for sinif in rapor.per_class:
        durumlar = ", ".join(f"{k}×{v}" for k, v in sinif["observed_states"].items())
        print(f"    {sinif['epistemic_class']:<12}{sinif['total']:>4}"
              f"{sinif['correct']:>7}{sinif['accuracy']:>9.3f}  {durumlar}")

    print("\n  NEGATİF KONTROL (dejenere politikalar)")
    print(f"    {'kol':<18}{'doğruluk':>9}{'bilinen':>9}{'bilinmeyen':>12}{'y.güven':>9}")
    for kol in run_epistemic_baselines(veri):
        print(f"    {kol.arm:<18}{kol.accuracy:>9.3f}{kol.known_accuracy:>9.3f}"
              f"{kol.unknown_accuracy:>12.3f}{kol.false_confidence_rate:>9.3f}")
    print(f"    {'EVALUATOR':<18}{rapor.accuracy:>9.3f}{rapor.known_accuracy:>9.3f}"
          f"{rapor.unknown_accuracy:>12.3f}{rapor.false_confidence_rate:>9.3f}")

    ayrim = rapor.epistemic_resolution
    print("\n  EPİSTEMİK ÇÖZÜNÜRLÜK (dürüstlük notu)")
    print(f"    UNKNOWN ↔ UNCERTAIN ayrılabilir mi? : "
          f"{'EVET' if ayrim['distinguishable'] else 'HAYIR'}")
    print(f"    {ayrim['note']}")

    print("\n  KAPILAR")
    for ad, deger in rapor.checks.items():
        print(f"    [{'GEÇTİ' if deger else 'KALDI'}] {ad}")

    print("\n  SINIRLAR")
    for sinir in rapor.limitations:
        print(f"    - {sinir}")

    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(rapor.to_dict(), handle, ensure_ascii=False, indent=2,
                      sort_keys=True)
        print(f"\n  report: {out}")
    if markdown:
        os.makedirs(os.path.dirname(os.path.abspath(markdown)) or ".", exist_ok=True)
        with open(markdown, "w", encoding="utf-8") as handle:
            handle.write("# Epistemik Benchmark (P0-007)\n\n")
            handle.write(f"Veri kümesi hash: `{rapor.dataset_hash}`\n\n")
            handle.write(f"- Yanlış güven oranı (↓): **{rapor.false_confidence_rate:.3f}**\n")
            handle.write(f"- Bilinmeyen doğruluğu (↑): **{rapor.unknown_accuracy:.3f}**\n")
            handle.write(f"- Sessiz kabul oranı (↓): **{rapor.silent_failure_rate:.3f}**\n")
            handle.write(f"- Genel doğruluk: **{rapor.accuracy:.3f}**\n\n")
            handle.write("## Sınıf bazında\n\n")
            handle.write("| Sınıf | n | Doğru | İsabet |\n|---|---:|---:|---:|\n")
            for sinif in rapor.per_class:
                handle.write(f"| {sinif['epistemic_class']} | {sinif['total']} | "
                             f"{sinif['correct']} | {sinif['accuracy']:.3f} |\n")
            handle.write("\n## Epistemik çözünürlük\n\n")
            handle.write(f"{ayrim['note']}\n\n")
            handle.write("## Sınırlar\n\n")
            for sinir in rapor.limitations:
                handle.write(f"- {sinir}\n")
        print(f"  markdown: {markdown}")


def _kronecker_rank(n_values, k_values, seed=1, out=None, markdown=None):
    """Faz 19/20: effective rank + n×K taraması ve zincir çöküş testi."""
    from hga.evaluation.kronecker_rank import run_nk_rank_sweep

    n_list = [int(v.strip()) for v in n_values.split(",") if v.strip()]
    k_list = [int(v.strip()) for v in k_values.split(",") if v.strip()]
    rapor = run_nk_rank_sweep(n_values=n_list, k_values=k_list, seed=int(seed))
    print("Kronecker effective rank ve n×K taraması (Faz 19/20):\n")
    print(rapor.markdown())
    print()
    for bulgu in rapor.findings:
        print(f"  - {bulgu}")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(rapor.to_dict(), handle, ensure_ascii=False, indent=2,
                      sort_keys=True)
        print(f"  report: {out}")
    if markdown:
        os.makedirs(os.path.dirname(os.path.abspath(markdown)) or ".", exist_ok=True)
        with open(markdown, "w", encoding="utf-8") as handle:
            handle.write("# Faz 19/20 — Kronecker Effective Rank ve n×K Taraması\n\n")
            handle.write(rapor.markdown() + "\n\n")
            for bulgu in rapor.findings:
                handle.write(f"- {bulgu}\n")
        print(f"  markdown: {markdown}")


def _provenance(out=None, markdown=None):
    """Faz 27/28: köken denetimi ve belge hash doğrulaması demosu."""
    from hga.evaluation.provenance import (
        audit_provenance,
        build_provenance_chain,
        ingest_with_provenance,
        verify_document_hashes,
    )
    from hga.knowledge import KaynakTuru, KnowledgeStore

    store = KnowledgeStore()
    belge = ("Ali ata bindi. Ayşe kitabı okudu. Mehmet topu attı. "
             "Kuş gökyüzünde uçtu.")
    cumleler = [c.strip() + "." for c in belge.split(".") if c.strip()]
    bilgi = ingest_with_provenance(store, cumleler,
                                   source_url="ornek://turkce-mini-korpus",
                                   content=belge)
    print("Köken damgalı aktarım (Faz 27/28):")
    for anahtar, deger in bilgi.items():
        print(f"  {anahtar}: {deger}")

    zincir = None
    if store.relations.olgular():
        zincir = build_provenance_chain(store, store.relations.olgular()[0])
        print("\nÖrnek provenance zinciri:")
        print("  " + " → ".join(zincir.kinds()))
        print(f"  tam dış iz: {zincir.complete_external_trace}; eksikler: {zincir.missing}")

    # Kökensiz bir olgu ekleyip denetimin bunu yakaladığını göster.
    store.varlik_ekle("yetim_ozne", entity_id="E_YETIM_S")
    store.varlik_ekle("yetim_nesne", entity_id="E_YETIM_O")
    store.iliski_tanimla("yetim_iliski", relation_id="R_YETIM")
    store.olgu_kaydet("E_YETIM_S", "R_YETIM", "E_YETIM_O", 1.0,
                      source=KaynakTuru.REAL_DATA, confidence=1.0)

    rapor = audit_provenance(store)
    print("\nKöken denetimi:")
    print(rapor.markdown())
    print()
    for bulgu in rapor.findings:
        print(f"  - {bulgu}")
    print(f"  temiz mi: {rapor.clean} (kökensiz olgu bilerek eklendi)")

    dogrulama = verify_document_hashes(
        store, {"ornek://turkce-mini-korpus": belge})
    print(f"\nHash doğrulaması: kontrol={dogrulama.checked} "
          f"eşleşen={dogrulama.matched} tutmayan={dogrulama.mismatched}")
    bozuk = verify_document_hashes(
        store, {"ornek://turkce-mini-korpus": belge + " Belge değişti."})
    print(f"  belge değiştirildiğinde yakalanan: {bozuk.mismatched}")

    cikti = {"ingest": bilgi, "audit": rapor.to_dict(),
             "provenance_chain_example": zincir.to_dict() if zincir else None,
             "hash_check": dogrulama.to_dict(),
             "tampered_check": bozuk.to_dict()}
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(cikti, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"  report: {out}")
    if markdown:
        os.makedirs(os.path.dirname(os.path.abspath(markdown)) or ".", exist_ok=True)
        with open(markdown, "w", encoding="utf-8") as handle:
            handle.write("# Faz 27/28 — Köken (Provenance) Denetimi\n\n")
            handle.write(rapor.markdown() + "\n\n")
            if zincir is not None:
                handle.write("## Örnek Provenance Zinciri\n\n")
                handle.write("`" + " → ".join(zincir.kinds()) + "`\n\n")
                handle.write(f"- Tam dış iz: `{zincir.complete_external_trace}`\n")
                handle.write(f"- Eksikler: `{zincir.missing}`\n\n")
            for bulgu in rapor.findings:
                handle.write(f"- {bulgu}\n")
        print(f"  markdown: {markdown}")


def _priority(k=10, out=None, markdown=None):
    """Faz 25: Priority(E) ağırlıkları ve terim ablasyonu."""
    import random as _random

    from hga.experience.exploration import (
        ExplorationEngine,
        agirlik_ablasyonu,
    )
    from hga.knowledge import (
        DeneyimDurumu,
        ExperienceCandidate,
        KaynakTuru,
        KnowledgeStore,
    )

    store = KnowledgeStore()
    rng = _random.Random(1)
    store.iliski_tanimla("iliski", relation_id="R_P")
    adaylar = []
    for index in range(40):
        ozne, nesne = f"PS{index}", f"PO{index}"
        store.varlik_ekle(ozne, entity_id=ozne)
        store.varlik_ekle(nesne, entity_id=nesne)
        for j in range(3):
            store.ozellik_koy(nesne, f"p{j}", rng.choice([0.0, 1.0]),
                              source=KaynakTuru.REAL_DATA, confidence=0.9)
        if index % 3 == 0:
            store.olgu_kaydet(ozne, "R_P", nesne, 1.0,
                              confidence=rng.uniform(0.2, 0.99))
        aday = ExperienceCandidate(f"PE{index}", ozne, "R_P", nesne)
        if index % 7 == 0:
            aday.state = DeneyimDurumu.CONFLICT
        adaylar.append(aday)

    motor = ExplorationEngine()
    agirliklar = motor.agirliklar()
    print("Priority(E) = w_gain·InfoGain + w_novelty·Novelty "
          "+ w_uncertainty·Uncertainty − w_conflict_penalty·Conflict\n")
    print("Varsayılan ağırlıklar:")
    for ad in ("w_gain", "w_novelty", "w_uncertainty", "w_conflict_penalty"):
        print(f"  {ad:22s} = {agirliklar[ad]:.2f}  "
              f"({agirliklar['descriptions'][ad]})")

    ornek = motor.priority_dokumu(store, adaylar[0])
    print(f"\nÖrnek döküm ({ornek['experience_id']}): "
          f"priority={ornek['priority']}")
    for ad, terim in ornek["terms"].items():
        print(f"  {ad:20s} değer={terim['value']:.4f} "
              f"ağırlık={terim['weight']:.2f} katkı={terim['contribution']:+.4f}")

    ablasyon = agirlik_ablasyonu(store, adaylar, k=int(k))
    print("\nAğırlık ablasyonu (her terim tek tek kapatıldı):")
    print(ablasyon.markdown())
    print()
    for bulgu in ablasyon.findings:
        print(f"  - {bulgu}")

    cikti = {"weights": agirliklar, "example_breakdown": ornek,
             "ablation": ablasyon.to_dict()}
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(cikti, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"  report: {out}")
    if markdown:
        os.makedirs(os.path.dirname(os.path.abspath(markdown)) or ".", exist_ok=True)
        with open(markdown, "w", encoding="utf-8") as handle:
            handle.write("# Faz 25 — Priority(E) Ağırlıkları ve Ablasyon\n\n")
            handle.write(ablasyon.markdown() + "\n\n")
            for bulgu in ablasyon.findings:
                handle.write(f"- {bulgu}\n")
        print(f"  markdown: {markdown}")


def _kapasite(operands_max, out=None):
    """P / C_I^UB / C_M^UB / C_E / C_V kapasite çerçevesi ölçümü."""
    from hga.evaluation import run_capacity_benchmark
    rapor = run_capacity_benchmark(operands_max=int(operands_max))
    print("HGA Kapasite Çerçevesi (P, C_I^UB, C_M^UB, C_E, C_V):")
    print(rapor.markdown())
    print(f"\n  C_V / C_E          : {rapor.c_v_over_c_e}")
    print(f"  karar verilebilirlik: {rapor.decidability}")
    print(f"  sıralama geçerli    : {rapor.ordering_holds} (C_V ≤ C_E ≤ C_M^UB)")
    for not_ in rapor.notes:
        print(f"  not: {not_}")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(rapor.to_dict(), handle, ensure_ascii=False, indent=2,
                      sort_keys=True)
        print(f"  report: {out}")


def _bilgi_surum(out=None):
    """Knowledge versioning + rollback demosu (K0→K1→K2→rollback)."""
    from hga.knowledge import KnowledgeVersionStore

    kv = KnowledgeVersionStore()
    kv.store.varlik_ekle("Ali", entity_type="insan", entity_id="E_001",
                         ozel_isim=True)
    kv.store.varlik_ekle("Ata", entity_type="hayvan", entity_id="E_002",
                         properties={"binilebilir": 1})
    kv.store.varlik_ekle("Gökyüzü", entity_type="mekan", entity_id="E_003",
                         properties={"binilebilir": 0})
    kv.store.iliski_tanimla("Binmek", relation_id="R_001",
                            subject_types=["insan"],
                            requires_object_props={"binilebilir": 1.0})
    kv.store.olgu_kaydet("E_001", "R_001", "E_002", score=1.0)
    kv.commit("K1: Ali ata bindi (doğru)")
    kv.store.olgu_kaydet("E_001", "R_001", "E_003", score=1.0)
    kv.commit("K2: Ali gökyüzüne bindi (HATALI)")
    geri = kv.rollback("K1", label="K2 hatalıydı; K1'e dönüldü")

    print("Bilgi sürüm zinciri (Faz 23):")
    for kayit in kv.gecmis():
        print(f"  {kayit['version_id']:<4} {kayit['kind']:<8} "
              f"kanit={kayit['ozet']['kanit']} hash={kayit['content_hash'][:12]} "
              f"{kayit['label']}")
    fark = kv.diff("K2", geri.version_id)
    print(f"\nK2 → {geri.version_id} farkı:")
    print(f"  silinen olgular : {fark.silinen_olgular}")
    print(f"  eklenen olgular : {fark.eklenen_olgular}")
    print(f"  zincir geçerli  : {kv.zincir_dogrula()}")
    print("  not: rollback geçmişi SİLMEZ; hatalı K2 denetim için zincirde kalır.")
    if out:
        kv.kaydet(out)
        print(f"  report: {os.path.abspath(out)}")


def _defter(out=None):
    """Immutable experience ledger demosu (Faz 24)."""
    from hga.experience import ExperienceLedger
    from hga.knowledge import DeneyimDurumu, ExperienceCandidate, KaynakTuru

    defter = ExperienceLedger()
    ornekler = [
        ("E_001|R_001|E_002", DeneyimDurumu.VERIFIED, "bağımsız doğrulayıcı onayladı"),
        ("E_001|R_001|E_003", DeneyimDurumu.INVALID, "binilebilir=0 kuralı ihlal edildi"),
        ("E_001|R_002|E_009", DeneyimDurumu.UNCERTAIN, "kanıt yetersiz (Mars bilinmiyor)"),
        ("E_001|R_001|E_004", DeneyimDurumu.CONFLICT, "kayıtlı kanıtla çelişiyor"),
    ]
    for index, (uclu, durum, neden) in enumerate(ornekler):
        s, r, o = uclu.split("|")
        aday = ExperienceCandidate(f"LDG-{index:03d}", s, r, o,
                                   source=KaynakTuru.MODEL_GENERATED,
                                   state=durum)
        if durum == DeneyimDurumu.VERIFIED:
            aday.verified_by = "deterministik-dogrulayici"
        defter.kaydet(aday, knowledge_version="K12", reason=neden, cycle=1)

    print("Immutable Experience Ledger (Faz 24):")
    for kayit in defter:
        print(f"  #{kayit.seq} {kayit.status:<10} {kayit.experience:<22} "
              f"{kayit.reason}")
    print("\nÖzet:")
    for k, v in defter.ozet().items():
        print(f"  {k:<20}: {v}")
    print("  not: reddedilen deneyim de silinmez; 'model nerede hata yaptı?' "
          "sorusu ancak böyle cevaplanır.")
    if out:
        defter.kaydet_jsonl(out)
        print(f"  report: {os.path.abspath(out)}")


def _sweep():
    """n/K/context hızlı kapasite taraması."""
    from hga.config import model_config_yukle
    from hga.evaluation import nk_taramasi
    mcfg = model_config_yukle()["model"]
    nler = sorted({max(8, mcfg.n // 2), mcfg.n, mcfg.n * 2})
    satirlar = nk_taramasi(n_degerleri=nler, vocab=mcfg.sozluk_boyutu,
                           emb=mcfg.emb_dim, heads=mcfg.num_heads,
                           seyrek_satir=mcfg.seyrek_tablo_boyutu,
                           seyrek_boyut=mcfg.seyrek_boyut)
    print("n/K/context taraması (tahmini):")
    print("  n     K   ctx   parametre(M)  fp32_ram(MB)  zincir_ops      attn_ops")
    for r in satirlar:
        print(f"  {r['n']:<5} {r['K']:<3} {r['baglam']:<5} "
              f"{r['tahmini_parametre']/1e6:>10.2f}  "
              f"{r['tahmini_ram_mb_fp32']:>11.1f}  "
              f"{r['zincir_ops_yaklasik']:>10.2e}  "
              f"{r['attention_ops_yaklasik']:>9.2e}")


def _tokenizer_benchmark():
    """Mini Türkçe tokenizer benchmark'ı."""
    from bpe_tokenizer import BPETokenizer

    from hga.config import model_config_yukle
    from hga.evaluation import mini_turkce_corpus, tokenizer_kapsami
    corpus = mini_turkce_corpus()
    sozluk_boyutu = model_config_yukle()["model"].sozluk_boyutu
    tok = BPETokenizer(max_vocab_size=sozluk_boyutu, min_freq=1)
    tok.fit_on_text("\n".join(corpus), verbose=False)
    r = tokenizer_kapsami(tok, corpus)
    print("Mini Türkçe tokenizer benchmark:")
    for k, v in r.items():
        print(f"  {k:<24}: {v}")


def _perplexity_benchmark(yol=None, checkpoint=None, tokenizer_yol=None,
                          tiny: bool = True, batch_size: int = 64,
                          n: Optional[int] = None, katman: Optional[int] = None,
                          baglam: Optional[int] = None, vocab: Optional[int] = None):
    """Mini Türkçe perplexity smoke'u; torch yoksa güvenli bilgi ver."""
    try:
        import torch  # noqa: F401
    except Exception:
        print("Perplexity benchmark için torch gerekli. Örn: .venv/bin/python -m hga perplexity --tiny")
        return

    from bpe_tokenizer import BPETokenizer
    from kuresel_model import HiperGeometrikAI, agirlik_yukle

    from hga.config import model_config_yukle
    from hga.evaluation import mini_turkce_corpus, perplexity_benchmark

    if yol:
        with open(yol, "r", encoding="utf-8") as f:
            metin = f.read()
        cumleler = [c.strip() for c in metin.splitlines() if c.strip()] or [metin]
    else:
        cumleler = mini_turkce_corpus()
    metin = "\n".join(cumleler)

    cfg = model_config_yukle()["model"]
    baglam = int(baglam if baglam is not None else (8 if tiny else cfg.baglam_penceresi))
    vocab = int(vocab if vocab is not None else (256 if tiny else cfg.sozluk_boyutu))
    n = int(n if n is not None else (8 if tiny else cfg.n))
    katman = int(katman if katman is not None else (1 if tiny else cfg.katman_sayisi))

    tok = BPETokenizer(baglam_penceresi=baglam, max_vocab_size=vocab, min_freq=1)
    notlar = []
    sozluk = tokenizer_yol or os.path.join(KOK, "bpe_sozluk.json")
    if os.path.exists(sozluk):
        tok.yukle(sozluk)
        notlar.append(f"tokenizer={sozluk}")
    else:
        tok.fit_on_text(metin, verbose=False)
        notlar.append("tokenizer aynı metinde fit edildi: sadece smoke, held-out ölçüm değil")

    model = HiperGeometrikAI(n=n, katman_sayisi=katman, baglam_penceresi=baglam,
                             emb_dim=(8 if tiny else cfg.emb_dim),
                             num_heads=(2 if tiny else cfg.num_heads),
                             sozluk_boyutu=max(len(tok.sozluk), 64),
                             dropout=0.0 if tiny else cfg.dropout,
                             seyrek_tablo_boyutu=0 if tiny else cfg.seyrek_tablo_boyutu,
                             seyrek_boyut=cfg.seyrek_boyut,
                             bilgilendir=False)
    if checkpoint:
        agirlik_yukle(model, checkpoint, strict=True)
        notlar.append(f"checkpoint={checkpoint}")
    else:
        notlar.append("checkpoint yok: rastgele ağırlıklarla smoke ölçümü")

    rapor = perplexity_benchmark(model, tok, cumleler, batch_size=batch_size)
    rapor.notlar.extend(notlar)
    print("Mini Türkçe perplexity benchmark:")
    for k, v in rapor.to_dict().items():
        print(f"  {k:<16}: {v}")


def _checkpoint_rapor(yol, config_yol=None, strict: bool = True,
                      n: Optional[int] = None, katman: Optional[int] = None,
                      baglam: Optional[int] = None, vocab: Optional[int] = None,
                      emb: Optional[int] = None, heads: Optional[int] = None,
                      seyrek_satir: Optional[int] = None,
                      seyrek_boyut: Optional[int] = None):
    """Checkpoint'i yüklemeden model anahtar/şekil uyumluluğunu raporla."""
    if not yol:
        raise SystemExit("checkpoint-rapor komutu için checkpoint yolu gerekli")
    try:
        import torch  # noqa: F401
    except Exception:
        print("Checkpoint raporu için torch gerekli. Örn: .venv/bin/python -m hga checkpoint-rapor ckpt.pt")
        return
    if not os.path.exists(yol):
        raise SystemExit(f"checkpoint bulunamadı: {yol}")

    from kuresel_model import model_olustur

    from egitim.saglamlik import checkpoint_uyumluluk_raporu
    from hga.config import model_config_yukle

    cfg = model_config_yukle(config_yol)["model"]
    n = int(n if n is not None else cfg.n)
    katman = int(katman if katman is not None else cfg.katman_sayisi)
    baglam = int(baglam if baglam is not None else cfg.baglam_penceresi)
    vocab = int(vocab if vocab is not None else cfg.sozluk_boyutu)
    emb = int(emb if emb is not None else cfg.emb_dim)
    heads = int(heads if heads is not None else cfg.num_heads)
    seyrek_satir = int(seyrek_satir if seyrek_satir is not None else cfg.seyrek_tablo_boyutu)
    seyrek_boyut = int(seyrek_boyut if seyrek_boyut is not None else cfg.seyrek_boyut)

    model = model_olustur(sozluk_boyutu=vocab, n=n, baglam_penceresi=baglam,
                          emb_dim=emb, num_heads=heads, katman_sayisi=katman,
                          dropout=cfg.dropout,
                          encoder_aktivasyon=cfg.encoder_aktivasyon,
                          zincir_aktivasyon=cfg.zincir_aktivasyon,
                          checkpoint_kullan=False, bilgilendir=False,
                          seyrek_tablo_boyutu=seyrek_satir,
                          seyrek_boyut=seyrek_boyut,
                          seyrek_tablo_sayisi=cfg.seyrek_tablo_sayisi,
                          seyrek_erisim_izleme=cfg.seyrek_erisim_izleme)
    rapor = checkpoint_uyumluluk_raporu(model, yol, strict=strict)
    print("Checkpoint uyumluluk raporu:")
    for k in ["ok", "strict", "checkpoint_version", "model_meta", "ortak_anahtar"]:
        print(f"  {k:<20}: {rapor.get(k)}")
    print(f"  eksik               : {len(rapor['eksik'])}")
    print(f"  fazla               : {len(rapor['fazla'])}")
    print(f"  sekil_uyumsuz       : {len(rapor['sekil_uyumsuz'])}")
    for ad, sekil in list(rapor["sekil_uyumsuz"].items())[:8]:
        print(f"    - {ad}: model={sekil['model']} checkpoint={sekil['checkpoint']}")
    if rapor["eksik"]:
        print("  ilk_eksik           :", rapor["eksik"][:8])
    if rapor["fazla"]:
        print("  ilk_fazla           :", rapor["fazla"][:8])


def _halusinasyon():
    """Aritmetik alanda somut hallucination/factual consistency metriği."""
    from hga.evaluation import hallucination_metrics
    from hga.experience import (
        AritmetikOrtam,
        ExperienceEvaluator,
        ExperienceGenerator,
        aritmetik_etki_alani,
    )
    store = aritmetik_etki_alani()
    gen = ExperienceGenerator(tip_filtresi=False)
    ev = ExperienceEvaluator()
    adaylar = gen.uret(store)
    for a in adaylar:
        ev.degerlendir(a, store)
    r = hallucination_metrics(adaylar, store=store,
                              validator=AritmetikOrtam().aday_dogrula)
    print("Halüsinasyon / factual consistency raporu:")
    for k, v in r.to_dict().items():
        print(f"  {k:<28}: {v}")


def _veri_kalite(yol):
    """Dosya veya mini metin üzerinde veri kalite filtresi."""
    from hga.data import temizle_cumleler
    from hga.experience.corpus import cumlelere_bol
    if yol:
        with open(yol, "r", encoding="utf-8") as f:
            metin = f.read()
    else:
        metin = "Ali ataya bindi. Ali ataya bindi. bozuk � metin. 12345 !!! ???."
    temiz, rapor = temizle_cumleler(cumlelere_bol(metin))
    print("Veri kalite raporu:")
    for k, v in rapor.to_dict().items():
        print(f"  {k:<18}: {v}")
    print("Temiz örnekler:")
    for c in temiz[:5]:
        print(f"  - {c}")


def _manifest(yol):
    """Tek dosya için SHA-256 veri manifesti yazdır."""
    if not yol:
        raise SystemExit("manifest komutu için dosya yolu gerekli")
    from hga.data import dosya_hashle
    m = dosya_hashle(yol)
    print("Veri manifesti:")
    for k, v in m.to_dict().items():
        print(f"  {k:<8}: {v}")


def _veri_canli_smoke(konular=None, cikis=None, out=None, kontrollu: bool = False):
    """Canlı Wikipedia → kalite → manifest smoke raporu."""
    from hga.data import canli_wikipedia_smoke
    konu_listesi = None if konular is None else [k.strip() for k in konular.split(",") if k.strip()]
    fetcher = None
    if kontrollu:
        def fetcher(konu):
            return (f"{konu} Türkçe veri hattı için kontrollü bir smoke metnidir. "
                    f"İstanbul, Iğdır, şeker, çay, öğüt ve üzüm örnekleri Türkçe karakterleri korur. "
                    f"Bu cümleler ağ bağımlılığı olmadan kalite filtresinden geçirilir.")
    rapor = canli_wikipedia_smoke(konu_listesi, proje_kok=KOK, cikis=cikis, fetcher=fetcher)
    if kontrollu:
        rapor.notlar.append("kontrollü fetcher kullanıldı; ağ bağımlılığı yok")
    d = rapor.to_dict()
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"Canlı veri smoke raporu yazıldı: {out}")
    else:
        print("Canlı veri smoke raporu:")
        print(json.dumps(d, ensure_ascii=False, indent=2, sort_keys=True))


def _benchmark_rapor(out=None, markdown=None, checkpoint=None, tokenizer_yol=None,
                     n=None, katman=None, baglam=None, vocab=None):
    """Tokenizer/perplexity/halüsinasyon/bellek/cihaz tek rapor."""
    from hga.evaluation import benchmark_raporu_kaydet, benchmark_raporu_olustur
    rapor = benchmark_raporu_olustur(checkpoint=checkpoint, tokenizer_yol=tokenizer_yol,
                                     n=int(n or 8), katman=int(katman or 1),
                                     baglam=int(baglam or 8), vocab=int(vocab or 256))
    yollar = benchmark_raporu_kaydet(rapor, json_yol=out, markdown_yol=markdown)
    if yollar:
        for tur, yol in yollar.items():
            print(f"Benchmark raporu ({tur}) yazıldı: {yol}")
    else:
        print(json.dumps(rapor, ensure_ascii=False, indent=2, sort_keys=True))


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



def _yaz_rapor(cikti, out=None, markdown=None, md_metin=None):
    """Ortak JSON/Markdown yazıcı — rapor komutları bunu paylaşır."""
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(cikti, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"  report: {out}")
    if markdown and md_metin is not None:
        os.makedirs(os.path.dirname(os.path.abspath(markdown)) or ".", exist_ok=True)
        with open(markdown, "w", encoding="utf-8") as handle:
            handle.write(md_metin)
        print(f"  markdown: {markdown}")


def _oncelik_zincir(k=10, seeds=None, out=None, markdown=None):
    """P0-1: Priority(E) nedensel zincir ablasyonu."""
    from hga.evaluation.priority_ablation import (
        priority_ablation_markdown,
        run_priority_weight_ablation,
    )
    tohumlar = [int(v) for v in (seeds or "1,2,3,4,5").split(",") if v.strip()]
    rapor = run_priority_weight_ablation(k=int(k), seeds=tohumlar)
    print(priority_ablation_markdown(rapor))
    _yaz_rapor(rapor.to_dict(), out, markdown, priority_ablation_markdown(rapor))


def _operator_baseline(n=8, steps=300, seeds=None, device="cpu",
                       out=None, markdown=None):
    """P0-2: Kronecker vs gerçek Dense baseline ailesi."""
    from hga.evaluation.operator_baselines import (
        operator_baseline_markdown,
        run_operator_baseline_benchmark,
    )
    tohumlar = [int(v) for v in (seeds or "1,2,3").split(",") if v.strip()]
    rapor = run_operator_baseline_benchmark(
        n=int(n), steps=int(steps), seeds=tohumlar, device=device)
    print(operator_baseline_markdown(rapor))
    _yaz_rapor(rapor.to_dict(), out, markdown, operator_baseline_markdown(rapor))


def _cikarim_derinligi(profile="standard", seeds=None, threshold=1.0,
                       out=None, markdown=None):
    """P0-7: C_R ve C_RD ölçümü."""
    from hga.evaluation.reasoning_depth import (
        measure_reasoning_depth,
        reasoning_depth_markdown,
    )
    tohumlar = ([int(v) for v in seeds.split(",") if v.strip()]
                if seeds else None)
    rapor = measure_reasoning_depth(profile=profile, seeds=tohumlar,
                                    reliability_threshold=float(threshold))
    print(reasoning_depth_markdown(rapor))
    _yaz_rapor(rapor.to_dict(), out, markdown, reasoning_depth_markdown(rapor))


def _signature(profile="smoke", seeds=None, out=None, markdown=None):
    """P0-4: HGA Signature Benchmark."""
    from hga.evaluation.signature import (
        run_signature_benchmark,
        signature_markdown,
    )
    tohumlar = [int(v) for v in (seeds or "1,2,3").split(",") if v.strip()]
    rapor = run_signature_benchmark(profile=profile, seeds=tohumlar)
    print(signature_markdown(rapor))
    _yaz_rapor(rapor.to_dict(), out, markdown, signature_markdown(rapor))


def _signature_gate(yol=None, profile="standard", seeds=None,
                    out=None, markdown=None):
    """Signature Benchmark release gate."""
    from hga.evaluation.signature import (
        run_signature_benchmark,
        run_signature_release_gate,
        signature_release_gate_markdown,
    )
    if yol:
        rapor_dict = json.loads(Path(yol).read_text(encoding="utf-8"))
    else:
        tohumlar = [int(v) for v in (seeds or "1").split(",") if v.strip()]
        rapor_dict = run_signature_benchmark(
            profile=profile, seeds=tohumlar).to_dict()
    gate = run_signature_release_gate(rapor_dict)
    md = signature_release_gate_markdown(gate)
    print(md)
    _yaz_rapor(gate.to_dict(), out, markdown, md)


def _verifier_ensemble(out=None, markdown=None):
    """Çoklu verifier ensemble benchmarkı."""
    from hga.evaluation.verifier_ensemble import (
        run_verifier_ensemble_benchmark,
        verifier_ensemble_markdown,
    )
    rapor = run_verifier_ensemble_benchmark()
    md = verifier_ensemble_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)


def _verifier_adversarial(out=None, markdown=None):
    """Genişletilmiş verifier adversarial suite v2."""
    from hga.evaluation.verifier_adversarial import (
        run_verifier_adversarial_v2_benchmark,
    )
    rapor = run_verifier_adversarial_v2_benchmark()
    md = rapor.markdown()
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)


def _semantik(out=None, markdown=None):
    """P0-5: Türkçe semantik çıkarım v2 gold benchmarkı."""
    from hga.evaluation.semantic_extraction import (
        run_semantic_extraction_v2_benchmark,
        semantic_extraction_markdown,
    )
    rapor = run_semantic_extraction_v2_benchmark()
    print(semantic_extraction_markdown(rapor))
    _yaz_rapor(rapor.to_dict(), out, markdown,
               semantic_extraction_markdown(rapor))


def _genelleme_v2(out=None, markdown=None):
    """P0-6: ham metinden keşif + kompozisyon (C_G v2)."""
    from hga.evaluation.compositional_v2 import (
        compositional_v2_markdown,
        run_compositional_v2_benchmark,
    )
    rapor = run_compositional_v2_benchmark()
    print(compositional_v2_markdown(rapor))
    _yaz_rapor(rapor.to_dict(), out, markdown, compositional_v2_markdown(rapor))


def _twt_sonuc(seeds=None, profile="smoke", out=None, markdown=None):
    """P0-3: TWT gerçek sonuç tablosu (çoklu tohum + FLOPs)."""
    from hga.evaluation.twt_results import results_markdown, run_twt_results
    tohumlar = [int(v) for v in (seeds or "1,2,3").split(",") if v.strip()]
    rapor = run_twt_results(seeds=tohumlar, profile=profile)
    md = results_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)


def _english_ewt(seeds=None, profile="smoke", out=None, markdown=None):
    """Pinned UD English EWT üzerinde Transformer/BERT/GPT-style kontroller."""
    from hga.evaluation.english_ewt import english_ewt_markdown, run_english_ewt_baselines

    tohumlar = [int(v) for v in (seeds or "1,2,3,4,5").split(",") if v.strip()]
    rapor = run_english_ewt_baselines(seeds=tohumlar, profile=profile)
    md = english_ewt_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)


def _english_scaling(seeds=None, profile="smoke", out=None, markdown=None):
    """HGA'nın gerçek English EWT üzerinde küçük parameter-scaling probe'u."""
    from hga.evaluation.english_ewt import (
        english_hga_scaling_markdown,
        run_english_hga_scaling_probe,
    )

    tohumlar = [int(v) for v in (seeds or "1,2,3,4,5").split(",") if v.strip()]
    rapor = run_english_hga_scaling_probe(seeds=tohumlar, profile=profile)
    md = english_hga_scaling_markdown(rapor)
    print(md)
    _yaz_rapor(rapor, out, markdown, md)


def _turkce_lm(seeds=None, profile="smoke", steps=None, out=None,
               markdown=None):
    """P1: Gerçek Türkçe LM (smoke: TWT; full: tr_corpus_v1 1.11M kelime)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print("turkce-lm için torch gerekli. Örn: .venv/bin/python -m hga turkce-lm")
        return
    from hga.evaluation.turkish_lm import (
        run_turkish_lm_benchmark,
        turkish_lm_markdown,
    )
    tohumlar = ([int(v) for v in seeds.split(",") if v.strip()]
                if seeds else None)
    # CLI --signature-profile {smoke,standard} → LM profilleri {smoke,full}
    lm_profil = "smoke" if profile == "smoke" else "full"
    # --steps yalnız CI/smoke hızlandırmasıdır; override hash'e dahildir,
    # yani kısaltılmış bir koşu tam koşunun imzasını taklit edemez.
    ezmeler = {"steps": int(steps)} if steps else None
    rapor = run_turkish_lm_benchmark(profile=lm_profil, seeds=tohumlar,
                                     overrides=ezmeler)
    md = turkish_lm_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)


def _uzun_baglam(profile="smoke", seeds=None, out=None, markdown=None):
    """P1: Uzun bağlam LM taraması (opsiyonel 512/1024 smoke)."""
    from hga.evaluation.long_context import (
        long_context_markdown,
        run_long_context_benchmark,
    )
    tohumlar = ([int(v) for v in seeds.split(",") if v.strip()]
                if seeds else None)
    rapor = run_long_context_benchmark(profile=profile, seeds=tohumlar)
    md = long_context_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)


def _bellek_hiyerarsi(profile="smoke", seeds=None, out=None, markdown=None):
    """P0-8: Hot/Warm/Cold/Archive hiyerarşik bellek benchmarkı."""
    from hga.evaluation.memory_hierarchy import (
        memory_hierarchy_markdown,
        run_memory_hierarchy_benchmark,
    )
    tohum = int((seeds or "1").split(",")[0])
    rapor = run_memory_hierarchy_benchmark(profile=profile, seed=tohum)
    md = memory_hierarchy_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)


def _bellek_streaming(target_records=100_000_000, sample_records=10_000,
                      shard_records=1_000_000, checkpoint_interval=1_000,
                      seeds=None, out=None, markdown=None):
    """P0-8: 100M-ready streaming/checkpoint dry-run harness."""
    from hga.evaluation.memory_hierarchy import (
        memory_streaming_harness_markdown,
        run_memory_streaming_harness,
    )
    tohum = int((seeds or "1").split(",")[0])
    rapor = run_memory_streaming_harness(
        target_records=int(target_records),
        sample_records=int(sample_records),
        shard_records=int(shard_records),
        checkpoint_interval=int(checkpoint_interval),
        seed=tohum,
    )
    md = memory_streaming_harness_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)


def _cok_ortam(seeds=None, out=None, markdown=None):
    """P1: Çoklu ortam + cross-domain verifier izolasyonu."""
    from hga.evaluation.multi_environment import (
        multi_environment_markdown,
        run_multi_environment_benchmark,
    )
    tohumlar = [int(v) for v in (seeds or "1,2,3,4,5").split(",") if v.strip()]
    rapor = run_multi_environment_benchmark(seeds=tohumlar)
    md = multi_environment_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)


def _ogrenme_transfer(seeds=None, source_examples=120,
                       target_support_examples=40,
                       target_unseen_examples=40,
                       out=None, markdown=None):
    """P1: Cross-domain self-learning transfer negatif-kontrol benchmarkı."""
    from hga.evaluation.self_learning_transfer import (
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


def _ogrenme_olcek(profile="smoke", seeds=None, out=None, markdown=None):
    """P1: Self-learning ölçeklendirme (100 → 1000 → 3000 cycle)."""
    from hga.evaluation.self_learning_scaling import (
        run_self_learning_scaling,
        self_learning_scaling_markdown,
    )
    tohumlar = [int(v) for v in (seeds or "1").split(",") if v.strip()]
    rapor = run_self_learning_scaling(profile=profile, seeds=tohumlar)
    md = self_learning_scaling_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)


def _tohum_istatistik(profile="smoke", seeds=None, out=None, markdown=None):
    """P1: Çekirdek protokollerde 20 tohum + istatistiksel çıkarım."""
    from hga.evaluation.seed_statistics import (
        run_core_seed_statistics,
        seed_statistics_markdown,
    )
    tohumlar = ([int(v) for v in seeds.split(",") if v.strip()]
                if seeds else None)
    rapor = run_core_seed_statistics(profile=profile, seeds=tohumlar)
    md = seed_statistics_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)


def _insan_degerlendirme_raporu(package_dir=None):
    """Varsa tamamlanmış puanları, yoksa yalnız protokolü yükle."""
    from hga.evaluation.human_eval_fill import build_human_evaluation_from_csv
    from hga.evaluation.human_evaluation import build_human_evaluation_protocol

    paket_dizini = Path(package_dir or "insan_degerlendirme_paketleri")
    try:
        return build_human_evaluation_from_csv(paket_dizini)
    except ValueError:
        return build_human_evaluation_protocol()


def _insan_degerlendirme(package_dir=None, out=None, markdown=None):
    """P1: İnsan değerlendirme protokolü + Krippendorff alfa aracı."""
    from hga.evaluation.human_evaluation import human_evaluation_markdown

    rapor = _insan_degerlendirme_raporu(package_dir)
    md = human_evaluation_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)


def _insan_import(package_dir=None, out=None, markdown=None):
    """P1: İnsan puanı CSV import + α + kol agregasyonu."""
    from hga.evaluation.human_eval_fill import (
        human_rating_import_markdown,
        human_rating_import_report,
    )
    paket_dizini = Path(package_dir or "insan_degerlendirme_paketleri")
    rapor = human_rating_import_report(paket_dizini)
    md = human_rating_import_markdown(rapor)
    print(md)
    _yaz_rapor(rapor, out, markdown, md)


def _korpus_genisletme(candidate_dir=None, target_words=2_000_000,
                       use_smoke_candidates=True, out=None, markdown=None):
    """P1: Türkçe korpus genişletme pipeline/release gate."""
    from hga.evaluation.turkish_corpus_expansion import (
        run_turkish_corpus_expansion_pipeline,
        turkish_corpus_expansion_markdown,
    )
    rapor = run_turkish_corpus_expansion_pipeline(
        Path(candidate_dir) if candidate_dir else None,
        use_smoke_candidates=bool(use_smoke_candidates),
        target_words=int(target_words),
    )
    md = turkish_corpus_expansion_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)


def _derinlik_teshis(profile="smoke", out=None, markdown=None):
    """Çıkarım derinliği çöküşünün kök neden teşhisi (bellek mi, çıkarım mı)."""
    from hga.evaluation.depth_diagnosis import (
        depth_diagnosis_markdown,
        diagnose_depth_collapse,
    )
    rapor = diagnose_depth_collapse(profile=profile)
    md = depth_diagnosis_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)


def _oncelik_optimizasyon(profile="smoke", out=None, markdown=None):
    """Priority(E) ağırlık araması + held-out doğrulama."""
    from hga.evaluation.priority_optimization import (
        optimize_priority_weights,
        priority_optimization_markdown,
    )
    rapor = optimize_priority_weights(profile=profile)
    md = priority_optimization_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)


def _muhendislik(out=None, markdown=None):
    """Mühendislik sözleşmesi denetimi (CI / paketleme / test)."""
    from hga.evaluation.engineering_audit import (
        audit_engineering,
        engineering_markdown,
    )
    rapor = audit_engineering()
    md = engineering_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)


def _yeniden_uretilebilirlik(out=None, markdown=None):
    """Yeniden üretilebilirlik denetimi (manifest / determinizm / tohum)."""
    from hga.evaluation.compositional_v2 import run_compositional_v2_benchmark
    from hga.evaluation.engineering_audit import (
        audit_reproducibility,
        reproducibility_markdown,
    )
    from hga.evaluation.multi_environment import (
        run_multi_environment_benchmark,
    )
    from hga.evaluation.semantic_extraction import (
        run_semantic_extraction_v2_benchmark,
    )
    from hga.evaluation.verifier_ensemble import run_verifier_ensemble_benchmark
    raporlar = {
        "compositional_v2": run_compositional_v2_benchmark().to_dict(),
        "semantic_extraction": run_semantic_extraction_v2_benchmark().to_dict(),
        "multi_environment": run_multi_environment_benchmark(
            seeds=tuple(range(1, 21))).to_dict(),
        "verifier_adversarial": run_verifier_ensemble_benchmark().to_dict(),
    }
    rapor = audit_reproducibility(raporlar)
    md = reproducibility_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)


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


def _observability_demo(out=None, markdown=None, html_yol=None):
    """Torch gerektirmeyen gözlemlenebilirlik demoları."""
    from hga.knowledge import DeneyimDurumu, ExperienceCandidate
    from hga.memory import DeneyimSlotlari
    from hga.observability import gozlem_paneli_kaydet, gozlem_paneli_olustur
    adaylar = [
        ExperienceCandidate("X1", "E1", "R1", "E2", state=DeneyimDurumu.VALID),
        ExperienceCandidate("X2", "E1", "R1", "E3", state=DeneyimDurumu.INVALID),
        ExperienceCandidate("X3", "E1", "R1", "E4", state=DeneyimDurumu.CONFLICT),
    ]
    slot = DeneyimSlotlari(slot_sayisi=16)
    slot.yaz("X1", ("E1", "R1", "E2"))
    slot.yaz("X3", ("E1", "R1", "E4"))
    panel = gozlem_paneli_olustur(adaylar, slot, [([1, 0], [1, 0]), ([0, 1], [1, 0])])
    yollar = gozlem_paneli_kaydet(panel, json_yol=out, markdown_yol=markdown, html_yol=html_yol)
    if yollar:
        for tur, yol in yollar.items():
            print(f"Gözlem paneli ({tur}) yazıldı: {yol}")
    else:
        print("Deneyim akışı:", panel["deneyim_akisi"])
        print("Bellek doluluk haritası:")
        print(panel["bellek_ascii"])
        print("Katman benzerliği:", panel["geometri"])


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m hga",
                                description="HGA Experience Engine CLI")
    p.add_argument("komut", choices=["bilgi", "gercek-veri", "benchmark",
                                     "dogrulama", "halusinasyon", "sweep",
                                     "tokenizer", "perplexity", "checkpoint-rapor",
                                     "benchmark-rapor", "veri-kalite", "veri-canli-smoke",
                                     "manifest", "observability", "ozet",
                                     "graf", "kesif", "golden-benchmark",
                                     "memory-benchmark", "kronecker-benchmark",
                                     "self-learning-benchmark", "research-benchmark", "milestone",
                                     "kapasite", "bilgi-surum", "defter",
                                     "memory-interference", "paradigma",
                                     "olcekli-golden", "kronecker-rank",
                                     "epistemik", "verim", "cok-adimli",
                                     "koken", "oncelik",
                                     "oncelik-zincir", "operator-baseline",
                                     "cikarim-derinligi", "signature",
                                     "signature-gate",
                                     "verifier-adversarial",
                                     "verifier-ensemble",
                                     "semantik", "genelleme-v2",
                                     "twt-sonuc", "english-ewt", "english-scaling", "turkce-lm",
                                     "uzun-baglam",
                                     "bellek-hiyerarsi",
                                     "bellek-streaming",
                                     "cok-ortam", "ogrenme-transfer",
                                     "ogrenme-olcek",
                                     "tohum-istatistik",
                                     "insan-degerlendirme",
                                     "insan-import", "korpus-genisletme",
                                     "derinlik-teshis",
                                     "oncelik-optimizasyon",
                                     "muhendislik",
                                     "yeniden-uretilebilirlik",
                                     "championship-benchmark"])
    p.add_argument("yol", nargs="?", default=None,
                   help="dosya yolu: ozet/veri-kalite/manifest/perplexity/checkpoint-rapor")
    p.add_argument("--config", default=None,
                   help="checkpoint-rapor için model/eğitim YAML config yolu")
    p.add_argument("--checkpoint", default=None,
                   help="perplexity için strict yüklenecek model checkpoint yolu")
    p.add_argument("--tokenizer-yol", default=None,
                   help="perplexity için BPE sözlük yolu (varsayılan: bpe_sozluk.json)")
    p.add_argument("--tiny", action="store_true",
                   help="perplexity smoke için küçük model kullan (varsayılan; --full-config bunu kapatır)")
    p.add_argument("--full-config", action="store_true",
                   help="perplexity için tiny yerine model_config.yaml boyutlarını kullan")
    p.add_argument("--batch", type=int, default=64,
                   help="perplexity batch boyutu")
    p.add_argument("--n", type=int, default=None,
                   help="perplexity tiny/full model n override")
    p.add_argument("--katman", type=int, default=None,
                   help="perplexity K override")
    p.add_argument("--baglam", type=int, default=None,
                   help="perplexity bağlam override")
    p.add_argument("--vocab", type=int, default=None,
                   help="perplexity/checkpoint model vocab override")
    p.add_argument("--emb", type=int, default=None,
                   help="checkpoint-rapor için emb_dim override")
    p.add_argument("--heads", type=int, default=None,
                   help="checkpoint-rapor için attention head override")
    p.add_argument("--seyrek-satir", type=int, default=None,
                   help="checkpoint-rapor için seyrek tablo satır override")
    p.add_argument("--seyrek-boyut", type=int, default=None,
                   help="checkpoint-rapor için seyrek vektör boyutu override")
    p.add_argument("--no-strict", action="store_true",
                   help="checkpoint-rapor için eksik/fazla anahtarı hata sayma")
    p.add_argument("--out", default=None,
                   help="rapor üreten komutlar için JSON çıktı yolu")
    p.add_argument("--markdown", default=None,
                   help="rapor üreten komutlar için Markdown çıktı yolu")
    p.add_argument("--html", default=None,
                   help="observability/research-benchmark için HTML çıktı yolu")
    p.add_argument("--konular", default=None,
                   help="veri-canli-smoke için virgüllü Wikipedia konu listesi")
    p.add_argument("--cikis", default=None,
                   help="veri-canli-smoke için temizlenmiş küçük corpus çıktı yolu")
    p.add_argument("--kontrollu", action="store_true",
                   help="veri-canli-smoke için ağsız/deterministik fetcher kullan")
    p.add_argument("--seeds", default=None,
                   help="çoklu-seed benchmarklar için virgüllü liste (örn. 1,2,3,4,5)")
    p.add_argument("--profile", choices=["smoke", "full"], default="smoke",
                   help="research-benchmark çalışma profili")
    p.add_argument("--sections", default=None,
                   help="research-benchmark bölüm filtresi (virgüllü; varsayılan: tümü)")
    p.add_argument("--strict", action="store_true",
                   help="research-benchmark bölüm atlanır/hata verirse non-zero çık")
    p.add_argument("--n-values", default="4,8,16",
                   help="kronecker-rank için n değerleri (virgüllü)")
    p.add_argument("--k-values", default="1,2,4",
                   help="kronecker-rank için K değerleri (virgüllü)")
    p.add_argument("--sizes", default="100,1000,10000",
                   help="olcekli-golden test seti boyutları (virgüllü)")
    p.add_argument("--hard", action="store_true",
                   help="olcekli-golden: sınır vakalarını (eşik/öncelik/kaynak) ekle")
    p.add_argument("--epochs", type=int, default=60,
                   help="paradigma nöral kolu eğitim epoch sayısı")
    p.add_argument("--experiment-root", default="experiments",
                   help="EXP-NNNN çalışma dizinlerinin kökü")
    p.add_argument("--scales", default="1000,10000,100000",
                   help="memory-benchmark context ölçekleri")
    p.add_argument("--slots", type=int, default=65536,
                   help="memory-benchmark için tablo başına slot sayısı")
    p.add_argument("--tables", default="1,2",
                   help="memory-benchmark tablo sayıları (1,2 veya ikisi)")
    p.add_argument("--hops", default=None,
                   help="cok-adimli: zincir derinlikleri (örn 1,2,3,4,5)")
    p.add_argument("--distractors", default=None,
                   help="cok-adimli: araya giren dolgu olgu sayıları (örn 0,16,64,256)")
    p.add_argument("--active-dynamic", action="store_true",
                   help="memory-benchmark: aynı streamde aktif bounded Dynamic KV'yi de kır")
    p.add_argument("--audit-samples", type=int, default=256,
                   help="aktif memory stress checkpoint exact audit örnek sayısı")
    p.add_argument("--steps", type=int, default=100,
                   help="kronecker-benchmark optimizasyon adımı")
    p.add_argument("--lm-steps", type=int, default=None,
                   help="turkce-lm eğitim adımı override (yalnız smoke/CI "
                        "hızlandırması; hash'e dahildir)")
    p.add_argument("--device", default="cpu",
                   help="kronecker-benchmark torch cihazı (cpu/cuda)")
    p.add_argument("--cycles", type=int, default=100,
                   help="self-learning/collapse döngü sayısı")
    p.add_argument("--initial-facts", type=int, default=100,
                   help="self-learning başlangıç doğrulanmış bilgi sayısı")
    p.add_argument("--operands-max", type=int, default=31,
                   help="self-learning aritmetik domain üst operandı")
    p.add_argument("--negatives-per-fact", type=int, default=7,
                   help="her doğru frontier olgusu başına yanlış aday")
    p.add_argument("--memory-slots", type=int, default=4096,
                   help="self-learning/milestone deney belleği slot sayısı")
    p.add_argument("--checkpoints", default=None,
                   help="milestone tablosu kontrol noktaları (örn. 0,10,50,100)")
    p.add_argument("--forced-collisions", type=int, default=25,
                   help="memory-interference için kasıtlı çakışma sayısı")
    p.add_argument("--signature-profile", choices=["smoke", "standard"],
                   default="smoke",
                   help="signature / championship-benchmark ölçek profili")
    p.add_argument("--depth-profile", choices=["smoke", "standard", "deep"],
                   default="standard",
                   help="cikarim-derinligi tarama profili")
    p.add_argument("--long-context-profile", choices=["smoke", "smoke_1024", "full"],
                   default="smoke",
                   help="uzun-baglam profili; smoke_1024 512/1024 yolunu koşar")
    p.add_argument("--reliability-threshold", type=float, default=1.0,
                   help="C_R/C_RD güvenilirlik eşiği (0,1]")
    p.add_argument("--skip-torch", action="store_true",
                   help="championship-benchmark: nöral bölümleri atla")
    p.add_argument("--ledger", default=None,
                   help="milestone için immutable experience ledger JSONL yolu")
    p.add_argument("--target-records", type=int, default=100_000_000,
                   help="bellek-streaming için hedef kayıt sayısı")
    p.add_argument("--sample-records", type=int, default=10_000,
                   help="bellek-streaming smoke koşusunda gerçek yazılacak kayıt")
    p.add_argument("--shard-records", type=int, default=1_000_000,
                   help="bellek-streaming planında shard başına kayıt")
    p.add_argument("--checkpoint-interval", type=int, default=1_000,
                   help="bellek-streaming checkpoint aralığı")
    p.add_argument("--source-examples", type=int, default=120,
                   help="ogrenme-transfer için kaynak ortam örnek sayısı")
    p.add_argument("--target-support-examples", type=int, default=40,
                   help="ogrenme-transfer için hedef destek örneği")
    p.add_argument("--target-unseen-examples", type=int, default=40,
                   help="ogrenme-transfer için hedef unseen eval örneği")
    p.add_argument("--package-dir", default="insan_degerlendirme_paketleri",
                   help="insan-import/degerlendirme paket dizini")
    p.add_argument("--candidate-dir", default=None,
                   help="korpus-genisletme için aday JSONL dizini")
    p.add_argument("--no-smoke-candidates", action="store_true",
                   help="korpus-genisletme: aday dizini yoksa gömülü smoke aday kullanma")
    p.add_argument("--target-words", type=int, default=2_000_000,
                   help="korpus-genisletme release hedef kelime sayısı")
    args = p.parse_args(argv)
    {"bilgi": _bilgi_demo, "gercek-veri": _gercek_veri,
     "benchmark": _benchmark, "dogrulama": _dogrulama,
     "halusinasyon": _halusinasyon, "sweep": _sweep,
     "tokenizer": _tokenizer_benchmark,
     "graf": _graf_demo, "kesif": _kesif_demo,
     "perplexity": lambda: _perplexity_benchmark(
         args.yol, checkpoint=args.checkpoint,
         tokenizer_yol=args.tokenizer_yol,
         tiny=(args.tiny or not args.full_config), batch_size=args.batch,
         n=args.n, katman=args.katman, baglam=args.baglam, vocab=args.vocab),
     "checkpoint-rapor": lambda: _checkpoint_rapor(
         args.yol, config_yol=args.config, strict=not args.no_strict,
         n=args.n, katman=args.katman, baglam=args.baglam, vocab=args.vocab,
         emb=args.emb, heads=args.heads, seyrek_satir=args.seyrek_satir,
         seyrek_boyut=args.seyrek_boyut),
     "benchmark-rapor": lambda: _benchmark_rapor(
         out=args.out, markdown=args.markdown, checkpoint=args.checkpoint,
         tokenizer_yol=args.tokenizer_yol, n=args.n, katman=args.katman,
         baglam=args.baglam, vocab=args.vocab),
     "veri-kalite": lambda: _veri_kalite(args.yol),
     "veri-canli-smoke": lambda: _veri_canli_smoke(
         konular=args.konular, cikis=args.cikis, out=args.out,
         kontrollu=args.kontrollu),
     "manifest": lambda: _manifest(args.yol),
     "golden-benchmark": lambda: _golden_benchmark(
         args.out, seeds=args.seeds, experiment_root=args.experiment_root),
     "memory-benchmark": lambda: _memory_benchmark(
         args.scales, args.slots, args.tables, args.seeds or "42",
         args.experiment_root, out=args.out, markdown=args.markdown,
         active_dynamic=args.active_dynamic, audit_samples=args.audit_samples),
     "kronecker-benchmark": lambda: _kronecker_benchmark(
         int(args.n or 16), args.steps, args.batch, args.seeds or "1,2,3,4,5",
         args.device, args.experiment_root, out=args.out),
     "self-learning-benchmark": lambda: _self_learning_benchmark(
         args.cycles, args.batch, args.initial_facts, args.operands_max,
         args.negatives_per_fact, args.memory_slots, args.seeds or "42",
         args.experiment_root, out=args.out),
     "research-benchmark": lambda: _research_benchmark(
         args.profile, args.seeds or "1,2,3,4,5", args.experiment_root,
         sections=args.sections, out=args.out, markdown=args.markdown,
         html_yol=args.html, strict=args.strict),
     "milestone": lambda: _milestone(
         args.cycles, args.batch, args.initial_facts, args.operands_max,
         args.negatives_per_fact, args.memory_slots, args.seeds or "42",
         args.experiment_root, out=args.out, markdown=args.markdown,
         ledger=args.ledger, checkpoints=args.checkpoints),
     "memory-interference": lambda: _memory_interference(
         args.forced_collisions, args.slots, 1,
         args.scales if args.scales != "1000,10000,100000" else "100,1000,10000,100000",
         out=args.out, markdown=args.markdown),
     "paradigma": lambda: _paradigma(args.seeds or "1,2,3,4,5", out=args.out,
                                     markdown=args.markdown,
                                     epochs=args.epochs),
     "olcekli-golden": lambda: _olcekli_golden(
         args.sizes, args.seeds or "1,2,3,4,5", hard=args.hard,
         out=args.out, markdown=args.markdown),
     "kronecker-rank": lambda: _kronecker_rank(
         args.n_values, args.k_values, seed=1,
         out=args.out, markdown=args.markdown),
     "epistemik": lambda: _epistemik(args.seeds, out=args.out,
                                     markdown=args.markdown),
     "cok-adimli": lambda: _cok_adimli(args.seeds, hops=args.hops,
                                       distractors=args.distractors,
                                       slots=args.memory_slots,
                                       out=args.out, markdown=args.markdown),
     "verim": lambda: _verim(args.cycles, args.batch, args.initial_facts,
                             args.operands_max, args.negatives_per_fact,
                             args.seeds, out=args.out, markdown=args.markdown),
     "koken": lambda: _provenance(out=args.out, markdown=args.markdown),
     "oncelik": lambda: _priority(k=10, out=args.out, markdown=args.markdown),
     "oncelik-zincir": lambda: _oncelik_zincir(
         k=10, seeds=args.seeds, out=args.out, markdown=args.markdown),
     "operator-baseline": lambda: _operator_baseline(
         n=int(args.n or 8), steps=args.steps, seeds=args.seeds,
         device=args.device, out=args.out, markdown=args.markdown),
     "cikarim-derinligi": lambda: _cikarim_derinligi(
         profile=args.depth_profile, seeds=args.seeds,
         threshold=args.reliability_threshold, out=args.out,
         markdown=args.markdown),
     "signature": lambda: _signature(
         profile=args.signature_profile, seeds=args.seeds, out=args.out,
         markdown=args.markdown),
     "signature-gate": lambda: _signature_gate(
         yol=args.yol, profile=args.signature_profile, seeds=args.seeds,
         out=args.out, markdown=args.markdown),
     "verifier-adversarial": lambda: _verifier_adversarial(
         out=args.out, markdown=args.markdown),
     "verifier-ensemble": lambda: _verifier_ensemble(
         out=args.out, markdown=args.markdown),
     "semantik": lambda: _semantik(out=args.out, markdown=args.markdown),
     "genelleme-v2": lambda: _genelleme_v2(out=args.out, markdown=args.markdown),
     "twt-sonuc": lambda: _twt_sonuc(
         seeds=args.seeds, profile=args.signature_profile, out=args.out,
         markdown=args.markdown),
     "english-ewt": lambda: _english_ewt(
         seeds=args.seeds, profile=args.signature_profile, out=args.out,
         markdown=args.markdown),
     "english-scaling": lambda: _english_scaling(
         seeds=args.seeds, profile=args.signature_profile, out=args.out,
         markdown=args.markdown),
     "turkce-lm": lambda: _turkce_lm(
         seeds=args.seeds, profile=args.signature_profile,
         steps=args.lm_steps, out=args.out, markdown=args.markdown),
     "uzun-baglam": lambda: _uzun_baglam(
         profile=args.long_context_profile, seeds=args.seeds,
         out=args.out, markdown=args.markdown),
     "bellek-hiyerarsi": lambda: _bellek_hiyerarsi(
         profile=args.depth_profile, seeds=args.seeds, out=args.out,
         markdown=args.markdown),
     "bellek-streaming": lambda: _bellek_streaming(
         target_records=args.target_records,
         sample_records=args.sample_records,
         shard_records=args.shard_records,
         checkpoint_interval=args.checkpoint_interval,
         seeds=args.seeds, out=args.out, markdown=args.markdown),
     "cok-ortam": lambda: _cok_ortam(
         seeds=args.seeds, out=args.out, markdown=args.markdown),
     "ogrenme-transfer": lambda: _ogrenme_transfer(
         seeds=args.seeds, source_examples=args.source_examples,
         target_support_examples=args.target_support_examples,
         target_unseen_examples=args.target_unseen_examples,
         out=args.out, markdown=args.markdown),
     "ogrenme-olcek": lambda: _ogrenme_olcek(
         profile=args.signature_profile, seeds=args.seeds, out=args.out,
         markdown=args.markdown),
     "tohum-istatistik": lambda: _tohum_istatistik(
         profile=args.signature_profile, seeds=args.seeds, out=args.out,
         markdown=args.markdown),
     "insan-degerlendirme": lambda: _insan_degerlendirme(
         package_dir=args.package_dir, out=args.out, markdown=args.markdown),
     "insan-import": lambda: _insan_import(
         package_dir=args.package_dir, out=args.out, markdown=args.markdown),
     "korpus-genisletme": lambda: _korpus_genisletme(
         candidate_dir=args.candidate_dir, target_words=args.target_words,
         use_smoke_candidates=not args.no_smoke_candidates,
         out=args.out, markdown=args.markdown),
     "derinlik-teshis": lambda: _derinlik_teshis(
         profile=args.depth_profile, out=args.out, markdown=args.markdown),
     "oncelik-optimizasyon": lambda: _oncelik_optimizasyon(
         profile=args.signature_profile, out=args.out,
         markdown=args.markdown),
     "muhendislik": lambda: _muhendislik(out=args.out, markdown=args.markdown),
     "yeniden-uretilebilirlik": lambda: _yeniden_uretilebilirlik(
         out=args.out, markdown=args.markdown),
     "championship-benchmark": lambda: _championship(
         profile=args.signature_profile, seeds=args.seeds, out=args.out,
         markdown=args.markdown, skip_torch=args.skip_torch),
     "kapasite": lambda: _kapasite(args.operands_max, out=args.out),
     "bilgi-surum": lambda: _bilgi_surum(out=args.out),
     "defter": lambda: _defter(out=args.out),
     "observability": lambda: _observability_demo(
         out=args.out, markdown=args.markdown, html_yol=args.html),
     "ozet": lambda: _ozet(args.yol)}[args.komut]()


if __name__ == "__main__":
    main()
