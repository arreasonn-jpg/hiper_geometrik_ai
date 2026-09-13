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
        run_self_learning_experiment,
        run_self_training_collapse_test,
        run_verifier_fault_injection,
    )
    from hga.memory import run_memory_capacity_sweep

    seed_values = [int(value.strip()) for value in seeds.split(",") if value.strip()]
    config = {
        "benchmark": "self-learning-collapse-robustness-v2",
        "cycles": int(cycles), "batch_size": int(batch),
        "initial_facts": int(initial_facts), "operands_max": int(operands_max),
        "negatives_per_fact": int(negatives_per_fact),
        "memory_slots": int(memory_slots),
        "verifier_fault_rates": {"false_acceptance": 0.25, "false_rejection": 0.25},
        "memory_recall_target": 0.95,
    }
    dataset_hash = canonical_hash({
        "generator": "arithmetic-frontier-holdout-v2",
        "operands_max": int(operands_max),
        "negatives_per_fact": int(negatives_per_fact),
        "test_holdout_fraction": 0.10,
    })

    def run_one(seed):
        expansion = run_self_learning_experiment(
            cycles=cycles, batch_size=batch, initial_facts=initial_facts,
            operands_max=operands_max, negatives_per_fact=negatives_per_fact,
            seed=seed, memory_slots=memory_slots,
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
            "collapse_probe": collapse.to_dict(),
            "verifier_robustness": robustness.to_dict(),
            "memory_capacity": capacity.to_dict(),
        }

    report = run_seed_sweep(
        run_one, seeds=seed_values, root=experiment_root, config=config,
        dataset_hash=dataset_hash,
        parameters={
            "ground_truth": "independent-arithmetic-environment",
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


def _memory_benchmark(scales, slots, tables, seeds, experiment_root, out=None):
    from hga.evaluation import canonical_hash, run_seed_sweep
    from hga.memory import run_memory_stress

    context_counts = [int(value.strip()) for value in scales.split(",") if value.strip()]
    table_counts = [int(value.strip()) for value in tables.split(",") if value.strip()]
    seed_values = [int(value.strip()) for value in seeds.split(",") if value.strip()]
    config = {
        "benchmark": "sparse-memory-collision-v1",
        "context_counts": context_counts,
        "slot_count": int(slots),
        "table_counts": table_counts,
        "collision_sample_limit": 1000,
    }
    dataset_hash = canonical_hash({
        "generator": "hga-memory-context-v1",
        "context_counts": context_counts,
    })

    def run_one(seed):
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
        parameters={"synthetic": True, "streaming_context_generation": True},
    ).to_dict()
    print("Sparse memory benchmark özeti:")
    print(f"  experiments: {', '.join(report['experiment_ids'])}")
    for metric, values in report["aggregate"].items():
        if metric.endswith(("collision_event_rate", "retrieval_accuracy", "interference_rate")):
            print(f"  {metric:<48} {values['mean']:.6f} ± {values['std']:.6f}")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"  report: {out}")


def _ozet(yol):
    from hga.knowledge import KnowledgeStore
    if yol:
        k = KnowledgeStore.yukle(yol)
    else:
        k = KnowledgeStore()
    print("Bilgi tabanı özeti:")
    for k, v in k.ozet().items():
        print(f"  {k:<10}: {v}")


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
    from hga.evaluation import mini_turkce_corpus, tokenizer_kapsami
    from hga.config import model_config_yukle
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
                          n: int = None, katman: int = None,
                          baglam: int = None, vocab: int = None):
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
                      n: int = None, katman: int = None, baglam: int = None,
                      vocab: int = None, emb: int = None, heads: int = None,
                      seyrek_satir: int = None, seyrek_boyut: int = None):
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

    from egitim.saglamlik import checkpoint_uyumluluk_raporu
    from hga.config import model_config_yukle
    from kuresel_model import model_olustur

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
    from hga.experience import AritmetikOrtam, ExperienceEvaluator, ExperienceGenerator, aritmetik_etki_alani
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
    from hga.evaluation import benchmark_raporu_olustur, benchmark_raporu_kaydet
    rapor = benchmark_raporu_olustur(checkpoint=checkpoint, tokenizer_yol=tokenizer_yol,
                                     n=int(n or 8), katman=int(katman or 1),
                                     baglam=int(baglam or 8), vocab=int(vocab or 256))
    yollar = benchmark_raporu_kaydet(rapor, json_yol=out, markdown_yol=markdown)
    if yollar:
        for tur, yol in yollar.items():
            print(f"Benchmark raporu ({tur}) yazıldı: {yol}")
    else:
        print(json.dumps(rapor, ensure_ascii=False, indent=2, sort_keys=True))


def _observability_demo(out=None, markdown=None, html_yol=None):
    """Torch gerektirmeyen gözlemlenebilirlik demoları."""
    from hga.knowledge import ExperienceCandidate, DeneyimDurumu
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
                                     "self-learning-benchmark"])
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
                   help="benchmark-rapor, veri-canli-smoke veya observability için JSON çıktı yolu")
    p.add_argument("--markdown", default=None,
                   help="benchmark-rapor/observability için Markdown çıktı yolu")
    p.add_argument("--html", default=None,
                   help="observability için tek dosya HTML panel yolu")
    p.add_argument("--konular", default=None,
                   help="veri-canli-smoke için virgüllü Wikipedia konu listesi")
    p.add_argument("--cikis", default=None,
                   help="veri-canli-smoke için temizlenmiş küçük corpus çıktı yolu")
    p.add_argument("--kontrollu", action="store_true",
                   help="veri-canli-smoke için ağsız/deterministik fetcher kullan")
    p.add_argument("--seeds", default=None,
                   help="golden-benchmark için virgüllü seed listesi (örn. 1,2,3,4,5)")
    p.add_argument("--experiment-root", default="experiments",
                   help="EXP-NNNN çalışma dizinlerinin kökü")
    p.add_argument("--scales", default="1000,10000,100000",
                   help="memory-benchmark context ölçekleri")
    p.add_argument("--slots", type=int, default=65536,
                   help="memory-benchmark için tablo başına slot sayısı")
    p.add_argument("--tables", default="1,2",
                   help="memory-benchmark tablo sayıları (1,2 veya ikisi)")
    p.add_argument("--steps", type=int, default=100,
                   help="kronecker-benchmark optimizasyon adımı")
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
                   help="self-learning deney belleği slot sayısı")
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
         args.experiment_root, out=args.out),
     "kronecker-benchmark": lambda: _kronecker_benchmark(
         int(args.n or 16), args.steps, args.batch, args.seeds or "1,2,3,4,5",
         args.device, args.experiment_root, out=args.out),
     "self-learning-benchmark": lambda: _self_learning_benchmark(
         args.cycles, args.batch, args.initial_facts, args.operands_max,
         args.negatives_per_fact, args.memory_slots, args.seeds or "42",
         args.experiment_root, out=args.out),
     "observability": lambda: _observability_demo(
         out=args.out, markdown=args.markdown, html_yol=args.html),
     "ozet": lambda: _ozet(args.yol)}[args.komut]()


if __name__ == "__main__":
    main()
