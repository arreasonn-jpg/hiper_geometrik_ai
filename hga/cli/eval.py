"""HGA CLI komutu (Faz 2 refactor)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from hga.cli.common import _toplu_sinirlar, _yaz_rapor


def _yeniden_uretilebilirlik(out=None, markdown=None):
    """Yeniden üretilebilirlik denetimi (manifest / determinizm / tohum)."""
    from hga.experiments.compositional_v2 import run_compositional_v2_benchmark
    from hga.experiments.engineering_audit import (
        audit_reproducibility,
        reproducibility_markdown,
    )
    from hga.experiments.multi_environment import (
        run_multi_environment_benchmark,
    )
    from hga.experiments.semantic_extraction import (
        run_semantic_extraction_v2_benchmark,
    )
    from hga.experiments.verifier_ensemble import run_verifier_ensemble_benchmark
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

def _muhendislik(out=None, markdown=None):
    """Mühendislik sözleşmesi denetimi (CI / paketleme / test)."""
    from hga.experiments.engineering_audit import (
        audit_engineering,
        engineering_markdown,
    )
    rapor = audit_engineering()
    md = engineering_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)

def _oncelik_optimizasyon(profile="smoke", out=None, markdown=None):
    """Priority(E) ağırlık araması + held-out doğrulama."""
    from hga.experiments.priority_optimization import (
        optimize_priority_weights,
        priority_optimization_markdown,
    )
    rapor = optimize_priority_weights(profile=profile)
    md = priority_optimization_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)

def _derinlik_teshis(profile="smoke", out=None, markdown=None):
    """Çıkarım derinliği çöküşünün kök neden teşhisi (bellek mi, çıkarım mı)."""
    from hga.experiments.depth_diagnosis import (
        depth_diagnosis_markdown,
        diagnose_depth_collapse,
    )
    rapor = diagnose_depth_collapse(profile=profile)
    md = depth_diagnosis_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)

def _korpus_genisletme(candidate_dir=None, target_words=2_000_000,
                       use_smoke_candidates=True, out=None, markdown=None):
    """P1: Türkçe korpus genişletme pipeline/release gate."""
    from hga.experiments.turkish_corpus_expansion import (
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

def _insan_import(package_dir=None, out=None, markdown=None):
    """P1: İnsan puanı CSV import + α + kol agregasyonu."""
    from hga.experiments.human_eval_fill import (
        human_rating_import_markdown,
        human_rating_import_report,
    )
    paket_dizini = Path(package_dir or "insan_degerlendirme_paketleri")
    rapor = human_rating_import_report(paket_dizini)
    md = human_rating_import_markdown(rapor)
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
    from hga.experiments.turkish_lm import (
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

def _reproduce_all(artifact_root="artifacts/reproduce-all", seeds=None, profile="smoke"):
    """Tek komutta manifestli CKPT-001 foundation artefaktlarını üret."""
    from hga.experiments.reproduce import reproduce_all_markdown, run_reproduce_all

    tohumlar = [int(value) for value in (seeds or "1,2,3,4,5").split(",") if value.strip()]
    rapor = run_reproduce_all(root=artifact_root, seeds=tohumlar, profile=profile)
    print(reproduce_all_markdown(rapor))
    print(f"artifact directory: {rapor['directory']}")

def _twt_sonuc(seeds=None, profile="smoke", out=None, markdown=None):
    """P0-3: TWT gerçek sonuç tablosu (çoklu tohum + FLOPs)."""
    from hga.experiments.twt_results import results_markdown, run_twt_results
    tohumlar = [int(v) for v in (seeds or "1,2,3").split(",") if v.strip()]
    rapor = run_twt_results(seeds=tohumlar, profile=profile)
    md = results_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)

def _oncelik_zincir(k=10, seeds=None, out=None, markdown=None):
    """P0-1: Priority(E) nedensel zincir ablasyonu."""
    from hga.experiments.priority_ablation import (
        priority_ablation_markdown,
        run_priority_weight_ablation,
    )
    tohumlar = [int(v) for v in (seeds or "1,2,3,4,5").split(",") if v.strip()]
    rapor = run_priority_weight_ablation(k=int(k), seeds=tohumlar)
    print(priority_ablation_markdown(rapor))
    _yaz_rapor(rapor.to_dict(), out, markdown, priority_ablation_markdown(rapor))

def _benchmark_rapor(out=None, markdown=None, checkpoint=None, tokenizer_yol=None,
                     n=None, katman=None, baglam=None, vocab=None):
    """Tokenizer/perplexity/halüsinasyon/bellek/cihaz tek rapor."""
    from hga.experiments import benchmark_raporu_kaydet, benchmark_raporu_olustur
    rapor = benchmark_raporu_olustur(checkpoint=checkpoint, tokenizer_yol=tokenizer_yol,
                                     n=int(n or 8), katman=int(katman or 1),
                                     baglam=int(baglam or 8), vocab=int(vocab or 256))
    yollar = benchmark_raporu_kaydet(rapor, json_yol=out, markdown_yol=markdown)
    if yollar:
        for tur, yol in yollar.items():
            print(f"Benchmark raporu ({tur}) yazıldı: {yol}")
    else:
        print(json.dumps(rapor, ensure_ascii=False, indent=2, sort_keys=True))

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

def _kapasite(operands_max, out=None):
    """P / C_I^UB / C_M^UB / C_E / C_V kapasite çerçevesi ölçümü."""
    from hga.experiments import run_capacity_benchmark
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

def _cok_adimli(seeds=None, hops=None, distractors=None, slots=4096,
                out=None, markdown=None):
    """P1-004/P1-006: çok adımlı çıkarım + uzun bağlam dayanıklılığı."""
    from hga.experiments.multi_hop import (
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

def _verim(cycles=20, batch=32, initial_facts=40, operands_max=15,
           negatives_per_fact=3, seeds=None, out=None, markdown=None):
    """P1-005: EY'yi NY / UEY / GY / VID eksenlerine ayır."""
    from hga.experience.verim import run_yield_experiment
    from hga.experiments.statistics import summarize_seed_metric

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

def _olcekli_golden(sizes, seeds, hard=False, out=None, markdown=None):
    """Faz 3/6: golden benchmark'ı 100/1.000/10.000 ölçeğine çıkar."""
    from hga.experiments.scaled_golden import run_scaled_golden_sweep

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

def _paradigma(seeds, out=None, markdown=None, epochs=60):
    """Faz 21: neural-only vs symbolic-only vs hybrid kontrollü ablasyon."""
    from hga.experiments.paradigma import run_paradigm_sweep

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

def _memory_benchmark(
    scales, slots, tables, seeds, experiment_root, out=None, markdown=None,
    active_dynamic=False, audit_samples=256,
):
    from hga.experiments import canonical_hash, run_seed_sweep
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

def _kronecker_benchmark(n, steps, batch, seeds, device, experiment_root, out=None):
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        raise SystemExit("kronecker-benchmark için PyTorch gerekli")

    from hga.experiments import canonical_hash, run_kronecker_dense_trial, run_seed_sweep

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

def _golden_benchmark(out=None, seeds=None, experiment_root="experiments"):
    from hga.experiments import run_golden_benchmark, run_golden_seed_sweep

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
