# -*- coding: utf-8 -*-
"""P0-3 — TWT gerçek sonuç tablosu: çoklu tohum, FLOPs, disjoint dilimler.

Bu modül yeni bir benchmark DEĞİLDİR. ``twt_baselines`` zaten dört mimariyi
aynı gerçek Türkçe veri (TWT v1) üzerinde eşit bütçeyle eğitiyor ve
Accuracy/F1/FAR/FRR/ECE/Brier/AURC/Coverage üretiyor. Eksik olan üç şey vardı
ve bu modül yalnız onları kapatır:

1. **FLOPs.** Parametre eşitliği hesaplanıyordu, işlem maliyeti değil. Bir
   modelin aynı parametreyle çok daha fazla çarpma yapması mümkündür; o zaman
   "eşit bütçe" iddiası yarım kalır. Burada her mimari için ileri geçiş
   çarpma-toplama sayısı ANALİTİK olarak türetilir (profil değil, kapalı form).
2. **Çoklu tohum.** Tek tohumlu bir tablo mimari farkını değil başlangıç
   şansını ölçebilir. Burada her model her tohumda yeniden eğitilir ve
   mean/std/%95 bootstrap CI raporlanır.
3. **Tek birleşik tablo.** Metrikler rapor içine dağılmıştı; kullanıcı
   isteği "tek gerçek sonuç tablosu" idi. ``results_markdown`` bunu üretir.

Ayrıca HGA ile en güçlü rakibi arasında eşleşmiş (paired) istatistiksel
karşılaştırma yapılır: aynı tohum aynı veriyi gördüğü için eşleşme geçerlidir.

Bu modülün üretmediği şey: dil modelleme iddiası. TWT görevi ikili arc
doğrulamadır (bağımlılık yayı doğru mu?), metin üretimi değildir.
"""
from __future__ import annotations

import hashlib
import json
import statistics as _stat
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .real_turkish import DIMENSIONS, prepare_real_turkish_task
from .statistics import compare_paired, summarize_seed_metric
from .twt_baselines import (
    ARCHITECTURE_CONFIG,
    MODEL_ORDER,
    run_twt_architecture_baselines,
)

PROTOCOL = "twt_real_results_v1"
SCHEMA_VERSION = 1

#: Tabloda raporlanan ana metrikler. "↑" yüksek iyi, "↓" düşük iyi.
HEADLINE_METRICS: Tuple[Tuple[str, str], ...] = (
    ("accuracy", "↑"),
    ("f1", "↑"),
    ("precision", "↑"),
    ("recall", "↑"),
    ("far", "↓"),
    ("frr", "↓"),
    ("coverage", "↑"),
)

#: Kalibrasyon metrikleri yalnız dev'de fit edilir, testte ölçülür.
CALIBRATION_METRICS: Tuple[Tuple[str, str], ...] = (
    ("ece", "↓"),
    ("brier", "↓"),
    ("nll", "↓"),
    ("aurc", "↓"),
)

#: Maliyet metrikleri — eşit bütçe iddiasının denetlendiği yer.
COST_FIELDS: Tuple[str, ...] = (
    "physical_parameters",
    "architecture_body_parameters",
    "parameter_bytes",
    "forward_flops_per_example",
    "training_seconds",
    "test_inference_seconds",
)


# ── FLOPs muhasebesi ────────────────────────────────────────────────────────
def analytic_forward_flops(model_name: str,
                           config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Bir örnek için ileri geçiş çarpma-toplama (MAC) sayısını türet.

    Neden analitik bir forma ihtiyaç var: profiler çıktısı donanıma ve
    çekirdek seçimine göre değişebilir; kapalı form ise mimarinin kendisine
    aittir, gözden geçirilebilir ve ``measured_forward_flops`` ile
    çapraz-denetlenir.

    Gömme (embedding) araması MAC değildir — tablo aramasıdır — bu yüzden
    sayıma girmez. LayerNorm, aktivasyon ve bias toplamaları da MAC sayılmaz;
    bunlar tüm mimarilerde aynı mertebede olduğu için oranı bozmaz.

    Raises:
        ValueError: bilinmeyen model adı.
    """
    if model_name not in MODEL_ORDER:
        raise ValueError(
            f"bilinmeyen model: {model_name!r}; beklenen: {', '.join(MODEL_ORDER)}")
    cfg = dict(ARCHITECTURE_CONFIG if config is None else config)
    emb = int(cfg["embedding_dim"])
    seq = int(cfg["sequence_length"])
    flat = emb * seq
    detay: Dict[str, int] = {}

    if model_name == "dense":
        gizli = int(cfg["dense_hidden_dim"])
        detay["linear_in"] = flat * gizli
        detay["linear_out"] = gizli * 2

    elif model_name == "transformer":
        basliklar = int(cfg["transformer_heads"])
        ff = int(cfg["transformer_feedforward_dim"])
        katman = int(cfg["transformer_layers"])
        basli_boyut = emb // max(1, basliklar)
        detay["qkv_projection"] = katman * 3 * seq * emb * emb
        detay["attention_output_projection"] = katman * seq * emb * emb
        # QKᵀ ve (skor)V: ikisi de head başına seq × seq × head_dim.
        detay["attention_scores"] = katman * basliklar * seq * seq * basli_boyut
        detay["attention_values"] = katman * basliklar * seq * seq * basli_boyut
        detay["feedforward"] = katman * 2 * seq * emb * ff
        detay["classifier"] = emb * 2

    elif model_name == "kronecker":
        n = int(cfg["kronecker_n"])
        katman = int(cfg["kronecker_layers"])
        detay["input_projection"] = flat * (n * n)
        # A @ X @ B: iki n×n×n kontraksiyon = 2n³. Tam operatör n⁴ OLURDU;
        # Kronecker yapısının hesap avantajı tam olarak buradadır.
        detay["kronecker_chain"] = katman * 2 * n ** 3
        detay["readout"] = (2 * n) * 2

    elif model_name == "hga":
        n = int(cfg["hga_n"])
        katman = int(cfg["hga_layers"])
        basliklar = int(cfg["transformer_heads"])
        basli_boyut = emb // max(1, basliklar)
        detay["attention_qkv"] = 3 * seq * emb * emb
        detay["attention_scores"] = basliklar * seq * seq * basli_boyut
        detay["attention_values"] = basliklar * seq * seq * basli_boyut
        detay["attention_output_projection"] = seq * emb * emb
        # Encoder iki projeksiyon üretir (u ve v), dış çarpım MAC değildir.
        detay["encoder_projections"] = 2 * flat * n
        detay["kronecker_chain"] = katman * 2 * n ** 3
        detay["decoder_lens"] = 2 * n ** 3
        detay["decoder_readout"] = (2 * n) * 2

    toplam = sum(detay.values())
    return {
        "model": model_name,
        "forward_flops_per_example": int(toplam),
        "breakdown": {k: int(v) for k, v in sorted(detay.items())},
        "embedding_lookups": int(seq),
        "note": ("MAC (çarpma-toplama) sayımıdır; FLOP olarak okunmak "
                 "istenirse 2× alınmalıdır. Geri geçiş kabaca 2× ileri "
                 "geçiştir. Gömme araması sayılmaz."),
    }


def measured_forward_flops(model_name: str,
                           vocabulary_size: int = 512) -> Dict[str, Any]:
    """Aynı MAC'i PyTorch'un kendi sayacıyla ölç (yer gerçeği).

    ``FlopCounterMode`` gerçekte çalışan çekirdekleri sayar, bu yüzden
    analitik formdaki bir hatayı yakalar. Bilinen bir sapma vardır:
    füzyonlu SDPA (scaled dot-product attention) çekirdeği kullanıldığında
    dikkat skoru matmul'ları sayılmaz. Bu yüzden dikkat içeren mimarilerde
    ölçülen değer analitikten biraz DÜŞÜK çıkar ve bu bir hata değildir.

    Raises:
        ImportError: PyTorch yoksa.
        ValueError: bilinmeyen model adı.
    """
    if model_name not in MODEL_ORDER:
        raise ValueError(f"bilinmeyen model: {model_name!r}")
    try:
        import torch
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError as hata:  # pragma: no cover - ortama bağlı
        raise ImportError("FLOP ölçümü için PyTorch gereklidir") from hata

    from .twt_baselines import build_twt_models

    seq = int(ARCHITECTURE_CONFIG["sequence_length"])
    model = build_twt_models(int(vocabulary_size))[model_name]()
    model.train()  # eval() SDPA füzyonunu tetikleyip sayımı bozuyor
    girdi = torch.randint(0, int(vocabulary_size), (1, seq))
    sayac = FlopCounterMode(display=False)
    with sayac, torch.no_grad():
        model(girdi)
    toplam_flop = int(sayac.get_total_flops())
    return {
        "model": model_name,
        "measured_flops": toplam_flop,
        "measured_macs": toplam_flop // 2,
        "counter": "torch.utils.flop_counter.FlopCounterMode",
    }


def reconcile_flops(models: Sequence[str] = MODEL_ORDER,
                    vocabulary_size: int = 512,
                    tolerance: float = 0.05) -> Dict[str, Any]:
    """Analitik formülü ölçülen değerle karşılaştır.

    Bu, FLOP tablosunun kendi kendini denetleyen kısmıdır: analitik form
    ölçümden ``tolerance``dan fazla saparsa formül yanlıştır ve kapı düşer.
    """
    satirlar: Dict[str, Any] = {}
    uyumlu = True
    for model in models:
        analitik = analytic_forward_flops(model)["forward_flops_per_example"]
        try:
            olculen = measured_forward_flops(model, vocabulary_size)["measured_macs"]
        except ImportError:
            satirlar[model] = {"analytic_macs": analitik, "measured_macs": None,
                               "relative_error": None, "agrees": None}
            uyumlu = False
            continue
        sapma = abs(analitik - olculen) / max(1, olculen)
        kabul = sapma <= tolerance
        uyumlu = uyumlu and kabul
        satirlar[model] = {
            "analytic_macs": analitik,
            "measured_macs": olculen,
            "relative_error": round(sapma, 6),
            "agrees": bool(kabul),
        }
    return {
        "per_model": satirlar,
        "tolerance": float(tolerance),
        "all_agree": bool(uyumlu),
        "note": ("Sapmanın bilinen kaynağı füzyonlu SDPA çekirdeğidir: "
                 "dikkat skoru matmul'ları sayaçta görünmez. Analitik form "
                 "bu terimleri İÇERİR ve bu yüzden biraz yüksektir."),
    }


def flop_fairness(models: Sequence[str] = MODEL_ORDER,
                  config: Optional[Dict[str, Any]] = None,
                  tolerance: float = 2.0) -> Dict[str, Any]:
    """FLOP bütçesi adilliğini denetle.

    ``tolerance`` en yüksek/en düşük FLOP oranı için üst sınırdır. Parametre
    kapısı %1'dir ama FLOP için varsayılan 2×'tir: mimariler yapısal olarak
    farklı işlem profillerine sahiptir ve tam eşitlik ancak mimariyi bozarak
    sağlanabilir. Kapı geçmezse bu GİZLENMEZ, bulgu olarak yazılır — eşit
    parametre eşit hesap demek değildir.
    """
    if tolerance < 1.0:
        raise ValueError("tolerance >= 1.0 olmalı")
    sayimlar = {m: analytic_forward_flops(m, config)["forward_flops_per_example"]
                for m in models}
    en_az = min(sayimlar.values())
    en_cok = max(sayimlar.values())
    oran = en_cok / en_az if en_az else float("inf")
    return {
        "forward_macs_per_example": sayimlar,
        "flop_min": en_az,
        "flop_max": en_cok,
        "cheapest_model": min(sayimlar, key=lambda k: sayimlar[k]),
        "most_expensive_model": max(sayimlar, key=lambda k: sayimlar[k]),
        "flop_max_to_min_ratio": round(oran, 6),
        "tolerance": float(tolerance),
        "within_tolerance": bool(oran <= tolerance),
        "note": ("Parametre eşitliği FLOP eşitliğini GARANTİ ETMEZ. Bu oran "
                 "kapıyı geçmezse, sonuç farkı kısmen işlem bütçesi farkına "
                 "atfedilebilir ve öyle okunmalıdır."),
    }


# ── Rapor ───────────────────────────────────────────────────────────────────
@dataclass
class TWTResultsReport:
    """P0-3 birleşik sonuç raporu."""

    protocol: str
    schema_version: int
    profile: str
    seeds: List[int]
    dataset_hash: str
    dataset_config_hash: str
    split_hashes: Dict[str, str]
    task_summary: Dict[str, Any]
    cost: Dict[str, Dict[str, Any]]
    flop_fairness: Dict[str, Any]
    flop_reconciliation: Dict[str, Any]
    results: Dict[str, Dict[str, Dict[str, Any]]]
    calibration: Dict[str, Dict[str, Dict[str, Any]]]
    per_seed: Dict[str, Dict[str, List[float]]]
    comparison: Dict[str, Any]
    best_model: Dict[str, str]
    checks: Dict[str, bool]
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _collect(reports: Sequence[Any], model: str, dimension: str,
             metric: str) -> List[float]:
    """Bir (model, dilim, metrik) üçlüsünün tohum başına değerlerini topla."""
    degerler = []
    for rapor in reports:
        dilim = rapor.models[model]["test"].get(dimension)
        if dilim is not None and metric in dilim:
            degerler.append(float(dilim[metric]))
    return degerler


def _collect_calibration(reports: Sequence[Any], model: str, dimension: str,
                         metric: str) -> List[float]:
    degerler = []
    for rapor in reports:
        dilimler = rapor.models[model]["calibration"]["slices"]
        if dimension in dilimler:
            degerler.append(float(dilimler[dimension]["after"][metric]))
    return degerler


def _summary(values: Sequence[float], seed: int) -> Dict[str, Any]:
    """Tek tohumda CI hesaplanamaz; bu durumu sessizce gizleme."""
    if not values:
        return {"n": 0, "mean": None, "std_sample": None,
                "ci_lower": None, "ci_upper": None,
                "note": "veri yok"}
    if len(values) < 2:
        return {"n": 1, "mean": round(float(values[0]), 12),
                "std_sample": None, "ci_lower": None, "ci_upper": None,
                "note": ("Tek tohum: std ve güven aralığı TANIMSIZ. Bu satır "
                         "betimleyicidir, istatistiksel iddia taşımaz.")}
    ozet = summarize_seed_metric(values, seed=seed)
    ozet["note"] = ""
    return ozet


def run_twt_results(
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    profile: str = "smoke",
    device: str = "cpu",
    dimensions: Sequence[str] = DIMENSIONS,
    models: Sequence[str] = MODEL_ORDER,
    flop_tolerance: float = 2.0,
    ci_seed: int = 12345,
) -> TWTResultsReport:
    """TWT'yi çoklu tohumda koş ve tek birleşik sonuç tablosu üret.

    Args:
        seeds: Tohum listesi. Çekirdek bilimsel iddia için 20 önerilir;
            5 ve altı smoke/engineering sayılır ve rapor bunu işaretler.
        profile: ``twt_baselines`` profili (``smoke`` / ``full``).
        dimensions: Raporlanacak disjoint dilimler.
        models: Karşılaştırılan mimariler.
        flop_tolerance: FLOP max/min oranı kapısı.

    Raises:
        ValueError: boş tohum/model/dilim listesi ya da bilinmeyen dilim.
    """
    tohumlar = [int(s) for s in seeds]
    if not tohumlar:
        raise ValueError("en az bir tohum gerekir")
    if len(set(tohumlar)) != len(tohumlar):
        raise ValueError("tohumlar benzersiz olmalı")
    if not models:
        raise ValueError("en az bir model gerekir")
    bilinmeyen = [m for m in models if m not in MODEL_ORDER]
    if bilinmeyen:
        raise ValueError(f"bilinmeyen model(ler): {bilinmeyen}")
    if not dimensions:
        raise ValueError("en az bir dilim gerekir")
    hatali = [d for d in dimensions if d not in DIMENSIONS]
    if hatali:
        raise ValueError(f"bilinmeyen dilim(ler): {hatali}")

    gorev = prepare_real_turkish_task()
    raporlar = [run_twt_architecture_baselines(seed=s, profile=profile,
                                               device=device)
                for s in tohumlar]
    ilk = raporlar[0]

    # ── maliyet ve FLOP ────────────────────────────────────────────────────
    maliyet: Dict[str, Dict[str, Any]] = {}
    for model in models:
        m = ilk.models[model]
        flop = analytic_forward_flops(model)
        maliyet[model] = {
            "physical_parameters": int(m["physical_parameters"]),
            "architecture_body_parameters": int(m["architecture_body_parameters"]),
            "parameter_bytes": int(m["parameter_bytes"]),
            "forward_flops_per_example": flop["forward_flops_per_example"],
            "flop_breakdown": flop["breakdown"],
            "training_seconds": round(_stat.fmean(
                [float(r.models[model]["training_seconds"]) for r in raporlar]), 6),
            "test_inference_seconds": round(_stat.fmean(
                [float(r.models[model]["test_inference_seconds"]) for r in raporlar]), 6),
            "estimated_adamw_training_bytes": int(
                m["estimated_adamw_training_bytes"]),
        }
    flop_adillik = flop_fairness(models, tolerance=flop_tolerance)
    flop_uzlasma = reconcile_flops(models)

    # ── metrik tabloları ───────────────────────────────────────────────────
    sonuclar: Dict[str, Dict[str, Dict[str, Any]]] = {}
    kalibrasyon: Dict[str, Dict[str, Dict[str, Any]]] = {}
    tohum_basi: Dict[str, Dict[str, List[float]]] = {}
    for model in models:
        sonuclar[model] = {}
        kalibrasyon[model] = {}
        tohum_basi[model] = {}
        for boyut in dimensions:
            sonuclar[model][boyut] = {
                ad: _summary(_collect(raporlar, model, boyut, ad), ci_seed)
                for ad, _ in HEADLINE_METRICS
            }
            sonuclar[model][boyut]["n_candidates"] = int(
                ilk.models[model]["test"][boyut]["total"])
            kalibrasyon[model][boyut] = {
                ad: _summary(_collect_calibration(raporlar, model, boyut, ad),
                             ci_seed)
                for ad, _ in CALIBRATION_METRICS
            }
            tohum_basi[model][boyut] = _collect(raporlar, model, boyut, "f1")

    # ── en iyi model ve eşleşmiş karşılaştırma ─────────────────────────────
    en_iyi: Dict[str, str] = {}
    for boyut in dimensions:
        en_iyi[boyut] = max(
            models,
            key=lambda m: (sonuclar[m][boyut]["f1"]["mean"] or 0.0))

    karsilastirma: Dict[str, Any] = {}
    rakipler = [m for m in models if m != "hga"]
    if "hga" in models and rakipler and len(tohumlar) >= 2:
        for boyut in dimensions:
            en_guclu = max(rakipler,
                           key=lambda m: (sonuclar[m][boyut]["f1"]["mean"] or 0.0))
            hga_degerler = tohum_basi["hga"][boyut]
            rakip_degerler = tohum_basi[en_guclu][boyut]
            if len(hga_degerler) == len(rakip_degerler) >= 2:
                karsilastirma[boyut] = compare_paired(
                    metric=f"f1@{boyut}",
                    treatment=hga_degerler,
                    baseline=rakip_degerler,
                    treatment_label="hga",
                    baseline_label=en_guclu,
                    seed=ci_seed,
                ).to_dict()

    # ── kapılar ────────────────────────────────────────────────────────────
    tum_kapilar_alt = {}
    for rapor in raporlar:
        for ad, deger in rapor.checks.items():
            tum_kapilar_alt[ad] = tum_kapilar_alt.get(ad, True) and bool(deger)

    kapilar: Dict[str, bool] = {
        "all_seeds_completed": len(raporlar) == len(tohumlar),
        "all_dimensions_reported": all(
            boyut in sonuclar[m] for m in models for boyut in dimensions),
        "all_headline_metrics_present": all(
            sonuclar[m][b][ad]["n"] > 0
            for m in models for b in dimensions for ad, _ in HEADLINE_METRICS),
        "calibration_metrics_present": all(
            kalibrasyon[m][b][ad]["n"] > 0
            for m in models for b in dimensions for ad, _ in CALIBRATION_METRICS),
        "flops_reported_for_all_models": all(
            maliyet[m]["forward_flops_per_example"] > 0 for m in models),
        "flop_budget_within_tolerance": bool(flop_adillik["within_tolerance"]),
        "analytic_flops_match_measured": bool(flop_uzlasma["all_agree"]),
        "parameter_budget_within_one_percent": bool(
            tum_kapilar_alt.get("physical_parameters_within_one_percent", False)),
        "sentence_disjoint_reported": "sentence_disjoint" in dimensions,
        "seed_count_sufficient_for_inference": len(tohumlar) >= 20,
        "underlying_fairness_gates_pass": all(tum_kapilar_alt.values()),
        # Bu kapı "HGA kazanmalı" demez; yalnız yönün RAPORLANDIĞINI
        # garanti eder. Bir kolun kaybetmesi protokol hatası değildir,
        # ama sessizce geçilmesi olurdu.
        "comparison_direction_reported": bool(karsilastirma) or len(tohumlar) < 2,
    }

    # ── bulgular ───────────────────────────────────────────────────────────
    bulgular: List[str] = []
    bulgular.append(
        f"Profil `{profile}`, {len(tohumlar)} tohum, {len(models)} mimari, "
        f"{len(dimensions)} disjoint dilim; gerçek veri TWT v1 "
        f"(`{ilk.dataset_hash[:12]}…`).")

    if not kapilar["seed_count_sufficient_for_inference"]:
        bulgular.append(
            f"Tohum sayısı {len(tohumlar)} < 20. Bu tablo ENGINEERING kanıtıdır; "
            "çekirdek bilimsel iddia için yeterli değildir. Aşağıdaki p-değerleri "
            "ve güven aralıkları bu sınırla okunmalıdır.")

    if not flop_adillik["within_tolerance"]:
        bulgular.append(
            f"FLOP oranı {flop_adillik['flop_max_to_min_ratio']:.2f}× "
            f"(kapı {flop_tolerance:.1f}×): mimariler eşit parametrede ama eşit "
            "işlem maliyetinde DEĞİL. Performans farkı kısmen hesap bütçesine "
            "atfedilebilir.")
    else:
        bulgular.append(
            f"FLOP oranı {flop_adillik['flop_max_to_min_ratio']:.2f}× kapı içinde; "
            "parametre ve işlem bütçesi birlikte denetlendi.")

    # Dilimler arası düşüş: genelleme zorluğunun gerçek göstergesi.
    for model in models:
        taban = sonuclar[model]["all"]["f1"]["mean"]
        if taban is None:
            continue
        dususler = []
        for boyut in dimensions:
            if boyut == "all":
                continue
            deger = sonuclar[model][boyut]["f1"]["mean"]
            if deger is not None and taban - deger > 0.05:
                dususler.append(f"{boyut} (−{taban - deger:.3f})")
        if dususler:
            bulgular.append(
                f"`{model}`: `all` diliminden belirgin düşüş → {', '.join(dususler)}. "
                "Toplam skor bu zorluğu gizler; dilim tablosu bu yüzden var.")

    # Ayrışma YÖNSÜZ bir kelimedir: "istatistiksel olarak ayrıştı" cümlesi
    # HGA'nın kazandığını ima eder ama kaybettiğinde de doğrudur. Bu yüzden
    # yön burada açıkça ayrıştırılır.
    hga_ustun, hga_geride, ayrilmayanlar = [], [], []
    for boyut, k in karsilastirma.items():
        karar = k["verdict"]
        fark = k["treatment_mean"] - k["baseline_mean"]
        if karar.startswith("AYRIŞMA"):
            (hga_ustun if fark > 0 else hga_geride).append(
                f"{boyut} ({fark:+.4f} vs {k['baseline_label']})")
        elif karar.startswith("AYRIM YOK"):
            ayrilmayanlar.append(boyut)

    if karsilastirma:
        if hga_ustun:
            bulgular.append(
                "HGA en güçlü rakibini şu dilimlerde İSTATİSTİKSEL OLARAK "
                f"GEÇTİ: {hga_ustun}.")
        if hga_geride:
            bulgular.append(
                "HGA en güçlü rakibinin şu dilimlerde İSTATİSTİKSEL OLARAK "
                f"GERİSİNDE KALDI: {hga_geride}. Bu, eşit parametre bütçesinde "
                "HGA lehine bir üstünlük iddiasının VERİYLE ÇELİŞTİĞİ "
                "anlamına gelir ve gizlenmez.")
        if ayrilmayanlar:
            bulgular.append(
                f"Şu dilimlerde HGA ile en güçlü rakip AYRIŞTIRILAMADI "
                f"(fark CI'si sıfırı içeriyor): {ayrilmayanlar}. Bu bir "
                "üstünlük iddiasının reddidir, eksik ölçüm değil.")
        if not (hga_ustun or hga_geride or ayrilmayanlar):
            bulgular.append(
                "Hiçbir dilimde kesin ayrışma yok; tüm karşılaştırmalar "
                "kararsız ya da yetersiz güçte.")

    kazanan_sayisi: Dict[str, int] = {}
    for boyut, model in en_iyi.items():
        kazanan_sayisi[model] = kazanan_sayisi.get(model, 0) + 1
    bulgular.append(
        "F1 ortalamasına göre dilim kazanımları: "
        + ", ".join(f"{m}×{c}" for m, c in sorted(kazanan_sayisi.items(),
                                                  key=lambda x: -x[1])))

    imza = hashlib.sha256(json.dumps(
        {"protocol": PROTOCOL, "seeds": tohumlar, "profile": profile,
         "models": list(models), "dimensions": list(dimensions),
         "dataset": ilk.dataset_hash},
        sort_keys=True).encode("utf-8")).hexdigest()[:12]

    return TWTResultsReport(
        protocol=PROTOCOL,
        schema_version=SCHEMA_VERSION,
        profile=profile,
        seeds=tohumlar,
        dataset_hash=ilk.dataset_hash,
        dataset_config_hash=ilk.dataset_config_hash,
        split_hashes=dict(ilk.split_hashes),
        task_summary={
            "task": "binary dependency arc validation",
            "signature": imza,
            "train_candidates": len(gorev.train),
            "dev_candidates": len(gorev.dev),
            "test_candidates": len(gorev.test),
            "heldout_relations": list(gorev.heldout_relations),
            "is_language_modeling": False,
        },
        cost=maliyet,
        flop_fairness=flop_adillik,
        flop_reconciliation=flop_uzlasma,
        results=sonuclar,
        calibration=kalibrasyon,
        per_seed=tohum_basi,
        comparison=karsilastirma,
        best_model=en_iyi,
        checks=kapilar,
        findings=bulgular,
        limitations=[
            "Görev ikili bağımlılık-yayı doğrulamasıdır; dil modelleme, metin "
            "üretimi veya perplexity iddiası İÇERMEZ.",
            "FLOP sayıları analitik kapalı formdur; gerçek donanım verimliliği "
            "(bellek bant genişliği, çekirdek füzyonu) hesaba katılmaz.",
            "Gömme parametreleri modeller arasında paylaşılır; 'body' oranı bu "
            "yüzden toplam orandan daha anlamlı bir adillik göstergesidir.",
            "Kalibrasyon sıcaklığı yalnız dev'de fit edilir ve argmax'ı "
            "değiştirmez; bu yüzden accuracy/F1 kalibrasyondan etkilenmez.",
            f"Tohum sayısı {len(tohumlar)}; 20'nin altındaki her sonuç "
            "betimleyicidir.",
        ],
    )


def _fmt(summary: Dict[str, Any], digits: int = 4) -> str:
    """Ortalamayı ± std ile yaz; tek tohumda std yerine açıkça '—' koy."""
    if not summary or summary.get("mean") is None:
        return "n/a"
    ort = summary["mean"]
    std = summary.get("std_sample")
    if std is None:
        return f"{ort:.{digits}f}"
    return f"{ort:.{digits}f} ±{std:.{digits}f}"


def _human_flops(value: int) -> str:
    for birim, esik in (("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if value >= esik:
            return f"{value / esik:.2f}{birim}"
    return str(value)


def results_markdown(report: TWTResultsReport) -> str:
    """P0-3 sonuç tablosunu Markdown olarak üret."""
    s = report
    satirlar = [
        "# TWT Gerçek Sonuç Tablosu (P0-3)",
        "",
        f"- Protokol: `{s.protocol}` v{s.schema_version} "
        f"(imza `{s.task_summary['signature']}`)",
        f"- Görev: {s.task_summary['task']} — **dil modelleme değildir**",
        f"- Veri: TWT v1 `{s.dataset_hash[:16]}…`, "
        f"train/dev/test aday sayısı "
        f"{s.task_summary['train_candidates']}/"
        f"{s.task_summary['dev_candidates']}/"
        f"{s.task_summary['test_candidates']}",
        f"- Profil / tohumlar: `{s.profile}` / `{s.seeds}`",
        "",
        "## Maliyet bütçesi",
        "",
        "| Model | Parametre | Gövde | Bayt | İleri FLOP/örnek | Eğitim (s) | Çıkarım (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, c in s.cost.items():
        satirlar.append(
            f"| {model} | {c['physical_parameters']:,} | "
            f"{c['architecture_body_parameters']:,} | {c['parameter_bytes']:,} | "
            f"{_human_flops(c['forward_flops_per_example'])} | "
            f"{c['training_seconds']:.3f} | {c['test_inference_seconds']:.3f} |")
    ff = s.flop_fairness
    satirlar.extend([
        "",
        f"FLOP max/min oranı: **{ff['flop_max_to_min_ratio']:.3f}×** "
        f"(kapı {ff['tolerance']:.1f}×) → "
        f"{'GEÇTİ' if ff['within_tolerance'] else 'KALDI'}",
        "",
        f"> {ff['note']}",
        "",
        "## Ana sonuç tablosu (test, ortalama ± std)",
        "",
    ])

    for boyut in s.best_model:
        satirlar.extend([
            f"### `{boyut}` "
            f"(n={s.results[list(s.results)[0]][boyut]['n_candidates']})",
            "",
            "| Model | " + " | ".join(
                f"{ad} {yon}" for ad, yon in HEADLINE_METRICS) + " |",
            "|---" * (len(HEADLINE_METRICS) + 1) + "|",
        ])
        for model in s.results:
            hucreler = [_fmt(s.results[model][boyut][ad])
                        for ad, _ in HEADLINE_METRICS]
            isaret = " **←**" if s.best_model[boyut] == model else ""
            satirlar.append(f"| {model}{isaret} | " + " | ".join(hucreler) + " |")
        satirlar.append("")

    satirlar.extend([
        "## Kalibrasyon (dev'de fit, testte ölçüm — `all` dilimi)",
        "",
        "| Model | " + " | ".join(f"{ad} {yon}"
                                  for ad, yon in CALIBRATION_METRICS) + " |",
        "|---" * (len(CALIBRATION_METRICS) + 1) + "|",
    ])
    for model in s.calibration:
        hucreler = [_fmt(s.calibration[model]["all"][ad])
                    for ad, _ in CALIBRATION_METRICS]
        satirlar.append(f"| {model} | " + " | ".join(hucreler) + " |")

    if s.comparison:
        satirlar.extend([
            "",
            "## HGA vs en güçlü rakip (eşleşmiş, F1)",
            "",
            "| Dilim | Rakip | HGA | Rakip | Fark %95 CI | Hedges g | p | Karar |",
            "|---|---|---:|---:|---|---:|---:|---|",
        ])
        for boyut, k in s.comparison.items():
            ci = k["difference_ci"]
            satirlar.append(
                f"| {boyut} | {k['baseline_label']} | "
                f"{k['treatment_mean']:.4f} | {k['baseline_mean']:.4f} | "
                f"[{ci['lower']:.4f}, {ci['upper']:.4f}] | "
                f"{k['effect_size']['hedges_g']:.3f} | "
                f"{k['permutation_test']['p_value']:.4f} | "
                f"{k['verdict'].split(':')[0]} |")

    satirlar.extend(["", "## Kabul kapıları", ""])
    satirlar.extend(f"| {ad} | {'GEÇTİ' if v else 'KALDI'} |"
                    for ad, v in s.checks.items())
    satirlar.insert(len(satirlar) - len(s.checks),
                    "| Kapı | Sonuç |\n|---|---|")
    satirlar.extend(["", "## Bulgular", ""])
    satirlar.extend(f"- {b}" for b in s.findings)
    satirlar.extend(["", "## Sınırlar", ""])
    satirlar.extend(f"- {b}" for b in s.limitations)
    return "\n".join(satirlar) + "\n"


__all__ = [
    "CALIBRATION_METRICS",
    "COST_FIELDS",
    "HEADLINE_METRICS",
    "PROTOCOL",
    "SCHEMA_VERSION",
    "TWTResultsReport",
    "analytic_forward_flops",
    "flop_fairness",
    "measured_forward_flops",
    "reconcile_flops",
    "results_markdown",
    "run_twt_results",
]
