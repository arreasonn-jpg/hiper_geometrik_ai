# -*- coding: utf-8 -*-
"""Priority(E) ağırlıklarının veri odaklı aranması ve held-out doğrulaması.

Ablasyon (`priority_ablation`) şu rahatsız edici bulguyu üretmişti:

    YÖN UYARISI: ['w_gain', 'w_conflict_penalty'] terimlerini KAPATMAK
    doğrulama verimini ARTIRIYOR — bu havuzda katkıları negatif.

Yani elle seçilmiş varsayılan ağırlıklar (0.40 / 0.35 / 0.25 / 0.20)
savunulamıyordu: bazı terimleri sıfırlamak sonucu **iyileştiriyordu**.
Bu, "ağırlıklar etkili mi?" sorusundan farklı ve daha önemli bir soruyu
gündeme getirir: **ağırlıklar doğru mu?**

Bu modül üç şey yapar:

1. **Arama** — ağırlık uzayını eğitim tohumları üzerinde tarar ve
   downstream doğrulama verimini maksimize eden ağırlıkları bulur.
2. **Held-out doğrulama** — bulunan ağırlıkları **hiç görülmemiş**
   tohumlarda test eder. Bu olmadan arama sadece ezberdir.
3. **Aşırı uydurma teşhisi** — eğitim ve test kazancı arasındaki farkı
   ölçer ve genelleşmeyen bir kazancı açıkça işaretler.

Kritik tasarım kararı: arama **yalnız eğitim tohumlarını** görür. Test
tohumları aramadan tamamen izoledir; sızıntı olsa ölçülen kazanç sahte
olurdu. `checks["train_test_seed_isolation"]` bunu zorlar.

İkinci kritik nokta: kazanç **eşleşmiş** olarak test edilir (aynı tohum,
aynı havuz, iki ağırlık seti) ve `statistics.compare_paired` ile güven
aralığı + etki büyüklüğü üretilir. "Ortalama arttı" bir sonuç değildir.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..experience.exploration import ExplorationEngine
from .priority_ablation import (
    TERIMLER,
    _downstream,
    _secim,
    _skorla_hepsi,
    havuz_uret,
)
from .statistics import compare_paired, summarize_seed_metric

PROTOCOL = "priority_weight_optimization_v1"
SCHEMA_VERSION = 1

#: Aranan hedef metrik. Downstream doğrulama verimi = seçilen K adayın
#: dış doğrulayıcıdan geçme oranı.
OBJECTIVE = "verification_yield"

#: Ağırlık başına taranan değerler. 0.0 dahildir: arama bir terimi
#: tamamen kapatabilmelidir, çünkü ablasyon bunun iyi olabileceğini
#: gösterdi.
DEFAULT_GRID: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6)

PROFILES: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "grid": (0.0, 0.4),
        "train_seeds": (1, 2, 3),
        "test_seeds": (101, 102, 103),
        "candidate_count": 60,
    },
    "standard": {
        "grid": DEFAULT_GRID,
        "train_seeds": tuple(range(1, 11)),
        "test_seeds": tuple(range(101, 121)),
        "candidate_count": 120,
    },
}


@dataclass
class WeightEvaluation:
    """Tek bir ağırlık setinin bir tohum kümesindeki performansı."""

    weights: Dict[str, float]
    mean_objective: float
    per_seed: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PriorityOptimizationReport:
    """Ağırlık araması + held-out doğrulama raporu."""

    protocol: str
    schema_version: int
    profile: str
    objective: str
    grid: List[float]
    train_seeds: List[int]
    test_seeds: List[int]
    candidates_evaluated: int
    default_weights: Dict[str, float]
    best_weights: Dict[str, float]
    train: Dict[str, Any]
    holdout: Dict[str, Any]
    comparison: Dict[str, Any]
    overfitting: Dict[str, Any]
    weight_sensitivity: Dict[str, Any]
    checks: Dict[str, bool]
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _evaluate_weights(
    weights: Dict[str, float],
    seeds: Sequence[int],
    k: int,
    candidate_count: int,
    operands_max: int,
    objective: str = OBJECTIVE,
) -> List[float]:
    """Bir ağırlık setini verilen tohumlarda değerlendir.

    Her tohum kendi havuzunu üretir; havuz üretimi ağırlıklardan
    bağımsızdır, yani karşılaştırma eşleşmiştir.
    """
    sonuclar: List[float] = []
    for seed in seeds:
        havuz = havuz_uret(candidate_count=candidate_count,
                           operands_max=operands_max, seed=seed)
        motor = ExplorationEngine(
            w_gain=weights["w_gain"], w_novelty=weights["w_novelty"],
            w_uncertainty=weights["w_uncertainty"],
            w_conflict_penalty=weights["w_conflict_penalty"])
        skorlar = _skorla_hepsi(motor, havuz)
        secim = _secim(havuz, skorlar, k)
        sonuclar.append(float(_downstream(havuz, secim)[objective]))
    return sonuclar


def optimize_priority_weights(
    profile: str = "smoke",
    k: int = 10,
    grid: Optional[Sequence[float]] = None,
    train_seeds: Optional[Sequence[int]] = None,
    test_seeds: Optional[Sequence[int]] = None,
    candidate_count: Optional[int] = None,
    operands_max: int = 12,
    objective: str = OBJECTIVE,
) -> PriorityOptimizationReport:
    """Ağırlıkları eğitim tohumlarında ara, held-out tohumlarda doğrula.

    Raises:
        ValueError: bilinmeyen profil, boş ızgara, ya da eğitim/test
            tohumlarının kesişmesi (sızıntı).
    """
    if profile not in PROFILES:
        raise ValueError(
            f"profile şunlardan biri olmalı: {', '.join(PROFILES)}")
    ayar = PROFILES[profile]
    izgara = [float(v) for v in (grid if grid is not None else ayar["grid"])]
    egitim = [int(s) for s in (train_seeds if train_seeds is not None
                               else ayar["train_seeds"])]
    test = [int(s) for s in (test_seeds if test_seeds is not None
                             else ayar["test_seeds"])]
    aday_sayisi = int(candidate_count if candidate_count is not None
                      else ayar["candidate_count"])
    if not izgara:
        raise ValueError("grid boş olamaz")
    if any(v < 0.0 for v in izgara):
        raise ValueError("ağırlıklar negatif olamaz")
    if len(egitim) < 2 or len(test) < 2:
        raise ValueError("eğitim ve test için en az 2'şer tohum gerekir")
    ortak = sorted(set(egitim) & set(test))
    if ortak:
        raise ValueError(
            f"eğitim/test tohum sızıntısı: {ortak}. Held-out doğrulama "
            "ancak tohum kümeleri ayrıkken anlamlıdır.")
    if k < 1:
        raise ValueError("k >= 1 olmalı")

    varsayilan = {t: float(getattr(ExplorationEngine(), t)) for t in TERIMLER}

    # ── arama: YALNIZ eğitim tohumları görülür ─────────────────────────────
    izgara = sorted(set(izgara))
    tum_kombinasyonlar = list(itertools.product(izgara, repeat=len(TERIMLER)))
    degerlendirmeler: List[WeightEvaluation] = []
    for kombinasyon in tum_kombinasyonlar:
        agirliklar = dict(zip(TERIMLER, kombinasyon))
        if all(v == 0.0 for v in kombinasyon):
            continue  # tüm terimler sıfır: skor sabit, sıralama anlamsız
        per_seed = _evaluate_weights(
            agirliklar, egitim, k, aday_sayisi, operands_max, objective)
        degerlendirmeler.append(WeightEvaluation(
            weights=agirliklar,
            mean_objective=round(sum(per_seed) / len(per_seed), 9),
            per_seed=[round(v, 9) for v in per_seed]))

    if not degerlendirmeler:
        raise ValueError("değerlendirilebilir ağırlık kombinasyonu yok")

    # Beraberlikte: daha az terim kullanan (daha sade) model tercih edilir.
    degerlendirmeler.sort(
        key=lambda e: (-e.mean_objective,
                       sum(1 for v in e.weights.values() if v > 0),
                       tuple(e.weights[t] for t in TERIMLER)))
    en_iyi = degerlendirmeler[0]

    varsayilan_egitim = _evaluate_weights(
        varsayilan, egitim, k, aday_sayisi, operands_max, objective)

    # ── held-out doğrulama: test tohumları ilk kez BURADA görülüyor ────────
    en_iyi_test = _evaluate_weights(
        en_iyi.weights, test, k, aday_sayisi, operands_max, objective)
    varsayilan_test = _evaluate_weights(
        varsayilan, test, k, aday_sayisi, operands_max, objective)

    kiyas = compare_paired(
        metric=f"holdout::{objective}",
        treatment=en_iyi_test, baseline=varsayilan_test,
        treatment_label="optimized", baseline_label="default").to_dict()

    egitim_kazanc = (en_iyi.mean_objective
                     - sum(varsayilan_egitim) / len(varsayilan_egitim))
    test_kazanc = ((sum(en_iyi_test) / len(en_iyi_test))
                   - (sum(varsayilan_test) / len(varsayilan_test)))
    kayip_oran = (1.0 - test_kazanc / egitim_kazanc) if egitim_kazanc > 0 \
        else None

    asiri_uydurma = {
        "train_gain": round(egitim_kazanc, 9),
        "holdout_gain": round(test_kazanc, 9),
        "generalization_ratio": (round(test_kazanc / egitim_kazanc, 6)
                                 if egitim_kazanc > 0 else None),
        "gain_lost_to_overfitting": (round(kayip_oran, 6)
                                     if kayip_oran is not None else None),
        "gain_survives_holdout": test_kazanc > 0.0,
        "note": ("Eğitim kazancının held-out'ta korunan kısmı. Oran 1.0'a "
                 "yakınsa kazanç gerçek; 0'a yakınsa arama ezberlemiş, "
                 "negatifse ağırlıklar held-out'ta ZARAR veriyor."),
    }

    # ── terim duyarlılığı: en iyi setin etrafında tek terim değiştir ───────
    duyarlilik: Dict[str, Any] = {}
    for terim in TERIMLER:
        satir = []
        atlanan = 0
        for deger in izgara:
            aday = dict(en_iyi.weights)
            aday[terim] = deger
            if all(v == 0.0 for v in aday.values()):
                # Tüm terimler sıfır: skor sabit olur, sıralama tanımsızdır.
                # Bu noktayı atlamak zorundayız ama SESSİZCE atlamak
                # yanıltıcı olurdu: geriye tek nokta kalırsa "fark
                # yaratmıyor" sonucu ölçüm artefaktı olur.
                atlanan += 1
                continue
            puanlar = _evaluate_weights(
                aday, test, k, aday_sayisi, operands_max, objective)
            satir.append({
                "value": deger,
                "holdout_mean": round(sum(puanlar) / len(puanlar), 9),
            })
        if not satir:
            continue
        en_iyi_deger = max(satir, key=lambda r: r["holdout_mean"])
        aralik = (max(r["holdout_mean"] for r in satir)
                  - min(r["holdout_mean"] for r in satir))
        # Tek başına kalan bir terimde ÖLÇEK sıralamayı değiştirmez
        # (monoton dönüşüm); değişen tek şey işaretin varlığıdır.
        tek_aktif = sum(1 for t2, v2 in en_iyi.weights.items()
                        if v2 > 0.0 and t2 != terim) == 0
        duyarlilik[terim] = {
            "curve": satir,
            "best_value": en_iyi_deger["value"],
            "holdout_range": round(aralik, 9),
            "points_evaluated": len(satir),
            "points_skipped_all_zero": atlanan,
            "scale_invariant": tek_aktif,
            "matters": (aralik > 1e-9) or tek_aktif,
            "note": ("Bu terim tek aktif terim olduğu için ölçeği "
                     "sıralamayı değiştirmez; belirleyici olan varlığıdır. "
                     "Sabit eğri 'ölü ağırlık' anlamına GELMEZ."
                     if tek_aktif else
                     "Diğer terimler sabitken bu terimin held-out etkisi."),
        }

    onemsiz = [t for t, v in duyarlilik.items() if not v["matters"]]
    sifirlanan = [t for t, v in en_iyi.weights.items() if v == 0.0]

    kapilar: Dict[str, bool] = {
        "train_test_seed_isolation": not ortak,
        "search_space_includes_zero": 0.0 in izgara,
        "holdout_evaluated": bool(en_iyi_test),
        "optimized_at_least_matches_default_on_holdout": (
            test_kazanc >= 0.0),
        "gain_survives_holdout": test_kazanc > 0.0,
        "gain_statistically_distinguishable": "AYRIŞMA" in kiyas["verdict"],
        "overfitting_measured": True,
        "sensitivity_measured": bool(duyarlilik),
        "paired_design": len(en_iyi_test) == len(varsayilan_test),
    }

    # ── bulgular ───────────────────────────────────────────────────────────
    bulgular: List[str] = [
        f"{len(degerlendirmeler)} ağırlık kombinasyonu {len(egitim)} eğitim "
        f"tohumunda tarandı; en iyisi {len(test)} HELD-OUT tohumda test "
        "edildi.",
        f"Varsayılan (elle seçilmiş) ağırlıklar: {varsayilan}.",
        f"Aramanın bulduğu ağırlıklar: {en_iyi.weights}.",
    ]
    if sifirlanan:
        bulgular.append(
            f"Arama şu terimleri TAMAMEN KAPATTI: {sifirlanan}. Bu, "
            "ablasyonun 'bu terimleri kapatmak verimi artırıyor' "
            "bulgusuyla tutarlıdır — elle seçilmiş ağırlıklar yalnız "
            "etkisiz değil, bazı yerlerde zararlıydı.")
    bulgular.append(
        f"Eğitim kazancı {egitim_kazanc:+.4f}, held-out kazancı "
        f"{test_kazanc:+.4f}.")
    if asiri_uydurma["generalization_ratio"] is not None:
        bulgular.append(
            f"Kazancın %{asiri_uydurma['generalization_ratio'] * 100:.0f}'ı "
            "held-out'ta korundu; kalanı aşırı uydurmadır.")
    bulgular.append(f"Held-out hükmü: {kiyas['verdict']}")
    if onemsiz:
        bulgular.append(
            f"Held-out'ta hiçbir fark yaratmayan terimler: {onemsiz}. Bu "
            "terimler bu havuzda ölü ağırlıktır.")
    if not kapilar["gain_survives_holdout"]:
        bulgular.append(
            "DÜRÜST NEGATİF: arama eğitim tohumlarında kazanç buldu ama "
            "held-out'ta korunmadı. Varsayılan ağırlıklar değiştirilmemeli.")

    imza = hashlib.sha256(json.dumps(
        {"protocol": PROTOCOL, "grid": izgara, "train": egitim,
         "test": test, "k": k, "candidates": aday_sayisi,
         "objective": objective},
        sort_keys=True).encode("utf-8")).hexdigest()[:12]

    return PriorityOptimizationReport(
        protocol=PROTOCOL,
        schema_version=SCHEMA_VERSION,
        profile=profile,
        objective=objective,
        grid=izgara,
        train_seeds=egitim,
        test_seeds=test,
        candidates_evaluated=len(degerlendirmeler),
        default_weights=varsayilan,
        best_weights=en_iyi.weights,
        train={
            "best_mean": en_iyi.mean_objective,
            "best_per_seed": en_iyi.per_seed,
            "default_mean": round(
                sum(varsayilan_egitim) / len(varsayilan_egitim), 9),
            "default_per_seed": [round(v, 9) for v in varsayilan_egitim],
            "top5": [e.to_dict() for e in degerlendirmeler[:5]],
        },
        holdout={
            "optimized": summarize_seed_metric(en_iyi_test),
            "default": summarize_seed_metric(varsayilan_test),
            "optimized_per_seed": [round(v, 9) for v in en_iyi_test],
            "default_per_seed": [round(v, 9) for v in varsayilan_test],
            "signature": imza,
        },
        comparison=kiyas,
        overfitting=asiri_uydurma,
        weight_sensitivity=duyarlilik,
        checks=kapilar,
        findings=bulgular,
        limitations=[
            "Arama ızgarası kabadır; sürekli optimizasyon daha iyi bir nokta "
            "bulabilir. Bulunan ağırlıklar 'ızgaranın en iyisi'dir, küresel "
            "optimum değildir.",
            "Hedef tek metriktir (doğrulama verimi). Çeşitlilik ya da "
            "kapsama gibi başka hedefler farklı ağırlıklar seçtirirdi.",
            "Havuz aritmetiktir ve doğrulayıcı kapalı formdur; gerçek "
            "alanlarda optimal ağırlıklar farklı olabilir.",
            "Held-out tohumlar aynı havuz üreticisinden gelir; bu bir tohum "
            "genellemesidir, ALAN genellemesi değildir.",
        ],
    )


def priority_optimization_markdown(
        report: PriorityOptimizationReport) -> str:
    """Raporu Markdown'a çevir."""
    s = report
    satirlar = [
        "# Priority(E) Ağırlık Optimizasyonu + Held-Out Doğrulama",
        "",
        f"- Protokol: `{s.protocol}` v{s.schema_version} "
        f"(imza `{s.holdout['signature']}`)",
        f"- Profil: `{s.profile}` · hedef: `{s.objective}`",
        f"- Izgara: `{s.grid}` · değerlendirilen kombinasyon: "
        f"`{s.candidates_evaluated}`",
        f"- Eğitim tohumları: `{s.train_seeds}`",
        f"- **Held-out tohumları: `{s.test_seeds}`** (aramada HİÇ görülmedi)",
        "",
        "## Ağırlıklar",
        "",
        "| Terim | Varsayılan (elle) | Aramanın bulduğu |",
        "|---|---:|---:|",
    ]
    for terim in TERIMLER:
        satirlar.append(
            f"| `{terim}` | {s.default_weights[terim]:.2f} | "
            f"**{s.best_weights[terim]:.2f}** |")

    o = s.overfitting
    h = s.holdout
    satirlar.extend([
        "",
        "## Eğitim vs held-out",
        "",
        "| Küme | Varsayılan | Optimize | Kazanç |",
        "|---|---:|---:|---:|",
        f"| Eğitim | {s.train['default_mean']:.4f} | "
        f"{s.train['best_mean']:.4f} | {o['train_gain']:+.4f} |",
        f"| **Held-out** | {h['default']['mean']:.4f} | "
        f"{h['optimized']['mean']:.4f} | **{o['holdout_gain']:+.4f}** |",
        "",
        f"- Genelleşme oranı: `{o['generalization_ratio']}`",
        f"- Held-out'ta kazanç korundu mu: "
        f"**{'EVET' if o['gain_survives_holdout'] else 'HAYIR'}**",
        "",
        f"> {o['note']}",
        "",
        "## Held-out istatistiksel karşılaştırma (eşleşmiş)",
        "",
        f"- Fark %95 CI: `[{s.comparison['difference_ci']['lower']:.4f}, "
        f"{s.comparison['difference_ci']['upper']:.4f}]`",
        f"- Etki büyüklüğü: `{s.comparison['effect_size']['value']:+.3f}` "
        f"({s.comparison['effect_size'].get('magnitude', '')})",
        f"- Permütasyon p: `{s.comparison['permutation_test']['p_value']:.5f}`",
        f"- Hüküm: {s.comparison['verdict']}",
    ])

    if s.weight_sensitivity:
        satirlar.extend([
            "",
            "## Terim duyarlılığı (held-out)",
            "",
            "| Terim | En iyi değer | Held-out aralığı | Fark yaratıyor mu |",
            "|---|---:|---:|---|",
        ])
        for terim, veri in s.weight_sensitivity.items():
            satirlar.append(
                f"| `{terim}` | {veri['best_value']:.2f} | "
                f"{veri['holdout_range']:.4f} | "
                f"{'evet' if veri['matters'] else 'HAYIR (ölü ağırlık)'} |")

    satirlar.extend(["", "## Kabul kapıları", "", "| Kapı | Sonuç |", "|---|---|"])
    satirlar.extend(f"| {ad} | {'GEÇTİ' if v else 'KALDI'} |"
                    for ad, v in s.checks.items())
    satirlar.extend(["", "## Bulgular", ""])
    satirlar.extend(f"- {b}" for b in s.findings)
    satirlar.extend(["", "## Sınırlar", ""])
    satirlar.extend(f"- {b}" for b in s.limitations)
    return "\n".join(satirlar) + "\n"


__all__ = [
    "DEFAULT_GRID",
    "OBJECTIVE",
    "PROFILES",
    "PROTOCOL",
    "SCHEMA_VERSION",
    "PriorityOptimizationReport",
    "WeightEvaluation",
    "optimize_priority_weights",
    "priority_optimization_markdown",
]
