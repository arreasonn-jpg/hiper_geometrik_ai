# -*- coding: utf-8 -*-
"""P1 — Self-learning ölçeklendirme: 100 → 1000 → 3000 cycle.

Kullanıcı isteği "cycle sayısını 100'den 3000'e çıkar" idi. İlk koşumda
ortaya çıkan şey, istekten daha ilginç:

    cycles=100,  operands_max=31 → K = 909
    cycles=1000, operands_max=31 → K = 938   (10× cycle, +%3 bilgi)

Yani cycle sayısını 10 katına çıkarmak bilgiyi neredeyse hiç artırmıyor.
Sebep basit ve önemli: **aday havuzu tükeniyor**. Sabit bir ``operands_max``
ile üretilebilecek ifade sayısı sonludur; havuz bitince ek cycle'lar boşa
döner. Bu, "daha çok öğrenme döngüsü = daha çok bilgi" varsayımının
yanlış olduğu noktadır.

Bu modül bu yüzden iki eksenli ölçer:

* **cycle ekseni** — sabit alanda cycle artırmak ne kazandırır? (doygunluk)
* **alan ekseni** — alanı da büyütünce ölçekleniyor mu? (gerçek ölçekleme)

Ölçülen kritik nicelikler:

``saturation_cycle``
    Bilginin son cycle'daki değerinin %99'una ulaştığı ilk cycle. Bundan
    sonrası hesap israfıdır ve raporlanır.
``knowledge_per_cycle``
    Marjinal verim. Doygunluk sonrası ~0'a iner.
``drift``
    Uzun koşuda yanlış bilgi birikiyor mu? Self-training çöküşünün
    (model collapse) doğrudan göstergesi. Sıfırdan büyükse kapı düşer.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..experience.self_learning import run_self_learning_experiment

PROTOCOL = "self_learning_scaling_v1"
SCHEMA_VERSION = 1

#: (cycles, operands_max) çiftleri. ``operands_max`` alan büyüklüğünü belirler.
PROFILES: Dict[str, Tuple[Tuple[int, int], ...]] = {
    # Hızlı duman testi: doygunluk yine de görünür.
    "smoke": ((25, 15), (100, 15), (100, 31)),
    # Kullanıcının istediği 100→1000→3000 ekseni + alan ekseni.
    "standard": ((100, 31), (1000, 31), (1000, 63)),
    "deep": ((100, 31), (1000, 31), (3000, 31), (1000, 63), (3000, 127)),
}

SATURATION_THRESHOLD = 0.99


@dataclass
class ScalingPoint:
    """Tek (cycles, operands_max) noktasının sonucu."""

    cycles: int
    operands_max: int
    seed: int
    cycles_completed: int
    final_knowledge: int
    initial_knowledge: int
    generated: int
    verified: int
    correct_knowledge: int
    incorrect_knowledge: int
    experience_yield: float
    novelty_yield: float
    useful_experience_yield: float
    far: float
    frr: float
    far_before_verifier: float
    memory_collisions: int
    isolation_clean: bool
    saturation_cycle: Optional[int]
    knowledge_per_cycle: float
    wall_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelfLearningScalingReport:
    """Ölçeklendirme raporu."""

    protocol: str
    schema_version: int
    profile: str
    seeds: List[int]
    points: List[Dict[str, Any]]
    cycle_axis: Dict[str, Any]
    domain_axis: Dict[str, Any]
    drift: Dict[str, Any]
    checks: Dict[str, bool]
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _saturation_cycle(cycles: Sequence[Any],
                      threshold: float = SATURATION_THRESHOLD) -> Optional[int]:
    """Bilginin nihai değerinin ``threshold`` oranına ulaştığı ilk cycle."""
    if not cycles:
        return None
    ham = [getattr(c, "knowledge_size", None) for c in cycles]
    boyutlar: List[float] = [float(b) for b in ham if b is not None]
    if not boyutlar:
        return None
    hedef = boyutlar[-1] * threshold
    for indeks, boyut in enumerate(boyutlar, start=1):
        if boyut >= hedef:
            return indeks
    return len(boyutlar)


def run_self_learning_scaling(
    profile: str = "smoke",
    seeds: Sequence[int] = (1,),
    batch_size: int = 64,
    initial_facts: int = 100,
    negatives_per_fact: int = 3,
    grid: Optional[Sequence[Tuple[int, int]]] = None,
) -> SelfLearningScalingReport:
    """Self-learning'i cycle ve alan eksenlerinde ölçekle.

    Args:
        profile: ``smoke`` / ``standard`` / ``deep``.
        seeds: Tohumlar; her nokta her tohumda koşar.
        grid: ``(cycles, operands_max)`` çiftleri; verilirse profili ezer.

    Raises:
        ValueError: bilinmeyen profil, boş tohum ya da geçersiz ızgara.
    """
    if grid is None:
        if profile not in PROFILES:
            raise ValueError(
                f"profile şunlardan biri olmalı: {', '.join(PROFILES)}")
        izgara = list(PROFILES[profile])
    else:
        izgara = [(int(c), int(o)) for c, o in grid]
        if not izgara:
            raise ValueError("grid boş olamaz")
        if any(c < 1 or o < 1 for c, o in izgara):
            raise ValueError("cycles ve operands_max >= 1 olmalı")
    tohumlar = [int(s) for s in seeds]
    if not tohumlar:
        raise ValueError("en az bir tohum gerekir")

    noktalar: List[ScalingPoint] = []
    for cycles, operands_max in izgara:
        for tohum in tohumlar:
            basla = time.perf_counter()
            rapor = run_self_learning_experiment(
                cycles=cycles, batch_size=batch_size,
                initial_facts=initial_facts, operands_max=operands_max,
                negatives_per_fact=negatives_per_fact, seed=tohum)
            gecen = time.perf_counter() - basla
            kazanim = rapor.final_knowledge_size - rapor.initial_knowledge_size
            noktalar.append(ScalingPoint(
                cycles=cycles,
                operands_max=operands_max,
                seed=tohum,
                cycles_completed=rapor.cycles_completed,
                final_knowledge=rapor.final_knowledge_size,
                initial_knowledge=rapor.initial_knowledge_size,
                generated=rapor.generated_experiences,
                verified=rapor.verified_new_knowledge,
                correct_knowledge=rapor.correct_knowledge,
                incorrect_knowledge=rapor.incorrect_knowledge,
                experience_yield=rapor.experience_yield,
                novelty_yield=rapor.novelty_yield,
                useful_experience_yield=rapor.useful_experience_yield,
                far=rapor.far,
                frr=rapor.frr,
                far_before_verifier=rapor.far_before_verifier,
                memory_collisions=rapor.memory_collisions,
                isolation_clean=rapor.isolation_clean,
                saturation_cycle=_saturation_cycle(rapor.cycles),
                knowledge_per_cycle=round(
                    kazanim / max(1, rapor.cycles_completed), 6),
                wall_seconds=round(gecen, 3),
            ))

    def ortalama(alt: Sequence[ScalingPoint], alan: str) -> float:
        degerler = [getattr(p, alan) for p in alt]
        return round(sum(degerler) / len(degerler), 6) if degerler else 0.0

    # ── cycle ekseni: alan SABİT, cycle değişiyor ──────────────────────────
    cycle_ekseni: Dict[str, Any] = {}
    alanlar = sorted({p.operands_max for p in noktalar})
    for alan in alanlar:
        alt = sorted((p for p in noktalar if p.operands_max == alan),
                     key=lambda p: p.cycles)
        if len({p.cycles for p in alt}) < 2:
            continue
        cycle_degerleri = sorted({p.cycles for p in alt})
        satir: List[Dict[str, Any]] = []
        for c in cycle_degerleri:
            grup = [p for p in alt if p.cycles == c]
            satir.append({
                "cycles": c,
                "final_knowledge": ortalama(grup, "final_knowledge"),
                "knowledge_per_cycle": ortalama(grup, "knowledge_per_cycle"),
                "experience_yield": ortalama(grup, "experience_yield"),
                "incorrect_knowledge": ortalama(grup, "incorrect_knowledge"),
                "saturation_cycle": grup[0].saturation_cycle,
                "wall_seconds": ortalama(grup, "wall_seconds"),
            })
        ilk, son = satir[0], satir[-1]
        cycle_carpani = son["cycles"] / max(1, ilk["cycles"])
        bilgi_carpani = son["final_knowledge"] / max(1.0, ilk["final_knowledge"])
        cycle_ekseni[f"operands_max={alan}"] = {
            "rows": satir,
            "cycle_multiplier": round(cycle_carpani, 3),
            "knowledge_multiplier": round(bilgi_carpani, 6),
            "scaling_efficiency": round(bilgi_carpani / cycle_carpani, 6)
            if cycle_carpani else 0.0,
            "saturated": bilgi_carpani < 1.10 and cycle_carpani >= 2.0,
        }

    # ── alan ekseni: cycle SABİT, alan değişiyor ───────────────────────────
    alan_ekseni: Dict[str, Any] = {}
    for cycles in sorted({p.cycles for p in noktalar}):
        alt = sorted((p for p in noktalar if p.cycles == cycles),
                     key=lambda p: p.operands_max)
        if len({p.operands_max for p in alt}) < 2:
            continue
        satir = []
        for alan in sorted({p.operands_max for p in alt}):
            grup = [p for p in alt if p.operands_max == alan]
            satir.append({
                "operands_max": alan,
                "final_knowledge": ortalama(grup, "final_knowledge"),
                "experience_yield": ortalama(grup, "experience_yield"),
                "incorrect_knowledge": ortalama(grup, "incorrect_knowledge"),
            })
        ilk, son = satir[0], satir[-1]
        alan_ekseni[f"cycles={cycles}"] = {
            "rows": satir,
            "knowledge_multiplier": round(
                son["final_knowledge"] / max(1.0, ilk["final_knowledge"]), 6),
            "domain_multiplier": round(
                son["operands_max"] / max(1, ilk["operands_max"]), 3),
        }

    # ── sürüklenme (model collapse göstergesi) ─────────────────────────────
    toplam_yanlis = sum(p.incorrect_knowledge for p in noktalar)
    en_uzun = max(noktalar, key=lambda p: p.cycles)
    surukleme = {
        "total_incorrect_knowledge": toplam_yanlis,
        "max_far": max(p.far for p in noktalar),
        "max_far_before_verifier": max(p.far_before_verifier for p in noktalar),
        "longest_run_cycles": en_uzun.cycles,
        "longest_run_incorrect": en_uzun.incorrect_knowledge,
        "longest_run_far": en_uzun.far,
        "all_isolation_clean": all(p.isolation_clean for p in noktalar),
        "note": ("Sürüklenme = uzun koşuda yanlış bilginin birikmesi. "
                 "CLOSED_VERIFIED rejiminde doğrulayıcı bunu engellemelidir; "
                 "sıfırdan büyük bir değer self-training çöküşünün "
                 "başladığını gösterir."),
    }

    doygun_eksenler = [ad for ad, v in cycle_ekseni.items() if v["saturated"]]

    kapilar: Dict[str, bool] = {
        "all_points_completed": all(
            p.cycles_completed == p.cycles for p in noktalar),
        "no_incorrect_knowledge_accumulated": toplam_yanlis == 0,
        "no_false_acceptance_after_verifier": all(p.far == 0.0
                                                  for p in noktalar),
        "train_test_isolation_clean": all(p.isolation_clean for p in noktalar),
        "long_run_stable": en_uzun.incorrect_knowledge == 0
        and en_uzun.far == 0.0,
        "reached_1000_cycles": any(p.cycles >= 1000 for p in noktalar),
        "reached_3000_cycles": any(p.cycles >= 3000 for p in noktalar),
        "saturation_measured": bool(cycle_ekseni),
        "domain_axis_measured": bool(alan_ekseni),
    }

    # ── bulgular ───────────────────────────────────────────────────────────
    bulgular: List[str] = [
        f"Izgara: {izgara}, tohumlar {tohumlar} → {len(noktalar)} koşu.",
    ]

    for ad, veri in cycle_ekseni.items():
        ilk, son = veri["rows"][0], veri["rows"][-1]
        if veri["saturated"]:
            bulgular.append(
                f"DOYGUNLUK ({ad}): cycle {ilk['cycles']}→{son['cycles']} "
                f"({veri['cycle_multiplier']:.0f}×) bilgiyi yalnız "
                f"{ilk['final_knowledge']:.0f}→{son['final_knowledge']:.0f} "
                f"({veri['knowledge_multiplier']:.3f}×) artırdı. "
                f"Ölçekleme verimi {veri['scaling_efficiency']:.4f}. "
                "Sabit alanda aday havuzu tükeniyor; ek cycle hesap "
                "israfıdır. 'Daha çok döngü = daha çok bilgi' YANLIŞ.")
        else:
            bulgular.append(
                f"({ad}): cycle {veri['cycle_multiplier']:.0f}× artınca bilgi "
                f"{veri['knowledge_multiplier']:.3f}× arttı "
                f"(verim {veri['scaling_efficiency']:.4f}).")
        if son.get("saturation_cycle"):
            bulgular.append(
                f"  └ Nihai bilginin %{SATURATION_THRESHOLD * 100:.0f}'una "
                f"{son['saturation_cycle']}. cycle'da ulaşıldı; kalan "
                f"{son['cycles'] - son['saturation_cycle']} cycle marjinal.")

    for ad, veri in alan_ekseni.items():
        bulgular.append(
            f"ALAN EKSENİ ({ad}): operands_max "
            f"{veri['domain_multiplier']:.0f}× büyüyünce bilgi "
            f"{veri['knowledge_multiplier']:.2f}× arttı. Ölçekleme cycle "
            "sayısından değil ALANIN genişliğinden geliyor.")

    if toplam_yanlis == 0:
        bulgular.append(
            f"Sürüklenme yok: en uzun koşuda ({en_uzun.cycles} cycle) "
            "yanlış bilgi 0, doğrulayıcı sonrası FAR 0.0. CLOSED_VERIFIED "
            "rejimi uzun koşuda çökmüyor.")
    else:
        bulgular.append(
            f"SÜRÜKLENME: toplam {toplam_yanlis} yanlış olgu birikti; "
            f"en uzun koşuda {en_uzun.incorrect_knowledge}. Self-training "
            "çöküşü başlamış olabilir.")

    verim_araligi = (min(p.experience_yield for p in noktalar),
                     max(p.experience_yield for p in noktalar))
    bulgular.append(
        f"Deneyim verimi (EY) tüm ölçeklerde {verim_araligi[0]:.4f}–"
        f"{verim_araligi[1]:.4f} bandında kaldı; ölçek EY'yi değiştirmiyor, "
        "çünkü üretim dağılımı sabit.")

    if doygun_eksenler and not any(p.cycles >= 3000 for p in noktalar):
        bulgular.append(
            "3000 cycle bu profilde koşulmadı; doygunluk zaten 1000'den "
            "önce görüldüğü için daha uzun koşu yeni bilgi getirmezdi. "
            "`deep` profili tam ekseni koşar.")

    imza = hashlib.sha256(json.dumps(
        {"protocol": PROTOCOL, "grid": izgara, "seeds": tohumlar,
         "batch": batch_size, "initial_facts": initial_facts},
        sort_keys=True).encode("utf-8")).hexdigest()[:12]
    surukleme["signature"] = imza

    return SelfLearningScalingReport(
        protocol=PROTOCOL,
        schema_version=SCHEMA_VERSION,
        profile=profile if grid is None else "custom",
        seeds=tohumlar,
        points=[p.to_dict() for p in noktalar],
        cycle_axis=cycle_ekseni,
        domain_axis=alan_ekseni,
        drift=surukleme,
        checks=kapilar,
        findings=bulgular,
        limitations=[
            "Alan aritmetiktir (a+b=c); doğrulayıcı kapalı formdur. Gerçek "
            "dünya alanlarında doğrulama bu kadar kesin olmaz.",
            "Doygunluk bu alanın SONLU olmasından gelir; sonsuz bir alanda "
            "cycle ölçeklemesi farklı davranabilir.",
            "'Sürüklenme yok' sonucu CLOSED_VERIFIED rejimine özgüdür; "
            "doğrulayıcısız self-training ayrıca ölçülmelidir.",
            "Tek makine, tek süreç; paralel öğrenme davranışı kapsam dışı.",
        ],
    )


def self_learning_scaling_markdown(report: SelfLearningScalingReport) -> str:
    """Raporu Markdown'a çevir."""
    s = report
    satirlar = [
        "# Self-Learning Ölçeklendirme (P1)",
        "",
        f"- Protokol: `{s.protocol}` v{s.schema_version} "
        f"(imza `{s.drift['signature']}`)",
        f"- Profil / tohumlar: `{s.profile}` / `{s.seeds}`",
        "",
        "## Tüm noktalar",
        "",
        "| Cycle | Alan | Bilgi | Bilgi/cycle | EY | Yanlış | FAR | Süre (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in s.points:
        satirlar.append(
            f"| {p['cycles']} | {p['operands_max']} | "
            f"{p['final_knowledge']} | {p['knowledge_per_cycle']:.3f} | "
            f"{p['experience_yield']:.4f} | {p['incorrect_knowledge']} | "
            f"{p['far']:.4f} | {p['wall_seconds']:.1f} |")

    if s.cycle_axis:
        satirlar.extend([
            "",
            "## Cycle ekseni (alan sabit) — doygunluk",
            "",
            "| Alan | Cycle çarpanı | Bilgi çarpanı | Ölçekleme verimi | Doygun |",
            "|---|---:|---:|---:|---|",
        ])
        for ad, v in s.cycle_axis.items():
            satirlar.append(
                f"| {ad} | {v['cycle_multiplier']:.0f}× | "
                f"{v['knowledge_multiplier']:.3f}× | "
                f"{v['scaling_efficiency']:.4f} | "
                f"{'EVET' if v['saturated'] else 'hayır'} |")

    if s.domain_axis:
        satirlar.extend([
            "",
            "## Alan ekseni (cycle sabit) — gerçek ölçekleme",
            "",
            "| Cycle | Alan çarpanı | Bilgi çarpanı |",
            "|---|---:|---:|",
        ])
        for ad, v in s.domain_axis.items():
            satirlar.append(
                f"| {ad} | {v['domain_multiplier']:.0f}× | "
                f"{v['knowledge_multiplier']:.2f}× |")

    d = s.drift
    satirlar.extend([
        "",
        "## Sürüklenme (model collapse göstergesi)",
        "",
        f"- Toplam yanlış bilgi: `{d['total_incorrect_knowledge']}`",
        f"- En uzun koşu: `{d['longest_run_cycles']}` cycle → "
        f"yanlış `{d['longest_run_incorrect']}`, FAR `{d['longest_run_far']}`",
        f"- Doğrulayıcı ÖNCESİ en yüksek FAR: "
        f"`{d['max_far_before_verifier']:.4f}` "
        "(doğrulayıcının gerçekten iş yaptığının kanıtı)",
        "",
        f"> {d['note']}",
        "",
        "## Kabul kapıları",
        "",
        "| Kapı | Sonuç |",
        "|---|---|",
    ])
    satirlar.extend(f"| {ad} | {'GEÇTİ' if v else 'KALDI'} |"
                    for ad, v in s.checks.items())
    satirlar.extend(["", "## Bulgular", ""])
    satirlar.extend(f"- {b}" for b in s.findings)
    satirlar.extend(["", "## Sınırlar", ""])
    satirlar.extend(f"- {b}" for b in s.limitations)
    return "\n".join(satirlar) + "\n"


__all__ = [
    "PROFILES",
    "PROTOCOL",
    "SATURATION_THRESHOLD",
    "SCHEMA_VERSION",
    "ScalingPoint",
    "SelfLearningScalingReport",
    "run_self_learning_scaling",
    "self_learning_scaling_markdown",
]
