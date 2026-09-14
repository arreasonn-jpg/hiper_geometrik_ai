# -*- coding: utf-8 -*-
"""P1 — Çekirdek protokoller için 20 tohum + istatistiksel çıkarım.

Kullanıcı kuralı nettir: **çekirdek bilimsel sonuçlar 20 tohum**, 5 tohum
yalnız duman/mühendislik içindir. Ama tohum sayısını artırmak tek başına
hiçbir şey kanıtlamaz. 20 tohumun anlamı, üzerine şunların kurulabilmesidir:

* ortalama **ve** örneklem standart sapması,
* %95 bootstrap güven aralığı,
* etki büyüklüğü (eşleşmiş Cohen's d),
* eşleşmiş permütasyon testi ve Wilcoxon işaretli sıra testi.

Kritik nokta **eşleşme**dir: aynı tohum, aynı veri havuzu, aynı başlangıç
koşulları altında iki kol karşılaştırılır. Eşleşmiş tasarım, tohumdan gelen
varyansı ortadan kaldırır; eşleşmemiş bir karşılaştırma aynı n ile çok daha
zayıftır.

İkinci kritik nokta: bu modül **kararı p-değerine bırakmaz**. Farkın güven
aralığı sıfırı içeriyorsa sonuç "anlamlı değil" diye raporlanır, p<0.05 olsa
bile. `hga/evaluation/statistics.py` içindeki `compare_paired` zaten bu
ihtiyatlı hükmü üretir; burada onu protokol çıktılarına bağlıyoruz.

Ayrıca **güç (power) sınırı** açıkça yazılır: n=20 eşleşmiş permütasyon
testinin ulaşabileceği en küçük iki yönlü p-değeri sonludur
(`minimum_two_sided_p`), yani çok küçük etkiler bu tasarımda saptanamaz.
Bunu gizlemek yerine raporluyoruz.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .statistics import (
    compare_paired,
    minimum_two_sided_p,
    summarize_seed_metric,
)

PROTOCOL = "core_seed_statistics_v1"
SCHEMA_VERSION = 1

#: Çekirdek bilimsel sonuç için zorunlu tohum sayısı.
CORE_SEED_REQUIREMENT = 20

#: Varsayılan çekirdek tohum kümesi.
CORE_SEEDS: Tuple[int, ...] = tuple(range(1, CORE_SEED_REQUIREMENT + 1))

PROFILES: Dict[str, Tuple[int, ...]] = {
    "smoke": (1, 2, 3),
    "core": CORE_SEEDS,
}


@dataclass
class SeedStatisticsReport:
    """20 tohum denetimi + eşleşmiş karşılaştırmalar."""

    protocol: str
    schema_version: int
    profile: str
    seeds: List[int]
    core_requirement: int
    protocols: Dict[str, Any]
    comparisons: List[Dict[str, Any]]
    power: Dict[str, Any]
    checks: Dict[str, bool]
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _paired_series(report: Dict[str, Any], path: Sequence[str]) -> List[float]:
    """İç içe sözlükten tohum-başı seriyi çek."""
    dugum: Any = report
    for anahtar in path:
        if not isinstance(dugum, dict) or anahtar not in dugum:
            return []
        dugum = dugum[anahtar]
    if isinstance(dugum, list) and all(
            isinstance(v, (int, float)) for v in dugum):
        return [float(v) for v in dugum]
    return []


def _compare(metric: str, treatment: Sequence[float],
             baseline: Sequence[float], treatment_label: str,
             baseline_label: str, context: str) -> Optional[Dict[str, Any]]:
    """Eşleşmiş karşılaştırma; veri yetersizse sessizce atla."""
    if len(treatment) != len(baseline) or len(treatment) < 2:
        return None
    rapor = compare_paired(
        metric=metric, treatment=treatment, baseline=baseline,
        treatment_label=treatment_label, baseline_label=baseline_label)
    kayit = rapor.to_dict()
    kayit["context"] = context
    kayit["n_pairs"] = len(treatment)
    return kayit


def _collect_signature(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Signature benchmark: her görevde HGA'yı her rakiple eşleşmiş kıyasla."""
    karsilastirmalar: List[Dict[str, Any]] = []
    sonuclar = report.get("results") or {}
    for task, kollar in sonuclar.items():
        hga = (kollar or {}).get("hga") or {}
        hga_seri = hga.get("per_seed_accuracy") or []
        if len(hga_seri) < 2:
            continue
        for kol, veri in (kollar or {}).items():
            if kol == "hga":
                continue
            rakip = (veri or {}).get("per_seed_accuracy") or []
            kayit = _compare(
                metric=f"accuracy::{task}", treatment=hga_seri,
                baseline=rakip, treatment_label="hga", baseline_label=kol,
                context=f"signature/{task}")
            if kayit:
                karsilastirmalar.append(kayit)
    return karsilastirmalar


def _collect_operator(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Operatör baseline: Kronecker'ı her rakiple eşleşmiş kıyasla.

    Metrik normalize MSE'dir; **küçük olan iyidir**. Bu yüzden Kronecker
    ``treatment`` olarak verilir ve negatif fark Kronecker lehinedir.
    """
    karsilastirmalar: List[Dict[str, Any]] = []
    for ogretmen, kollar in (report.get("results") or {}).items():
        kron = (kollar or {}).get("kronecker") or {}
        kron_seri = kron.get("per_seed_test_normalized_mse") or []
        if len(kron_seri) < 2:
            continue
        for kol, veri in (kollar or {}).items():
            if kol == "kronecker":
                continue
            rakip = (veri or {}).get("per_seed_test_normalized_mse") or []
            kayit = _compare(
                metric=f"test_normalized_mse::{ogretmen}",
                treatment=kron_seri, baseline=rakip,
                treatment_label="kronecker", baseline_label=kol,
                context=f"operator_baselines/{ogretmen}")
            if kayit:
                kayit["lower_is_better"] = True
                karsilastirmalar.append(kayit)
    return karsilastirmalar


def _collect_priority(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Priority(E): her ağırlık ablasyonunun downstream etkisini özetle.

    Burada eşleşmiş kıyas değil, **tek kolun tohum dağılımı** özetlenir:
    ablasyon farkının kendisi zaten baseline'a göre hesaplanmıştır, yani
    seri baştan eşleşmiş bir fark serisidir.
    """
    ozetler: List[Dict[str, Any]] = []
    for terim, veri in (report.get("variants") or {}).items():
        per_seed = (veri or {}).get("per_seed") or []
        if len(per_seed) < 2:
            continue
        for metrik in ("kendall_tau", "topk_overlap"):
            degerler = [float(k[metrik]) for k in per_seed if metrik in k]
            if len(degerler) < 2:
                continue
            ozet = summarize_seed_metric(degerler)
            ozet.update({"context": f"priority_ablation/{terim}",
                         "metric": metrik, "weight": terim})
            ozetler.append(ozet)
        farklar = [float((k.get("downstream_delta") or {}).get(
            "verification_yield", 0.0)) for k in per_seed]
        if len(farklar) >= 2:
            ozet = summarize_seed_metric(farklar)
            ozet.update({
                "context": f"priority_ablation/{terim}",
                "metric": "downstream_delta::verification_yield",
                "weight": terim,
                "excludes_zero": (ozet["ci_lower"] > 0.0
                                  or ozet["ci_upper"] < 0.0),
            })
            ozetler.append(ozet)
    return ozetler


def run_core_seed_statistics(
    profile: str = "smoke",
    seeds: Optional[Sequence[int]] = None,
    signature_profile: str = "smoke",
    operator_n: int = 8,
    operator_steps: int = 200,
    include_operator: bool = True,
) -> SeedStatisticsReport:
    """Çekirdek protokolleri çok tohumla koş ve istatistiksel çıkarım üret.

    Args:
        profile: ``smoke`` (3 tohum) ya da ``core`` (20 tohum).
        seeds: Verilirse profili ezer.
        signature_profile: Signature benchmark iç profili.
        include_operator: PyTorch gerektiren operatör kolunu koş.

    Raises:
        ValueError: bilinmeyen profil veya boş tohum listesi.
    """
    if seeds is None:
        if profile not in PROFILES:
            raise ValueError(
                f"profile şunlardan biri olmalı: {', '.join(PROFILES)}")
        tohumlar = list(PROFILES[profile])
    else:
        tohumlar = [int(s) for s in seeds]
    if len(tohumlar) < 2:
        raise ValueError("istatistiksel çıkarım için en az 2 tohum gerekir")

    from .priority_ablation import run_priority_weight_ablation
    from .signature import run_signature_benchmark

    protokoller: Dict[str, Any] = {}
    karsilastirmalar: List[Dict[str, Any]] = []

    oncelik = run_priority_weight_ablation(k=10, seeds=tohumlar).to_dict()
    protokoller["priority_ablation"] = {
        "seeds": tohumlar,
        "seed_count": len(tohumlar),
        "meets_core_requirement": len(tohumlar) >= CORE_SEED_REQUIREMENT,
        "dataset_hash": oncelik.get("dataset_hash"),
        "seed_summaries": _collect_priority(oncelik),
    }

    imza = run_signature_benchmark(
        profile=signature_profile, seeds=tohumlar).to_dict()
    imza_kiyas = _collect_signature(imza)
    protokoller["signature"] = {
        "seeds": tohumlar,
        "seed_count": len(tohumlar),
        "meets_core_requirement": len(tohumlar) >= CORE_SEED_REQUIREMENT,
        "dataset_hash": imza.get("dataset_hash"),
        "comparisons": len(imza_kiyas),
    }
    karsilastirmalar.extend(imza_kiyas)

    if include_operator:
        try:
            from .operator_baselines import run_operator_baseline_benchmark
            operator = run_operator_baseline_benchmark(
                n=operator_n, steps=operator_steps, seeds=tohumlar).to_dict()
            op_kiyas = _collect_operator(operator)
            protokoller["operator_baselines"] = {
                "seeds": tohumlar,
                "seed_count": len(tohumlar),
                "meets_core_requirement": (
                    len(tohumlar) >= CORE_SEED_REQUIREMENT),
                "dataset_hash": operator.get("dataset_hash"),
                "comparisons": len(op_kiyas),
            }
            karsilastirmalar.extend(op_kiyas)
        except ImportError as hata:  # PyTorch yoksa dürüstçe atla
            protokoller["operator_baselines"] = {
                "skipped": True, "reason": f"torch yok: {hata}"}

    # ── güç sınırı ─────────────────────────────────────────────────────────
    en_kucuk_p = minimum_two_sided_p(len(tohumlar))
    guc = {
        "n_seeds": len(tohumlar),
        "minimum_attainable_two_sided_p": en_kucuk_p,
        "can_reach_p_0_05": en_kucuk_p <= 0.05,
        "can_reach_p_0_01": en_kucuk_p <= 0.01,
        "note": ("Eşleşmiş permütasyon testinde ulaşılabilecek en küçük iki "
                 "yönlü p-değeri tohum sayısıyla sınırlıdır. Bu sınırın "
                 "üstünde bir 'anlamlılık' iddia edilemez."),
    }

    # ── hüküm dağılımı ─────────────────────────────────────────────────────
    hukumler: Dict[str, int] = {}
    for kiyas in karsilastirmalar:
        hukumler[kiyas["verdict"]] = hukumler.get(kiyas["verdict"], 0) + 1
    anlamli = [k for k in karsilastirmalar
               if not _ci_sifir_iceriyor(k)]

    cekirdek_protokoller = [ad for ad, v in protokoller.items()
                            if not v.get("skipped")]
    kapilar: Dict[str, bool] = {
        "all_core_protocols_use_same_seeds": len({
            tuple(v.get("seeds") or ()) for ad, v in protokoller.items()
            if not v.get("skipped")}) == 1,
        "all_core_protocols_meet_20_seeds": bool(cekirdek_protokoller) and all(
            protokoller[ad].get("meets_core_requirement")
            for ad in cekirdek_protokoller),
        "comparisons_are_paired": all(
            k["n_pairs"] == len(tohumlar) for k in karsilastirmalar),
        "every_comparison_reports_ci": all(
            k.get("difference_ci") for k in karsilastirmalar),
        "every_comparison_reports_effect_size": all(
            k.get("effect_size") for k in karsilastirmalar),
        "every_comparison_reports_two_tests": all(
            k.get("permutation_test") and k.get("wilcoxon_test")
            for k in karsilastirmalar),
        "power_limit_documented": True,
        "design_can_reach_p_0_05": bool(guc["can_reach_p_0_05"]),
    }

    # ── bulgular ───────────────────────────────────────────────────────────
    bulgular: List[str] = [
        f"{len(tohumlar)} tohum × {len(cekirdek_protokoller)} çekirdek "
        f"protokol; {len(karsilastirmalar)} eşleşmiş karşılaştırma üretildi.",
    ]
    if len(tohumlar) >= CORE_SEED_REQUIREMENT:
        bulgular.append(
            f"Çekirdek tohum kuralı ({CORE_SEED_REQUIREMENT}) karşılandı; "
            "bu protokoller artık 'duman testi' değil bilimsel sonuçtur.")
    else:
        bulgular.append(
            f"UYARI: {len(tohumlar)} tohum < {CORE_SEED_REQUIREMENT}. Bu koşum "
            "duman/mühendislik seviyesindedir, bilimsel sonuç sayılmaz.")

    bulgular.append(
        f"Ulaşılabilir en küçük iki yönlü p = {en_kucuk_p:.6f}. "
        + ("Tasarım p<0.05 saptayabilir."
           if guc["can_reach_p_0_05"]
           else "Tasarım p<0.05'e ULAŞAMAZ; tohum artırılmalı."))

    if karsilastirmalar:
        bulgular.append(
            f"Karşılaştırmaların {len(anlamli)}/{len(karsilastirmalar)} "
            "tanesinde farkın %95 güven aralığı sıfırı DIŞLIYOR; geri kalanı "
            "istatistiksel olarak ayırt edilemez. Ayırt edilemeyen sonuçlar "
            "gizlenmedi.")
        for ad, sayi in sorted(hukumler.items()):
            bulgular.append(f"  └ hüküm '{ad}': {sayi} karşılaştırma")

    sifir_dislayan = [o for o in protokoller.get(
        "priority_ablation", {}).get("seed_summaries", [])
        if o.get("excludes_zero")]
    if sifir_dislayan:
        bulgular.append(
            f"Priority(E): {len(sifir_dislayan)} ağırlık ablasyonunda "
            "downstream verification_yield farkının %95 CI'si sıfırı "
            "dışlıyor — bu ağırlıkların nedensel etkisi gerçek.")

    imza_hash = hashlib.sha256(json.dumps(
        {"protocol": PROTOCOL, "seeds": tohumlar,
         "signature_profile": signature_profile},
        sort_keys=True).encode("utf-8")).hexdigest()[:12]
    guc["signature"] = imza_hash

    return SeedStatisticsReport(
        protocol=PROTOCOL,
        schema_version=SCHEMA_VERSION,
        profile=profile if seeds is None else "custom",
        seeds=tohumlar,
        core_requirement=CORE_SEED_REQUIREMENT,
        protocols=protokoller,
        comparisons=karsilastirmalar,
        power=guc,
        checks=kapilar,
        findings=bulgular,
        limitations=[
            "Tohum artırmak varyansı daha iyi tahmin ettirir; sistematik "
            "yanlılığı (veri üretimi, görev tasarımı) düzeltmez.",
            "Eşleşmiş testler aynı tohumun aynı veri havuzunu ürettiğini "
            "varsayar; protokoller bunu garanti eder, dışarıdan verilen "
            "seriler için doğrulanmamıştır.",
            "Çoklu karşılaştırma düzeltmesi (Bonferroni/FDR) uygulanmadı; "
            "çok sayıda kıyasta tek tek p-değerleri iyimserdir.",
            "Bootstrap CI küçük n'de asimptotik değildir; n=20'de aralıklar "
            "gerçek kapsamanın biraz altında kalabilir.",
        ],
    )


def _ci_sifir_iceriyor(kiyas: Dict[str, Any]) -> bool:
    """Farkın güven aralığı sıfırı içeriyor mu?"""
    ci = kiyas.get("difference_ci") or {}
    alt, ust = ci.get("lower"), ci.get("upper")
    if alt is None or ust is None:
        return True
    return bool(alt <= 0.0 <= ust)


def seed_statistics_markdown(report: SeedStatisticsReport) -> str:
    """Raporu Markdown'a çevir."""
    s = report
    satirlar = [
        "# Çekirdek Tohum İstatistikleri (P1)",
        "",
        f"- Protokol: `{s.protocol}` v{s.schema_version} "
        f"(imza `{s.power['signature']}`)",
        f"- Profil: `{s.profile}` · tohum sayısı: **{len(s.seeds)}** "
        f"(çekirdek kural: {s.core_requirement})",
        "",
        "## Protokol tohum disiplini",
        "",
        "| Protokol | Tohum | 20-tohum kuralı |",
        "|---|---:|---|",
    ]
    for ad, veri in s.protocols.items():
        if veri.get("skipped"):
            satirlar.append(f"| {ad} | — | atlandı ({veri.get('reason')}) |")
            continue
        satirlar.append(
            f"| {ad} | {veri['seed_count']} | "
            f"{'GEÇTİ' if veri['meets_core_requirement'] else 'KALDI'} |")

    g = s.power
    satirlar.extend([
        "",
        "## İstatistiksel güç sınırı",
        "",
        f"- n = `{g['n_seeds']}` eşleşmiş tohum",
        f"- Ulaşılabilir en küçük iki yönlü p: `{g['minimum_attainable_two_sided_p']:.6f}`",
        f"- p<0.05 saptanabilir mi: **{'evet' if g['can_reach_p_0_05'] else 'HAYIR'}**",
        f"- p<0.01 saptanabilir mi: **{'evet' if g['can_reach_p_0_01'] else 'HAYIR'}**",
        "",
        f"> {g['note']}",
    ])

    if s.comparisons:
        ayirt_edilen = [k for k in s.comparisons if not _ci_sifir_iceriyor(k)]
        satirlar.extend([
            "",
            f"## Eşleşmiş karşılaştırmalar ({len(s.comparisons)} adet, "
            f"{len(ayirt_edilen)} tanesinde CI sıfırı dışlıyor)",
            "",
            "| Bağlam | İşlem | Kontrol | Δ ortalama | Δ %95 CI | Cohen's d | perm p | Hüküm |",
            "|---|---|---|---:|---|---:|---:|---|",
        ])
        for k in s.comparisons[:60]:
            ci = k.get("difference_ci") or {}
            etki = k.get("effect_size") or {}
            perm = k.get("permutation_test") or {}
            fark = k["treatment_mean"] - k["baseline_mean"]
            satirlar.append(
                f"| {k['context']} | {k['treatment_label']} | "
                f"{k['baseline_label']} | {fark:+.4f} | "
                f"[{ci.get('lower', float('nan')):.4f}, "
                f"{ci.get('upper', float('nan')):.4f}] | "
                f"{etki.get('value', float('nan')):+.3f} | "
                f"{perm.get('p_value', float('nan')):.4f} | "
                f"{k['verdict']} |")
        if len(s.comparisons) > 60:
            satirlar.append(
                f"| … | | | | | | | ({len(s.comparisons) - 60} satır daha) |")

    ozetler = (s.protocols.get("priority_ablation") or {}).get(
        "seed_summaries") or []
    if ozetler:
        satirlar.extend([
            "",
            "## Priority(E) ağırlık ablasyonu — tohum dağılımı",
            "",
            "| Ağırlık | Metrik | n | Ortalama | Std | %95 CI |",
            "|---|---|---:|---:|---:|---|",
        ])
        for o in ozetler:
            satirlar.append(
                f"| {o['weight']} | {o['metric']} | {o['n']} | "
                f"{o['mean']:.4f} | {o['std_sample']:.4f} | "
                f"[{o['ci_lower']:.4f}, {o['ci_upper']:.4f}] |")

    satirlar.extend(["", "## Kabul kapıları", "", "| Kapı | Sonuç |", "|---|---|"])
    satirlar.extend(f"| {ad} | {'GEÇTİ' if v else 'KALDI'} |"
                    for ad, v in s.checks.items())
    satirlar.extend(["", "## Bulgular", ""])
    satirlar.extend(f"- {b}" for b in s.findings)
    satirlar.extend(["", "## Sınırlar", ""])
    satirlar.extend(f"- {b}" for b in s.limitations)
    return "\n".join(satirlar) + "\n"


__all__ = [
    "CORE_SEEDS",
    "CORE_SEED_REQUIREMENT",
    "PROFILES",
    "PROTOCOL",
    "SCHEMA_VERSION",
    "SeedStatisticsReport",
    "run_core_seed_statistics",
    "seed_statistics_markdown",
]
