"""Operatör öğrenme baseline ailesi: Kronecker vs **gerçek** Dense (P0-2).

Neden bu modül var
------------------
``hga/evaluation/kronecker.py`` Kronecker'ı yalnız ``n² → 1 → n²`` rank-1
darboğazına karşı ölçüyordu. O baseline kasıtlı olarak zayıftır; ondan
``Kronecker > Dense`` sonucu çıkarılamaz. Bu modül karşılaştırmayı gerçek
dense operatörlere yükseltir ve **iki ayrı adalet rejimini** ayırır:

``P`` = eğitilebilir parametre, ``F`` = forward çarpma-toplama sayısı,
``d = n²`` (operatörün vektör boyutu):

===================  ===================  =====================  ==========================
Kol                  P                    F                      Rejim
===================  ===================  =====================  ==========================
``kronecker``        ``2n²``              ``2n³``                referans
``rank1``            ``2n²``              ``2n²``                eşit-parametre, ucuz FLOP
``low_rank_r``       ``2rn²``             ``2rn²``               eşit-parametre (r seçilir)
``low_rank_flop``    ``2n³``              ``2n³``                **eşit-FLOP** dense
``kron_sum_R``       ``2Rn²``             ``2Rn³``               R terimli Kronecker toplamı
``full_dense``       ``n⁴``               ``n⁴``                 kısıtsız tavan
===================  ===================  =====================  ==========================

``full_dense`` kasıtlı olarak bütçe-dışıdır: o bir rakip değil **tavandır**.
"Kronecker kazandı" demek için tavanı yenmek gerekmez; ama tavanın ne kadar
gerisinde kalındığı raporlanmadan hiçbir şey iddia edilemez.

Hangi problem sınıfı?
---------------------
Tek bir öğretmen görevinden evrensel üstünlük çıkmaz. Dört öğretmen ailesi
koşulur ve her birinde kolların sırası ayrı raporlanır:

* ``kronecker_teacher``   — ``Y = A* X B*`` (ayrılabilir/geometrik yapı)
* ``rank1_teacher``       — ``vec(Y) = u (vᵀ vec(X))`` (darboğaz yapısı)
* ``low_rank_teacher``    — rank-``r`` dense operatör (kısmen yapılı)
* ``full_dense_teacher``  — tamamen rastgele dense operatör (yapısız)

Beklenti açıkça yazılıdır ve testle kilitlidir: Kronecker inductive bias'ı
**yapılı** görevlerde avantaj sağlamalı, **yapısız** dense öğretmende eşit
parametre bütçesinde geri kalmalıdır. İkincisi gerçekleşmezse ölçüm
şüphelidir, çünkü 2n² parametreyle n⁴ serbestlikli rastgele bir operatör
öğrenilemez.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence

TEACHERS = ("kronecker_teacher", "rank1_teacher", "low_rank_teacher",
            "full_dense_teacher")
ARMS = ("kronecker", "rank1", "low_rank_param_matched", "low_rank_flop_matched",
        "kron_sum_2", "full_dense")
PROTOCOL = "operator_baseline_family_v1"
OFFICIAL_SEED_REQUIREMENT = 20
OFFICIAL_SEEDS = tuple(range(1, OFFICIAL_SEED_REQUIREMENT + 1))

#: Yapılı (ayrılabilir/düşük-ranklı) öğretmenler — geometrik bias lehine olması
#: BEKLENEN görevler. Yapısız öğretmende beklenti tersidir.
STRUCTURED_TEACHERS = ("kronecker_teacher",)
UNSTRUCTURED_TEACHERS = ("full_dense_teacher",)


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as error:  # pragma: no cover - ortama bağlı
        raise ImportError(
            "Operatör baseline ailesi için PyTorch gereklidir") from error
    return torch, nn


# ── analitik bütçe muhasebesi (torch gerektirmez) ───────────────────────────
def arm_budget(arm: str, n: int, low_rank_r: int = 2, kron_terms: int = 2) -> Dict[str, int]:
    """Bir kolun parametre ve forward FLOP (çarpma-toplama) bütçesi."""
    if n < 2:
        raise ValueError("n >= 2 olmalı")
    d = n * n
    if arm == "kronecker":
        return {"parameters": 2 * d, "flops": 2 * n ** 3}
    if arm == "rank1":
        return {"parameters": 2 * d, "flops": 2 * d}
    if arm == "low_rank_param_matched":
        return {"parameters": 2 * low_rank_r * d, "flops": 2 * low_rank_r * d}
    if arm == "low_rank_flop_matched":
        return {"parameters": 2 * n * d, "flops": 2 * n * d}
    if arm.startswith("kron_sum_"):
        terms = int(arm.rsplit("_", 1)[1])
        return {"parameters": 2 * terms * d, "flops": 2 * terms * n ** 3}
    if arm == "full_dense":
        return {"parameters": d * d, "flops": d * d}
    raise ValueError(f"bilinmeyen kol: {arm}")


def budget_table(n: int, arms: Sequence[str] = ARMS,
                 low_rank_r: int = 2) -> Dict[str, Dict[str, Any]]:
    """Tüm kolların bütçesi + referansa (Kronecker) göre oranlar."""
    referans = arm_budget("kronecker", n)
    tablo: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        b = arm_budget(arm, n, low_rank_r=low_rank_r)
        tablo[arm] = {
            **b,
            "parameter_ratio_to_kronecker": round(
                b["parameters"] / referans["parameters"], 6),
            "flop_ratio_to_kronecker": round(b["flops"] / referans["flops"], 6),
            "parameter_matched": b["parameters"] == referans["parameters"],
            "flop_matched": b["flops"] == referans["flops"],
        }
    return tablo


# ── model kolları ───────────────────────────────────────────────────────────
def _build_arm(arm: str, n: int, torch, nn, low_rank_r: int = 2):
    d = n * n

    class KroneckerArm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.A = nn.Parameter(torch.empty(n, n))
            self.B = nn.Parameter(torch.empty(n, n))
            nn.init.xavier_uniform_(self.A)
            nn.init.xavier_uniform_(self.B)

        def forward(self, x):
            return torch.matmul(self.A, torch.matmul(x, self.B))

    class KronSumArm(nn.Module):
        """R terimli Kronecker toplamı: ``Σ_r A_r X B_r`` (Tensor-Train benzeri)."""

        def __init__(self, terms: int) -> None:
            super().__init__()
            self.A = nn.Parameter(torch.empty(terms, n, n))
            self.B = nn.Parameter(torch.empty(terms, n, n))
            for t in range(terms):
                nn.init.xavier_uniform_(self.A[t])
                nn.init.xavier_uniform_(self.B[t])

        def forward(self, x):
            # x: (batch, n, n) → her terim için A_r X B_r, sonra topla.
            genis = x.unsqueeze(1)                      # (b, 1, n, n)
            sol = torch.matmul(self.A.unsqueeze(0), genis)
            return torch.matmul(sol, self.B.unsqueeze(0)).sum(dim=1)

    class LowRankArm(nn.Module):
        def __init__(self, rank: int) -> None:
            super().__init__()
            self.down = nn.Linear(d, rank, bias=False)
            self.up = nn.Linear(rank, d, bias=False)

        def forward(self, x):
            flat = x.reshape(x.shape[0], d)
            return self.up(self.down(flat)).reshape(x.shape[0], n, n)

    class FullDenseArm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(d, d, bias=False)

        def forward(self, x):
            flat = x.reshape(x.shape[0], d)
            return self.linear(flat).reshape(x.shape[0], n, n)

    if arm == "kronecker":
        return KroneckerArm()
    if arm == "rank1":
        return LowRankArm(1)
    if arm == "low_rank_param_matched":
        return LowRankArm(low_rank_r)
    if arm == "low_rank_flop_matched":
        return LowRankArm(n)
    if arm.startswith("kron_sum_"):
        return KronSumArm(int(arm.rsplit("_", 1)[1]))
    if arm == "full_dense":
        return FullDenseArm()
    raise ValueError(f"bilinmeyen kol: {arm}")


def _teacher(task: str, n: int, torch, generator, low_rank_r: int = 4):
    """Öğretmen operatörünü kapanış olarak üret."""
    d = n * n
    if task == "kronecker_teacher":
        left = torch.randn(n, n, generator=generator) / math.sqrt(n)
        right = torch.randn(n, n, generator=generator) / math.sqrt(n)

        def fn(x):
            return torch.matmul(left, torch.matmul(x, right))
        return fn
    if task == "rank1_teacher":
        out_vec = torch.randn(d, generator=generator)
        in_vec = torch.randn(d, generator=generator) / math.sqrt(d)

        def fn(x):
            flat = x.reshape(*x.shape[:-2], d)
            scalar = torch.matmul(flat, in_vec)
            return (scalar.unsqueeze(-1) * out_vec).reshape(*x.shape[:-2], n, n)
        return fn
    if task == "low_rank_teacher":
        u = torch.randn(d, low_rank_r, generator=generator) / math.sqrt(low_rank_r)
        v = torch.randn(low_rank_r, d, generator=generator) / math.sqrt(d)
        operator = u @ v

        def fn(x):
            flat = x.reshape(*x.shape[:-2], d)
            return torch.matmul(flat, operator.T).reshape(*x.shape[:-2], n, n)
        return fn
    if task == "full_dense_teacher":
        operator = torch.randn(d, d, generator=generator) / math.sqrt(d)

        def fn(x):
            flat = x.reshape(*x.shape[:-2], d)
            return torch.matmul(flat, operator.T).reshape(*x.shape[:-2], n, n)
        return fn
    raise ValueError(f"bilinmeyen öğretmen: {task}")


def _metrics(torch, prediction, target) -> Dict[str, float]:
    error = prediction - target
    mse = float(torch.mean(error.square()).item())
    energy = float(torch.mean(target.square()).item())
    centered = float(torch.mean((target - target.mean()).square()).item())
    return {
        "mse": round(mse, 10),
        "normalized_mse": round(mse / max(energy, 1e-12), 10),
        "r2": round(1.0 - mse / max(centered, 1e-12), 10),
    }


@dataclass
class ArmResult:
    arm: str
    parameters: int
    analytic_parameters: int
    analytic_flops: int
    train_normalized_mse: float
    test_normalized_mse: float
    test_r2: float
    training_seconds: float
    optimization_stable: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OperatorBaselineReport:
    protocol: str
    n: int
    seeds: List[int]
    arms: List[str]
    teachers: List[str]
    steps: int
    batch_size: int
    dataset_hash: str
    budget: Dict[str, Dict[str, Any]]
    results: Dict[str, Dict[str, Dict[str, Any]]]   # teacher → arm → özet
    rankings: Dict[str, List[str]]                  # teacher → en iyiden kötüye
    parameter_matched_rankings: Dict[str, List[str]]
    ceiling_gap: Dict[str, float]                   # teacher → kron nMSE − dense nMSE
    checks: Dict[str, bool]
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        return operator_baseline_markdown(self)


def _mean(values: Sequence[float]) -> float:
    return round(statistics.fmean(values), 10) if values else float("nan")


def _std(values: Sequence[float]) -> float:
    return round(statistics.stdev(values), 10) if len(values) > 1 else 0.0


def _ci95_mean(values: Sequence[float]) -> List[float]:
    """Normal-yaklaşım %95 ortalama güven aralığı (küçük rapor bağımlılıksız)."""
    if not values:
        return [float("nan"), float("nan")]
    ort = statistics.fmean(values)
    if len(values) == 1:
        return [round(ort, 10), round(ort, 10)]
    yaricap = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return [round(max(0.0, ort - yaricap), 10), round(ort + yaricap, 10)]


def run_operator_baseline_benchmark(
    n: int = 8,
    steps: int = 300,
    batch_size: int = 16,
    test_samples: int = 256,
    learning_rate: float = 0.01,
    seeds: Sequence[int] = (1, 2, 3),
    arms: Sequence[str] = ARMS,
    teachers: Sequence[str] = TEACHERS,
    low_rank_r: int = 2,
    device: str = "cpu",
) -> OperatorBaselineReport:
    """Aynı veri, optimizer, adım ve başlangıç protokolünde kol ailesini koş.

    Adalet sözleşmesi: her (öğretmen, tohum) için TÜM kollar **aynı** eğitim
    ve test tensörlerini görür; her kol aynı Adam ayarlarıyla aynı sayıda adım
    atar. Kollar arasında değişen tek şey operatörün parametrizasyonudur.
    """
    torch, nn = _torch()
    if n < 2 or steps < 1 or batch_size < 1 or test_samples < 1:
        raise ValueError("n>=2 ve steps/batch_size/test_samples >=1 olmalı")
    arms = list(arms)
    teachers = list(teachers)
    seeds = [int(s) for s in seeds]
    if not arms or not teachers or not seeds:
        raise ValueError("arms, teachers ve seeds boş olamaz")
    bilinmeyen = [a for a in arms if a not in ARMS]
    if bilinmeyen:
        raise ValueError(f"bilinmeyen kollar: {bilinmeyen}")
    bilinmeyen_t = [t for t in teachers if t not in TEACHERS]
    if bilinmeyen_t:
        raise ValueError(f"bilinmeyen öğretmenler: {bilinmeyen_t}")

    hedef = torch.device(device)
    butce = budget_table(n, arms, low_rank_r=low_rank_r)
    ham: Dict[str, Dict[str, List[ArmResult]]] = {t: {a: [] for a in arms}
                                                  for t in teachers}

    for task in teachers:
        for seed in seeds:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            train_x = torch.randn(steps, batch_size, n, n, generator=gen)
            test_x = torch.randn(test_samples, n, n, generator=gen)
            fn = _teacher(task, n, torch, gen)
            train_y, test_y = fn(train_x), fn(test_x)
            train_x, train_y = train_x.to(hedef), train_y.to(hedef)
            test_x, test_y = test_x.to(hedef), test_y.to(hedef)

            for arm in arms:
                torch.manual_seed(seed)
                model = _build_arm(arm, n, torch, nn, low_rank_r=low_rank_r).to(hedef)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                kararli = True
                basladi = time.perf_counter()
                for step in range(steps):
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.mean((model(train_x[step]) - train_y[step]).square())
                    if not bool(torch.isfinite(loss)):
                        kararli = False
                        continue
                    loss.backward()
                    optimizer.step()
                sure = time.perf_counter() - basladi
                with torch.no_grad():
                    tr = _metrics(torch, model(train_x[-1]), train_y[-1])
                    te = _metrics(torch, model(test_x), test_y)
                ham[task][arm].append(ArmResult(
                    arm=arm,
                    parameters=sum(p.numel() for p in model.parameters()
                                   if p.requires_grad),
                    analytic_parameters=butce[arm]["parameters"],
                    analytic_flops=butce[arm]["flops"],
                    train_normalized_mse=tr["normalized_mse"],
                    test_normalized_mse=te["normalized_mse"],
                    test_r2=te["r2"],
                    training_seconds=round(sure, 6),
                    optimization_stable=kararli,
                ))

    # ── özet ───────────────────────────────────────────────────────────────
    sonuclar: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for task in teachers:
        sonuclar[task] = {}
        for arm in arms:
            kosular = ham[task][arm]
            nmse = [r.test_normalized_mse for r in kosular]
            sonuclar[task][arm] = {
                "parameters": kosular[0].parameters,
                "analytic_parameters": kosular[0].analytic_parameters,
                "analytic_flops": kosular[0].analytic_flops,
                "test_normalized_mse_mean": _mean(nmse),
                "test_normalized_mse_std": _std(nmse),
                "test_normalized_mse_ci95": _ci95_mean(nmse),
                "seed_count": len(kosular),
                "test_r2_mean": _mean([r.test_r2 for r in kosular]),
                "train_normalized_mse_mean": _mean(
                    [r.train_normalized_mse for r in kosular]),
                "training_seconds_mean": _mean([r.training_seconds for r in kosular]),
                "optimization_stable": all(r.optimization_stable for r in kosular),
                "per_seed_test_normalized_mse": nmse,
            }
        # Analitik parametre sayısı gerçek sayımla tutmalı; tutmuyorsa
        # bütün adalet iddiası çöker.
        for arm in arms:
            gercek = sonuclar[task][arm]["parameters"]
            analitik = sonuclar[task][arm]["analytic_parameters"]
            if gercek != analitik:
                raise AssertionError(
                    f"{arm}: gerçek parametre {gercek} != analitik {analitik}")

    siralama = {
        task: sorted(arms, key=lambda a: sonuclar[task][a]["test_normalized_mse_mean"])
        for task in teachers
    }
    esit_param = [a for a in arms if butce[a]["parameter_matched"]]
    esit_param_siralama = {
        task: sorted(esit_param,
                     key=lambda a: sonuclar[task][a]["test_normalized_mse_mean"])
        for task in teachers
    }
    tavan_farki = {}
    if "full_dense" in arms and "kronecker" in arms:
        for task in teachers:
            tavan_farki[task] = round(
                sonuclar[task]["kronecker"]["test_normalized_mse_mean"]
                - sonuclar[task]["full_dense"]["test_normalized_mse_mean"], 10)

    kontroller: Dict[str, bool] = {
        "parameter_accounting_exact": True,   # yukarıda assert ile garanti
        "all_arms_optimization_stable": all(
            sonuclar[t][a]["optimization_stable"] for t in teachers for a in arms),
        "full_dense_is_over_budget": (
            butce.get("full_dense", {}).get("parameter_ratio_to_kronecker", 0) > 1.0
            if "full_dense" in arms else True),
        "all_teacher_families_present": set(teachers) == set(TEACHERS),
        "all_baseline_arms_present": set(arms) == set(ARMS),
        "official_20_seed_rule_met": len(set(seeds)) >= OFFICIAL_SEED_REQUIREMENT,
    }
    if "kronecker" in arms and "rank1" in arms:
        kontroller["kronecker_beats_rank1_on_structured"] = all(
            sonuclar[t]["kronecker"]["test_normalized_mse_mean"]
            < sonuclar[t]["rank1"]["test_normalized_mse_mean"]
            for t in teachers if t in STRUCTURED_TEACHERS)
    if "kronecker" in arms and "full_dense" in arms:
        # Yapısız öğretmende 2n² parametre ile n⁴ serbestliği yakalamak
        # beklenmez; beklenirse ölçüm şüphelidir.
        kontroller["kronecker_loses_to_dense_on_unstructured"] = all(
            sonuclar[t]["kronecker"]["test_normalized_mse_mean"]
            > sonuclar[t]["full_dense"]["test_normalized_mse_mean"]
            for t in teachers if t in UNSTRUCTURED_TEACHERS)
    if "kronecker" in arms and "low_rank_flop_matched" in arms:
        kontroller["kronecker_beats_flop_matched_dense_on_structured"] = all(
            sonuclar[t]["kronecker"]["test_normalized_mse_mean"]
            < sonuclar[t]["low_rank_flop_matched"]["test_normalized_mse_mean"]
            for t in teachers if t in STRUCTURED_TEACHERS)

    imza = hashlib.sha256(json.dumps({
        "protocol": PROTOCOL, "n": n, "steps": steps, "batch": batch_size,
        "test": test_samples, "lr": learning_rate, "seeds": seeds,
        "arms": arms, "teachers": teachers, "low_rank_r": low_rank_r,
    }, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    bulgular = [
        f"n={n}, d=n²={n * n}; {len(seeds)} tohum × {len(teachers)} öğretmen × "
        f"{len(arms)} kol, her kol {steps} adım.",
    ]
    for task in teachers:
        en_iyi = siralama[task][0]
        bulgular.append(
            f"`{task}`: en iyi kol `{en_iyi}` "
            f"(nMSE {sonuclar[task][en_iyi]['test_normalized_mse_mean']:.6f}); "
            f"eşit-parametre kolları arasında en iyi "
            f"`{esit_param_siralama[task][0]}`.")
    if tavan_farki:
        bulgular.append(
            "Kronecker'ın kısıtsız dense tavanına göre nMSE farkı: "
            + ", ".join(f"{t}: {v:+.6f}" for t, v in tavan_farki.items())
            + ". Pozitif değer Kronecker'ın tavanın gerisinde olduğunu gösterir.")

    tohum_notu = (
        f"Tohum sayısı {len(set(seeds))}; 20 tohum kuralı karşılandı ve rapor "
        "çekirdek istatistiksel iddia için kullanılabilir."
        if len(set(seeds)) >= OFFICIAL_SEED_REQUIREMENT
        else f"Tohum sayısı {len(set(seeds))}; çekirdek bilimsel iddia için "
        "20 tohum hedefi ayrıca koşulmalıdır."
    )
    sinirlar = [
        "Görev tek katmanlı operatör regresyonudur; derin ağ, dil modeli veya "
        "sınıflandırma sonucu DEĞİLDİR.",
        "full_dense kolu parametre bütçesini kasıtlı aşar; rakip değil tavandır. "
        "Onu yenmek beklenmez, ona olan mesafe raporlanır.",
        "FLOP sayıları analitik çarpma-toplama sayımıdır; gerçek donanım "
        "verimi (BLAS, cache, paralellik) ölçülmez — wall-clock ayrıca verilir.",
        "Öğretmenler sentetiktir; 'HGA şu problem sınıfında iyidir' sonucu "
        "yalnız bu sentetik sınıflar için geçerlidir.",
        tohum_notu,
    ]

    return OperatorBaselineReport(
        protocol=PROTOCOL, n=n, seeds=seeds, arms=arms, teachers=teachers,
        steps=steps, batch_size=batch_size, dataset_hash=imza, budget=butce,
        results=sonuclar, rankings=siralama,
        parameter_matched_rankings=esit_param_siralama,
        ceiling_gap=tavan_farki, checks=kontroller,
        findings=bulgular, limitations=sinirlar,
    )


def run_operator_baseline_official_report(
    n: int = 8,
    steps: int = 300,
    batch_size: int = 16,
    test_samples: int = 256,
    learning_rate: float = 0.01,
    seeds: Sequence[int] = OFFICIAL_SEEDS,
    device: str = "cpu",
) -> OperatorBaselineReport:
    """20-tohum resmi operatör baseline raporunu üret.

    Parametreler testlerde hızlandırma için override edilebilir; varsayılanlar
    dokümante edilen resmi rapor ayarlarıdır.
    """
    return run_operator_baseline_benchmark(
        n=n, steps=steps, batch_size=batch_size, test_samples=test_samples,
        learning_rate=learning_rate, seeds=seeds, arms=ARMS, teachers=TEACHERS,
        device=device,
    )


def operator_baseline_markdown(report: OperatorBaselineReport) -> str:
    satirlar = [
        "# Operatör Baseline Ailesi — Kronecker vs Gerçek Dense (P0-2)",
        "",
        f"Protokol: `{report.protocol}` · veri imzası: `{report.dataset_hash}` · "
        f"n={report.n} · tohumlar: {report.seeds} · adım: {report.steps}",
        "",
        "## Bütçe muhasebesi",
        "",
        "| Kol | Parametre | FLOP | P oranı (Kron=1) | F oranı | Eşit-P | Eşit-F |",
        "|---|---:|---:|---:|---:|:--:|:--:|",
    ]
    for arm in report.arms:
        b = report.budget[arm]
        satirlar.append(
            f"| `{arm}` | {b['parameters']:,} | {b['flops']:,} | "
            f"{b['parameter_ratio_to_kronecker']:.3f} | "
            f"{b['flop_ratio_to_kronecker']:.3f} | "
            f"{'✓' if b['parameter_matched'] else '—'} | "
            f"{'✓' if b['flop_matched'] else '—'} |")

    for task in report.teachers:
        satirlar += ["", f"## Öğretmen: `{task}`", "",
                     "| Kol | test nMSE (ort) | 95% CI | ± std | R² | eğitim sn | P |",
                     "|---|---:|---:|---:|---:|---:|---:|"]
        for arm in report.rankings[task]:
            r = report.results[task][arm]
            ci = r.get("test_normalized_mse_ci95", [
                r["test_normalized_mse_mean"], r["test_normalized_mse_mean"]])
            satirlar.append(
                f"| `{arm}` | {r['test_normalized_mse_mean']:.6f} | "
                f"[{ci[0]:.6f}, {ci[1]:.6f}] | "
                f"{r['test_normalized_mse_std']:.6f} | {r['test_r2_mean']:.4f} | "
                f"{r['training_seconds_mean']:.3f} | {r['parameters']:,} |")

    satirlar += ["", "## Kabul kapıları", "", "| kapı | sonuç |", "|---|---|"]
    for ad, sonuc in report.checks.items():
        satirlar.append(f"| {ad} | {'GEÇTİ' if sonuc else 'KALDI'} |")
    satirlar += ["", "## Bulgular", ""]
    satirlar += [f"- {b}" for b in report.findings]
    satirlar += ["", "## Sınırlar", ""]
    satirlar += [f"- {s}" for s in report.limitations]
    return "\n".join(satirlar) + "\n"


__all__ = [
    "PROTOCOL", "OFFICIAL_SEED_REQUIREMENT", "OFFICIAL_SEEDS", "ARMS", "TEACHERS",
    "STRUCTURED_TEACHERS", "UNSTRUCTURED_TEACHERS", "ArmResult",
    "OperatorBaselineReport", "arm_budget", "budget_table",
    "run_operator_baseline_benchmark", "run_operator_baseline_official_report",
    "operator_baseline_markdown",
]
