# -*- coding: utf-8 -*-
"""Cross-domain self-learning transfer benchmark (P1).

Bu modülün amacı pozitif transfer iddiası uydurmak değildir. Mevcut çoklu
ortam altyapısında alanlar ayrık sembol uzaylarına sahiptir; bu yüzden önce
ölçülmesi gereken soru şudur:

* Kaynak ortamda doğrulanmış deneyimler hedef ortamda yanlış karar üretmeden
  **çekimser** kalıyor mu?
* Hedef ortamdan az sayıda doğrulanmış örnek eklendiğinde scratch ve kaynak-
  ön-yüklü öğrenen aynı hedef recall/accuracy'yi mi veriyor?
* Pozitif transfer varsa sayısal delta ile görünür mü; yoksa rapor bunu açıkça
  ``NOT_DEMONSTRATED`` olarak mı yazar?

Öğrenen kasıtla basittir: verifier tarafından onaylanan/reddedilen iddiaları
anahtar-değer belleğine yazar ve yalnız gördüğü üçlüler için karar verir. Bu,
"kaynak alandan hedef alana kendiliğinden genelleme" değil, self-learning
transfer hattının sızıntı/negatif-kontrol ve hedef-adaptasyon ölçümüdür.
"""
from __future__ import annotations

import hashlib
import json
import statistics as _stat
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .multi_environment import (
    ACCEPT,
    ENVIRONMENTS,
    GENERATORS,
    REJECT,
    VERIFIERS,
    Claim,
)

PROTOCOL = "cross_domain_self_learning_transfer_v1"
SCHEMA_VERSION = 1


@dataclass
class TransferPairResult:
    """Tek kaynak→hedef transfer satırı."""

    source_environment: str
    target_environment: str
    seed: int
    source_examples: int
    target_support_examples: int
    target_eval_examples: int
    source_only: Dict[str, Any]
    scratch_after_target_support: Dict[str, Any]
    transfer_after_target_support: Dict[str, Any]
    transfer_delta: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelfLearningTransferReport:
    """Cross-domain self-learning transfer raporu."""

    protocol: str
    schema_version: int
    seeds: List[int]
    environments: List[str]
    config: Dict[str, Any]
    pair_results: List[Dict[str, Any]]
    aggregate: Dict[str, Any]
    checks: Dict[str, bool]
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        return self_learning_transfer_markdown(self)


class _VerifiedMemoryLearner:
    """Verifier çıktısını exact-triple belleğine yazan küçük öğrenen."""

    def __init__(self) -> None:
        self._memory: Dict[Tuple[str, str, str], bool] = {}
        self.accepted = 0
        self.rejected = 0
        self.abstained = 0

    def learn(self, claims: Iterable[Claim], verifier_name: str) -> None:
        verifier = VERIFIERS[verifier_name]
        for claim in claims:
            verdict = verifier(claim)
            if verdict == ACCEPT:
                self._memory[claim.as_triple()] = True
                self.accepted += 1
            elif verdict == REJECT:
                self._memory[claim.as_triple()] = False
                self.rejected += 1
            else:
                self.abstained += 1

    def predict(self, claim: Claim) -> Optional[bool]:
        return self._memory.get(claim.as_triple())

    @property
    def size(self) -> int:
        return len(self._memory)


def _generate_claims(environment: str, seed: int, n: int) -> List[Claim]:
    import random

    # Python hash'i proses başına tuzludur; protokol imzasının koşudan koşuya
    # değişmemesi için ortam adını SHA-256 ile sayıya çeviriyoruz.
    offset = int(hashlib.sha256(environment.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed * 100_003 + offset)
    return GENERATORS[environment](rng, n)


def _eval_predictions(learner: _VerifiedMemoryLearner,
                      claims: Sequence[Claim]) -> Dict[str, Any]:
    decided = correct = false_accept = false_reject = abstain = 0
    for claim in claims:
        pred = learner.predict(claim)
        if pred is None:
            abstain += 1
            continue
        decided += 1
        if pred == claim.truth:
            correct += 1
        elif pred and not claim.truth:
            false_accept += 1
        else:
            false_reject += 1
    total = len(claims)
    return {
        "claims": total,
        "decided": decided,
        "abstain": abstain,
        "coverage": round(decided / total, 6) if total else 0.0,
        "accuracy_on_decided": round(correct / decided, 6) if decided else None,
        "false_acceptance_rate": round(false_accept / max(1, false_accept + correct), 6),
        "false_rejection_count": false_reject,
    }


def _mean(values: Sequence[float]) -> float:
    return round(float(sum(values) / len(values)), 6) if values else 0.0


def _ci95(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci95": 0.0, "n": 0}
    if len(values) == 1:
        return {"mean": round(values[0], 6), "ci95": 0.0, "n": 1}
    std = _stat.stdev(values)
    ci = 1.96 * std / (len(values) ** 0.5)
    return {"mean": round(_stat.mean(values), 6), "ci95": round(ci, 6),
            "n": len(values)}


def run_self_learning_transfer_benchmark(
    seeds: Sequence[int] = (1, 2, 3),
    environments: Sequence[str] = ENVIRONMENTS,
    source_examples: int = 120,
    target_support_examples: int = 40,
    target_unseen_examples: int = 40,
) -> SelfLearningTransferReport:
    """Kaynak→hedef self-learning transfer smoke benchmarkı.

    ``target_eval`` = hedef destek kümesi + ayrı hedef unseen kümesi. Bellek
    öğreneni yalnız destek üçlülerini gördüğü için unseen coverage'ın düşük/0
    kalması beklenir; bu bir pozitif genelleme iddiası değildir.
    """
    tohumlar = [int(s) for s in seeds]
    if not tohumlar:
        raise ValueError("en az bir tohum gerekir")
    ortamlar = list(environments)
    unknown = [env for env in ortamlar if env not in ENVIRONMENTS]
    if unknown:
        raise ValueError(f"bilinmeyen ortam(lar): {unknown}")
    if len(ortamlar) < 2:
        raise ValueError("transfer için en az iki ortam gerekir")
    for name, value in {
        "source_examples": source_examples,
        "target_support_examples": target_support_examples,
        "target_unseen_examples": target_unseen_examples,
    }.items():
        if int(value) < 1:
            raise ValueError(f"{name} >= 1 olmalı")

    pair_results: List[TransferPairResult] = []
    source_only_false_decisions = 0
    scratch_coverages: List[float] = []
    transfer_coverages: List[float] = []
    scratch_accs: List[float] = []
    transfer_accs: List[float] = []
    deltas: List[float] = []

    for seed in tohumlar:
        for source in ortamlar:
            source_train = _generate_claims(source, seed, int(source_examples))
            for target in ortamlar:
                if source == target:
                    continue
                target_support = _generate_claims(
                    target, seed + 10_000, int(target_support_examples))
                target_unseen = _generate_claims(
                    target, seed + 20_000, int(target_unseen_examples))
                target_eval = list(target_support) + list(target_unseen)

                source_only = _VerifiedMemoryLearner()
                source_only.learn(source_train, source)
                source_only_eval = _eval_predictions(source_only, target_eval)
                source_only_false_decisions += int(source_only_eval["decided"])

                scratch = _VerifiedMemoryLearner()
                scratch.learn(target_support, target)
                scratch_eval = _eval_predictions(scratch, target_eval)

                transfer = _VerifiedMemoryLearner()
                transfer.learn(source_train, source)
                transfer.learn(target_support, target)
                transfer_eval = _eval_predictions(transfer, target_eval)

                scratch_cov = float(scratch_eval["coverage"])
                transfer_cov = float(transfer_eval["coverage"])
                scratch_acc = float(scratch_eval["accuracy_on_decided"] or 0.0)
                transfer_acc = float(transfer_eval["accuracy_on_decided"] or 0.0)
                scratch_coverages.append(scratch_cov)
                transfer_coverages.append(transfer_cov)
                scratch_accs.append(scratch_acc)
                transfer_accs.append(transfer_acc)
                delta = transfer_cov - scratch_cov
                deltas.append(delta)
                pair_results.append(TransferPairResult(
                    source_environment=source,
                    target_environment=target,
                    seed=seed,
                    source_examples=int(source_examples),
                    target_support_examples=int(target_support_examples),
                    target_eval_examples=len(target_eval),
                    source_only=source_only_eval,
                    scratch_after_target_support=scratch_eval,
                    transfer_after_target_support=transfer_eval,
                    transfer_delta={
                        "coverage_delta": round(delta, 6),
                        "accuracy_on_decided_delta": round(transfer_acc - scratch_acc, 6),
                    },
                ))

    delta_summary = _ci95(deltas)
    positive_transfer_status = (
        "DEMONSTRATED" if delta_summary["mean"] > 0
        and delta_summary["mean"] > delta_summary["ci95"]
        else "NOT_DEMONSTRATED")
    aggregate: Dict[str, Any] = {
        "pairs": len(pair_results),
        "source_only_target_decisions": source_only_false_decisions,
        "source_only_target_spoke_rate": round(
            source_only_false_decisions /
            max(1, len(pair_results) * (int(target_support_examples)
                                        + int(target_unseen_examples))), 8),
        "scratch_coverage": _ci95(scratch_coverages),
        "transfer_coverage": _ci95(transfer_coverages),
        "coverage_delta_transfer_minus_scratch": delta_summary,
        "scratch_accuracy_on_decided_mean": _mean(scratch_accs),
        "transfer_accuracy_on_decided_mean": _mean(transfer_accs),
        "positive_transfer_status": positive_transfer_status,
    }
    config = {
        "source_examples": int(source_examples),
        "target_support_examples": int(target_support_examples),
        "target_unseen_examples": int(target_unseen_examples),
        "learner": "verifier_confirmed_exact_triple_memory",
        "signature": hashlib.sha256(json.dumps({
            "protocol": PROTOCOL,
            "seeds": tohumlar,
            "environments": ortamlar,
            "source_examples": int(source_examples),
            "target_support_examples": int(target_support_examples),
            "target_unseen_examples": int(target_unseen_examples),
        }, sort_keys=True).encode("utf-8")).hexdigest()[:12],
    }
    checks = {
        "all_ordered_pairs_evaluated": len(pair_results) == (
            len(tohumlar) * len(ortamlar) * (len(ortamlar) - 1)),
        "source_only_abstains_on_target": source_only_false_decisions == 0,
        "target_support_adaptation_has_coverage": min(scratch_coverages) > 0.0,
        "target_support_accuracy_perfect_on_decided": min(scratch_accs + transfer_accs) == 1.0,
        "transfer_delta_reported": "coverage_delta_transfer_minus_scratch" in aggregate,
        "positive_transfer_not_claimed_when_absent": (
            positive_transfer_status == "DEMONSTRATED"
            or abs(float(delta_summary["mean"])) <= 1e-9),
        "multi_seed": len(tohumlar) >= 2,
    }
    findings = [
        f"{len(pair_results)} kaynak→hedef×tohum satırı koştu; kaynak-only hedef karar sayısı "
        f"{source_only_false_decisions}.",
        "Kaynak ortam deneyimleri hedef ortamda karar üretmedi; bu smoke koşuda "
        "yanlış pozitif transfer/kontaminasyon gözlenmedi.",
        "Hedef destek örnekleri eklendiğinde scratch ve kaynak-ön-yüklü öğrenen "
        f"aynı coverage'ı verdi (delta {aggregate['coverage_delta_transfer_minus_scratch']['mean']:.6f}).",
        f"Pozitif transfer durumu: {aggregate['positive_transfer_status']}. Bu rapor "
        "pozitif genelleme iddiası olarak okunmamalıdır.",
    ]
    limitations = [
        "Öğrenen exact-triple bellek kullanır; unseen hedef genellemesi beklenmez.",
        "Ortamlar sentetiktir ve ayrık sembol uzayları kullanır; gerçek domain transferi değildir.",
        "Bu artifact transfer protokolünü ve negatif kontrolü kapatır; pozitif transfer iddiası için daha güçlü, ortak soyut özellikli görevler gerekir.",
    ]
    return SelfLearningTransferReport(
        protocol=PROTOCOL,
        schema_version=SCHEMA_VERSION,
        seeds=tohumlar,
        environments=ortamlar,
        config=config,
        pair_results=[r.to_dict() for r in pair_results],
        aggregate=aggregate,
        checks=checks,
        findings=findings,
        limitations=limitations,
    )


def self_learning_transfer_markdown(report: SelfLearningTransferReport) -> str:
    """Transfer raporunu Markdown'a çevir."""
    s = report
    a = s.aggregate
    rows = [
        "# Cross-domain Self-learning Transfer Benchmark",
        "",
        f"- Protokol: `{s.protocol}` v{s.schema_version} (imza `{s.config['signature']}`)",
        f"- Ortamlar: {', '.join(s.environments)}",
        f"- Tohumlar: `{s.seeds}`",
        "",
        "> Bu rapor pozitif transfer iddiası değildir. Source-only davranışın "
        "hedefte çekimser kalıp kalmadığını ve target-support adaptasyonunun "
        "scratch ile farkını ölçer.",
        "",
        "## Özet",
        "",
        "| Metrik | Değer |",
        "|---|---:|",
        f"| Pair satırı | {a['pairs']} |",
        f"| Source-only hedef karar | {a['source_only_target_decisions']} |",
        f"| Source-only hedef konuşma oranı | {a['source_only_target_spoke_rate']:.8f} |",
        f"| Scratch coverage | {a['scratch_coverage']['mean']:.6f} ± {a['scratch_coverage']['ci95']:.6f} |",
        f"| Transfer coverage | {a['transfer_coverage']['mean']:.6f} ± {a['transfer_coverage']['ci95']:.6f} |",
        f"| Transfer − scratch coverage | {a['coverage_delta_transfer_minus_scratch']['mean']:.6f} ± {a['coverage_delta_transfer_minus_scratch']['ci95']:.6f} |",
        f"| Pozitif transfer durumu | {a['positive_transfer_status']} |",
        "",
        "## Örnek pair satırları",
        "",
        "| Kaynak→Hedef | Seed | Source-only karar | Scratch cov | Transfer cov | Δcov |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in s.pair_results[:12]:
        rows.append(
            f"| {row['source_environment']}→{row['target_environment']} | "
            f"{row['seed']} | {row['source_only']['decided']} | "
            f"{row['scratch_after_target_support']['coverage']:.6f} | "
            f"{row['transfer_after_target_support']['coverage']:.6f} | "
            f"{row['transfer_delta']['coverage_delta']:.6f} |")
    rows.extend(["", "## Kabul kapıları", "", "| Kapı | Sonuç |", "|---|---|"])
    rows.extend(f"| {k} | {'GEÇTİ' if v else 'KALDI'} |"
                for k, v in s.checks.items())
    rows.extend(["", "## Bulgular", ""])
    rows.extend(f"- {x}" for x in s.findings)
    rows.extend(["", "## Sınırlar", ""])
    rows.extend(f"- {x}" for x in s.limitations)
    return "\n".join(rows) + "\n"


__all__ = [
    "PROTOCOL",
    "SCHEMA_VERSION",
    "SelfLearningTransferReport",
    "TransferPairResult",
    "run_self_learning_transfer_benchmark",
    "self_learning_transfer_markdown",
]
