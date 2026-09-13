"""Birden çok bağımsız verifier environment'ında kapalı self-learning.

Aritmetik, modus-ponens mantık ve property-tutarlılık ortamları aynı
KnowledgeStore/DynamicKV üzerinde dengeli ve interleaved çalışır. Namespace,
verifier router ve holdout izolasyonu ayrı kabul kapılarıdır.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..knowledge import DeneyimDurumu, ExperienceCandidate, KaynakTuru, KnowledgeStore
from ..memory import BellekEntegrasyonu
from .dogrulama import DogrulamaHatti
from .evaluator import ExperienceEvaluator
from .mini_env import AritmetikOrtam, MantikOrtam, TutarlilikOrtam

Triple = Tuple[str, str, str]


@dataclass
class _EnvironmentDomain:
    name: str
    prefix: str
    relation_id: str
    verifier_name: str
    verifier: Callable[[KnowledgeStore, ExperienceCandidate], Optional[bool]]
    candidate_pool: List[Triple]
    holdout: List[Triple]
    initial_facts: List[Triple]

    def routed_verify(
        self, store: KnowledgeStore, candidate: ExperienceCandidate
    ) -> Optional[bool]:
        if candidate.relation_id != self.relation_id:
            return None
        return self.verifier(store, candidate)


@dataclass
class EnvironmentLearningMetrics:
    environment: str
    verifier: str
    generated: int
    truth_positive: int
    truth_negative: int
    valid_before_verifier: int
    verified: int
    rejected: int
    uncertain: int
    false_acceptance_before_verifier: int
    false_acceptance: int
    false_rejection: int
    initial_knowledge: int
    final_knowledge: int
    durable_new_knowledge: int
    incorrect_durable_knowledge: int
    experience_yield: float
    # P1-005: EY tek başına yanıltıcıdır (ayrıntı: docs/VERIM_METRIKLERI.md).
    novelty_yield: float           # kalıcı YENİ olgu / üretilen
    useful_experience_yield: float  # doğru + paylaşılan bellekte bulunan / üretilen
    holdout_size: int
    generation_holdout_overlap: int
    memory_holdout_overlap: int
    knowledge_holdout_overlap: int
    memory_retrieval_accuracy: float


@dataclass
class MultiEnvironmentLearningReport:
    protocol: str
    seed: int
    environments: List[str]
    cycles: int
    batch_per_environment: int
    schedule_hash: str
    shared_store_initial_facts: int
    shared_store_final_facts: int
    shared_memory_policy: str
    shared_memory_records: int
    shared_memory_retrieval_accuracy: float
    cross_verifier_attempts: int
    cross_verifier_acceptances: int
    cross_verifier_uncertain: int
    cross_environment_fact_contamination: int
    per_environment: Dict[str, EnvironmentLearningMetrics]
    checks: Dict[str, bool]
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        lines = [
            "| Environment | Generated | Verified | Rejected | FAR | FRR | "
            "K₀→Kₙ | Holdout leak | Memory recall |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name in self.environments:
            metric = self.per_environment[name]
            far = (
                metric.false_acceptance / metric.truth_negative
                if metric.truth_negative else 0.0
            )
            frr = (
                metric.false_rejection / metric.truth_positive
                if metric.truth_positive else 0.0
            )
            leak = (
                metric.generation_holdout_overlap
                + metric.memory_holdout_overlap
                + metric.knowledge_holdout_overlap
            )
            lines.append(
                f"| {name} | {metric.generated} | {metric.verified} | "
                f"{metric.rejected} | {far:.4f} | {frr:.4f} | "
                f"{metric.initial_knowledge}→{metric.final_knowledge} | {leak} | "
                f"{metric.memory_retrieval_accuracy:.4f} |"
            )
        return "\n".join(lines)


def _split_pool(
    positives: List[Triple], negatives: List[Triple], seed: int
) -> Tuple[List[Triple], List[Triple]]:
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    positive_holdout = max(1, len(positives) // 10)
    negative_holdout = max(1, len(negatives) // 10)
    holdout = positives[-positive_holdout:] + negatives[-negative_holdout:]
    pool = positives[:-positive_holdout] + negatives[:-negative_holdout]
    rng.shuffle(pool)
    rng.shuffle(holdout)
    return pool, holdout


def _add_initial_facts(
    store: KnowledgeStore, triples: List[Triple]
) -> None:
    for triple in triples:
        store.olgu_kaydet(
            *triple,
            score=1.0,
            source=KaynakTuru.VERIFIED_RULE,
            confidence=1.0,
        )


def _arithmetic_domain(store: KnowledgeStore, seed: int) -> _EnvironmentDomain:
    prefix = "ARITH"
    relation_id = f"{prefix}:R_EQUALS"
    store.iliski_tanimla("eşittir", relation_id=relation_id)
    result_ids = {}
    for value in range(16):
        entity_id = f"{prefix}:RESULT:{value}"
        result_ids[value] = entity_id
        store.varlik_ekle(str(value), entity_type="sayi", entity_id=entity_id)
    positives, negatives = [], []
    for index, (left, right) in enumerate(
        (pair for left in range(8) for right in range(4) for pair in [(left, right)])
    ):
        subject_id = f"{prefix}:EXPR:{index}"
        store.varlik_ekle(f"{left}+{right}", entity_type="ifade", entity_id=subject_id)
        truth = left + right
        positives.append((subject_id, relation_id, result_ids[truth]))
        negatives.append((subject_id, relation_id, result_ids[(truth + 1) % 16]))
        negatives.append((subject_id, relation_id, result_ids[(truth + 3) % 16]))
    initial = positives[:3]
    positive_pool = positives[3:]
    pool, holdout = _split_pool(positive_pool, negatives, seed + 101)
    _add_initial_facts(store, initial)
    environment = AritmetikOrtam()
    return _EnvironmentDomain(
        "arithmetic", prefix, relation_id, "arithmetic-env-v1",
        environment.aday_dogrula, pool, holdout, initial,
    )


def _logic_domain(store: KnowledgeStore, seed: int) -> _EnvironmentDomain:
    prefix = "LOGIC"
    relation_id = f"{prefix}:R_ENTAILS"
    store.iliski_tanimla("çıkarır", relation_id=relation_id)
    positives, negatives = [], []
    for index in range(60):
        subject_id = f"{prefix}:PREMISE:{index}"
        true_id = f"{prefix}:RESULT:{index}:TRUE"
        false_id = f"{prefix}:RESULT:{index}:FALSE"
        store.varlik_ekle(
            f"p{index} -> q{index}, p{index}",
            entity_type="mantik_onculu",
            entity_id=subject_id,
        )
        store.varlik_ekle(f"q{index}", entity_type="mantik_sonucu", entity_id=true_id)
        store.varlik_ekle(f"x{index}", entity_type="mantik_sonucu", entity_id=false_id)
        positives.append((subject_id, relation_id, true_id))
        negatives.append((subject_id, relation_id, false_id))
    initial = positives[:3]
    pool, holdout = _split_pool(positives[3:], negatives, seed + 211)
    _add_initial_facts(store, initial)
    environment = MantikOrtam()
    return _EnvironmentDomain(
        "logic", prefix, relation_id, "logic-modus-ponens-env-v1",
        environment.aday_dogrula, pool, holdout, initial,
    )


def _consistency_domain(store: KnowledgeStore, seed: int) -> _EnvironmentDomain:
    prefix = "CONSISTENCY"
    relation_id = f"{prefix}:R_CAN_USE"
    store.iliski_tanimla(
        "kullanabilir",
        relation_id=relation_id,
        requires_subject_props={"eligible": 1.0},
        requires_object_props={"allowed": 1.0},
    )
    subjects, allowed_objects, blocked_objects = [], [], []
    for index in range(64):
        subject_id = f"{prefix}:SUBJECT:{index}"
        allowed_id = f"{prefix}:OBJECT:{index}:ALLOWED"
        blocked_id = f"{prefix}:OBJECT:{index}:BLOCKED"
        store.varlik_ekle(
            f"kullanıcı-{index}", entity_type="kullanici",
            properties={"eligible": 1.0}, entity_id=subject_id,
        )
        store.varlik_ekle(
            f"kaynak-{index}", entity_type="kaynak",
            properties={"allowed": 1.0}, entity_id=allowed_id,
        )
        store.varlik_ekle(
            f"engelli-kaynak-{index}", entity_type="kaynak",
            properties={"allowed": 0.0}, entity_id=blocked_id,
        )
        subjects.append(subject_id)
        allowed_objects.append(allowed_id)
        blocked_objects.append(blocked_id)
    positives = [
        (subjects[index], relation_id, allowed_objects[index])
        for index in range(64)
    ]
    negatives = [
        (subjects[index], relation_id, blocked_objects[index])
        for index in range(64)
    ]
    initial = positives[:3]
    pool, holdout = _split_pool(positives[3:], negatives, seed + 307)
    _add_initial_facts(store, initial)
    environment = TutarlilikOrtam()
    return _EnvironmentDomain(
        "consistency", prefix, relation_id, "property-consistency-env-v1",
        environment.aday_dogrula, pool, holdout, initial,
    )


def _candidate(
    environment: str, triple: Triple, seed: int, cycle: int, index: int
) -> ExperienceCandidate:
    return ExperienceCandidate(
        experience_id=f"MULTI-{seed}-{environment}-{cycle:03d}-{index:03d}",
        subject_id=triple[0],
        relation_id=triple[1],
        object_id=triple[2],
        source=KaynakTuru.MODEL_GENERATED,
        source_confidence=0.5,
        generation_step=cycle,
    )


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


def run_multi_environment_self_learning(
    cycles: int = 6,
    batch_per_environment: int = 8,
    seed: int = 1,
    memory_capacity: int = 512,
) -> MultiEnvironmentLearningReport:
    """Üç environment'ı shared store/memory üzerinde interleaved çalıştır."""
    if cycles < 1 or batch_per_environment < 1:
        raise ValueError("cycles ve batch_per_environment >= 1 olmalı")
    store = KnowledgeStore()
    domains = [
        _arithmetic_domain(store, int(seed)),
        _logic_domain(store, int(seed)),
        _consistency_domain(store, int(seed)),
    ]
    required = cycles * batch_per_environment
    if any(len(domain.candidate_pool) < required for domain in domains):
        raise ValueError("İstenen cycle/batch için environment candidate pool yetersiz")

    evaluator = ExperienceEvaluator()
    memory = BellekEntegrasyonu(
        slot_sayisi=memory_capacity,
        replay_kapasitesi=memory_capacity,
        tohum=int(seed),
        politika="DYNAMIC_KV",
    )
    verifiers = {
        domain.name: DogrulamaHatti(
            domain.routed_verify,
            dogrulayici_adi=domain.verifier_name,
        )
        for domain in domains
    }
    stats: Dict[str, Dict[str, Any]] = {
        domain.name: {
            "generated": 0,
            "truth_positive": 0,
            "truth_negative": 0,
            "valid_before": 0,
            "verified": 0,
            "rejected": 0,
            "uncertain": 0,
            "false_before": 0,
            "false_acceptance": 0,
            "false_rejection": 0,
            "generated_triples": set(),
            "memory_entries": [],
        }
        for domain in domains
    }
    schedule = []
    cursors = {domain.name: 0 for domain in domains}
    rng = random.Random(int(seed) + 4_099)
    for cycle in range(1, cycles + 1):
        ordered = list(domains)
        rng.shuffle(ordered)
        schedule.append([domain.name for domain in ordered])
        for domain in ordered:
            start = cursors[domain.name]
            triples = domain.candidate_pool[start:start + batch_per_environment]
            cursors[domain.name] += len(triples)
            candidates = [
                _candidate(domain.name, triple, int(seed), cycle, index)
                for index, triple in enumerate(triples)
            ]
            truths = [domain.routed_verify(store, candidate) for candidate in candidates]
            for candidate in candidates:
                evaluator.degerlendir(candidate, store)
            valid_before = [
                candidate.state == DeneyimDurumu.VALID for candidate in candidates
            ]
            report = verifiers[domain.name].isle(store, candidates)
            environment_stats = stats[domain.name]
            environment_stats["generated"] += len(candidates)
            environment_stats["truth_positive"] += sum(value is True for value in truths)
            environment_stats["truth_negative"] += sum(value is False for value in truths)
            environment_stats["valid_before"] += sum(valid_before)
            environment_stats["verified"] += report.dogrulanan
            environment_stats["rejected"] += report.reddedilen + sum(
                (truth is False and not was_valid)
                for truth, was_valid in zip(truths, valid_before)
            )
            environment_stats["uncertain"] += report.belirsiz
            environment_stats["false_before"] += sum(
                truth is False and was_valid
                for truth, was_valid in zip(truths, valid_before)
            )
            environment_stats["false_acceptance"] += sum(
                truth is False and candidate.state == DeneyimDurumu.VERIFIED
                for truth, candidate in zip(truths, candidates)
            )
            environment_stats["false_rejection"] += sum(
                truth is True and candidate.state != DeneyimDurumu.VERIFIED
                for truth, candidate in zip(truths, candidates)
            )
            environment_stats["generated_triples"].update(
                candidate.uclusu for candidate in candidates
            )
            for candidate in candidates:
                if candidate.state == DeneyimDurumu.VERIFIED:
                    memory.yaz(candidate)
                    environment_stats["memory_entries"].append(candidate)

    facts = list(store.relations.olgular())
    facts_by_relation = {
        domain.name: [fact for fact in facts if fact.relation_id == domain.relation_id]
        for domain in domains
    }
    all_holdout = {triple for domain in domains for triple in domain.holdout}
    memory_keys = {record.key for record in memory.dynamic_kv.kayitlar()}
    cross_fact_contamination = 0
    per_environment: Dict[str, EnvironmentLearningMetrics] = {}
    for domain in domains:
        environment_stats = stats[domain.name]
        domain_facts = facts_by_relation[domain.name]
        incorrect = 0
        for fact in domain_facts:
            candidate = _candidate(domain.name, fact.uclusu, int(seed), 0, 0)
            if domain.routed_verify(store, candidate) is not True:
                incorrect += 1
            if not (
                fact.subject_id.startswith(f"{domain.prefix}:")
                and fact.object_id.startswith(f"{domain.prefix}:")
            ):
                cross_fact_contamination += 1
        entries = environment_stats["memory_entries"]
        memory_hits = sum(memory.icerir(candidate) for candidate in entries)

        # P1-005: kalıcı yeni olgular ve bunların doğru + bellekte bulunanı.
        domain_ucluleri = {fact.uclusu for fact in domain_facts}
        kalici_yeni = domain_ucluleri - set(domain.initial_facts)
        kullanisli = len({
            uclu for uclu in kalici_yeni
            if uclu in memory_keys
            and domain.routed_verify(
                store, _candidate(domain.name, uclu, int(seed), 0, 0)) is True
        })
        holdout = set(domain.holdout)
        per_environment[domain.name] = EnvironmentLearningMetrics(
            environment=domain.name,
            verifier=domain.verifier_name,
            generated=environment_stats["generated"],
            truth_positive=environment_stats["truth_positive"],
            truth_negative=environment_stats["truth_negative"],
            valid_before_verifier=environment_stats["valid_before"],
            verified=environment_stats["verified"],
            rejected=environment_stats["rejected"],
            uncertain=environment_stats["uncertain"],
            false_acceptance_before_verifier=environment_stats["false_before"],
            false_acceptance=environment_stats["false_acceptance"],
            false_rejection=environment_stats["false_rejection"],
            initial_knowledge=len(domain.initial_facts),
            final_knowledge=len(domain_facts),
            durable_new_knowledge=len(domain_facts) - len(domain.initial_facts),
            incorrect_durable_knowledge=incorrect,
            experience_yield=_ratio(
                environment_stats["verified"], environment_stats["generated"]
            ),
            novelty_yield=_ratio(len(kalici_yeni), environment_stats["generated"]),
            useful_experience_yield=_ratio(
                kullanisli, environment_stats["generated"]
            ),
            holdout_size=len(holdout),
            generation_holdout_overlap=len(
                environment_stats["generated_triples"] & holdout
            ),
            memory_holdout_overlap=len(memory_keys & holdout),
            knowledge_holdout_overlap=len({fact.uclusu for fact in domain_facts} & holdout),
            memory_retrieval_accuracy=_ratio(memory_hits, len(entries)),
        )

    cross_attempts = cross_acceptances = cross_uncertain = 0
    for domain in domains:
        probe = _candidate(domain.name, domain.candidate_pool[0], int(seed), 0, 0)
        for other in domains:
            if other.name == domain.name:
                continue
            cross_attempts += 1
            outcome = other.routed_verify(store, probe)
            cross_acceptances += outcome is True
            cross_uncertain += outcome is None

    generated_counts = {metric.generated for metric in per_environment.values()}
    all_generated = {
        triple
        for environment_stats in stats.values()
        for triple in environment_stats["generated_triples"]
    }
    shared_memory_entries = [
        candidate
        for environment_stats in stats.values()
        for candidate in environment_stats["memory_entries"]
    ]
    shared_memory_hits = sum(memory.icerir(candidate) for candidate in shared_memory_entries)
    initial_total = sum(len(domain.initial_facts) for domain in domains)
    final_total = len(facts)
    checks = {
        "three_independent_environments_present": (
            {domain.name for domain in domains}
            == {"arithmetic", "logic", "consistency"}
        ),
        "environment_schedule_balanced": len(generated_counts) == 1,
        "environment_namespaces_disjoint": all(
            domain.candidate_pool[0][0].startswith(f"{domain.prefix}:")
            for domain in domains
        ),
        "candidate_sets_disjoint": sum(len(stats[d.name]["generated_triples"]) for d in domains)
        == len(all_generated),
        "cross_verifiers_never_accept": cross_acceptances == 0,
        "cross_verifiers_abstain": cross_uncertain == cross_attempts,
        "shared_store_has_no_cross_environment_facts": cross_fact_contamination == 0,
        "no_false_acceptance_after_verifier": all(
            metric.false_acceptance == 0 for metric in per_environment.values()
        ),
        "no_false_rejection_after_verifier": all(
            metric.false_rejection == 0 for metric in per_environment.values()
        ),
        "all_durable_knowledge_correct": all(
            metric.incorrect_durable_knowledge == 0
            for metric in per_environment.values()
        ),
        "every_environment_grows": all(
            metric.durable_new_knowledge > 0 for metric in per_environment.values()
        ),
        "generation_memory_knowledge_holdouts_clean": all(
            metric.generation_holdout_overlap == 0
            and metric.memory_holdout_overlap == 0
            and metric.knowledge_holdout_overlap == 0
            for metric in per_environment.values()
        ),
        "shared_memory_is_active_dynamic_kv": memory.politika == "DYNAMIC_KV",
        "shared_memory_exact_retrieval": (
            shared_memory_hits == len(shared_memory_entries)
        ),
        "shared_memory_contains_no_holdout": not (memory_keys & all_holdout),
        "shared_store_growth_accounted": final_total == initial_total + sum(
            metric.verified for metric in per_environment.values()
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Multi-environment self-learning kabul kapısı başarısız: {failed}")
    schedule_hash = hashlib.sha256(
        json.dumps(schedule, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return MultiEnvironmentLearningReport(
        protocol="multi-environment-closed-verified-self-learning-v1",
        seed=int(seed),
        environments=[domain.name for domain in domains],
        cycles=int(cycles),
        batch_per_environment=int(batch_per_environment),
        schedule_hash=schedule_hash,
        shared_store_initial_facts=initial_total,
        shared_store_final_facts=final_total,
        shared_memory_policy=memory.politika,
        shared_memory_records=len(memory.dynamic_kv),
        shared_memory_retrieval_accuracy=_ratio(
            shared_memory_hits, len(shared_memory_entries)
        ),
        cross_verifier_attempts=cross_attempts,
        cross_verifier_acceptances=cross_acceptances,
        cross_verifier_uncertain=cross_uncertain,
        cross_environment_fact_contamination=cross_fact_contamination,
        per_environment=per_environment,
        checks=checks,
        limitations=[
            "Üç environment deterministik ve sentetiktir; gerçek dünya veya genel dil self-learning kanıtı değildir.",
            "Verifier router relation namespace'ini önceden bilir; environment discovery/induction ölçülmez.",
            "Shared store büyümesi yeni entity öğrenimi değil, bağımsız doğrulanmış relation fact eklenmesidir.",
            "Neural model eğitimi yoktur; bu epistemik durum, contamination ve yaşam döngüsü protokolüdür.",
            "Holdout yalnız sızıntı kontrolüdür; held-out task generalization skoru değildir.",
        ],
    )


__all__ = [
    "EnvironmentLearningMetrics",
    "MultiEnvironmentLearningReport",
    "run_multi_environment_self_learning",
]
