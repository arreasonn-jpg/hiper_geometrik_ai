"""Kapalı self-learning ve self-training collapse kontrollü deneyleri.

Bu modül genel zekâ/self-play iddiası taşımaz. Ground truth'u bağımsız olarak
hesaplanabilen aritmetik mini-environment içinde iki protokolü karşılaştırır:

* CLOSED_VERIFIED: yalnız Verifier onaylı yeni olgular Kₙ'e yazılır.
* UNVERIFIED_SELF_TRAINING: modelin VALID çıktıları verifier olmadan yeniden
  bilgi/bellek girdisi yapılır; tekrar, contamination ve collision ölçülür.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from ..knowledge import DeneyimDurumu, ExperienceCandidate, KaynakTuru, KnowledgeStore
from ..memory.sparse_memory import DeneyimSlotlari
from .consolidation import Consolidator
from .dogrulama import DogrulamaHatti
from .evaluator import ExperienceEvaluator
from .mini_env import AritmetikOrtam

Triple = Tuple[str, str, str]


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


def _normalized_entropy(values: Sequence[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    if len(counts) == 1:
        return 0.0
    total = len(values)
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return round(entropy / math.log(len(counts)), 8)


@dataclass
class LearningCycleMetrics:
    cycle: int
    generated: int
    novel: int
    valid_before_verifier: int
    verified: int
    invalid: int
    uncertain: int
    conflict: int
    false_acceptance_before_verifier: int
    false_acceptance: int
    false_rejection: int
    knowledge_size: int
    correct_knowledge: int
    incorrect_knowledge: int
    cumulative_generated: int
    cumulative_verified_new: int
    experience_yield: float
    novelty_rate: float
    cumulative_diversity: float
    knowledge_entropy: float
    mean_information_gain: float
    memory_collisions: int
    memory_retrieval_accuracy: float


@dataclass
class SelfLearningReport:
    protocol: str
    seed: int
    initial_knowledge_size: int
    final_knowledge_size: int
    cycles_requested: int
    cycles_completed: int
    generated_experiences: int
    verified_new_knowledge: int
    invalid_generated: int
    uncertain_generated: int
    conflict_generated: int
    false_acceptance_before_verifier: int
    false_acceptance: int
    false_rejection: int
    far_before_verifier: float
    far: float
    frr: float
    experience_yield: float
    # P1-005: EY tek başına yanıltıcıdır (ayrıntı: docs/VERIM_METRIKLERI.md).
    novelty_yield: float           # ayrık YENİ doğrulanmış olgu / üretilen
    useful_experience_yield: float  # doğru + bellekten geri çağrılabilir / üretilen
    correct_knowledge: int
    incorrect_knowledge: int
    memory_collisions: int
    test_holdout_size: int
    generation_test_overlap: int
    memory_test_overlap: int
    isolation_clean: bool
    cycles: List[LearningCycleMetrics] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CollapseCycleMetrics:
    cycle: int
    generated: int
    accepted: int
    true_accepted: int
    false_accepted: int
    acceptance_accuracy: float
    far: float
    novel: int
    novelty_rate: float
    mean_evaluator_novelty: float
    cumulative_diversity: float
    model_fact_records: int
    unique_model_facts: int
    incorrect_model_facts: int
    repetition_count: int
    memory_collisions: int
    memory_retrieval_accuracy: float


@dataclass
class CollapseReport:
    protocol: str
    seed: int
    cycles: int
    batch_size: int
    generated_experiences: int
    unique_experiences: int
    repetition_rate: float
    initial_novelty_rate: float
    final_novelty_rate: float
    initial_evaluator_novelty: float
    final_evaluator_novelty: float
    acceptance_accuracy: float
    far: float
    unique_model_facts: int
    incorrect_model_facts: int
    memory_collisions: int
    test_holdout_size: int
    generation_test_overlap: int
    memory_test_overlap: int
    isolation_clean: bool
    collapse_signals: Dict[str, bool]
    collapse_detected: bool
    cycle_metrics: List[CollapseCycleMetrics] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class _ArithmeticDomain:
    store: KnowledgeStore
    environment: AritmetikOrtam
    relation_id: str
    candidate_pool: List[Triple]
    test_holdout: List[Triple]
    initial_triples: List[Triple]


def _build_domain(
    operands_max: int,
    initial_facts: int,
    negatives_per_fact: int,
    seed: int,
) -> _ArithmeticDomain:
    if operands_max < 1 or initial_facts < 0 or negatives_per_fact < 1:
        raise ValueError("operands_max>=1, initial_facts>=0, negatives_per_fact>=1 olmalı")
    expressions = [(left, right) for left in range(operands_max + 1)
                   for right in range(operands_max + 1)]
    if initial_facts >= len(expressions):
        raise ValueError("initial_facts, toplam ifade sayısından küçük olmalı")

    store = KnowledgeStore()
    expression_ids: Dict[Tuple[int, int], str] = {}
    for index, (left, right) in enumerate(expressions):
        entity_id = f"E_EXPR_{index:05d}"
        expression_ids[(left, right)] = entity_id
        store.varlik_ekle(f"{left}+{right}", entity_type="ifade", entity_id=entity_id)
    result_ids: Dict[int, str] = {}
    result_count = 2 * operands_max + 1
    for value in range(result_count):
        entity_id = f"E_RESULT_{value:05d}"
        result_ids[value] = entity_id
        store.varlik_ekle(str(value), entity_type="sayi", entity_id=entity_id)
    relation_id = "R_EQUALS"
    store.iliski_tanimla("eşittir", relation_id=relation_id)

    initial_triples: List[Triple] = []
    for left, right in expressions[:initial_facts]:
        triple = (expression_ids[(left, right)], relation_id, result_ids[left + right])
        initial_triples.append(triple)
        store.olgu_kaydet(*triple, score=1.0, source=KaynakTuru.VERIFIED_RULE, confidence=1.0)

    pool: List[Triple] = []
    for left, right in expressions[initial_facts:]:
        truth = left + right
        subject_id = expression_ids[(left, right)]
        pool.append((subject_id, relation_id, result_ids[truth]))
        for offset in range(1, negatives_per_fact + 1):
            wrong = (truth + offset) % result_count
            pool.append((subject_id, relation_id, result_ids[wrong]))
    random.Random(seed).shuffle(pool)
    holdout_size = max(1, len(pool) // 10)
    candidate_pool, test_holdout = pool[:-holdout_size], pool[-holdout_size:]
    return _ArithmeticDomain(
        store, AritmetikOrtam(), relation_id,
        candidate_pool, test_holdout, initial_triples,
    )


def _candidate(triple: Triple, experience_id: str, cycle: int) -> ExperienceCandidate:
    return ExperienceCandidate(
        experience_id=experience_id,
        subject_id=triple[0], relation_id=triple[1], object_id=triple[2],
        source=KaynakTuru.MODEL_GENERATED, source_confidence=0.5,
        generation_step=cycle,
    )


def _unique_facts(store: KnowledgeStore) -> Dict[Triple, Any]:
    result = {}
    for fact in store.relations.olgular():
        result[fact.uclusu] = fact
    return result


def _fact_correct(domain: _ArithmeticDomain, triple: Triple) -> bool:
    candidate = _candidate(triple, "AUDIT", 0)
    return domain.environment.aday_dogrula(domain.store, candidate) is True


def _knowledge_audit(domain: _ArithmeticDomain) -> Tuple[int, int, float]:
    facts = _unique_facts(domain.store)
    correct = sum(_fact_correct(domain, triple) for triple in facts)
    objects = [triple[2] for triple in facts]
    return correct, len(facts) - correct, _normalized_entropy(objects)


def _memory_recall(memory: DeneyimSlotlari, entries: Sequence[Tuple[str, Triple]]) -> float:
    if not entries:
        return 1.0
    retained = sum(memory.icerir(experience_id, triple) for experience_id, triple in entries)
    return _ratio(retained, len(entries))


def run_self_learning_experiment(
    cycles: int = 100,
    batch_size: int = 64,
    initial_facts: int = 100,
    operands_max: int = 31,
    negatives_per_fact: int = 3,
    seed: int = 42,
    memory_slots: int = 4096,
) -> SelfLearningReport:
    """Yalnız bağımsız doğrulanmış olgularla K₀→Kₙ kapalı döngüsünü çalıştır."""
    if cycles < 1 or batch_size < 1:
        raise ValueError("cycles ve batch_size >= 1 olmalı")
    domain = _build_domain(operands_max, initial_facts, negatives_per_fact, seed)
    evaluator = ExperienceEvaluator()
    verifier = DogrulamaHatti(domain.environment.aday_dogrula, dogrulayici_adi="arithmetic-env-v1")
    memory = DeneyimSlotlari(slot_sayisi=memory_slots, cakisma_ornek_limiti=1000)
    generated_seen: set = set()
    verified_memory_entries: List[Tuple[str, Triple]] = []
    cursor = 0
    cumulative_generated = 0
    cumulative_verified = 0
    total_invalid = total_uncertain = total_conflict = 0
    total_false_acceptance_before = 0
    total_false_acceptance = total_false_rejection = 0
    cycle_reports: List[LearningCycleMetrics] = []

    for cycle in range(1, cycles + 1):
        triples = domain.candidate_pool[cursor:cursor + batch_size]
        cursor += len(triples)
        candidates = [
            _candidate(triple, f"SL-{seed}-{cycle:04d}-{index:04d}", cycle)
            for index, triple in enumerate(triples)
        ]
        novel = sum(candidate.uclusu not in generated_seen for candidate in candidates)
        generated_seen.update(candidate.uclusu for candidate in candidates)
        for candidate in candidates:
            evaluator.degerlendir(candidate, domain.store)
        valid_before = sum(candidate.state == DeneyimDurumu.VALID for candidate in candidates)
        information_gains = [candidate.scores.get("information_gain", 0.0)
                             for candidate in candidates]
        verification = verifier.isle(domain.store, candidates) if candidates else None
        verified = verification.dogrulanan if verification else 0
        invalid = verification.reddedilen if verification else 0
        uncertain = verification.belirsiz if verification else 0
        conflict = sum(candidate.state == DeneyimDurumu.CONFLICT for candidate in candidates)
        false_acceptance_before = verification.yanlis_kabul_oncesi if verification else 0
        false_acceptance = verification.yanlis_kabul_sonrasi if verification else 0
        false_rejection = 0

        for candidate in candidates:
            if candidate.state == DeneyimDurumu.VERIFIED:
                memory.yaz(candidate.experience_id, candidate.uclusu)
                verified_memory_entries.append((candidate.experience_id, candidate.uclusu))

        cumulative_generated += len(candidates)
        cumulative_verified += verified
        total_invalid += invalid
        total_uncertain += uncertain
        total_conflict += conflict
        total_false_acceptance_before += false_acceptance_before
        total_false_acceptance += false_acceptance
        total_false_rejection += false_rejection
        correct, incorrect, entropy = _knowledge_audit(domain)
        cycle_reports.append(LearningCycleMetrics(
            cycle=cycle, generated=len(candidates), novel=novel,
            valid_before_verifier=valid_before, verified=verified, invalid=invalid,
            uncertain=uncertain, conflict=conflict,
            false_acceptance_before_verifier=false_acceptance_before,
            false_acceptance=false_acceptance, false_rejection=false_rejection,
            knowledge_size=len(_unique_facts(domain.store)), correct_knowledge=correct,
            incorrect_knowledge=incorrect, cumulative_generated=cumulative_generated,
            cumulative_verified_new=cumulative_verified,
            experience_yield=_ratio(cumulative_verified, cumulative_generated),
            novelty_rate=_ratio(novel, len(candidates)),
            cumulative_diversity=_ratio(len(generated_seen), cumulative_generated),
            knowledge_entropy=entropy,
            mean_information_gain=(round(sum(information_gains) / len(information_gains), 8)
                                   if information_gains else 0.0),
            memory_collisions=memory.cakisma_sayisi,
            memory_retrieval_accuracy=_memory_recall(memory, verified_memory_entries),
        ))

    correct, incorrect, _ = _knowledge_audit(domain)
    valid_truths = cumulative_verified + total_false_rejection
    invalid_truths = total_invalid + total_false_acceptance

    # P1-005 ayrıştırması: ayrık yeni olgu ve bunların geri çağrılabilir olanı.
    son_olgular = set(_unique_facts(domain.store))
    baslangic_olgular = set(domain.initial_triples)
    ayrik_yeni = len(son_olgular - baslangic_olgular)
    kullanisli = len({
        triple for experience_id, triple in verified_memory_entries
        if triple in son_olgular and triple not in baslangic_olgular
        and memory.icerir(experience_id, triple)
        and _fact_correct(domain, triple)
    })

    holdout = set(domain.test_holdout)
    generation_overlap = len(generated_seen & holdout)
    memory_overlap = len({triple for _, triple in verified_memory_entries} & holdout)
    return SelfLearningReport(
        protocol="closed-verified-self-learning-v1", seed=seed,
        initial_knowledge_size=len(domain.initial_triples),
        final_knowledge_size=len(_unique_facts(domain.store)),
        cycles_requested=cycles, cycles_completed=len(cycle_reports),
        generated_experiences=cumulative_generated,
        verified_new_knowledge=cumulative_verified,
        invalid_generated=total_invalid, uncertain_generated=total_uncertain,
        conflict_generated=total_conflict,
        false_acceptance_before_verifier=total_false_acceptance_before,
        false_acceptance=total_false_acceptance,
        false_rejection=total_false_rejection,
        far_before_verifier=_ratio(total_false_acceptance_before, invalid_truths),
        far=_ratio(total_false_acceptance, invalid_truths),
        frr=_ratio(total_false_rejection, valid_truths),
        experience_yield=_ratio(cumulative_verified, cumulative_generated),
        novelty_yield=_ratio(ayrik_yeni, cumulative_generated),
        useful_experience_yield=_ratio(kullanisli, cumulative_generated),
        correct_knowledge=correct, incorrect_knowledge=incorrect,
        memory_collisions=memory.cakisma_sayisi,
        test_holdout_size=len(holdout), generation_test_overlap=generation_overlap,
        memory_test_overlap=memory_overlap,
        isolation_clean=(generation_overlap == 0 and memory_overlap == 0),
        cycles=cycle_reports,
        limitations=[
            "Ground truth bağımsız ama sentetik aritmetik environment tarafından sağlanır.",
            "Entity uzayı başlangıçta sabittir; ölçülen büyüme yeni doğrulanmış relation fact büyümesidir.",
            "Bu deney genel dilde otonom yeni bilgi keşfi kanıtı değildir.",
        ],
    )


@dataclass
class VerifierRobustnessReport:
    """Kontrollü verifier hata/epistemik durum enjeksiyonu confusion matrix'i."""

    protocol: str
    seed: int
    truth_positive: int
    truth_negative: int
    truth_unknown: int
    actual_conflicts: int
    true_acceptance: int
    true_rejection: int
    false_acceptance: int
    false_rejection: int
    uncertain: int
    conflict: int
    precision: float
    recall: float
    f1: float
    far: float
    frr: float
    initial_knowledge_size: int
    final_knowledge_size: int
    durable_new_knowledge: int
    correct_knowledge: int
    incorrect_knowledge: int
    injected_false_acceptance_rate: float
    injected_false_rejection_rate: float
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_verifier_fault_injection(
    sample_per_class: int = 8,
    unknown_count: int = 2,
    conflict_count: int = 2,
    false_acceptance_rate: float = 0.25,
    false_rejection_rate: float = 0.25,
    seed: int = 42,
) -> VerifierRobustnessReport:
    """FAR/FRR/UNCERTAIN/CONFLICT ölçümlerini kontrollü hatalarla kırma testi.

    Bu protokol üretim verifier'ının kalitesini temsil etmez. Bağımsız aritmetik
    oracle kararları ground truth olarak tutulurken verifier adaptörünün belirli
    doğru/yanlış adaylardaki kararı kasıtlı bozulur. Böylece confusion-matrix
    metriklerinin sıfır dışı hatalara tepki verdiği doğrulanır.
    """
    if sample_per_class < 1 or unknown_count < 1 or conflict_count < 1:
        raise ValueError("sample_per_class, unknown_count ve conflict_count >= 1 olmalı")
    if not 0.0 <= false_acceptance_rate <= 1.0:
        raise ValueError("false_acceptance_rate [0,1] aralığında olmalı")
    if not 0.0 <= false_rejection_rate <= 1.0:
        raise ValueError("false_rejection_rate [0,1] aralığında olmalı")

    domain = _build_domain(operands_max=7, initial_facts=max(4, conflict_count),
                           negatives_per_fact=3, seed=seed)
    evaluator = ExperienceEvaluator()
    positives: List[ExperienceCandidate] = []
    negatives: List[ExperienceCandidate] = []
    for index, triple in enumerate(domain.candidate_pool):
        candidate = _candidate(triple, f"ROB-{seed}-{index:04d}", 1)
        truth = domain.environment.aday_dogrula(domain.store, candidate)
        target = positives if truth is True else negatives
        if truth is not None and len(target) < sample_per_class:
            target.append(candidate)
        if len(positives) == sample_per_class and len(negatives) == sample_per_class:
            break
    if len(positives) < sample_per_class or len(negatives) < sample_per_class:
        raise ValueError("İstenen robustness örnekleri için aritmetik havuz yetersiz")

    unknowns: List[ExperienceCandidate] = []
    known_object = domain.initial_triples[0][2]
    for index in range(unknown_count):
        entity = domain.store.varlik_ekle(
            f"kanıtı_bilinmeyen_{index}", entity_type="ifade",
            entity_id=f"E_UNKNOWN_{index:04d}",
        )
        unknowns.append(_candidate(
            (entity.entity_id, domain.relation_id, known_object),
            f"ROB-{seed}-UNKNOWN-{index:04d}", 1,
        ))

    conflicts: List[ExperienceCandidate] = []
    result_count = 15
    for index, true_triple in enumerate(domain.initial_triples[:conflict_count]):
        true_value = int(true_triple[2].rsplit("_", 1)[1])
        wrong_object = f"E_RESULT_{(true_value + 1) % result_count:05d}"
        candidate = _candidate(
            (true_triple[0], domain.relation_id, wrong_object),
            f"ROB-{seed}-CONFLICT-{index:04d}", 1,
        )
        candidate.contradicts.append("|".join(true_triple))
        conflicts.append(candidate)

    candidates = positives + negatives + unknowns + conflicts
    for candidate in candidates:
        evaluator.degerlendir(candidate, domain.store)
    # Bu adaylar K₀'daki doğrulanmış fonksiyonel eşitlikle doğrudan çelişir.
    for candidate in conflicts:
        candidate.state = DeneyimDurumu.CONFLICT
        candidate.rationale.append("K₀ doğrulanmış eşitliğiyle gerçek contradiction")

    false_accept_count = min(len(negatives), round(len(negatives) * false_acceptance_rate))
    false_reject_count = min(len(positives), round(len(positives) * false_rejection_rate))
    forced_accept = {candidate.experience_id for candidate in negatives[:false_accept_count]}
    forced_reject = {candidate.experience_id for candidate in positives[:false_reject_count]}

    def faulted_verifier(store, candidate):
        if candidate.experience_id in forced_accept:
            return True
        if candidate.experience_id in forced_reject:
            return False
        return domain.environment.aday_dogrula(store, candidate)

    verifier = DogrulamaHatti(faulted_verifier, dogrulayici_adi="fault-injection-verifier-v1")
    verifier.isle(domain.store, candidates)

    true_acceptance = sum(c.state == DeneyimDurumu.VERIFIED for c in positives)
    false_rejection = len(positives) - true_acceptance
    false_acceptance = sum(c.state == DeneyimDurumu.VERIFIED for c in negatives)
    true_rejection = len(negatives) - false_acceptance
    uncertain = sum(c.state == DeneyimDurumu.UNCERTAIN for c in unknowns)
    conflict = sum(c.state == DeneyimDurumu.CONFLICT for c in conflicts)
    precision = _ratio(true_acceptance, true_acceptance + false_acceptance)
    recall = _ratio(true_acceptance, len(positives))
    f1 = round(2 * precision * recall / (precision + recall), 8) if precision + recall else 0.0
    correct, incorrect, _ = _knowledge_audit(domain)
    initial_size = len(domain.initial_triples)
    final_size = len(_unique_facts(domain.store))
    return VerifierRobustnessReport(
        protocol="verifier-epistemic-fault-injection-v1", seed=seed,
        truth_positive=len(positives), truth_negative=len(negatives),
        truth_unknown=len(unknowns), actual_conflicts=len(conflicts),
        true_acceptance=true_acceptance, true_rejection=true_rejection,
        false_acceptance=false_acceptance, false_rejection=false_rejection,
        uncertain=uncertain, conflict=conflict,
        precision=precision, recall=recall, f1=f1,
        far=_ratio(false_acceptance, len(negatives)),
        frr=_ratio(false_rejection, len(positives)),
        initial_knowledge_size=initial_size, final_knowledge_size=final_size,
        durable_new_knowledge=final_size - initial_size,
        correct_knowledge=correct, incorrect_knowledge=incorrect,
        injected_false_acceptance_rate=float(false_acceptance_rate),
        injected_false_rejection_rate=float(false_rejection_rate),
        limitations=[
            "FAR/FRR kasıtlı verifier fault injection ile üretilir; üretim kalite tahmini değildir.",
            "UNCERTAIN ayrıştırılamayan ama yanlışlığı bilinmeyen ifadeyi temsil eder.",
            "CONFLICT K₀'daki doğrulanmış fonksiyonel eşitliğe karşıt iddiadır ve kalıcılaşmaz.",
        ],
    )


def run_self_training_collapse_test(
    cycles: int = 100,
    batch_size: int = 64,
    initial_facts: int = 10,
    operands_max: int = 15,
    negatives_per_fact: int = 7,
    seed: int = 42,
    memory_slots: int = 256,
) -> CollapseReport:
    """Aynı model çıktısını verifier olmadan tekrar besleyerek collapse sinyallerini ölç."""
    if cycles < 2 or batch_size < 1:
        raise ValueError("collapse testi için cycles>=2 ve batch_size>=1 olmalı")
    domain = _build_domain(operands_max, initial_facts, negatives_per_fact, seed)
    fixed_triples = domain.candidate_pool[:batch_size]
    if len(fixed_triples) < batch_size:
        raise ValueError("İstenen batch için aday havuzu yetersiz")
    evaluator = ExperienceEvaluator()
    consolidator = Consolidator()
    memory = DeneyimSlotlari(slot_sayisi=memory_slots, cakisma_ornek_limiti=1000)
    seen: set = set()
    memory_entries: List[Tuple[str, Triple]] = []
    cycle_reports: List[CollapseCycleMetrics] = []
    total_generated = total_accepted = total_true = total_false = 0

    for cycle in range(1, cycles + 1):
        candidates = [
            _candidate(triple, f"COL-{seed}-{cycle:04d}-{index:04d}", cycle)
            for index, triple in enumerate(fixed_triples)
        ]
        novel = sum(candidate.uclusu not in seen for candidate in candidates)
        seen.update(candidate.uclusu for candidate in candidates)
        for candidate in candidates:
            evaluator.degerlendir(candidate, domain.store)
        evaluator_novelty = [candidate.scores.get("novelty", 0.0) for candidate in candidates]
        accepted_candidates = [candidate for candidate in candidates
                               if candidate.state == DeneyimDurumu.VALID]
        truth = [domain.environment.aday_dogrula(domain.store, candidate)
                 for candidate in accepted_candidates]
        true_accepted = sum(value is True for value in truth)
        false_accepted = sum(value is False for value in truth)
        invalid_truths = sum(value is False for value in truth)
        consolidator.konsolide_et(domain.store, candidates)
        for candidate in accepted_candidates:
            memory.yaz(candidate.experience_id, candidate.uclusu)
            memory_entries.append((candidate.experience_id, candidate.uclusu))

        total_generated += len(candidates)
        total_accepted += len(accepted_candidates)
        total_true += true_accepted
        total_false += false_accepted
        model_facts = [fact for fact in domain.store.relations.olgular()
                       if fact.source == KaynakTuru.MODEL_GENERATED]
        unique_model = {fact.uclusu for fact in model_facts}
        incorrect_model = sum(not _fact_correct(domain, triple) for triple in unique_model)
        cycle_reports.append(CollapseCycleMetrics(
            cycle=cycle, generated=len(candidates), accepted=len(accepted_candidates),
            true_accepted=true_accepted, false_accepted=false_accepted,
            acceptance_accuracy=_ratio(true_accepted, len(accepted_candidates)),
            far=_ratio(false_accepted, invalid_truths),
            novel=novel, novelty_rate=_ratio(novel, len(candidates)),
            mean_evaluator_novelty=(round(sum(evaluator_novelty) / len(evaluator_novelty), 8)
                                    if evaluator_novelty else 0.0),
            cumulative_diversity=_ratio(len(seen), total_generated),
            model_fact_records=len(model_facts), unique_model_facts=len(unique_model),
            incorrect_model_facts=incorrect_model,
            repetition_count=total_generated - len(seen),
            memory_collisions=memory.cakisma_sayisi,
            memory_retrieval_accuracy=_memory_recall(memory, memory_entries),
        ))

    first, last = cycle_reports[0], cycle_reports[-1]
    unique_model_facts = last.unique_model_facts
    incorrect_model_facts = last.incorrect_model_facts
    holdout = set(domain.test_holdout)
    generation_overlap = len(seen & holdout)
    memory_overlap = len({triple for _, triple in memory_entries} & holdout)
    signals = {
        "novelty_declined": last.novelty_rate < first.novelty_rate,
        "evaluator_novelty_declined": last.mean_evaluator_novelty < first.mean_evaluator_novelty,
        "repetition_increased": last.repetition_count > first.repetition_count,
        "unverified_contamination": incorrect_model_facts > 0,
        "memory_interference_increased": last.memory_collisions > first.memory_collisions,
    }
    return CollapseReport(
        protocol="unverified-self-training-collapse-v1", seed=seed,
        cycles=cycles, batch_size=batch_size, generated_experiences=total_generated,
        unique_experiences=len(seen),
        repetition_rate=_ratio(total_generated - len(seen), total_generated),
        initial_novelty_rate=first.novelty_rate, final_novelty_rate=last.novelty_rate,
        initial_evaluator_novelty=first.mean_evaluator_novelty,
        final_evaluator_novelty=last.mean_evaluator_novelty,
        acceptance_accuracy=_ratio(total_true, total_accepted),
        far=_ratio(total_false, total_false),
        unique_model_facts=unique_model_facts,
        incorrect_model_facts=incorrect_model_facts,
        memory_collisions=memory.cakisma_sayisi,
        test_holdout_size=len(holdout), generation_test_overlap=generation_overlap,
        memory_test_overlap=memory_overlap,
        isolation_clean=(generation_overlap == 0 and memory_overlap == 0),
        collapse_signals=signals,
        collapse_detected=sum(signals.values()) >= 3,
        cycle_metrics=cycle_reports,
        limitations=[
            "Collapse kasıtlı sabit aday replay'iyle failure injection olarak üretilir.",
            "Bu sonuç eğitilmiş neural modelin zaman içindeki kalite eğrisi değildir.",
            "Amaç verifier'sız öz-beslemenin tekrar, contamination ve interference riskini yakalamaktır.",
        ],
    )


__all__ = [
    "CollapseCycleMetrics", "CollapseReport", "LearningCycleMetrics", "SelfLearningReport",
    "VerifierRobustnessReport", "run_self_learning_experiment",
    "run_self_training_collapse_test", "run_verifier_fault_injection",
]
