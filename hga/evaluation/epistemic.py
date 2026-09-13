# -*- coding: utf-8 -*-
"""P0-007 — KNOWN / UNKNOWN / UNCERTAIN / CONFLICT / FALSE epistemik benchmarkı.

Bir sistemin "doğru cevap verme" oranı tek başına yanıltıcıdır. Asıl soru:

> Bilmediğini biliyor mu?

Bu modül karışık bir epistemik veri kümesinde **yanlış güven (false
confidence)** ile **dürüst çekimserliği** ayrı ayrı ölçer:

* ``KNOWN``     — kısıtlar sağlanır, kabul edilmeli (``VALID``)
* ``FALSE``     — kural/tip ihlali KAYITLI kanıtla bilinir, reddedilmeli (``INVALID``)
* ``UNKNOWN``   — varlık/ilişki bilgi tabanında hiç yok (``UNCERTAIN``)
* ``UNCERTAIN`` — varlık var ama gerekli özellik hiç yazılmamış (``UNCERTAIN``)
* ``CONFLICT``  — yapısal kural ile kayıtlı kanıt zıt yönde (``CONFLICT``)

## Ölçülen asıl şey: false confidence

``false_confidence_rate``, epistemik olarak **kesin cevap verilmemesi gereken**
(UNKNOWN + UNCERTAIN) örneklerde sistemin yine de kesin karar (VALID veya
INVALID) verme oranıdır. Bir halüsinasyon ölçüsüdür: "bilmiyorum" demesi
gereken yerde konuşmak.

``unknown_accuracy`` ise bu örneklerde doğru şekilde çekimser kalma oranıdır.

## Dürüstlük: UNKNOWN ile UNCERTAIN aynı duruma eşlenir

Mevcut ``ExperienceEvaluator`` hem "kayıt hiç yok" hem "özellik yazılmamış"
durumunu tek bir ``DeneyimDurumu.UNCERTAIN``'e indirger. Bu iki şey epistemik
olarak farklıdır (*bilmiyorum* vs *emin değilim*). Benchmark bunu gizlemez:
``epistemic_resolution`` bölümünde ayrımın yapılabilir olup olmadığı
``distinguishable`` bayrağıyla açıkça raporlanır. Şu an beklenen değer
``False``'tur ve bu bir başarısızlık değil, **ölçülmüş mimari sınırdır**.

Sonuçlar kontrollü bir ontoloji üzerindedir; gerçek dil anlama iddiası
taşımaz.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from importlib import resources
from typing import Any, Dict, List, Mapping, Optional, Sequence

from hga.experience.evaluator import ExperienceEvaluator
from hga.knowledge import (
    BelirsizlikSebebi,
    DeneyimDurumu,
    ExperienceCandidate,
    KaynakTuru,
    KnowledgeStore,
)

DATA_PACKAGE = "hga.evaluation.datasets"
DATA_FILE = "epistemic_tr_v1.json"

EPISTEMIC_CLASSES = ("KNOWN", "FALSE", "UNKNOWN", "UNCERTAIN", "CONFLICT")

# Kesin (committed) karar sayılan durumlar: sistem "biliyorum" demiştir.
COMMITTED_STATES = (DeneyimDurumu.VALID, DeneyimDurumu.INVALID)

# Epistemik olarak çekimser kalınması GEREKEN sınıflar.
ABSTENTION_REQUIRED = ("UNKNOWN", "UNCERTAIN")


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


@dataclass
class EpistemicCaseResult:
    case_id: str
    epistemic_class: str
    expected_state: str
    observed_state: str
    correct: bool
    committed: bool
    abstained: bool
    false_confidence: bool
    rationale: str
    # P0-007: UNCERTAIN ise belirsizliğin epistemik kaynağı.
    uncertainty_reason: str = BelirsizlikSebebi.YOK.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpistemicClassMetrics:
    epistemic_class: str
    total: int
    correct: int
    accuracy: float
    committed: int
    abstained: int
    observed_states: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpistemicReport:
    protocol: str
    dataset_hash: str
    total_cases: int
    correct: int
    accuracy: float
    false_confidence_cases: int
    false_confidence_rate: float
    unknown_accuracy: float
    known_accuracy: float
    false_detection_accuracy: float
    conflict_accuracy: float
    silent_failure_cases: int
    silent_failure_rate: float
    epistemic_resolution: Dict[str, Any]
    per_class: List[Dict[str, Any]]
    confusion: Dict[str, Dict[str, int]]
    cases: List[Dict[str, Any]]
    checks: Dict[str, bool]
    baselines: List[Dict[str, Any]] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EpistemicDataset:
    """Sürümlü, elle sabitlenmiş epistemik fixture."""

    def __init__(self, document: Optional[Mapping[str, Any]] = None):
        if document is None:
            raw = resources.files(DATA_PACKAGE).joinpath(DATA_FILE).read_text("utf-8")
            document = json.loads(raw)
        self.document: Dict[str, Any] = dict(document)
        self.entities: List[Dict[str, Any]] = list(self.document["entities"])
        self.relations: List[Dict[str, Any]] = list(self.document["relations"])
        self.cases: List[Dict[str, Any]] = list(self.document["cases"])
        self.conflict_evidence: List[Dict[str, Any]] = list(
            self.document.get("conflict_evidence", [])
        )
        self._validate()

    def _validate(self) -> None:
        if self.document.get("schema_version") != 1:
            raise ValueError("Desteklenmeyen epistemik fixture şeması")
        if not self.cases:
            raise ValueError("Epistemik fixture boş olamaz")

        gorulen = {str(case["epistemic_class"]) for case in self.cases}
        eksik = set(EPISTEMIC_CLASSES) - gorulen
        if eksik:
            raise ValueError(f"Epistemik sınıflar eksik: {sorted(eksik)}")

        kimlikler = [str(case["case_id"]) for case in self.cases]
        if len(kimlikler) != len(set(kimlikler)):
            raise ValueError("case_id değerleri benzersiz olmalı")

        gecerli_durumlar = {durum.value for durum in DeneyimDurumu}
        for case in self.cases:
            if str(case["expected_state"]) not in gecerli_durumlar:
                raise ValueError(f"Geçersiz expected_state: {case['case_id']}")
            if str(case["epistemic_class"]) not in EPISTEMIC_CLASSES:
                raise ValueError(f"Geçersiz epistemic_class: {case['case_id']}")

        def _uclu(kayit: Mapping[str, Any]) -> tuple:
            return (
                str(kayit["subject_id"]),
                str(kayit["relation_id"]),
                str(kayit["object_id"]),
            )

        # Aynı üçlü iki vakada geçerse biri diğerinin kanıtını kirletir.
        ucluler = [_uclu(case) for case in self.cases]
        if len(ucluler) != len(set(ucluler)):
            raise ValueError("Vaka üçlüleri (özne, ilişki, nesne) benzersiz olmalı")

        # Çelişki kanıtı YALNIZCA CONFLICT vakalarının üçlüsüne yazılmalıdır;
        # aksi halde KNOWN vakası da CONFLICT'e düşer ve fixture sessizce bozulur.
        kanit_ucluleri = {_uclu(kanit) for kanit in self.conflict_evidence}
        for case in self.cases:
            if _uclu(case) in kanit_ucluleri and str(case["epistemic_class"]) != "CONFLICT":
                raise ValueError(
                    f"Çelişki kanıtı CONFLICT olmayan vakayı kirletiyor: {case['case_id']}"
                )

    def dataset_hash(self) -> str:
        payload = json.dumps(
            self.document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def build_store(self) -> KnowledgeStore:
        """Fixture'dan deterministik bilgi tabanı kur."""
        store = KnowledgeStore()
        for entity in self.entities:
            store.varlik_ekle(
                str(entity["token"]),
                entity_type=str(entity["entity_type"]),
                entity_id=str(entity["entity_id"]),
                properties={str(k): float(v) for k, v in
                            (entity.get("properties") or {}).items()},
                ozel_isim=bool(entity.get("is_ozel", False)),
                source=KaynakTuru.VERIFIED_RULE,
                confidence=1.0,
            )
        for relation in self.relations:
            store.iliski_tanimla(
                str(relation["token"]),
                relation_id=str(relation["relation_id"]),
                subject_types=list(relation.get("subject_types") or []),
                object_types=list(relation.get("object_types") or []),
                requires_object_props={
                    str(k): float(v) for k, v in
                    (relation.get("requires_object_props") or {}).items()
                },
                requires_subject_props={
                    str(k): float(v) for k, v in
                    (relation.get("requires_subject_props") or {}).items()
                },
            )
        # CONFLICT sınıfı için: yapısal kural olumlu derken kayıtlı kanıt olumsuz.
        for evidence in self.conflict_evidence:
            store.olgu_kaydet(
                str(evidence["subject_id"]),
                str(evidence["relation_id"]),
                str(evidence["object_id"]),
                score=float(evidence["score"]),
                source=KaynakTuru[str(evidence.get("source", "REAL_DATA"))],
                confidence=float(evidence["confidence"]),
            )
        return store


def run_epistemic_benchmark(
    dataset: Optional[EpistemicDataset] = None,
    seed: int = 1,
) -> EpistemicReport:
    """Gerçek ``ExperienceEvaluator`` kararını epistemik sınıflara karşı ölç.

    ``seed`` yalnız manifest/tekrarlanabilirlik sözleşmesi içindir: protokol
    deterministiktir ve seed sonucu değiştirmez (bu bir kapıyla sınanır).
    """
    data = dataset or EpistemicDataset()
    store = data.build_store()
    evaluator = ExperienceEvaluator()

    sonuclar: List[EpistemicCaseResult] = []
    confusion: Dict[str, Dict[str, int]] = {sinif: {} for sinif in EPISTEMIC_CLASSES}

    for case in data.cases:
        aday = ExperienceCandidate(
            experience_id=str(case["case_id"]),
            subject_id=str(case["subject_id"]),
            relation_id=str(case["relation_id"]),
            object_id=str(case["object_id"]),
            source=KaynakTuru.MODEL_GENERATED,
        )
        evaluator.degerlendir(aday, store)

        sinif = str(case["epistemic_class"])
        beklenen = str(case["expected_state"])
        gozlenen = aday.state.value
        committed = aday.state in COMMITTED_STATES
        abstained = aday.state in (DeneyimDurumu.UNCERTAIN, DeneyimDurumu.CONFLICT)

        confusion[sinif][gozlenen] = confusion[sinif].get(gozlenen, 0) + 1
        sonuclar.append(EpistemicCaseResult(
            case_id=str(case["case_id"]),
            epistemic_class=sinif,
            expected_state=beklenen,
            observed_state=gozlenen,
            correct=(gozlenen == beklenen),
            committed=committed,
            abstained=abstained,
            # Çekimser kalınması gereken yerde kesin karar = yanlış güven.
            false_confidence=(sinif in ABSTENTION_REQUIRED and committed),
            rationale=(aday.rationale[0] if aday.rationale else ""),
            uncertainty_reason=aday.belirsizlik_sebebi.value,
        ))

    def _dilim(sinif: str) -> List[EpistemicCaseResult]:
        return [sonuc for sonuc in sonuclar if sonuc.epistemic_class == sinif]

    per_class: List[Dict[str, Any]] = []
    for sinif in EPISTEMIC_CLASSES:
        dilim = _dilim(sinif)
        per_class.append(EpistemicClassMetrics(
            epistemic_class=sinif,
            total=len(dilim),
            correct=sum(sonuc.correct for sonuc in dilim),
            accuracy=_ratio(sum(sonuc.correct for sonuc in dilim), len(dilim)),
            committed=sum(sonuc.committed for sonuc in dilim),
            abstained=sum(sonuc.abstained for sonuc in dilim),
            observed_states=dict(sorted(confusion[sinif].items())),
        ).to_dict())

    cekimser_gereken = [sonuc for sonuc in sonuclar
                        if sonuc.epistemic_class in ABSTENTION_REQUIRED]
    yanlis_guven = [sonuc for sonuc in cekimser_gereken if sonuc.false_confidence]

    # Sessiz başarısızlık: FALSE örneğini kabul etmek (en tehlikeli hata türü).
    sessiz = [sonuc for sonuc in _dilim("FALSE")
              if sonuc.observed_state == DeneyimDurumu.VALID.value]

    # UNKNOWN ile UNCERTAIN ayrılabiliyor mu?
    unknown_durumlari = {sonuc.observed_state for sonuc in _dilim("UNKNOWN")}
    uncertain_durumlari = {sonuc.observed_state for sonuc in _dilim("UNCERTAIN")}
    durum_koduyla_ayrilabilir = bool(unknown_durumlari.isdisjoint(uncertain_durumlari))

    # P0-007: durum kodu ikisini de UNCERTAIN'e indirger, fakat
    # ``belirsizlik_sebebi`` alanı epistemik kaynağı ayırır.
    unknown_sebepleri = {sonuc.uncertainty_reason for sonuc in _dilim("UNKNOWN")}
    uncertain_sebepleri = {sonuc.uncertainty_reason for sonuc in _dilim("UNCERTAIN")}
    sebeple_ayrilabilir = bool(
        unknown_sebepleri
        and uncertain_sebepleri
        and unknown_sebepleri.isdisjoint(uncertain_sebepleri)
    )
    # Her sınıf tek ve doğru sebebi üretmeli (karışık sebep = ayrım güvenilmez).
    sebep_tutarli = (
        unknown_sebepleri == {BelirsizlikSebebi.KAYIT_YOK.value}
        and uncertain_sebepleri == {BelirsizlikSebebi.OZELLIK_YOK.value}
    )

    epistemic_resolution = {
        "unknown_observed_states": sorted(unknown_durumlari),
        "uncertain_observed_states": sorted(uncertain_durumlari),
        "distinguishable_by_state": durum_koduyla_ayrilabilir,
        "unknown_reasons": sorted(unknown_sebepleri),
        "uncertain_reasons": sorted(uncertain_sebepleri),
        "distinguishable_by_reason": sebeple_ayrilabilir,
        "reasons_consistent": sebep_tutarli,
        "distinguishable": sebeple_ayrilabilir,
        "note": (
            "UNKNOWN (kayıt hiç yok) ve UNCERTAIN (özellik yazılmamış) epistemik "
            "olarak farklıdır. Durum KODU ikisini de DeneyimDurumu.UNCERTAIN'e "
            "indirger (distinguishable_by_state=False) fakat "
            "ExperienceCandidate.belirsizlik_sebebi alanı kaynağı ayırır: "
            "KAYIT_YOK ('bilmiyorum') ve OZELLIK_YOK ('emin değilim'). Ayrım "
            "durum makinesi geçişleri bozulmadan ölçülebilir."
        ),
    }

    dogru = sum(sonuc.correct for sonuc in sonuclar)
    known_acc = _ratio(sum(s.correct for s in _dilim("KNOWN")), len(_dilim("KNOWN")))
    false_acc = _ratio(sum(s.correct for s in _dilim("FALSE")), len(_dilim("FALSE")))
    conflict_acc = _ratio(sum(s.correct for s in _dilim("CONFLICT")),
                          len(_dilim("CONFLICT")))
    unknown_acc = _ratio(sum(sonuc.correct for sonuc in cekimser_gereken),
                         len(cekimser_gereken))

    # Negatif kontrol: dejenere politikalar bu benchmarkı geçemiyor mu?
    baselines = run_epistemic_baselines(data)
    baseline_dominated = all(
        known_acc >= arm.known_accuracy
        and unknown_acc >= arm.unknown_accuracy
        and _ratio(len(yanlis_guven), len(cekimser_gereken)) <= arm.false_confidence_rate
        and _ratio(dogru, len(sonuclar)) > arm.accuracy
        for arm in baselines
    )

    checks = {
        # Bilinen doğruyu kabul etmeli.
        "known_cases_accepted": known_acc == 1.0,
        # Bilinen yanlışı ASLA kabul etmemeli (sessiz başarısızlık yok).
        "no_silent_acceptance_of_false": len(sessiz) == 0,
        # Kanıt yokluğunda kesin karar vermemeli.
        "no_false_confidence_on_unknown": len(yanlis_guven) == 0,
        # Çelişkiyi çelişki olarak işaretlemeli.
        "conflicts_flagged": conflict_acc == 1.0,
        # Beş epistemik sınıf da fixture'da temsil edilmeli.
        "all_epistemic_classes_present": all(
            len(_dilim(sinif)) > 0 for sinif in EPISTEMIC_CLASSES
        ),
        # Benchmark dejenere politikalarla geçilememeli (negatif kontrol).
        "beats_degenerate_baselines": baseline_dominated,
        # "Bilmiyorum" ile "emin değilim" ayırt edilebilmeli (P0-007).
        "unknown_uncertain_distinguishable": sebeple_ayrilabilir,
        "uncertainty_reasons_consistent": sebep_tutarli,
    }

    return EpistemicReport(
        protocol="epistemic-known-unknown-uncertain-conflict-false-v1",
        dataset_hash=data.dataset_hash(),
        total_cases=len(sonuclar),
        correct=dogru,
        accuracy=_ratio(dogru, len(sonuclar)),
        false_confidence_cases=len(yanlis_guven),
        false_confidence_rate=_ratio(len(yanlis_guven), len(cekimser_gereken)),
        unknown_accuracy=unknown_acc,
        known_accuracy=known_acc,
        false_detection_accuracy=false_acc,
        conflict_accuracy=conflict_acc,
        silent_failure_cases=len(sessiz),
        silent_failure_rate=_ratio(len(sessiz), len(_dilim("FALSE"))),
        epistemic_resolution=epistemic_resolution,
        per_class=per_class,
        confusion={sinif: dict(sorted(deger.items()))
                   for sinif, deger in confusion.items()},
        cases=[sonuc.to_dict() for sonuc in sonuclar],
        checks=checks,
        baselines=[arm.to_dict() for arm in baselines],
        limitations=[
            "Kontrollü ontoloji üzerinde epistemik karar testidir; gerçek dil anlama değildir.",
            "UNKNOWN ve UNCERTAIN tek duruma eşlendiği için ayrım ölçülür, varsayılmaz.",
            "Sonuç Evaluator karar ağacının davranışıdır; neural üretim kalitesi değildir.",
            "false_confidence_rate yalnız bu fixture içindir; genel halüsinasyon oranı değildir.",
        ],
    )


@dataclass
class BaselineArm:
    """Dejenere bir karar politikasının aynı fixture üzerindeki sonucu."""

    arm: str
    description: str
    accuracy: float
    known_accuracy: float
    false_detection_accuracy: float
    unknown_accuracy: float
    false_confidence_rate: float
    silent_failure_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _score_policy(
    cases: Sequence[Mapping[str, Any]],
    policy: Any,
    arm: str,
    description: str,
) -> BaselineArm:
    """Bir politikayı (case -> DeneyimDurumu) fixture üzerinde puanla."""
    dogru = 0
    sinif_toplam: Dict[str, int] = {}
    sinif_dogru: Dict[str, int] = {}
    cekimser_gereken = 0
    cekimser_dogru = 0
    yanlis_guven = 0
    sessiz = 0

    for case in cases:
        sinif = str(case["epistemic_class"])
        durum = policy(case)
        isabet = durum.value == str(case["expected_state"])
        dogru += int(isabet)
        sinif_toplam[sinif] = sinif_toplam.get(sinif, 0) + 1
        sinif_dogru[sinif] = sinif_dogru.get(sinif, 0) + int(isabet)

        if sinif in ABSTENTION_REQUIRED:
            cekimser_gereken += 1
            cekimser_dogru += int(isabet)
            if durum in COMMITTED_STATES:
                yanlis_guven += 1
        if sinif == "FALSE" and durum == DeneyimDurumu.VALID:
            sessiz += 1

    return BaselineArm(
        arm=arm,
        description=description,
        accuracy=_ratio(dogru, len(cases)),
        known_accuracy=_ratio(sinif_dogru.get("KNOWN", 0), sinif_toplam.get("KNOWN", 0)),
        false_detection_accuracy=_ratio(
            sinif_dogru.get("FALSE", 0), sinif_toplam.get("FALSE", 0)
        ),
        unknown_accuracy=_ratio(cekimser_dogru, cekimser_gereken),
        false_confidence_rate=_ratio(yanlis_guven, cekimser_gereken),
        silent_failure_rate=_ratio(sessiz, sinif_toplam.get("FALSE", 0)),
    )


def run_epistemic_baselines(
    dataset: Optional[EpistemicDataset] = None,
) -> List[BaselineArm]:
    """Negatif kontrol: benchmark dejenere politikalarla geçilebiliyor mu?

    Tek bir metrik kolay kandırılır. Bu yüzden iki metrik birlikte raporlanır
    ve aşağıdaki dejenere kollar ikisini AYNI ANDA yükseltemez:

    * ``always_valid``  — her şeye "doğru" der: bilinenleri geçer ama
      ``false_confidence_rate = 1.0`` ve yanlışları sessizce kabul eder.
    * ``always_abstain`` — her şeye "bilmiyorum" der: ``unknown_accuracy = 1.0``
      olur ama hiçbir bilineni kabul edemez (``known_accuracy = 0.0``).
    * ``always_invalid`` — her şeyi reddeder: yanlışları yakalar ama
      bilinenleri de reddeder ve yine yüksek yanlış güven üretir.

    Gerçek değerlendiricinin anlamlı olabilmesi için her iki metrikte de bu
    kolların hepsini geçmesi gerekir; ölçüt budur.
    """
    data = dataset or EpistemicDataset()
    return [
        _score_policy(
            data.cases, lambda case: DeneyimDurumu.VALID, "always_valid",
            "Her adayı kabul eden aşırı güvenli politika",
        ),
        _score_policy(
            data.cases, lambda case: DeneyimDurumu.UNCERTAIN, "always_abstain",
            "Her aday için çekimser kalan politika",
        ),
        _score_policy(
            data.cases, lambda case: DeneyimDurumu.INVALID, "always_invalid",
            "Her adayı reddeden politika",
        ),
    ]


def run_epistemic_seed_sweep(
    root: Any,
    seeds: Sequence[int] = (1, 2, 3),
) -> Any:
    """Epistemik benchmarkı tam manifest sözleşmesiyle çok seed'te çalıştır."""
    from .experiment import run_seed_sweep

    data = EpistemicDataset()

    def _callback(seed: int) -> Dict[str, Any]:
        rapor = run_epistemic_benchmark(data, seed=seed)
        return {
            "metrics": {
                "accuracy": rapor.accuracy,
                "known_accuracy": rapor.known_accuracy,
                "false_detection_accuracy": rapor.false_detection_accuracy,
                "conflict_accuracy": rapor.conflict_accuracy,
                "unknown_accuracy": rapor.unknown_accuracy,
                "false_confidence_rate": rapor.false_confidence_rate,
                "silent_failure_rate": rapor.silent_failure_rate,
            },
            "report": rapor.to_dict(),
        }

    return run_seed_sweep(
        _callback,
        seeds=seeds,
        root=root,
        config={
            "protocol": "epistemic-known-unknown-uncertain-conflict-false-v1",
            "dataset": DATA_FILE,
        },
        dataset_hash=data.dataset_hash(),
    )


__all__ = [
    "ABSTENTION_REQUIRED",
    "COMMITTED_STATES",
    "EPISTEMIC_CLASSES",
    "EpistemicCaseResult",
    "EpistemicClassMetrics",
    "EpistemicDataset",
    "EpistemicReport",
    "run_epistemic_benchmark",
    "run_epistemic_seed_sweep",
]
