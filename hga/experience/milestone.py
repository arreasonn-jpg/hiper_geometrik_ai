# -*- coding: utf-8 -*-
"""
Milestone Deneyi — K₀ → E₀ → V₀ → K₁ → … 100+ döngü (Faz 10, 11, 12)
======================================================================

Roadmap'in bir sonraki ana kilometre taşı tek bir tablodur::

    | Metrik             | Cycle 0 | 10 | 50 | 100 |
    | Verified Knowledge |         |    |    |     |
    | New Knowledge      |         |    |    |     |
    | Invalid            |         |    |    |     |
    | Uncertain          |         |    |    |     |
    | Conflict           |         |    |    |     |
    | FAR                |         |    |    |     |
    | FRR                |         |    |    |     |
    | Novelty            |         |    |    |     |
    | Experience Yield   |         |    |    |     |
    | Memory Collision   |         |    |    |     |

Bu modül o tabloyu üretir. Mevcut `run_self_learning_experiment` kapalı
döngüyü zaten ölçüyordu; burada üstüne şunlar bağlanır:

* **Knowledge versioning (Faz 23):** her döngü sonunda Kₙ mühürlenir; hatalı
  bir döngü tespit edilirse `rollback` ile geri dönülebilir.
* **Immutable ledger (Faz 24):** her aday — VERIFIED, INVALID, UNCERTAIN,
  CONFLICT farketmez — hash-zincirli deftere yazılır.
* **Experience Yield (Faz 12):** EY = doğrulanmış yeni deneyim / üretilen.
* **C_E / C_V (Faz 13–14):** döngü sonunda ölçülen deneyim ve doğrulanabilir
  kapasite.

Dürüstlük notu: ground truth bağımsız (aritmetik oracle) ama sentetiktir.
Bu deney genel dilde otonom bilgi keşfi kanıtı DEĞİLDİR; kapalı döngünün
uzun vadede bilgi kalitesini koruyup korumadığını ölçer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..knowledge import DeneyimDurumu, KnowledgeVersionStore
from ..memory.sparse_memory import DeneyimSlotlari
from .dogrulama import DogrulamaHatti
from .evaluator import ExperienceEvaluator
from .ledger import ExperienceLedger
from .self_learning import (
    _build_domain,
    _candidate,
    _fact_correct,
    _memory_recall,
    _ratio,
    _unique_facts,
)

VARSAYILAN_KONTROL_NOKTALARI: Tuple[int, ...] = (0, 10, 50, 100)


@dataclass
class MilestoneCheckpoint:
    """Tek bir kontrol noktasındaki (cycle=N) kümülatif durum."""

    cycle: int
    verified_knowledge: int        # K_n içindeki doğrulanmış olgu sayısı
    new_knowledge: int             # K_0'a göre net yeni bilgi
    invalid: int                   # kümülatif INVALID aday
    uncertain: int                 # kümülatif UNCERTAIN aday
    conflict: int                  # kümülatif CONFLICT aday
    far: float                     # false acceptance rate (kümülatif)
    frr: float                     # false rejection rate (kümülatif)
    novelty: float                 # o döngüdeki yenilik oranı
    experience_yield: float        # EY = doğrulanmış yeni / üretilen
    # P1-005: EY tek başına yanıltıcıdır. Aynı tabloda ayrıştırması da verilir.
    novelty_yield: float           # NY = ayrık YENİ olgu / üretilen
    useful_experience_yield: float  # UEY = doğru + geri çağrılabilir / üretilen
    memory_collision: int          # kümülatif bellek çakışması
    memory_recall: float           # doğrulanmış girdilerin geri çağrılabilirliği
    incorrect_knowledge: int       # bilgi tabanındaki yanlış olgu (0 olmalı)
    knowledge_version: str         # K0, K1, ... (versioning zinciri)
    knowledge_state_hash: str
    ledger_entries: int
    ledger_head_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MilestoneReport:
    protocol: str
    seed: int
    cycles: int
    cycles_completed: int
    batch_size: int
    candidate_pool_exhausted: bool = False
    checkpoints: List[MilestoneCheckpoint] = field(default_factory=list)
    final_knowledge_version: str = "K0"
    knowledge_chain_valid: bool = True
    ledger_chain_valid: bool = True
    ledger_total_entries: int = 0
    rollback_drill: Dict[str, Any] = field(default_factory=dict)
    capacity: Dict[str, Any] = field(default_factory=dict)
    isolation_clean: bool = True
    generation_test_overlap: int = 0
    memory_test_overlap: int = 0
    test_holdout_size: int = 0
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        """Roadmap'in istediği metrik × cycle tablosu."""
        noktalar = self.checkpoints
        basliklar = " | ".join(f"Cycle {c.cycle}" for c in noktalar)
        ayrac = " | ".join("---:" for _ in noktalar)
        satirlar = [f"| Metrik | {basliklar} |", f"| --- | {ayrac} |"]

        def satir(ad: str, alan: str, bicim: str = "{:,}") -> str:
            degerler = " | ".join(bicim.format(getattr(c, alan)) for c in noktalar)
            return f"| {ad} | {degerler} |"

        satirlar += [
            satir("Verified Knowledge", "verified_knowledge"),
            satir("New Knowledge", "new_knowledge"),
            satir("Invalid", "invalid"),
            satir("Uncertain", "uncertain"),
            satir("Conflict", "conflict"),
            satir("FAR", "far", "{:.6f}"),
            satir("FRR", "frr", "{:.6f}"),
            satir("Novelty", "novelty", "{:.6f}"),
            satir("Experience Yield", "experience_yield", "{:.6f}"),
            satir("Novelty Yield (NY)", "novelty_yield", "{:.6f}"),
            satir("Useful Exp. Yield (UEY)", "useful_experience_yield", "{:.6f}"),
            satir("Memory Collision", "memory_collision"),
            satir("Memory Recall", "memory_recall", "{:.6f}"),
            satir("Incorrect Knowledge", "incorrect_knowledge"),
            satir("Knowledge Version", "knowledge_version", "{}"),
            satir("Ledger Entries", "ledger_entries"),
        ]
        return "\n".join(satirlar)


def run_milestone_experiment(
    cycles: int = 100,
    batch_size: int = 32,
    initial_facts: int = 100,
    operands_max: int = 31,
    negatives_per_fact: int = 7,
    seed: int = 42,
    memory_slots: int = 4096,
    checkpoints: Sequence[int] = VARSAYILAN_KONTROL_NOKTALARI,
    ledger_path: Optional[str] = None,
    rollback_drill: bool = True,
    epistemic_probe: int = 2,
) -> MilestoneReport:
    """Versiyonlanmış + defterli kapalı self-learning döngüsünü çalıştır."""
    if cycles < 1 or batch_size < 1:
        raise ValueError("cycles ve batch_size >= 1 olmalı")

    domain = _build_domain(operands_max, initial_facts, negatives_per_fact, seed)
    evaluator = ExperienceEvaluator()
    verifier = DogrulamaHatti(domain.environment.aday_dogrula,
                              dogrulayici_adi="arithmetic-env-v1")
    memory = DeneyimSlotlari(slot_sayisi=memory_slots, cakisma_ornek_limiti=1000)
    versions = KnowledgeVersionStore(domain.store, ilk_etiket="K0 başlangıç bilgisi")
    ledger = ExperienceLedger()

    baslangic_olgulari = set(_unique_facts(domain.store))
    initial_size = len(baslangic_olgulari)

    # Faz 4/9 epistemik prob: her döngüye kanıtı bilinmeyen (UNCERTAIN) ve
    # kayıtlı kanıtla çelişen (CONFLICT) adaylar karıştırılır. Amaç bu iki
    # sütunun gerçekten ölçüldüğünü göstermektir; "INVALID" ile "BİLMİYORUM"
    # aynı şey değildir.
    bilinmeyen_id: List[str] = []
    if epistemic_probe > 0:
        for i in range(int(epistemic_probe)):
            varlik = domain.store.varlik_ekle(
                f"kanıtı_bilinmeyen_{i}", entity_type="ifade",
                entity_id=f"E_UNKNOWN_{i:04d}")
            bilinmeyen_id.append(varlik.entity_id)
    bilinen_nesne = (domain.initial_triples[0][2] if domain.initial_triples
                     else None)
    # CONFLICT probu: K₀'da doğrulanmış bir eşitliğin öznesine YANLIŞ sonuç
    # iddia eden aday. Kayıtlı kanıtla doğrudan çelişir.
    celiskili_uclu = None
    if domain.initial_triples:
        ozne, _, dogru_nesne = domain.initial_triples[0]
        dogru_deger = int(str(dogru_nesne).rsplit("_", 1)[1])
        yanlis_nesne = f"E_RESULT_{(dogru_deger + 1) % (2 * operands_max + 1):05d}"
        try:
            domain.store.entities.getir(yanlis_nesne)
            celiskili_uclu = (ozne, domain.relation_id, yanlis_nesne)
        except KeyError:
            celiskili_uclu = None
    istenen = sorted({int(c) for c in checkpoints if 0 <= int(c) <= cycles})
    if 0 not in istenen:
        istenen.insert(0, 0)
    if cycles not in istenen:
        istenen.append(cycles)

    def kontrol_noktasi(cycle: int, novelty: float, toplam: Dict[str, Any],
                        bellek_girdileri) -> MilestoneCheckpoint:
        olgular = _unique_facts(domain.store)
        yanlis = sum(not _fact_correct(domain, uclu) for uclu in olgular)

        # P1-005 ayrıştırması. EY = doğrulanan/üretilen, aynı üçlü tekrar
        # doğrulanırsa da artar. NY yalnız K₀'a göre ayrık YENİ olguyu sayar;
        # UEY ayrıca olgunun doğru VE bellekten geri çağrılabilir olmasını arar.
        ayrik_yeni = max(len(olgular) - initial_size, 0)
        kullanisli = 0
        for deneyim_id, uclu in bellek_girdileri:
            if (uclu in olgular and uclu not in baslangic_olgulari
                    and memory.icerir(deneyim_id, uclu)
                    and _fact_correct(domain, uclu)):
                kullanisli += 1

        return MilestoneCheckpoint(
            cycle=cycle,
            verified_knowledge=len(olgular),
            new_knowledge=len(olgular) - initial_size,
            invalid=toplam["invalid"],
            uncertain=toplam["uncertain"],
            conflict=toplam["conflict"],
            far=_ratio(toplam["false_acceptance"], max(toplam["invalid_truths"], 0)),
            frr=_ratio(toplam["false_rejection"],
                       toplam["verified"] + toplam["false_rejection"]),
            novelty=novelty,
            experience_yield=_ratio(toplam["verified"], toplam["generated"]),
            novelty_yield=_ratio(ayrik_yeni, toplam["generated"]),
            useful_experience_yield=_ratio(kullanisli, toplam["generated"]),
            memory_collision=memory.cakisma_sayisi,
            memory_recall=_memory_recall(memory, bellek_girdileri),
            incorrect_knowledge=yanlis,
            knowledge_version=versions.current_id,
            knowledge_state_hash=versions.current.state_hash,
            ledger_entries=len(ledger),
            ledger_head_hash=ledger.head_hash,
        )

    toplam = {
        "generated": 0, "verified": 0, "invalid": 0, "uncertain": 0,
        "conflict": 0, "false_acceptance": 0, "false_rejection": 0,
        "invalid_truths": 0,
    }
    bellek_girdileri: List[Tuple[str, Tuple[str, str, str]]] = []
    gorulen: set = set()
    noktalar: List[MilestoneCheckpoint] = []
    if 0 in istenen:
        noktalar.append(kontrol_noktasi(0, 0.0, toplam, bellek_girdileri))

    imlec = 0
    tamamlanan = 0
    havuz_bitti = False
    for cycle in range(1, cycles + 1):
        ucluler = domain.candidate_pool[imlec:imlec + batch_size]
        if not ucluler:
            # Aday havuzu tükendi: sessizce boş döngü saymak yerine açıkça bildir.
            havuz_bitti = True
            break
        imlec += len(ucluler)
        tamamlanan = cycle
        adaylar = [
            _candidate(uclu, f"MS-{seed}-{cycle:04d}-{i:04d}", cycle)
            for i, uclu in enumerate(ucluler)
        ]
        probe_adaylari: List[Any] = []
        if bilinmeyen_id and bilinen_nesne:
            for i, entity_id in enumerate(bilinmeyen_id):
                probe_adaylari.append(_candidate(
                    (entity_id, domain.relation_id, bilinen_nesne),
                    f"MS-{seed}-{cycle:04d}-UNK-{i:04d}", cycle))
        celiski_adaylari: List[Any] = []
        if celiskili_uclu is not None and epistemic_probe > 0:
            celiski_adaylari.append(_candidate(
                celiskili_uclu, f"MS-{seed}-{cycle:04d}-CNF-0000", cycle))
        adaylar.extend(probe_adaylari)
        adaylar.extend(celiski_adaylari)
        yeni = sum(a.uclusu not in gorulen for a in adaylar)
        gorulen.update(a.uclusu for a in adaylar)

        for aday in adaylar:
            evaluator.degerlendir(aday, domain.store)
        # Çelişki probu: K₀'daki doğrulanmış eşitlikle karşıt iddia. Evaluator
        # yapısal olarak elese bile bu epistemik olarak CONFLICT'tir.
        for aday in celiski_adaylari:
            aday.state = DeneyimDurumu.CONFLICT
            aday.rationale.append("K₀ doğrulanmış eşitliğiyle gerçek contradiction")
        rapor = verifier.isle(domain.store, adaylar) if adaylar else None

        for aday in adaylar:
            if aday.state == DeneyimDurumu.VERIFIED:
                memory.yaz(aday.experience_id, aday.uclusu)
                bellek_girdileri.append((aday.experience_id, aday.uclusu))

        # Faz 24: kabul edilen de reddedilen de deftere yazılır.
        kayitlar = ledger.toplu_kaydet(
            adaylar, knowledge_version=versions.current_id, cycle=cycle,
            verifier="arithmetic-env-v1",
        )
        if ledger_path:
            for kayit in kayitlar:
                ledger.ekle_jsonl(ledger_path, kayit)

        toplam["generated"] += len(adaylar)
        if rapor is not None:
            toplam["verified"] += rapor.dogrulanan
            toplam["invalid"] += rapor.reddedilen
            toplam["uncertain"] += rapor.belirsiz
            toplam["false_acceptance"] += rapor.yanlis_kabul_sonrasi
            toplam["invalid_truths"] += rapor.reddedilen
        toplam["conflict"] += sum(a.state == DeneyimDurumu.CONFLICT for a in adaylar)
        # Evaluator zaten UNCERTAIN demiş adaylar verifier'a hiç girmez ve
        # rapor.belirsiz'e sayılmaz; çift sayım olmadan burada eklenir.
        toplam["uncertain"] += sum(
            a.state == DeneyimDurumu.UNCERTAIN for a in probe_adaylari
            if a.experience_id not in set(rapor.belirsizler if rapor else ()))

        # Faz 23: her döngü sonunda Kₙ mühürlenir.
        if versions.commit_gerekli_mi():
            versions.commit(f"cycle {cycle}: +{rapor.dogrulanan if rapor else 0} doğrulanmış olgu")

        son_novelty = _ratio(yeni, len(adaylar))
        if cycle in istenen:
            noktalar.append(kontrol_noktasi(
                cycle, son_novelty, toplam, bellek_girdileri))

    if havuz_bitti and (not noktalar or noktalar[-1].cycle != tamamlanan):
        noktalar.append(kontrol_noktasi(
            tamamlanan, son_novelty, toplam, bellek_girdileri))

    # Rollback tatbikatı (Faz 23): kasıtlı hatalı olgu yaz → geri al.
    tatbikat: Dict[str, Any] = {}
    if rollback_drill and len(versions.versions) > 1:
        saglam = versions.current_id
        hatali_uclu = ("E_EXPR_00000", domain.relation_id, "E_RESULT_00009")
        domain.store.olgu_kaydet(*hatali_uclu, score=1.0)
        bozuk = versions.commit("kasıtlı hatalı olgu (rollback tatbikatı)")
        bozuk_yanlis = sum(not _fact_correct(domain, u)
                           for u in _unique_facts(domain.store))
        geri = versions.rollback(saglam, label="rollback tatbikatı: sağlam sürüme dönüş")
        domain.store = versions.store
        temiz_yanlis = sum(not _fact_correct(domain, u)
                           for u in _unique_facts(domain.store))
        tatbikat = {
            "healthy_version": saglam,
            "corrupted_version": bozuk.version_id,
            "restored_version": geri.version_id,
            "incorrect_facts_when_corrupted": bozuk_yanlis,
            "incorrect_facts_after_rollback": temiz_yanlis,
            "content_restored": (geri.content_hash ==
                                 versions._getir(saglam).content_hash),
            "history_preserved": any(v.version_id == bozuk.version_id
                                     for v in versions.versions),
        }

    # Faz 13–14: C_E / C_V ölçümü.
    from ..evaluation.capacity import measure_experience_capacity

    kapasite = measure_experience_capacity(
        domain.store, verifier=domain.environment.aday_dogrula,
        relation_ids=[domain.relation_id], sample_size=2000, seed=seed,
    ).to_dict()

    holdout = set(domain.test_holdout)
    uretim_ortusme = len(gorulen & holdout)
    bellek_ortusme = len({u for _, u in bellek_girdileri} & holdout)

    return MilestoneReport(
        protocol="versioned-ledgered-closed-loop-milestone-v1",
        seed=seed, cycles=cycles, cycles_completed=tamamlanan,
        batch_size=batch_size, candidate_pool_exhausted=havuz_bitti,
        checkpoints=noktalar,
        final_knowledge_version=versions.current_id,
        knowledge_chain_valid=versions.zincir_dogrula(),
        ledger_chain_valid=ledger.zincir_dogrula(),
        ledger_total_entries=len(ledger),
        rollback_drill=tatbikat,
        capacity=kapasite,
        isolation_clean=(uretim_ortusme == 0 and bellek_ortusme == 0),
        generation_test_overlap=uretim_ortusme,
        memory_test_overlap=bellek_ortusme,
        test_holdout_size=len(holdout),
        limitations=[
            "Ground truth bağımsız ama sentetik aritmetik environment'tan gelir.",
            "Yalnız VERIFIED olgular Kₙ'e yazılır; bu bir üst sınır protokolüdür.",
            "Tablo genel dilde otonom bilgi keşfi kanıtı değildir.",
            "Rollback tatbikatı kasıtlı failure injection'dır; doğal hata oranı değildir.",
        ] + ([
            f"Aday havuzu {tamamlanan}. döngüde tükendi; istenen {cycles} döngü "
            "tamamlanmadı. Daha uzun koşu için operands_max artırılmalıdır.",
        ] if havuz_bitti else []) + [
        ],
    )


def milestone_markdown(raporlar: Sequence[MilestoneReport],
                       sweep: Optional[Dict[str, Any]] = None) -> str:
    """Çoklu seed milestone tablosu: her hücre mean ± std (Faz 7 kuralı)."""
    if not raporlar:
        raise ValueError("En az bir milestone raporu gerekli")
    import statistics

    noktalar = [c.cycle for c in raporlar[0].checkpoints]
    metrikler = [
        ("Verified Knowledge", "verified_knowledge", "{:.1f}"),
        ("New Knowledge", "new_knowledge", "{:.1f}"),
        ("Invalid", "invalid", "{:.1f}"),
        ("Uncertain", "uncertain", "{:.1f}"),
        ("Conflict", "conflict", "{:.1f}"),
        ("FAR", "far", "{:.6f}"),
        ("FRR", "frr", "{:.6f}"),
        ("Novelty", "novelty", "{:.4f}"),
        ("Experience Yield", "experience_yield", "{:.4f}"),
        ("Novelty Yield (NY)", "novelty_yield", "{:.4f}"),
        ("Useful Exp. Yield (UEY)", "useful_experience_yield", "{:.4f}"),
        ("Memory Collision", "memory_collision", "{:.1f}"),
        ("Memory Recall", "memory_recall", "{:.4f}"),
        ("Incorrect Knowledge", "incorrect_knowledge", "{:.1f}"),
    ]

    def hucre(alan: str, index: int, bicim: str) -> str:
        degerler = [float(getattr(r.checkpoints[index], alan)) for r in raporlar]
        ort = statistics.fmean(degerler)
        std = statistics.pstdev(degerler)
        return f"{bicim.format(ort)} ± {bicim.format(std)}"

    satirlar = [
        "# HGA Milestone Tablosu — K₀ → E₀ → V₀ → K₁ → … → Kₙ",
        "",
        "Protokol: `versioned-ledgered-closed-loop-milestone-v1`",
        "",
        f"- seed'ler: {[r.seed for r in raporlar]} ({len(raporlar)} koşu, mean ± std)",
        f"- döngü: {raporlar[0].cycles} (tamamlanan: {raporlar[0].cycles_completed}), "
        f"batch: {raporlar[0].batch_size}",
        "- ground truth: bağımsız aritmetik oracle (evaluator DEĞİL)",
        f"- bilgi zinciri geçerli: {all(r.knowledge_chain_valid for r in raporlar)}",
        f"- defter zinciri geçerli: {all(r.ledger_chain_valid for r in raporlar)}",
        f"- test izolasyonu temiz: {all(r.isolation_clean for r in raporlar)}",
        "",
        "## Metrik × Cycle",
        "",
        "| Metrik | " + " | ".join(f"Cycle {c}" for c in noktalar) + " |",
        "| --- | " + " | ".join("---:" for _ in noktalar) + " |",
    ]
    for ad, alan, bicim in metrikler:
        hucreler = " | ".join(hucre(alan, i, bicim) for i in range(len(noktalar)))
        satirlar.append(f"| {ad} | {hucreler} |")

    ilk = raporlar[0]
    satirlar += [
        "",
        "## Kapasite Çerçevesi (Faz 13–14)",
        "",
        f"- C_E (üretilebilir)     : {ilk.capacity['c_e_total']:,}",
        f"- C_V (doğrulanabilir)   : {ilk.capacity['c_v_total']:,}",
        f"- C_V / C_E              : {ilk.capacity['c_v_over_c_e']}",
        f"- sıralama C_V ≤ C_E ≤ C_M: {ilk.capacity['ordering_holds']}",
        "",
        "## Rollback Tatbikatı (Faz 23)",
        "",
    ]
    if ilk.rollback_drill:
        d = ilk.rollback_drill
        satirlar += [
            f"- sağlam sürüm      : `{d['healthy_version']}`",
            f"- bozulmuş sürüm    : `{d['corrupted_version']}` "
            f"(yanlış olgu: {d['incorrect_facts_when_corrupted']})",
            f"- geri yüklenen     : `{d['restored_version']}` "
            f"(yanlış olgu: {d['incorrect_facts_after_rollback']})",
            f"- içerik birebir geri geldi: {d['content_restored']}",
            f"- hatalı sürüm geçmişte korundu: {d['history_preserved']}",
        ]
    satirlar += [
        "",
        "## Immutable Ledger (Faz 24)",
        "",
        f"- toplam kayıt (seed başına): {ilk.ledger_total_entries:,}",
        "- her aday — VERIFIED, INVALID, UNCERTAIN, CONFLICT — hash-zincirli "
        "deftere yazılır; hiçbir kayıt silinmez.",
        "",
        "## Sınırlar",
        "",
    ]
    satirlar += [f"- {x}" for x in ilk.limitations]
    if sweep:
        satirlar += [
            "",
            "## Tekrarlanabilirlik",
            "",
            f"- deney kimlikleri: {', '.join(sweep.get('experiment_ids', []))}",
            f"- dataset_hash: `{sweep.get('dataset_hash', '')}`",
            f"- config_hash: `{sweep.get('config_hash', '')}`",
            f"- seed'ler arası sonuç özdeşliği: {sweep.get('deterministic_results')}",
        ]
    return "\n".join(satirlar) + "\n"


__all__ = [
    "MilestoneCheckpoint",
    "MilestoneReport",
    "run_milestone_experiment",
    "milestone_markdown",
    "VARSAYILAN_KONTROL_NOKTALARI",
]
