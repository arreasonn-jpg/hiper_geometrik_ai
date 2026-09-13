# -*- coding: utf-8 -*-
"""P1-005 — Deneyim verimi metrikleri: EY'nin ötesinde.

Depoda şu ana kadar tek bir verim ölçüsü vardı::

    EY = doğrulanmış / üretilen

Bu ölçü **yanıltıcıdır**, çünkü şu üç durumu birbirinden ayıramaz:

1. Sistem zaten bildiği bir şeyi tekrar üretip doğrular (yeni bilgi yok).
2. Sistem bir olguyu doğrular ama olgu bellekte kaybolur (kullanılamaz).
3. Sistem bir olguyu ezberler ama hiçbir yeni duruma genelleyemez.

Üçünde de EY yüksektir. Bu modül EY'yi dört ayrı eksene ayırır:

``novelty_yield`` (NY)
    Üretilen deneyim başına **ayrık yeni** doğrulanmış olgu. Tekrar üretilen
    aynı üçlüler paydayı şişirir ama payı şişiremez.

``useful_experience_yield`` (UEY)
    Üretilen deneyim başına, doğrulanmış **ve** doğru **ve** bellekten geri
    çağrılabilir olgu. Doğrulanıp sonra bellek çakışmasında kaybolan bilgi
    işe yaramaz; UEY bunu cezalandırır, EY görmezden gelir.

``generalization_yield`` (GY)
    Öğrenme sonucunda **holdout** (hiç görülmemiş) kümede doğru karar
    verilebilir hale gelen örneklerin oranı. Ezberleme ile genelleme
    arasındaki farkı ölçen tek metrik budur.

``verified_information_density`` (VID)
    Üretilen deneyim başına **bit** cinsinden doğrulanmış bilgi. Bir olguyu
    ``R`` olası sonuç arasından sabitlemek ``log2(R)`` bit bilgidir.

Bu metriklerin anlamlı olabilmesi için EY'den **ayrışmaları** gerekir; hepsi
EY ile birebir aynı değeri veriyorsa yeni bilgi taşımıyorlar demektir. Bu
ayrışma ``divergence`` bölümünde açıkça raporlanır ve testlerle sabitlenir.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..knowledge import DeneyimDurumu, KaynakTuru, KnowledgeStore
from ..memory import DeneyimSlotlari
from .dogrulama import DogrulamaHatti
from .evaluator import ExperienceEvaluator
from .self_learning import Triple, _build_domain, _candidate, _unique_facts


def _ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


# Bir olgunun "öğrenilmiş pozitif" sayılması için kaynağının bunlardan biri
# olması gerekir; model üretimi tek başına kanıt değildir.
_DOGRULANMIS_KAYNAKLAR = frozenset({
    KaynakTuru.VERIFIED_RULE,
    KaynakTuru.EXTERNAL_VERIFIED,
    KaynakTuru.HUMAN_CONFIRMED,
    KaynakTuru.REAL_DATA,
})


@dataclass
class HoldoutDecision:
    """Holdout kümesinde tek bir kararın sonucu."""

    total: int = 0
    decided: int = 0
    correct: int = 0
    abstained: int = 0
    coverage: float = 0.0
    accuracy_on_decided: float = 0.0
    correct_rate_overall: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class YieldReport:
    protocol: str
    seed: int
    cycles: int
    generated: int

    # Klasik (tek başına yanıltıcı) ölçü.
    experience_yield: float

    # P1-005 ile eklenen dört eksen.
    novelty_yield: float
    useful_experience_yield: float
    generalization_yield: float
    verified_information_density: float

    # Ham sayımlar — her oranın payı/paydası denetlenebilir olmalı.
    verified: int
    distinct_new_facts: int
    useful_facts: int
    correct_facts: int
    incorrect_facts: int
    duplicate_generations: int
    memory_collisions: int
    retrievable_facts: int
    bits_per_fact: float
    total_verified_bits: float

    holdout_before: Dict[str, Any]
    holdout_after: Dict[str, Any]
    divergence: Dict[str, Any]
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _infer_holdout(store: KnowledgeStore, triple: Triple) -> Optional[bool]:
    """Öğrenilmiş olgulardan holdout üçlüsü hakkında çıkarım yap.

    Kullanılan kural **fonksiyonel teklik**tir: ``eşittir`` ilişkisinde bir
    ifadenin tek bir doğru sonucu vardır. Dolayısıyla ``(ifade, =, X)``
    doğrulanmışsa, aynı ifade için ``(ifade, =, Y≠X)`` yanlıştır.

    Dönüş: ``True`` = doğru olmalı, ``False`` = yanlış olmalı, ``None`` =
    çekimser (bu ifade hakkında hiç öğrenilmiş pozitif yok).

    ``ExperienceEvaluator`` burada kasıtlı olarak kullanılmaz: ``R_EQUALS``
    ilişkisinin tip/özellik kısıtı olmadığı için evaluator holdout'un
    tamamına ``VALID`` der ve öğrenme öncesi/sonrası hiç değişmez. Yani
    evaluator ile ölçülen GY yapısal olarak daima 0.0 çıkar — bu ölçüm
    değil, ölçüm aracının körlüğüdür.
    """
    subject_id, relation_id, object_id = triple
    # Doğrulama hattı olguyu score=1.0 ile DEĞİL, doğrulayıcı güvenine göre
    # ölçeklenmiş bir skorla yazar (ör. 0.6375). Bu yüzden eşik skora değil,
    # kaynağın doğrulanmış olmasına bakılır: aksi halde öğrenilen her olgu
    # sessizce elenir ve GY yapay biçimde 0.0 görünür.
    pozitifler = {
        fact.object_id
        for fact in store.relations.olgular(subject_id, relation_id)
        if fact.source in _DOGRULANMIS_KAYNAKLAR and fact.score > 0.5
    }
    if not pozitifler:
        return None
    return object_id in pozitifler


def _decide_holdout(
    store: KnowledgeStore,
    environment: Any,
    holdout: Sequence[Triple],
) -> HoldoutDecision:
    """Holdout üçlülerini öğrenilmiş bilgiden çıkarımla karara bağla.

    Doğruluk bağımsız ortam oracle'ı ile kıyaslanır; karar verilemeyen
    örnekler çekimser sayılır ve doğruluk payına girmez.
    """
    sonuc = HoldoutDecision(total=len(holdout))
    for triple in holdout:
        tahmin = _infer_holdout(store, triple)
        if tahmin is None:
            sonuc.abstained += 1
            continue
        gercek = environment.aday_dogrula(
            store, _candidate(triple, "TRUTH", 0)
        ) is True
        sonuc.decided += 1
        sonuc.correct += int(tahmin == gercek)
    sonuc.coverage = _ratio(sonuc.decided, sonuc.total)
    sonuc.accuracy_on_decided = _ratio(sonuc.correct, sonuc.decided)
    sonuc.correct_rate_overall = _ratio(sonuc.correct, sonuc.total)
    return sonuc


def run_yield_experiment(
    cycles: int = 20,
    batch_size: int = 32,
    initial_facts: int = 40,
    operands_max: int = 15,
    negatives_per_fact: int = 3,
    seed: int = 42,
    memory_slots: int = 512,
    duplicate_rate: int = 4,
) -> YieldReport:
    """Kapalı döngü öğrenmeyi çalıştırıp dört verim eksenini ölç.

    ``duplicate_rate``: her N'inci döngüde daha önce üretilmiş üçlüler yeniden
    üretilir. Bu yapay değil, gerçekçidir: üretken bir sistem aynı şeyi tekrar
    önerir. EY bunu fark etmez, NY eder. ``0`` verilirse tekrar üretilmez.
    """
    if cycles < 1 or batch_size < 1:
        raise ValueError("cycles ve batch_size >= 1 olmalı")
    if duplicate_rate < 0:
        raise ValueError("duplicate_rate >= 0 olmalı")

    domain = _build_domain(operands_max, initial_facts, negatives_per_fact, seed)
    evaluator = ExperienceEvaluator()
    verifier = DogrulamaHatti(
        domain.environment.aday_dogrula, dogrulayici_adi="arithmetic-env-v1"
    )
    memory = DeneyimSlotlari(slot_sayisi=memory_slots, cakisma_ornek_limiti=1000)

    # Öğrenmeden ÖNCE holdout başarımı — genelleme bundan sonraki artıştır.
    holdout_before = _decide_holdout(
        domain.store, domain.environment, domain.test_holdout
    )

    baslangic_olgulari = set(_unique_facts(domain.store))
    uretilenler: List[Triple] = []
    gorulen: set = set()
    tekrar_uretim = 0
    dogrulanan = 0
    verified_entries: List[Tuple[str, Triple]] = []
    cursor = 0

    for cycle in range(1, cycles + 1):
        if duplicate_rate and cycle % duplicate_rate == 0 and uretilenler:
            # Daha önce üretilmiş üçlüleri tekrar öner.
            triples = uretilenler[:batch_size]
        else:
            triples = domain.candidate_pool[cursor:cursor + batch_size]
            cursor += len(triples)
        if not triples:
            break

        adaylar = [
            _candidate(triple, f"VY-{seed}-{cycle:04d}-{index:04d}", cycle)
            for index, triple in enumerate(triples)
        ]
        for aday in adaylar:
            if aday.uclusu in gorulen:
                tekrar_uretim += 1
            gorulen.add(aday.uclusu)
            uretilenler.append(aday.uclusu)
            evaluator.degerlendir(aday, domain.store)

        verification = verifier.isle(domain.store, adaylar)
        dogrulanan += verification.dogrulanan if verification else 0

        for aday in adaylar:
            if aday.state == DeneyimDurumu.VERIFIED:
                memory.yaz(aday.experience_id, aday.uclusu)
                verified_entries.append((aday.experience_id, aday.uclusu))

    uretilen_sayisi = len(uretilenler)

    # --- Ayrık yeni bilgi -------------------------------------------------
    son_olgular = set(_unique_facts(domain.store))
    yeni_olgular = son_olgular - baslangic_olgulari
    ayrik_yeni = len(yeni_olgular)

    # --- Doğruluk denetimi ------------------------------------------------
    dogru = sum(
        domain.environment.aday_dogrula(domain.store, _candidate(t, "AUDIT", 0)) is True
        for t in yeni_olgular
    )
    yanlis = ayrik_yeni - dogru

    # --- Kullanışlılık: doğrulanmış + doğru + bellekten geri çağrılabilir --
    geri_cagrilabilir = 0
    kullanisli: set = set()
    for experience_id, triple in verified_entries:
        if memory.icerir(experience_id, triple):
            geri_cagrilabilir += 1
            if triple in yeni_olgular:
                dogru_mu = domain.environment.aday_dogrula(
                    domain.store, _candidate(triple, "AUDIT", 0)
                ) is True
                if dogru_mu:
                    kullanisli.add(triple)

    # --- Genelleme: öğrenmeden SONRA holdout ------------------------------
    holdout_after = _decide_holdout(
        domain.store, domain.environment, domain.test_holdout
    )
    genelleme_artisi = (
        holdout_after.correct_rate_overall - holdout_before.correct_rate_overall
    )

    # --- Bilgi yoğunluğu: bir olguyu sabitlemek log2(R) bittir -------------
    sonuc_uzayi = 2 * operands_max + 1
    bit_basi = math.log2(sonuc_uzayi) if sonuc_uzayi > 1 else 0.0
    toplam_bit = dogru * bit_basi

    ey = _ratio(dogrulanan, uretilen_sayisi)
    ny = _ratio(ayrik_yeni, uretilen_sayisi)
    uey = _ratio(len(kullanisli), uretilen_sayisi)
    gy = round(max(0.0, genelleme_artisi), 8)
    vid = _ratio(toplam_bit, uretilen_sayisi)

    # --- Ayrışma: bu metrikler EY'den farklı bir şey söylüyor mu? ---------
    ayrisma = {
        "ey_vs_ny": round(ey - ny, 8),
        "ey_vs_uey": round(ey - uey, 8),
        "ny_vs_uey": round(ny - uey, 8),
        "ey_overstates_novelty": ny < ey,
        "ey_overstates_usefulness": uey < ey,
        "memorization_without_generalization": (ey > 0.0 and gy <= 0.0),
        "note": (
            "Bu farklar sıfırsa yeni metrikler EY'nin yeniden adlandırılmasıdır. "
            "Sıfırdan büyükse EY, yenilik ve kullanışlılığı sistematik olarak "
            "abartıyor demektir."
        ),
    }

    bulgular: List[str] = []
    if ny < ey:
        bulgular.append(
            f"EY={ey:.4f} fakat NY={ny:.4f}: üretimin bir kısmı zaten bilinen "
            f"üçlülerin tekrarı ({tekrar_uretim} tekrar üretim)."
        )
    if uey < ny:
        bulgular.append(
            f"NY={ny:.4f} fakat UEY={uey:.4f}: doğrulanan bilginin bir kısmı "
            f"bellekte kalıcı/kullanılabilir değil ({memory.cakisma_sayisi} çakışma)."
        )
    if gy <= 0.0:
        bulgular.append(
            "GY=0.0000: doğrulanmış deneyim holdout kümesine hiç genellemiyor. "
            "Bu domainde öğrenme EZBERLEMEDİR; yüksek EY genelleme kanıtı değildir."
        )
    else:
        bulgular.append(
            f"GY={gy:.4f}: holdout doğruluğu öğrenmeyle "
            f"{holdout_before.correct_rate_overall:.4f} → "
            f"{holdout_after.correct_rate_overall:.4f} arttı "
            f"({holdout_after.decided}/{holdout_after.total} örnekte karar verilebildi, "
            f"karar verilenlerde isabet {holdout_after.accuracy_on_decided:.4f}). "
            "Genelleme, görülmemiş negatifleri fonksiyonel teklikten çıkarmaktır."
        )
    if yanlis:
        bulgular.append(f"UYARI: bilgi tabanına {yanlis} yanlış olgu girdi.")
    bulgular.append(
        f"VID={vid:.4f} bit/deneyim (olgu başına {bit_basi:.4f} bit, "
        f"sonuç uzayı R={sonuc_uzayi})."
    )

    return YieldReport(
        protocol="experience-yield-decomposition-v1",
        seed=seed,
        cycles=cycles,
        generated=uretilen_sayisi,
        experience_yield=ey,
        novelty_yield=ny,
        useful_experience_yield=uey,
        generalization_yield=gy,
        verified_information_density=vid,
        verified=dogrulanan,
        distinct_new_facts=ayrik_yeni,
        useful_facts=len(kullanisli),
        correct_facts=dogru,
        incorrect_facts=yanlis,
        duplicate_generations=tekrar_uretim,
        memory_collisions=memory.cakisma_sayisi,
        retrievable_facts=geri_cagrilabilir,
        bits_per_fact=round(bit_basi, 8),
        total_verified_bits=round(toplam_bit, 8),
        holdout_before=holdout_before.to_dict(),
        holdout_after=holdout_after.to_dict(),
        divergence=ayrisma,
        findings=bulgular,
        limitations=[
            "Sentetik aritmetik domain; sonuçlar genel dil görevlerine taşınmaz.",
            "GY tek bir çıkarım kuralıyla (fonksiyonel teklik) ölçülür: öğrenilen "
            "pozitiften görülmemiş negatifi elemek. Bu gerçek ama DAR bir genellemedir; "
            "yeni toplama bağıntısı keşfetmek değildir.",
            "GY'nin paydası holdout kümesidir; kapsam (coverage) düşükken GY de düşük "
            "görünür — bu öğrenme eksikliğidir, metrik kusuru değil.",
            "UEY bellek geri çağrımına bağlıdır; farklı slot sayısı farklı UEY verir.",
            "VID, her olgunun sonuç uzayında tekdüze dağıldığını varsayar.",
            "Her rapor tek tohumun tek koşusudur; güven aralığı için çok tohumlu "
            "koşu (CLI --seeds veya run_yield_seed_sweep) gerekir.",
        ],
    )


def run_yield_seed_sweep(
    root: Any,
    seeds: Sequence[int] = (1, 2, 3),
    cycles: int = 20,
    batch_size: int = 32,
    initial_facts: int = 40,
    operands_max: int = 15,
    negatives_per_fact: int = 3,
    memory_slots: int = 512,
) -> Any:
    """Verim metriklerini çok tohumla çalıştır (manifest + istatistik ile)."""
    from ..evaluation.experiment import run_seed_sweep

    def _callback(seed: int) -> Dict[str, Any]:
        rapor = run_yield_experiment(
            cycles=cycles, batch_size=batch_size, initial_facts=initial_facts,
            operands_max=operands_max, negatives_per_fact=negatives_per_fact,
            seed=seed, memory_slots=memory_slots,
        )
        return {
            "metrics": {
                "experience_yield": rapor.experience_yield,
                "novelty_yield": rapor.novelty_yield,
                "useful_experience_yield": rapor.useful_experience_yield,
                "generalization_yield": rapor.generalization_yield,
                "verified_information_density": rapor.verified_information_density,
            },
            "report": rapor.to_dict(),
        }

    return run_seed_sweep(
        _callback,
        seeds=seeds,
        root=root,
        config={
            "protocol": "experience-yield-decomposition-v1",
            "cycles": cycles,
            "batch_size": batch_size,
            "operands_max": operands_max,
        },
        dataset_hash="synthetic-arithmetic-domain",
    )


__all__ = [
    "HoldoutDecision",
    "YieldReport",
    "run_yield_experiment",
    "run_yield_seed_sweep",
]
