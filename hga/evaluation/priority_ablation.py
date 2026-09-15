"""Priority(E) ağırlıklarının **nedensel zincirini** ölçen kontrollü ablasyon.

Neden bu modül var
------------------
``hga/experience/exploration.py`` içindeki :func:`agirlik_ablasyonu` yalnız tek
bir soruyu sorar: "bu terimi kapatınca ilk-K seçimi değişti mi?". Bu, bir
ağırlığın *ayarlanabilir* olmasının *etkili* olmakla aynı şey olmadığını
gösterir ama yeterli değildir: seçim değişse bile **kararın sonucu**
değişmeyebilir. O zaman ağırlık istatistiksel bir süs olur.

Bu modül zincirin tamamını ölçer::

    ağırlık değişti
          ↓  (score_changed_ratio, mean_abs_score_delta)
    aday skoru değişti
          ↓  (kendall_tau, spearman_rho, mean_rank_displacement)
    sıralama değişti
          ↓  (topk_overlap, topk_jaccard, selection_changed)
    seçilen deneyim değişti
          ↓  (verification_yield, novel_knowledge_yield, diversity)
    downstream sonuç değişti

Her halka ayrı raporlanır. Bir halkada zincir kopuyorsa bu bir başarısızlık
değil **bulgudur** ve ``chain`` sözlüğünde açıkça yazılır: örneğin skor
değişip sıralama değişmiyorsa terim monotonik bir sabit kaydırma yapıyordur.

Downstream neden gerçek?
------------------------
Adaylar aritmetik mini-environment üzerinden üretilir; bir adayın doğru olup
olmadığına **Priority'den tamamen bağımsız** deterministik bir doğrulayıcı
(``AritmetikOrtam``) karar verir. Yani "seçilen deneyim daha mı işe yaradı?"
sorusunun cevabı puanlama sisteminin kendi kendini onaylamasıyla değil, dış
bir hakemle verilir.

Sınırlar (dürüstlük)
--------------------
* Domain aritmetiktir; doğal dil için genellenemez.
* Downstream ölçüt "seçilen K adayın doğrulanabilir ve yeni olma oranı"dır;
  bu bir eğitim kazancı değil, aktif öğrenme örneklem kalitesidir.
* Ölçüm tek bir sabit aday havuzunda yapılır; havuz kompozisyonu değişirse
  hangi terimin etkili olduğu da değişebilir — bu yüzden rapor havuz
  parametrelerini ve veri imzasını taşır.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..experience.exploration import (
    VARSAYILAN_PRIORITY_AGIRLIKLARI,
    ExplorationEngine,
)
from ..experience.mini_env import AritmetikOrtam
from ..knowledge import (
    DeneyimDurumu,
    ExperienceCandidate,
    KaynakTuru,
    KnowledgeStore,
)

TERIMLER = ("w_gain", "w_novelty", "w_uncertainty", "w_conflict_penalty")
PROTOCOL = "priority_causal_chain_ablation_v1"

#: Her terimin TASARIM HEDEFİ olan downstream metrik ve yönü (+1: yüksek iyi,
#: -1: düşük iyi). "Zarar" kapısı tek bir metriğe göre kesilemez: Priority(E)
#: çok amaçlıdır ve keşif terimini (novelty) doğrulama verimiyle yargılamak
#: kategori hatasıdır — novelty'yi kapatmak onay oranını YÜKSELTİR çünkü
#: sistem artık yeni bilgi aramaz; bu zarar değil, keşif/sömürü takasının ta
#: kendisidir. Zarar, terimin KENDİ hedef metriğini kötüleştirmesidir.
HEDEF_METRIK: Dict[str, Tuple[str, int]] = {
    "w_gain": ("novel_knowledge_yield", +1),          # ayırt edici → yeni bilgi
    "w_novelty": ("novel_knowledge_yield", +1),       # görülmemiş → yeni bilgi
    "w_uncertainty": ("verification_yield", +1),      # zayıf kanıt → doğrulat
    "w_conflict_penalty": ("false_selection_rate", -1),  # çelişki → yanlıştan kaç
}

#: Havuz üretiminin varsayılan kompozisyon parametreleri — TEK KAYNAK.
#: `havuz_uret` imzasındaki varsayılanlar da buradan okunur; veri imzası bu
#: değerleri bağlar ki farklı havuz kompozisyonları aynı kimlikle raporlanamasın.
VARSAYILAN_HAVUZ: Dict[str, float] = {
    "known_fact_ratio": 0.88,
    "wrong_ratio": 0.45,
    "conflict_ratio": 0.15,
}


# ── yardımcı istatistikler (scipy yok; hepsi açıkça yazılı) ─────────────────
def _siralama(degerler: Sequence[float]) -> List[float]:
    """Beraberliklerde ortalama rank veren sıralama (1 = en yüksek değer)."""
    indeksli = sorted(range(len(degerler)), key=lambda i: -degerler[i])
    ranklar = [0.0] * len(degerler)
    i = 0
    while i < len(indeksli):
        j = i
        while (j + 1 < len(indeksli)
               and degerler[indeksli[j + 1]] == degerler[indeksli[i]]):
            j += 1
        ortalama = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranklar[indeksli[k]] = ortalama
        i = j + 1
    return ranklar


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    """Tau-b (beraberlik düzeltmeli). Boş/tek elemanlı girdi → 1.0."""
    n = len(a)
    if n != len(b):
        raise ValueError("kendall_tau: diziler aynı uzunlukta olmalı")
    if n < 2:
        return 1.0
    uyumlu = uyumsuz = 0
    bag_a = bag_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            carpim = da * db
            if da == 0 and db == 0:
                bag_a += 1
                bag_b += 1
            elif da == 0:
                bag_a += 1
            elif db == 0:
                bag_b += 1
            elif carpim > 0:
                uyumlu += 1
            else:
                uyumsuz += 1
    toplam = n * (n - 1) / 2
    payda = ((toplam - bag_a) * (toplam - bag_b)) ** 0.5
    if payda == 0:
        return 1.0
    return float(round((uyumlu - uyumsuz) / payda, 8))


def spearman_rho(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank korelasyonu (Pearson-on-ranks; beraberlikleri doğru ele alır)."""
    n = len(a)
    if n != len(b):
        raise ValueError("spearman_rho: diziler aynı uzunlukta olmalı")
    if n < 2:
        return 1.0
    ra, rb = _siralama(a), _siralama(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    pay = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    na = sum((x - ma) ** 2 for x in ra) ** 0.5
    nb = sum((y - mb) ** 2 for y in rb) ** 0.5
    if na == 0 or nb == 0:
        return 1.0
    return float(round(pay / (na * nb), 8))


def _oran(pay: int, payda: int) -> float:
    return round(pay / payda, 8) if payda else 0.0


def _entropi(sayimlar: Sequence[int]) -> float:
    toplam = sum(sayimlar)
    if toplam <= 0:
        return 0.0
    deger = 0.0
    for c in sayimlar:
        if c <= 0:
            continue
        p = c / toplam
        deger -= p * math.log(p, 2)
    return round(deger, 8)


# ── aday havuzu ─────────────────────────────────────────────────────────────
@dataclass
class PriorityPool:
    """Aritmetik domainde, dış doğrulayıcısı olan aday havuzu."""

    store: KnowledgeStore
    candidates: List[ExperienceCandidate]
    truth: Dict[str, Optional[bool]]        # experience_id → verifier kararı
    already_known: Dict[str, bool]          # experience_id → depoda kayıtlı mı
    subject_of: Dict[str, str]              # experience_id → özne token

    def ozet(self) -> Dict[str, Any]:
        return {
            "candidates": len(self.candidates),
            "verifier_true": sum(1 for v in self.truth.values() if v is True),
            "verifier_false": sum(1 for v in self.truth.values() if v is False),
            "verifier_undecidable": sum(1 for v in self.truth.values() if v is None),
            "already_known": sum(1 for v in self.already_known.values() if v),
        }


def havuz_uret(
    candidate_count: int = 120,
    operands_max: int = 12,
    seed: int = 1,
    known_fact_ratio: float = VARSAYILAN_HAVUZ["known_fact_ratio"],
    wrong_ratio: float = VARSAYILAN_HAVUZ["wrong_ratio"],
    conflict_ratio: float = VARSAYILAN_HAVUZ["conflict_ratio"],
) -> PriorityPool:
    """Aritmetik aday havuzu üret: doğru/yanlış/çelişkili karışık.

    Havuz, Priority(E) tasarımının varsaydığı rejimi modeller: **yanlış
    kalibre edilmiş, dış doğrulamaya muhtaç bir öğrenici**. Deneyim döngüsünde
    Priority'nin işi tam bu rejimde aday seçmektir — kendi güven puanına
    körü körüne güvenilemeyen MODEL_GENERATED kanıtlar arasından, doğrulayıcıya
    göndermeye değer olanları ayıklamak. Bu yüzden:

    * kayıtlı üçlülerin kanıt SAYISI değişir (novelty = 1/(1+n) ayrışır),
    * kanıt CONFIDENCE'ı doğrulukla TERS ilişkilidir: yanlış iddialar yüksek,
      doğru iddialar düşük güvenle kayıtlıdır (yanlış kalibrasyon; uncertainty
      teriminin nedensel rolü tam burada doğar),
    * CONFLICT bayrağı önce YANLIŞ iddialara düşer (çelişki tespiti gerçek
      sistemlerde yanlışlıkla koreledir; ceza teriminin rolü budur),
    * nesne varlıkları sürekli + benzersiz özellik boyutları taşır
      (information_gain [0,1] içinde DERECELİ ayrışır).

    ÖLÇÜM DERSİ (varsayılan known_fact_ratio neden yüksek): motorda hiç
    kaydı olmayan bir aday novelty=1.0 VE uncertainty=1.0 köşesine oturur.
    Kayıtsız adaylar çoğunluktaysa ilk-K tamamen bu köşeden seçilir; köşede
    her terim sabit olduğu için w_novelty/w_uncertainty'yi sıfırlamak bütün
    ilk-K skorlarını AYNI sabitle kaydırır ve seçim tanım gereği değişemez.
    Bu, terimlerin işlevsizliği değil ölçüm havuzunun körlüğüdür; eski
    varsayılan (0.30) tam bu körlüğü üretiyordu. Kayıtlı oran ve çelişki
    sayısı binom zarıyla değil TAM sayıyla uygulanır ki hiçbir tohum havuzu
    yeniden köşe rejimine düşürmesin.
    """
    if candidate_count < 4:
        raise ValueError("candidate_count >= 4 olmalı")
    if operands_max < 2:
        raise ValueError("operands_max >= 2 olmalı")
    for ad, deger in (("known_fact_ratio", known_fact_ratio),
                      ("wrong_ratio", wrong_ratio),
                      ("conflict_ratio", conflict_ratio)):
        if not 0.0 <= deger <= 1.0:
            raise ValueError(f"{ad} 0..1 aralığında olmalı")

    rng = random.Random(seed)
    store = KnowledgeStore()
    store.iliski_tanimla("eşittir", relation_id="R_EQUALS",
                         subject_types=["ifade"], object_types=["sayi"])

    tavan = 2 * operands_max
    sonuc_idleri: Dict[int, str] = {}
    for deger in range(tavan + 1):
        entity_id = f"E_SONUC_{deger:03d}"
        store.varlik_ekle(str(deger), entity_type="sayi", entity_id=entity_id)
        # Sayısal özellikler: information_gain'in ayrışması için gereklidir.
        # Yalnız ikili (0/1) özellikler gain'i {0,1}'e çökertti (her varlığın
        # aynı imzalı bir "ikizi" vardı → benzerlik hep 1.0 ya da 0.0); sürekli
        # özellikler kosinüs benzerliğini ve dolayısıyla gain'i DERECELİ yapar.
        store.ozellik_koy(entity_id, "cift", 1.0 if deger % 2 == 0 else 0.0,
                          source=KaynakTuru.VERIFIED_RULE, confidence=1.0)
        store.ozellik_koy(entity_id, "buyuk", 1.0 if deger > operands_max else 0.0,
                          source=KaynakTuru.VERIFIED_RULE, confidence=1.0)
        store.ozellik_koy(entity_id, "ucekatli", 1.0 if deger % 3 == 0 else 0.0,
                          source=KaynakTuru.VERIFIED_RULE, confidence=1.0)
        store.ozellik_koy(entity_id, "buyukluk",
                          round(deger / tavan, 6) if tavan else 0.0,
                          source=KaynakTuru.VERIFIED_RULE, confidence=1.0)
        store.ozellik_koy(entity_id, "bolen_sayisi",
                          round(sum(1 for b in range(1, deger + 1)
                                    if deger % b == 0) / max(1, tavan), 6),
                          source=KaynakTuru.VERIFIED_RULE, confidence=1.0)
        # Her değerin BENZERSİZ bir boyutu vardır: yalnız paylaşılan
        # özelliklerle her sayının kosinüs-ikizi bulunur ve gain herkes için
        # ~0'a çöker; benzersiz boyut en-yakın-komşu benzerliğini dereceli
        # yapar ve gain [~0.1, ~0.6] bandına açılır.
        store.ozellik_koy(entity_id, f"deger_{deger:03d}", 1.0,
                          source=KaynakTuru.VERIFIED_RULE, confidence=1.0)
        sonuc_idleri[deger] = entity_id

    ortam = AritmetikOrtam()
    adaylar: List[ExperienceCandidate] = []
    truth: Dict[str, Optional[bool]] = {}
    known: Dict[str, bool] = {}
    subject_of: Dict[str, str] = {}

    kayitli_sayisi = int(round(candidate_count * known_fact_ratio))
    kayitli_indeksler = set(rng.sample(range(candidate_count), kayitli_sayisi))
    aday_bilgisi: List[Tuple[ExperienceCandidate, bool, bool]] = []

    for index in range(candidate_count):
        sol = rng.randint(0, operands_max)
        sag = rng.randint(0, operands_max)
        ifade = f"{sol}+{sag}"
        # DÜZELTME: varlik_ekle token bazında idempotenttir — aynı ifade
        # ikinci kez üretildiğinde istenen entity_id OLUŞMAZ, mevcut varlık
        # döner. Eski kod istenen id'yi köre kullanıyordu; kopya ifadeler
        # var olmayan id'lere işaret ediyor, aday çözülemiyor (gain/novelty/
        # uncertainty zorla 1.0) ve doğrulayıcı None veriyordu. Bu, ablasyonu
        # dev bir tavan-beraberliğine çeviren ölçüm hatasıydı.
        ifade_id = store.varlik_ekle(
            ifade, entity_type="ifade",
            entity_id=f"E_IFADE_{index:04d}").entity_id
        dogru = sol + sag
        if rng.random() < wrong_ratio:
            sapma = rng.choice([-2, -1, 1, 2])
            hedef = max(0, min(tavan, dogru + sapma))
        else:
            hedef = dogru
        nesne_id = sonuc_idleri[hedef]
        iddia_dogru = (hedef == dogru)

        aday = ExperienceCandidate(f"PE_{index:04d}", ifade_id, "R_EQUALS", nesne_id)
        if index in kayitli_indeksler:
            # Kanıt SAYISI (novelty = 1/(1+n)) kanıt GÜVENİ'nden (uncertainty
            # = 1-confidence) bağımsız zarla değişir; iki terim aynı değişkenin
            # iki adı olsaydı ablasyon onları ayırt edemezdi. Confidence ise
            # YANLIŞ KALİBREDİR: yanlış iddia yüksek, doğru iddia düşük güven
            # taşır — belirsizliği önceleyen w_uncertainty'nin doğrulama
            # verimine nedensel katkısı ancak bu rejimde ölçülebilir.
            for _ in range(rng.randint(1, 3)):
                if iddia_dogru:
                    conf = rng.uniform(0.10, 0.55)
                else:
                    conf = rng.uniform(0.55, 0.99)
                store.olgu_kaydet(ifade_id, "R_EQUALS", nesne_id, score=1.0,
                                  source=KaynakTuru.MODEL_GENERATED,
                                  confidence=round(conf, 4))

        adaylar.append(aday)
        truth[aday.experience_id] = ortam.aday_dogrula(store, aday)
        subject_of[aday.experience_id] = ifade
        aday_bilgisi.append((aday, iddia_dogru, index in kayitli_indeksler))

    # CONFLICT bayrağı önce yanlış-kayıtsız, sonra yanlış-kayıtlı, en son
    # doğru iddialara düşer: çelişki tespiti yanlışlıkla koreledir. Sayı
    # TAM uygulanır (binom zarı değil) ki ceza teriminin ölçümü tohumdan
    # tohuma boş kümeye düşmesin.
    celiski_sayisi = int(round(candidate_count * conflict_ratio))
    yanlis_kayitsiz = [a for a, g, kayit in aday_bilgisi if not g and not kayit]
    yanlis_kayitli = [a for a, g, kayit in aday_bilgisi if not g and kayit]
    dogrular = [a for a, g, _ in aday_bilgisi if g]
    rng.shuffle(yanlis_kayitsiz)
    rng.shuffle(yanlis_kayitli)
    rng.shuffle(dogrular)
    for aday in (yanlis_kayitsiz + yanlis_kayitli + dogrular)[:celiski_sayisi]:
        aday.state = DeneyimDurumu.CONFLICT

    # already_known depo durumundan okunur, yerel zar atışından değil: kopya
    # ifadeler aynı üçlüyü paylaşabilir; bir kopyanın kaydettiği kanıt diğer
    # kopyanın adayını da "kayıtlı" yapar. Ölçüm anındaki gerçek durum budur.
    for aday in adaylar:
        known[aday.experience_id] = bool(store.relations.olgular(
            aday.subject_id, aday.relation_id, aday.object_id))

    return PriorityPool(store=store, candidates=adaylar, truth=truth,
                        already_known=known, subject_of=subject_of)


# ── ölçüm ───────────────────────────────────────────────────────────────────
def _downstream(pool: PriorityPool, secim: Sequence[str]) -> Dict[str, float]:
    """Seçilen K adayın dış doğrulayıcı karşısındaki verimi."""
    if not secim:
        return {"verification_yield": 0.0, "novel_knowledge_yield": 0.0,
                "decidability": 0.0, "false_selection_rate": 0.0,
                "subject_diversity": 0.0, "subject_entropy": 0.0}
    kararlar = [pool.truth[e] for e in secim]
    dogru = sum(1 for k in kararlar if k is True)
    yanlis = sum(1 for k in kararlar if k is False)
    kararli = sum(1 for k in kararlar if k is not None)
    yeni_dogru = sum(1 for e in secim
                     if pool.truth[e] is True and not pool.already_known[e])
    ozneler: Dict[str, int] = {}
    for e in secim:
        ozneler[pool.subject_of[e]] = ozneler.get(pool.subject_of[e], 0) + 1
    return {
        "verification_yield": _oran(dogru, len(secim)),
        "novel_knowledge_yield": _oran(yeni_dogru, len(secim)),
        "decidability": _oran(kararli, len(secim)),
        "false_selection_rate": _oran(yanlis, len(secim)),
        "subject_diversity": _oran(len(ozneler), len(secim)),
        "subject_entropy": _entropi(list(ozneler.values())),
    }


def _skorla_hepsi(motor: ExplorationEngine, pool: PriorityPool) -> List[float]:
    return [motor.bilgi_kazanci_skoru(pool.store, a) for a in pool.candidates]


def _ham_skorla_hepsi(motor: ExplorationEngine, pool: PriorityPool) -> List[float]:
    """Kırpılmamış (clip öncesi) Priority — clip, etkiyi gizleyebilir."""
    return [float(motor.priority_dokumu(pool.store, a)["raw_priority"])
            for a in pool.candidates]


def _secim(pool: PriorityPool, skorlar: Sequence[float], k: int) -> List[str]:
    """Deterministik ilk-K: skor azalan, beraberlikte experience_id artan."""
    sirali = sorted(
        zip(pool.candidates, skorlar),
        key=lambda item: (-item[1], item[0].experience_id),
    )
    return [aday.experience_id for aday, _ in sirali[:k]]


@dataclass
class VariantResult:
    """Tek bir ağırlık varyantının zincir boyunca ölçümü."""

    variant: str
    weights: Dict[str, float]
    # 1. halka: skor
    score_changed_ratio: float
    mean_abs_score_delta: float
    max_abs_score_delta: float
    # 2. halka: sıralama
    kendall_tau: float
    spearman_rho: float
    mean_rank_displacement: float
    max_rank_displacement: float
    # 3. halka: seçim
    topk_overlap: float
    topk_jaccard: float
    selection_changed: int
    # 4. halka: downstream
    downstream: Dict[str, float]
    downstream_delta: Dict[str, float]
    chain: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PriorityAblationReport:
    protocol: str
    seeds: List[int]
    k: int
    pool_summary: Dict[str, Any]
    dataset_hash: str
    baseline_weights: Dict[str, Any]
    baseline_downstream: Dict[str, float]
    variants: Dict[str, Dict[str, Any]]
    checks: Dict[str, bool]
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        return priority_ablation_markdown(self)


def _ortalama(degerler: Sequence[float]) -> float:
    return round(statistics.fmean(degerler), 8) if degerler else 0.0


def _std(degerler: Sequence[float]) -> float:
    return round(statistics.stdev(degerler), 8) if len(degerler) > 1 else 0.0


def run_priority_weight_ablation(
    k: int = 10,
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    candidate_count: int = 120,
    operands_max: int = 12,
) -> PriorityAblationReport:
    """Her Priority(E) terimi için tam nedensel zincir ablasyonu koş.

    Baseline dışında her varyantta tek bir ağırlık sıfırlanır; diğer her şey
    (havuz, tohum, K, sıralama kuralı) sabittir.
    """
    k = int(k)
    if k < 1:
        raise ValueError("k >= 1 olmalı")
    seeds = [int(s) for s in seeds]
    if not seeds:
        raise ValueError("en az bir tohum gerekli")

    per_variant: Dict[str, List[VariantResult]] = {t: [] for t in TERIMLER}
    baseline_downstreams: List[Dict[str, float]] = []
    havuz_ozetleri: List[Dict[str, Any]] = []

    for seed in seeds:
        pool = havuz_uret(candidate_count=candidate_count,
                          operands_max=operands_max, seed=seed)
        havuz_ozetleri.append(pool.ozet())
        temel = ExplorationEngine()
        temel_skor = _ham_skorla_hepsi(temel, pool)
        temel_secim = _secim(pool, temel_skor, k)
        temel_down = _downstream(pool, temel_secim)
        baseline_downstreams.append(temel_down)
        temel_rank = _siralama(temel_skor)

        for terim in TERIMLER:
            kwargs: Dict[str, Any] = {terim: 0.0}
            motor = ExplorationEngine(**kwargs)
            skor = _ham_skorla_hepsi(motor, pool)
            secim = _secim(pool, skor, k)
            down = _downstream(pool, secim)
            rank = _siralama(skor)

            deltalar = [abs(a - b) for a, b in zip(skor, temel_skor)]
            degisen = sum(1 for d in deltalar if d > 1e-9)
            yer_degistirme = [abs(a - b) for a, b in zip(rank, temel_rank)]
            ortak = len(set(secim) & set(temel_secim))
            birlesim = len(set(secim) | set(temel_secim))

            down_delta = {ad: round(down[ad] - temel_down[ad], 8) for ad in down}
            zincir = {
                "score_changed": degisen > 0,
                "ranking_changed": kendall_tau(skor, temel_skor) < 1.0 - 1e-9,
                "selection_changed": ortak < len(temel_secim),
                "downstream_changed": any(abs(v) > 1e-9 for v in down_delta.values()),
            }
            zincir["chain_complete"] = all(zincir.values())

            per_variant[terim].append(VariantResult(
                variant=terim,
                weights={ad: float(getattr(motor, ad)) for ad in TERIMLER},
                score_changed_ratio=_oran(degisen, len(deltalar)),
                mean_abs_score_delta=_ortalama(deltalar),
                max_abs_score_delta=round(max(deltalar), 8) if deltalar else 0.0,
                kendall_tau=kendall_tau(skor, temel_skor),
                spearman_rho=spearman_rho(skor, temel_skor),
                mean_rank_displacement=_ortalama(yer_degistirme),
                max_rank_displacement=round(max(yer_degistirme), 8) if yer_degistirme else 0.0,
                topk_overlap=_oran(ortak, len(temel_secim)),
                topk_jaccard=_oran(ortak, birlesim),
                selection_changed=len(temel_secim) - ortak,
                downstream=down,
                downstream_delta=down_delta,
                chain=zincir,
            ))

    # ── tohumlar arası özet ────────────────────────────────────────────────
    sayisal_alanlar = ("score_changed_ratio", "mean_abs_score_delta",
                       "max_abs_score_delta", "kendall_tau", "spearman_rho",
                       "mean_rank_displacement", "max_rank_displacement",
                       "topk_overlap", "topk_jaccard", "selection_changed")
    varyantlar: Dict[str, Dict[str, Any]] = {}
    for terim, kosular in per_variant.items():
        ozet: Dict[str, Any] = {"weights": kosular[0].weights, "per_seed": []}
        for ad in sayisal_alanlar:
            degerler = [float(getattr(r, ad)) for r in kosular]
            ozet[f"{ad}_mean"] = _ortalama(degerler)
            ozet[f"{ad}_std"] = _std(degerler)
        for metrik in kosular[0].downstream:
            degerler = [r.downstream[metrik] for r in kosular]
            deltalar = [r.downstream_delta[metrik] for r in kosular]
            ozet[f"downstream_{metrik}_mean"] = _ortalama(degerler)
            ozet[f"downstream_{metrik}_delta_mean"] = _ortalama(deltalar)
        for halka in ("score_changed", "ranking_changed", "selection_changed",
                      "downstream_changed", "chain_complete"):
            ozet[f"{halka}_seed_ratio"] = _oran(
                sum(1 for r in kosular if r.chain[halka]), len(kosular))
        # Bir halka TÜM tohumlarda değişiyorsa o halka "sağlam" sayılır.
        ozet["chain"] = {
            halka: ozet[f"{halka}_seed_ratio"] >= 1.0
            for halka in ("score_changed", "ranking_changed",
                          "selection_changed", "downstream_changed")
        }
        ozet["chain"]["chain_complete"] = all(ozet["chain"].values())
        ozet["effective_on_selection"] = ozet["topk_overlap_mean"] < 1.0
        ozet["per_seed"] = [r.to_dict() for r in kosular]
        varyantlar[terim] = ozet

    temel_ozet = {
        ad: _ortalama([d[ad] for d in baseline_downstreams])
        for ad in baseline_downstreams[0]
    }
    # İmza havuzun ÜRETİM PARAMETRELERİNİ de bağlar: kayıt oranı / yanlışlık
    # oranı / çelişki oranı havuz kompozisyonunu (dolayısıyla her sonucu)
    # belirler; bunlar imza dışında kalırsa iki farklı havuz aynı kimlikle
    # raporlanabilir.
    imza = hashlib.sha256(json.dumps({
        "protocol": PROTOCOL, "seeds": seeds, "k": k,
        "candidate_count": candidate_count, "operands_max": operands_max,
        "pool": {"known_fact_ratio": VARSAYILAN_HAVUZ["known_fact_ratio"],
                 "wrong_ratio": VARSAYILAN_HAVUZ["wrong_ratio"],
                 "conflict_ratio": VARSAYILAN_HAVUZ["conflict_ratio"]},
    }, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    tam_zincir = [t for t in TERIMLER if varyantlar[t]["chain"]["chain_complete"]]
    kopuk = {t: [h for h, v in varyantlar[t]["chain"].items()
                 if h != "chain_complete" and not v]
             for t in TERIMLER if not varyantlar[t]["chain"]["chain_complete"]}

    kontroller = {
        # En az bir terim zinciri uçtan uca taşıyor mu? Hiçbiri taşımıyorsa
        # Priority(E) downstream üzerinde ölçülebilir bir etkiye sahip değildir.
        "at_least_one_full_causal_chain": bool(tam_zincir),
        # Her terim en azından skoru değiştiriyor mu (ölü kod kontrolü)?
        "all_terms_move_scores": all(
            varyantlar[t]["chain"]["score_changed"] for t in TERIMLER),
        # Her terim sıralamayı değiştiriyor mu?
        "all_terms_move_ranking": all(
            varyantlar[t]["chain"]["ranking_changed"] for t in TERIMLER),
        # Her terim seçimi değiştiriyor mu?
        "all_terms_move_selection": all(
            varyantlar[t]["chain"]["selection_changed"] for t in TERIMLER),
        # Her terim downstream sonucu değiştiriyor mu?
        "all_terms_move_downstream": all(
            varyantlar[t]["chain"]["downstream_changed"] for t in TERIMLER),
    }

    bulgular = [
        f"{len(seeds)} tohum × {candidate_count} aday; ilk-{k} seçimi ölçüldü.",
        f"Baseline downstream doğrulama verimi "
        f"{temel_ozet['verification_yield']:.4f}, yeni bilgi verimi "
        f"{temel_ozet['novel_knowledge_yield']:.4f}.",
    ]
    if tam_zincir:
        bulgular.append(
            f"Zinciri uçtan uca (skor→sıralama→seçim→downstream) taşıyan "
            f"terimler: {tam_zincir}.")
    else:
        bulgular.append(
            "UYARI: hiçbir terim zinciri uçtan uca taşımıyor — Priority(E) "
            "ayarlanabilir ama downstream sonuç üzerinde ölçülebilir etkisi yok.")
    for terim, eksikler in kopuk.items():
        bulgular.append(
            f"`{terim}` zinciri şu halkada kopuyor: {eksikler}. Bu bir hata "
            f"değil bulgudur: terim bu havuzda o halkanın ötesine geçmiyor.")

    # Etkinin YÖNÜ de raporlanmalı — ama her terim KENDİ tasarım hedefine
    # göre: terimi kapatmak kendi hedef metriğini İYİLEŞTİRİYORSA terim o
    # havuzda zararlıdır. Tek ortak metrikle yargılamak (eski kural: herkes
    # verification_yield'e göre) keşif terimlerini yapısal olarak cezalandıran
    # bir kategori hatasıydı; HEDEF_METRIK tablosu bunun düzeltmesidir.
    zararli = []
    faydali = []
    for t in TERIMLER:
        metrik, yon = HEDEF_METRIK[t]
        delta = varyantlar[t][f"downstream_{metrik}_delta_mean"] * yon
        if delta > 1e-9:
            zararli.append((t, metrik, delta))
        elif delta < -1e-9:
            faydali.append((t, metrik, delta))
    if zararli:
        bulgular.append(
            f"YÖN UYARISI: {[(t, m_) for t, m_, _ in zararli]} — terimi "
            f"KAPATMAK kendi hedef metriğini İYİLEŞTİRİYOR; bu havuzda "
            f"katkısı negatif. 'Etkili olmak' 'faydalı olmak' değildir; "
            f"ağırlıklar bu bulgu incelenmeden savunulamaz.")
    if faydali:
        bulgular.append(
            f"Kendi hedef metriğinde pozitif katkı taşıyan terimler: "
            f"{[(t, m_) for t, m_, _ in faydali]} — terimi kapatmak hedef "
            f"metriğini kötüleştiriyor.")
    verim_artislari = [t for t in TERIMLER
                       if varyantlar[t]["downstream_verification_yield_delta_mean"]
                       > 1e-9]
    if verim_artislari:
        bulgular.append(
            f"NOT: {verim_artislari} kapatılınca doğrulama verimi yükseliyor; "
            f"bu keşif/sömürü takasıdır (yeni bilgi verimi aynı anda düşer), "
            f"tek metrikli 'zarar' kanıtı değildir.")
    kontroller["no_term_harms_downstream_yield"] = not zararli

    sinirlar = [
        "Domain aritmetik mini-environment'tır; doğal dil aday havuzlarına "
        "genellenemez.",
        "Downstream ölçütü seçilen K adayın dış doğrulayıcı karşısındaki "
        "verimidir; model eğitim kazancı DEĞİLDİR.",
        "Priority skorları kırpılmadan (raw_priority) karşılaştırılır; CLI'nin "
        "gösterdiği kırpılmış skor bazı etkileri gizleyebilir.",
        "Beraberlikler experience_id'ye göre deterministik kırılır; farklı bir "
        "kırma kuralı ilk-K örtüşmesini değiştirebilir.",
    ]

    return PriorityAblationReport(
        protocol=PROTOCOL,
        seeds=seeds,
        k=k,
        pool_summary={
            "candidate_count": candidate_count,
            "operands_max": operands_max,
            "per_seed": havuz_ozetleri,
        },
        dataset_hash=imza,
        baseline_weights=dict(VARSAYILAN_PRIORITY_AGIRLIKLARI),
        baseline_downstream=temel_ozet,
        variants=varyantlar,
        checks=kontroller,
        findings=bulgular,
        limitations=sinirlar,
    )


def priority_ablation_markdown(report: PriorityAblationReport) -> str:
    satirlar = [
        "# Priority(E) Nedensel Zincir Ablasyonu (P0-1)",
        "",
        f"Protokol: `{report.protocol}` · veri imzası: `{report.dataset_hash}` · "
        f"tohumlar: {report.seeds} · K={report.k}",
        "",
        "## Zincir halkaları",
        "",
        "| Kapatılan terim | Skor Δ (ort) | Kendall τ | Spearman ρ | "
        "Rank kayması | Top-K örtüşme | Downstream Δ verim | Zincir tam mı |",
        "|---|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    for terim in TERIMLER:
        v = report.variants[terim]
        satirlar.append(
            f"| `{terim}` | {v['mean_abs_score_delta_mean']:.4f} | "
            f"{v['kendall_tau_mean']:.4f} | {v['spearman_rho_mean']:.4f} | "
            f"{v['mean_rank_displacement_mean']:.3f} | "
            f"{v['topk_overlap_mean']:.3f} | "
            f"{v['downstream_verification_yield_delta_mean']:+.4f} | "
            f"{'EVET' if v['chain']['chain_complete'] else 'hayır'} |"
        )

    satirlar += ["", "## Downstream (seçilen K aday, dış doğrulayıcı)", "",
                 "| Kol | Doğrulama verimi | Yeni bilgi verimi | "
                 "Yanlış seçim oranı | Özne çeşitliliği |",
                 "|---|---:|---:|---:|---:|"]
    b = report.baseline_downstream
    satirlar.append(
        f"| baseline | {b['verification_yield']:.4f} | "
        f"{b['novel_knowledge_yield']:.4f} | {b['false_selection_rate']:.4f} | "
        f"{b['subject_diversity']:.4f} |")
    for terim in TERIMLER:
        v = report.variants[terim]
        satirlar.append(
            f"| {terim}=0 | {v['downstream_verification_yield_mean']:.4f} | "
            f"{v['downstream_novel_knowledge_yield_mean']:.4f} | "
            f"{v['downstream_false_selection_rate_mean']:.4f} | "
            f"{v['downstream_subject_diversity_mean']:.4f} |")

    satirlar += ["", "## Kabul kapıları", "", "| kapı | sonuç |", "|---|---|"]
    for ad, sonuc in report.checks.items():
        satirlar.append(f"| {ad} | {'GEÇTİ' if sonuc else 'KALDI'} |")
    satirlar += ["", "## Bulgular", ""]
    satirlar += [f"- {b}" for b in report.findings]
    satirlar += ["", "## Sınırlar", ""]
    satirlar += [f"- {s}" for s in report.limitations]
    return "\n".join(satirlar) + "\n"


__all__ = [
    "PROTOCOL",
    "TERIMLER",
    "PriorityPool",
    "PriorityAblationReport",
    "VariantResult",
    "havuz_uret",
    "kendall_tau",
    "spearman_rho",
    "run_priority_weight_ablation",
    "priority_ablation_markdown",
]
