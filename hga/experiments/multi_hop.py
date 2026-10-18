"""Çok adımlı çıkarım (multi-hop) ve uzun bağlam dayanıklılığı benchmarkı.

Bu protokol tek bir soruyu ölçer: **sistem, tek tek doğrulanmış olgulardan
zincirleme çıkarım yapabiliyor mu ve araya giren alakasız olgular biriktikçe
bu yeteneği koruyor mu?**

Neyi ölçtüğü
------------
``a --öncesi--> b --öncesi--> c`` zinciri kurulur ve ``a --öncesi--> c``
sorulur. Bu, bilgi deposunda **yazılı olmayan** ama geçişlilikle (transitivity)
çıkarılabilen bir üçlüdür. İki eksen bağımsız olarak taranır:

* **derinlik** (``hop``): zincirdeki kenar sayısı. 1 adım = doğrudan geri
  çağırma (çıkarım değil); 2+ adım = gerçek çok adımlı çıkarım.
* **bağlam yükü** (``distractor``): zincir kenarlarının ARASINA serpiştirilen,
  soruyla ilgisiz olgu sayısı. Uzun bağlam baskısı budur.

Neyi ÖLÇMEDİĞİ (dürüstlük sınırı)
---------------------------------
Bu bir dil modeli "uzun context window" testi DEĞİLDİR. Token penceresi,
attention span veya doğal metin anlama ölçülmez. Ölçülen şey, sembolik bilgi
deposu + seyrek bellek katmanının, araya giren yazmalar altında zincir
takibini sürdürebilmesidir. Zincir geçişli tek bir ilişki (``R_ONCESI``)
üzerinden kurulur; bu DAR ama gerçek bir çıkarımdır.

Ayrıca ``hop=1`` kasıtlı olarak bir **negatif kontroldür**: orada başarı
çıkarımı değil yalnızca geri çağırmayı gösterir. Çıkarım iddiası ancak
``hop>=2`` sütunlarında anlamlıdır.

Çürütülebilirlik
----------------
Protokol iki dejenere kolu da koşar (``always_yes`` / ``always_no``). Bunlar
zinciri hiç takip etmeden sabit cevap verir. Gerçek motor bu kollardan
istatistiksel olarak ayrışamıyorsa çok adımlı çıkarım iddiası DÜŞER; rapor bunu
``beats_degenerate`` kapısıyla açıkça bildirir.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from hga.knowledge import KaynakTuru, KnowledgeStore
from hga.memory.sparse_memory import DeneyimSlotlari

# Zincirin kurulduğu geçişli ilişki. "a, b'den öncedir" — eğer a<b ve b<c ise
# a<c. Geçişlilik burada AÇIKÇA varsayılan tek çıkarım kuralıdır.
RELATION_TOKEN = "öncesi"
RELATION_ID = "R_ONCESI"

DEFAULT_HOPS = (1, 2, 3, 4, 5)
DEFAULT_DISTRACTORS = (0, 16, 64, 256)


def _oran(pay: int, payda: int) -> float:
    return round(pay / payda, 8) if payda else 0.0


@dataclass
class MultiHopCase:
    """Tek bir zincir sorgusunun sonucu."""

    hop: int
    distractors: int
    seed: int
    chain: List[str]
    query: Tuple[str, str, str]
    expected: bool
    predicted: bool
    correct: bool
    steps_followed: int
    memory_recall_ok: bool

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["query"] = list(self.query)
        return d


@dataclass
class MultiHopCell:
    """(hop × distractor) ızgarasında tek bir hücrenin toplamı."""

    hop: int
    distractors: int
    total: int
    correct: int
    accuracy: float
    positive_total: int
    positive_correct: int
    positive_accuracy: float
    negative_total: int
    negative_correct: int
    negative_accuracy: float
    mean_steps_followed: float
    memory_recall_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DegenerateArm:
    """Zinciri hiç takip etmeyen sabit-cevap kolu."""

    arm: str
    accuracy: float
    positive_accuracy: float
    negative_accuracy: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MultiHopReport:
    protocol: str
    seeds: List[int]
    hops: List[int]
    distractor_levels: List[int]
    dataset_hash: str
    cells: List[MultiHopCell]
    degenerate_arms: List[DegenerateArm]
    overall_accuracy: float
    inference_accuracy: float
    recall_accuracy: float
    accuracy_at_min_context: float
    accuracy_at_max_context: float
    context_degradation: float
    deepest_reliable_hop: int
    checks: Dict[str, bool]
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    cases: List[MultiHopCase] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol,
            "seeds": list(self.seeds),
            "hops": list(self.hops),
            "distractor_levels": list(self.distractor_levels),
            "dataset_hash": self.dataset_hash,
            "cells": [c.to_dict() for c in self.cells],
            "degenerate_arms": [a.to_dict() for a in self.degenerate_arms],
            "overall_accuracy": self.overall_accuracy,
            "inference_accuracy": self.inference_accuracy,
            "recall_accuracy": self.recall_accuracy,
            "accuracy_at_min_context": self.accuracy_at_min_context,
            "accuracy_at_max_context": self.accuracy_at_max_context,
            "context_degradation": self.context_degradation,
            "deepest_reliable_hop": self.deepest_reliable_hop,
            "checks": dict(self.checks),
            "findings": list(self.findings),
            "limitations": list(self.limitations),
        }


class _ZincirDepo:
    """Bilgi deposu + seyrek bellek üzerine kurulmuş zincir takipçisi.

    Olgular hem ``KnowledgeStore``a (sembolik gerçek) hem ``DeneyimSlotlari``na
    (seyrek bellek) yazılır. Zincir takibi sembolik depodan yapılır; bellek
    katmanı ayrıca ölçülür ki "bellek çakışması yüzünden mi kaybettik, çıkarım
    yapamadığı için mi?" sorusu ayrılabilsin.
    """

    def __init__(self, slot_sayisi: int = 4096, tablo_sayisi: int = 2) -> None:
        self.store = KnowledgeStore()
        self.store.iliski_tanimla(RELATION_TOKEN, relation_id=RELATION_ID)
        self.memory = DeneyimSlotlari(slot_sayisi=slot_sayisi, tablo_sayisi=tablo_sayisi)
        self._kenarlar: Dict[str, set] = {}

    def varlik(self, token: str) -> str:
        entity = self.store.varlik_ekle(token, entity_type="kavram")
        return entity.entity_id

    def kenar_ekle(self, kaynak: str, hedef: str) -> None:
        self.store.olgu_kaydet(
            kaynak, RELATION_ID, hedef, score=1.0, source=KaynakTuru.VERIFIED_RULE
        )
        self._kenarlar.setdefault(kaynak, set()).add(hedef)
        self.memory.yaz(f"{kaynak}|{RELATION_ID}|{hedef}", (kaynak, RELATION_ID, hedef))

    def bellekte_var_mi(self, kaynak: str, hedef: str) -> bool:
        return self.memory.icerir(
            f"{kaynak}|{RELATION_ID}|{hedef}", (kaynak, RELATION_ID, hedef)
        )

    def zincir_takip(self, kaynak: str, hedef: str, limit: int) -> Tuple[bool, int]:
        """``kaynak``tan ``hedef``e giden yolu ara. (bulundu, atılan_adım).

        KRİTİK: her kenar, ilerlemeden önce seyrek BELLEKTEN doğrulanır. Aksi
        halde takip Python sözlüğü üzerinden yapılırdı ve bellek katmanı süs
        olurdu — doğruluk her koşulda 1.0 çıkar, metrik ölürdü. Bellek katmanı
        bir kenarı çakışma yüzünden kaybettiyse zincir orada KOPAR.
        """
        if kaynak == hedef:
            return True, 0
        sinir = {kaynak}
        gorulen = {kaynak}
        for adim in range(1, limit + 1):
            sonraki = set()
            for dugum in sinir:
                for komsu in self._kenarlar.get(dugum, ()):
                    # Kenar bellekte hayatta değilse bu yol izlenemez.
                    if not self.bellekte_var_mi(dugum, komsu):
                        continue
                    if komsu == hedef:
                        return True, adim
                    if komsu not in gorulen:
                        gorulen.add(komsu)
                        sonraki.add(komsu)
            if not sonraki:
                return False, adim
            sinir = sonraki
        return False, limit


def _vaka_uret(
    hop: int, distractors: int, seed: int, pozitif: bool
) -> Tuple[List[str], Tuple[str, str, str], bool]:
    """Zinciri ve sorguyu üret. Negatif vakada zincir KASITLI olarak kopuktur.

    Düğüm adları tohum/derinlik/bağlam üçlüsünden türetilir; protokol
    deterministiktir, rastgelelik kullanılmaz.
    """
    etiket = f"s{seed}h{hop}d{distractors}{'p' if pozitif else 'n'}"
    zincir = [f"dugum_{etiket}_{i}" for i in range(hop + 1)]
    return zincir, (zincir[0], RELATION_ID, zincir[-1]), pozitif


def run_multi_hop_benchmark(
    hops: Sequence[int] = DEFAULT_HOPS,
    distractor_levels: Sequence[int] = DEFAULT_DISTRACTORS,
    seeds: Sequence[int] = (1, 2, 3),
    slot_sayisi: int = 4096,
    tablo_sayisi: int = 2,
) -> MultiHopReport:
    """Çok adımlı çıkarımı derinlik × bağlam yükü ızgarasında ölç."""
    hops = [int(h) for h in hops]
    distractor_levels = [int(d) for d in distractor_levels]
    seeds = [int(s) for s in seeds]

    vakalar: List[MultiHopCase] = []

    for hop in hops:
        for distractors in distractor_levels:
            for seed in seeds:
                for pozitif in (True, False):
                    zincir, _sorgu, beklenen = _vaka_uret(
                        hop, distractors, seed, pozitif
                    )
                    depo = _ZincirDepo(slot_sayisi=slot_sayisi, tablo_sayisi=tablo_sayisi)
                    dugumler = [depo.varlik(t) for t in zincir]

                    # Negatif vakada zincirin SON kenarını kasıtlı olarak atla:
                    # yol kopuk olmalı, dolayısıyla doğru cevap "hayır".
                    kenarlar = list(zip(dugumler, dugumler[1:]))
                    if not beklenen and kenarlar:
                        kenarlar = kenarlar[:-1]

                    # Dolgu olguları zincir kenarlarının ARASINA serpiştir; böylece
                    # araya giren yazmalar zincirin ortasına denk gelir.
                    dolgu_id = 0
                    for idx, (kaynak, hedef) in enumerate(kenarlar):
                        depo.kenar_ekle(kaynak, hedef)
                        pay = distractors // max(1, len(kenarlar))
                        if idx == len(kenarlar) - 1:
                            pay = distractors - pay * (len(kenarlar) - 1)
                        for _ in range(max(0, pay)):
                            a = depo.varlik(f"dolgu_{seed}_{hop}_{dolgu_id}_a")
                            b = depo.varlik(f"dolgu_{seed}_{hop}_{dolgu_id}_b")
                            depo.kenar_ekle(a, b)
                            dolgu_id += 1

                    hedef_id = depo.varlik(zincir[-1])
                    kaynak_id = depo.varlik(zincir[0])
                    tahmin, adim = depo.zincir_takip(
                        kaynak_id, hedef_id, limit=max(hops) + 1
                    )
                    bellek_ok = all(
                        depo.bellekte_var_mi(k, h) for k, h in kenarlar
                    ) if kenarlar else True

                    vakalar.append(
                        MultiHopCase(
                            hop=hop,
                            distractors=distractors,
                            seed=seed,
                            chain=list(zincir),
                            query=(zincir[0], RELATION_ID, zincir[-1]),
                            expected=beklenen,
                            predicted=bool(tahmin),
                            correct=bool(tahmin) == beklenen,
                            steps_followed=int(adim),
                            memory_recall_ok=bool(bellek_ok),
                        )
                    )

    # ── Izgara toplamları ────────────────────────────────────────────────
    hucreler: List[MultiHopCell] = []
    for hop in hops:
        for distractors in distractor_levels:
            alt = [v for v in vakalar if v.hop == hop and v.distractors == distractors]
            poz = [v for v in alt if v.expected]
            neg = [v for v in alt if not v.expected]
            hucreler.append(
                MultiHopCell(
                    hop=hop,
                    distractors=distractors,
                    total=len(alt),
                    correct=sum(1 for v in alt if v.correct),
                    accuracy=_oran(sum(1 for v in alt if v.correct), len(alt)),
                    positive_total=len(poz),
                    positive_correct=sum(1 for v in poz if v.correct),
                    positive_accuracy=_oran(sum(1 for v in poz if v.correct), len(poz)),
                    negative_total=len(neg),
                    negative_correct=sum(1 for v in neg if v.correct),
                    negative_accuracy=_oran(sum(1 for v in neg if v.correct), len(neg)),
                    mean_steps_followed=round(
                        sum(v.steps_followed for v in alt) / len(alt), 4
                    ) if alt else 0.0,
                    memory_recall_rate=_oran(
                        sum(1 for v in alt if v.memory_recall_ok), len(alt)
                    ),
                )
            )

    # ── Dejenere kollar: zinciri hiç takip etmeden sabit cevap ───────────
    kollar: List[DegenerateArm] = []
    for ad, sabit in (("always_yes", True), ("always_no", False)):
        poz = [v for v in vakalar if v.expected]
        neg = [v for v in vakalar if not v.expected]
        kollar.append(
            DegenerateArm(
                arm=ad,
                accuracy=_oran(sum(1 for v in vakalar if sabit == v.expected), len(vakalar)),
                positive_accuracy=_oran(sum(1 for v in poz if sabit == v.expected), len(poz)),
                negative_accuracy=_oran(sum(1 for v in neg if sabit == v.expected), len(neg)),
            )
        )

    genel = _oran(sum(1 for v in vakalar if v.correct), len(vakalar))
    cikarim = [v for v in vakalar if v.hop >= 2]
    geri_cagirma = [v for v in vakalar if v.hop == 1]
    cikarim_acc = _oran(sum(1 for v in cikarim if v.correct), len(cikarim))
    geri_cagirma_acc = _oran(sum(1 for v in geri_cagirma if v.correct), len(geri_cagirma))

    min_ctx = min(distractor_levels)
    max_ctx = max(distractor_levels)
    min_vakalar = [v for v in vakalar if v.distractors == min_ctx]
    max_vakalar = [v for v in vakalar if v.distractors == max_ctx]
    acc_min = _oran(sum(1 for v in min_vakalar if v.correct), len(min_vakalar))
    acc_max = _oran(sum(1 for v in max_vakalar if v.correct), len(max_vakalar))

    # En derin güvenilir hop: TÜM bağlam seviyelerinde doğruluk 1.0 olan en büyük hop.
    en_derin = 0
    for hop in sorted(hops):
        hop_hucreleri = [c for c in hucreler if c.hop == hop]
        if hop_hucreleri and all(c.accuracy >= 1.0 for c in hop_hucreleri):
            en_derin = hop

    en_iyi_dejenere = max(k.accuracy for k in kollar)

    imza = hashlib.sha256(
        json.dumps(
            {"hops": hops, "distractors": distractor_levels, "seeds": seeds},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]

    kontroller = {
        # Çıkarım gerçekten oluyor mu (1 adımlık geri çağırmanın ötesinde)?
        "multi_hop_inference_works": cikarim_acc >= 0.99,
        # Dejenere sabit-cevap kollarını yeniyor mu? Yenmiyorsa iddia düşer.
        "beats_degenerate": genel > en_iyi_dejenere,
        # Negatifleri de doğru reddediyor mu (her şeye "evet" demiyor)?
        "rejects_broken_chains": all(c.negative_accuracy >= 0.99 for c in hucreler),
        # Uzun bağlam altında bozulma yok mu?
        "robust_to_long_context": acc_max >= acc_min,
        # Bellek katmanı zinciri koruyabildi mi?
        "memory_preserved_chain": all(c.memory_recall_rate >= 0.99 for c in hucreler),
    }

    bulgular = [
        f"Çok adımlı çıkarım (hop>=2) doğruluğu {cikarim_acc:.4f}; "
        f"tek adımlı geri çağırma {geri_cagirma_acc:.4f}.",
        f"Bağlam yükü {min_ctx} → {max_ctx} dolgu olguya çıkarıldığında doğruluk "
        f"{acc_min:.4f} → {acc_max:.4f} ({acc_max - acc_min:+.4f}).",
        f"Tüm bağlam seviyelerinde tam doğru kalan en derin zincir: {en_derin} adım.",
        f"En iyi dejenere kol {en_iyi_dejenere:.4f}; motor {genel:.4f} "
        f"({'ayrışıyor' if genel > en_iyi_dejenere else 'AYRIŞAMIYOR'}).",
    ]

    sinirlar = [
        "Bu bir dil modeli 'context window' testi DEĞİLDİR; token penceresi, "
        "attention span veya doğal metin anlama ölçülmez. Sembolik depo + seyrek "
        "bellek üzerinde zincir takibi ölçülür.",
        "Zincir tek bir geçişli ilişki (öncesi) üzerinden kurulur. Bu gerçek ama "
        "DAR bir çıkarımdır; çok ilişkili karma akıl yürütme kapsam dışıdır.",
        "hop=1 sütunu çıkarım değil geri çağırmadır ve negatif kontrol olarak "
        "okunmalıdır; çıkarım iddiası yalnız hop>=2 için geçerlidir.",
        "Dolgu olgular sentetiktir ve zincirle aynı ilişkiyi kullanır; doğal "
        "metindeki anlamsal karışıklığı temsil etmez.",
        "Zincir takibi genişlik-öncelikli aramadır; maliyeti kenar sayısıyla "
        "büyür. Bu bir öğrenilmiş yetenek değil, deterministik bir çıkarımdır.",
    ]

    return MultiHopReport(
        protocol="multi_hop_long_context_v1",
        seeds=seeds,
        hops=hops,
        distractor_levels=distractor_levels,
        dataset_hash=imza,
        cells=hucreler,
        degenerate_arms=kollar,
        overall_accuracy=genel,
        inference_accuracy=cikarim_acc,
        recall_accuracy=geri_cagirma_acc,
        accuracy_at_min_context=acc_min,
        accuracy_at_max_context=acc_max,
        context_degradation=round(acc_max - acc_min, 8),
        deepest_reliable_hop=en_derin,
        checks=kontroller,
        findings=bulgular,
        limitations=sinirlar,
        cases=vakalar,
    )


def multi_hop_markdown(report: MultiHopReport) -> str:
    """Raporu Markdown'a çevir."""
    satirlar = [
        "# Çok Adımlı Çıkarım ve Uzun Bağlam (P1-004 / P1-006)",
        "",
        f"Protokol: `{report.protocol}`  ·  veri imzası: `{report.dataset_hash}`",
        f"Tohumlar: {report.seeds}",
        "",
        "## Doğruluk ızgarası (satır = zincir derinliği, sütun = dolgu olgu sayısı)",
        "",
    ]
    baslik = "| hop \\ dolgu | " + " | ".join(
        str(d) for d in report.distractor_levels
    ) + " |"
    ayrac = "|---|" + "---:|" * len(report.distractor_levels)
    satirlar.extend([baslik, ayrac])
    for hop in report.hops:
        hucreler = {c.distractors: c for c in report.cells if c.hop == hop}
        etiket = f"{hop} adım" + (" (geri çağırma)" if hop == 1 else "")
        satir = f"| {etiket} | " + " | ".join(
            f"{hucreler[d].accuracy:.4f}" for d in report.distractor_levels
        ) + " |"
        satirlar.append(satir)

    satirlar.extend(["", "## Dejenere kontrol kolları", "",
                     "| kol | doğruluk | pozitif | negatif |",
                     "|---|---:|---:|---:|"])
    for kol in report.degenerate_arms:
        satirlar.append(
            f"| {kol.arm} | {kol.accuracy:.4f} | {kol.positive_accuracy:.4f} | "
            f"{kol.negative_accuracy:.4f} |"
        )

    satirlar.extend(["", "## Kabul kapıları", "", "| kapı | sonuç |", "|---|---|"])
    for ad, sonuc in report.checks.items():
        satirlar.append(f"| {ad} | {'GEÇTİ' if sonuc else 'KALDI'} |")

    satirlar.extend(["", "## Bulgular", ""])
    satirlar.extend(f"- {b}" for b in report.findings)
    satirlar.extend(["", "## Sınırlar", ""])
    satirlar.extend(f"- {s}" for s in report.limitations)
    return "\n".join(satirlar) + "\n"


__all__ = [
    "MultiHopCase",
    "MultiHopCell",
    "DegenerateArm",
    "MultiHopReport",
    "run_multi_hop_benchmark",
    "multi_hop_markdown",
    "RELATION_ID",
    "DEFAULT_HOPS",
    "DEFAULT_DISTRACTORS",
]
