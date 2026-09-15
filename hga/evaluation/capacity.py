# -*- coding: utf-8 -*-
"""
Kapasite Çerçevesi — P, C_I, C_M, C_E, C_V (Faz 13–14)
=======================================================

README bugüne kadar üç büyüklüğü ayırıyordu:

* ``P``    — fiziksel/eğitilebilir parametre (RAM'de gerçekten ayrılan).
* ``C_I``  — etkileşim kapasitesi (Kronecker zincirinin temsil ettiği operatör
  üst sınırı; **parametre değildir**).
* ``C_M``  — bellek adres kapasitesi (``sözlük^pencere``; fiziksel tablo değil).

Bu üçlü doğru ama eksiktir, çünkü **adreslenebilir olmak ile üretilebilir
olmak ve doğrulanabilir olmak aynı şey değildir**:

    C_M = adreslenebilir     (10^62 mertebesinde kavramsal anahtar uzayı)
    C_E = üretilebilir       (generator'ın kısıtlar altında ürettiği anlamlı
                              deneyim uzayı)
    C_V = doğrulanabilir     (bağımsız verifier'ın karar verebildiği alt küme)

Zorunlu sıralama::

    C_V  ≤  C_E  ≤  C_M

`C_E` ve `C_V` **ölçülen** büyüklüklerdir: bir bilgi tabanı ve doğrulayıcı
verildiğinde sayılır ya da örnekleme ile tahmin edilir. Teorik iddia değil,
deney çıktısıdır.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..knowledge.schemas import DeneyimDurumu, ExperienceCandidate, KaynakTuru

Verifier = Callable[[Any, ExperienceCandidate], Optional[bool]]


def _oran(pay: int, payda: int) -> float:
    return round(pay / payda, 8) if payda else 0.0


def _log10(deger: float) -> float:
    return round(math.log10(deger), 4) if deger > 0 else float("-inf")


@dataclass
class CapacityReport:
    """P / C_I / C_M / C_E / C_V ölçüm raporu."""

    # Ölçülen kapasiteler
    c_e_total: int                 # üretilebilir anlamlı deneyim sayısı
    c_v_total: int                 # bunlardan bağımsız verifier'ın karara bağladığı
    c_v_true: int                  # doğrulanan (True)
    c_v_false: int                 # çürütülen (False)
    c_e_undecidable: int           # verifier'ın None döndürdüğü (UNCERTAIN)
    # Ham üretim uzayı
    generated_candidates: int
    structurally_valid: int
    sampled: bool
    sample_size: int
    # Oranlar
    c_v_over_c_e: float
    decidability: float
    # Teorik üst sınırlar (bağlam için; ölçüm değildir)
    c_m_conceptual: Optional[float] = None
    c_i_interaction: Optional[float] = None
    physical_parameters: Optional[int] = None
    log10: Dict[str, float] = field(default_factory=dict)
    ordering_holds: bool = True
    notes: List[str] = field(default_factory=list)
    #: Raporu ürettiği konfigürasyona bağlayan deterministik imza.
    #: Yeniden-üretilebilirlik denetimi bu alanı arar; imzasız rapor
    #: hangi girdiyle üretildiğini kanıtlayamaz.
    config_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        satirlar = [
            "| Kapasite | Sembol | Değer | Anlam |",
            "|---|---|---:|---|",
        ]
        if self.physical_parameters is not None:
            satirlar.append(
                f"| Fiziksel parametre | P | {self.physical_parameters:,} | "
                "RAM'de ayrılan, optimizer'ın güncellediği |")
        if self.c_i_interaction is not None:
            satirlar.append(
                f"| Etkileşim kapasitesi | C_I | {self.c_i_interaction:.3e} | "
                "operatör üst sınırı — parametre DEĞİL |")
        if self.c_m_conceptual is not None:
            satirlar.append(
                f"| Bellek adres kapasitesi | C_M | {self.c_m_conceptual:.3e} | "
                "adreslenebilir kavramsal anahtar |")
        satirlar.append(
            f"| Deneyim kapasitesi | C_E | {self.c_e_total:,} | "
            "kısıtlar altında gerçekten üretilebilen |")
        satirlar.append(
            f"| Doğrulanabilir kapasite | C_V | {self.c_v_total:,} | "
            "bağımsız verifier'ın karara bağladığı |")
        return "\n".join(satirlar)


def capacity_contract(
    physical_parameters: Optional[int] = None,
    n: Optional[int] = None,
    k: Optional[int] = None,
    vocab: Optional[int] = None,
    window: Optional[int] = None,
) -> Dict[str, Any]:
    """Teorik üst sınırları (P, C_I, C_M) dürüst etiketlerle hesapla."""
    sozlesme: Dict[str, Any] = {
        "physical_parameters": physical_parameters,
        "c_i_interaction": None,
        "c_m_conceptual": None,
        "c_i_is_parameter_count": False,
        "c_m_is_physical_table_size": False,
        "statement": (
            "C_I ve C_M üst sınırdır; fiziksel parametre veya fiziksel tablo "
            "boyutu değildir. Ölçülen kapasiteler C_E ve C_V'dir."
        ),
    }
    if n is not None and k is not None:
        if n < 2 or k < 1:
            raise ValueError("n >= 2 ve K >= 1 olmalı")
        sozlesme["c_i_interaction"] = float(n) ** (2 * int(k))
        sozlesme["operator_entries_per_layer_n4"] = float(n) ** 4
    if vocab is not None and window is not None:
        if vocab < 2 or window < 1:
            raise ValueError("vocab >= 2 ve window >= 1 olmalı")
        sozlesme["c_m_conceptual"] = float(vocab) ** int(window)
    return sozlesme


def measure_experience_capacity(
    store,
    verifier: Verifier,
    generator=None,
    relation_ids: Optional[Sequence[str]] = None,
    evaluator=None,
    sample_size: Optional[int] = None,
    seed: int = 42,
    physical_parameters: Optional[int] = None,
    n: Optional[int] = None,
    k: Optional[int] = None,
    vocab: Optional[int] = None,
    window: Optional[int] = None,
) -> CapacityReport:
    """C_E ve C_V'yi verilen bilgi tabanı + bağımsız doğrulayıcı ile ölç.

    * ``C_E`` = generator'ın kısıtlar altında ürettiği ve evaluator'ın
      yapısal olarak elemediği (INVALID olmayan) aday sayısı.
    * ``C_V`` = bu adaylardan bağımsız verifier'ın ``True``/``False`` karar
      verebildiği sayı. ``None`` dönenler karara bağlanamayan (UNCERTAIN)
      bölgedir ve ``C_V``'ye girmez.

    Üretim uzayı büyükse ``sample_size`` ile örnekleme yapılır; rapor
    ``sampled=True`` ile bunu açıkça bildirir.
    """
    from ..experience.evaluator import ExperienceEvaluator
    from ..experience.generator import ExperienceGenerator

    generator = generator or ExperienceGenerator()
    evaluator = evaluator or ExperienceEvaluator()
    adaylar: List[ExperienceCandidate] = generator.uret(
        store, relation_ids=list(relation_ids) if relation_ids else None)
    uretilen = len(adaylar)

    sampled = False
    if sample_size is not None and sample_size < uretilen:
        adaylar = random.Random(seed).sample(adaylar, int(sample_size))
        sampled = True

    yapisal_gecerli = 0
    c_v_true = c_v_false = kararsiz = 0
    for aday in adaylar:
        evaluator.degerlendir(aday, store)
        if aday.state == DeneyimDurumu.INVALID:
            continue
        yapisal_gecerli += 1
        sonuc = verifier(store, aday)
        if sonuc is True:
            c_v_true += 1
        elif sonuc is False:
            c_v_false += 1
        else:
            kararsiz += 1

    olcek = (uretilen / len(adaylar)) if (sampled and adaylar) else 1.0
    c_e = int(round(yapisal_gecerli * olcek))
    c_v = int(round((c_v_true + c_v_false) * olcek))

    sozlesme = capacity_contract(physical_parameters, n, k, vocab, window)
    c_m = sozlesme.get("c_m_conceptual")
    ordering = c_v <= c_e and (c_m is None or c_e <= c_m)

    log10_tablosu = {"c_e": _log10(c_e), "c_v": _log10(c_v)}
    if c_m:
        log10_tablosu["c_m"] = _log10(c_m)
    if sozlesme.get("c_i_interaction"):
        log10_tablosu["c_i"] = _log10(float(sozlesme["c_i_interaction"]))

    notlar = [
        "C_E ve C_V ölçülmüş değerlerdir; C_I ve C_M teorik üst sınırdır.",
        "C_V ≤ C_E ≤ C_M sıralaması sistemin epistemik daralmasını gösterir.",
    ]
    if sampled:
        notlar.append(
            f"Örnekleme yapıldı ({len(adaylar)}/{uretilen}); C_E/C_V ölçeklenmiş tahmindir.")

    return CapacityReport(
        c_e_total=c_e,
        c_v_total=c_v,
        c_v_true=c_v_true,
        c_v_false=c_v_false,
        c_e_undecidable=kararsiz,
        generated_candidates=uretilen,
        structurally_valid=yapisal_gecerli,
        sampled=sampled,
        sample_size=len(adaylar),
        c_v_over_c_e=_oran(c_v, c_e),
        decidability=_oran(c_v_true + c_v_false, yapisal_gecerli),
        c_m_conceptual=c_m,
        c_i_interaction=sozlesme.get("c_i_interaction"),
        physical_parameters=physical_parameters,
        log10=log10_tablosu,
        ordering_holds=ordering,
        notes=notlar,
    )


def run_capacity_benchmark(
    operands_max: int = 9,
    initial_facts: int = 5,
    sample_size: Optional[int] = 2000,
    seed: int = 42,
    n: Optional[int] = 256,
    k: Optional[int] = 4,
    vocab: Optional[int] = 8000,
    window: Optional[int] = 16,
    physical_parameters: Optional[int] = 40_524_865,
) -> CapacityReport:
    """Aritmetik mini-environment üzerinde C_E/C_V referans ölçümü.

    Ground truth bağımsızdır (aritmetik oracle), bu yüzden C_V gerçekten
    "bağımsız olarak doğrulanabilir" alt kümeyi ölçer.
    """
    from ..experience.mini_env import AritmetikOrtam
    from ..knowledge import KnowledgeStore

    if operands_max < 1:
        raise ValueError("operands_max >= 1 olmalı")

    store = KnowledgeStore()
    ifade_sayaci = 0
    for sol in range(operands_max + 1):
        for sag in range(operands_max + 1):
            store.varlik_ekle(f"{sol}+{sag}", entity_type="ifade",
                              entity_id=f"E_EXPR_{ifade_sayaci:05d}")
            ifade_sayaci += 1
    for deger in range(2 * operands_max + 1):
        store.varlik_ekle(str(deger), entity_type="sayi",
                          entity_id=f"E_RESULT_{deger:05d}")
    store.iliski_tanimla("eşittir", relation_id="R_EQUALS",
                         subject_types=["ifade"], object_types=["sayi"])
    for index in range(min(initial_facts, ifade_sayaci)):
        sol, sag = divmod(index, operands_max + 1)
        store.olgu_kaydet(f"E_EXPR_{index:05d}", "R_EQUALS",
                          f"E_RESULT_{sol + sag:05d}", score=1.0,
                          source=KaynakTuru.VERIFIED_RULE, confidence=1.0)

    ortam = AritmetikOrtam()
    rapor = measure_experience_capacity(
        store, verifier=ortam.aday_dogrula, relation_ids=["R_EQUALS"],
        sample_size=sample_size, seed=seed,
        physical_parameters=physical_parameters, n=n, k=k,
        vocab=vocab, window=window,
    )
    rapor.notes.append(
        "Domain: aritmetik mini-environment; genel dil kapasitesi iddiası yoktur.")
    rapor.config_hash = hashlib.sha256(json.dumps({
        "protocol": "capacity_framework_v1",
        "operands_max": operands_max, "initial_facts": initial_facts,
        "sample_size": sample_size, "seed": seed, "n": n, "k": k,
        "vocab": vocab, "window": window,
        "physical_parameters": physical_parameters,
    }, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return rapor


__all__ = [
    "CapacityReport",
    "capacity_contract",
    "measure_experience_capacity",
    "run_capacity_benchmark",
]
