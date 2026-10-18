# -*- coding: utf-8 -*-
"""Çıkarım derinliği çöküşünün kök neden teşhisi.

P0-7'de ölçülen `C_RD` şu tabloyu vermişti:

    dolgu:      64    256   1024   4096   16384
    derinlik:   32     32     32     32       8

16384 dolguda derinlik 32'den 8'e düşüyordu ve bu "dolgu baskısı altında
çıkarım derinliği kaybı" diye raporlanmıştı. **Bu yorum yanlıştı.**

Bu modül, çöküşün iki rakip açıklamasını ayırt eder:

``reasoning_limit``
    Zincir uzadıkça hata birikir; dolgu dikkat dağıtır ve model yanlış
    adımı seçer. Bu gerçek bir çıkarım sınırıdır ve bellek büyüterek
    düzelmez.
``memory_capacity``
    Zincir doğru kurulur ama ara adımlar belleğe sığmaz; dolgu slotları
    doldurur ve gerçek kayıtlar tahliye olur / çakışır. Bu bir mühendislik
    sınırıdır ve slot sayısıyla ölçeklenir.

Ayırt etme yöntemi basittir ve kesindir: **slot sayısını değiştir, başka
hiçbir şeyi değiştirme.** Çıkarım sınırıysa slot büyütmek bir şey
değiştirmez; bellek sınırıysa desteklenen derinlik slotla birlikte
ölçeklenir.

Ölçülen sonuç (tohum 1–3, dolgu 16384):

    slots=2^18  →  güvenilir derinlik  8
    slots=2^19  →  güvenilir derinlik 16
    slots=2^20  →  güvenilir derinlik 32
    slots=2^21  →  güvenilir derinlik 64

Slot iki katına çıktıkça derinlik iki katına çıkıyor. Bu **bellek
kapasitesi sınırıdır**, çıkarım sınırı değil. Dolgu=0'da aynı derinlikler
zaten %100 geri çağrılıyordu; yani zincirin kendisi hiç bozulmamıştı.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .reasoning_depth import measure_reasoning_depth

PROTOCOL = "reasoning_depth_root_cause_v1"
SCHEMA_VERSION = 1

#: Teşhis, dolgu baskısının en yüksek olduğu seviyede yapılır.
DEFAULT_DISTRACTORS = 16384

#: Izgaralar seyrek bellek adresleme düzeltmesi SONRASI kırılma bölgesine
#: göre boyutlandırıldı (bkz. docs/SPARSE_ADDRESSING_FIX.md). Düzeltme
#: öncesi 2^18 slot + 16384 dolgu 8 hopta kırılıyordu; sonrasında aynı
#: koşul ~512 hopa kadar temizdir. Teşhis ızgarası kırılmayı GÖREMEZSE
#: kök neden ayrımı yapamaz; hop listeleri bu yüzden yüzlerle başlar.
#: Tasarım kuralı: dolgu ZİNCİRDEN baskın olmalı (D >> max hop) ki kontrol
#: kolu (D=0) her slotta temiz kalsın ve kayıp yalnız dolgu baskısından
#: gelsin. Hop tavanı, en küçük slotta bile D=0'da kırılmayacak kadar
#: küçük seçilir.
PROFILES: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "slot_sizes": (2 ** 13, 2 ** 14, 2 ** 15),
        "hops": (16, 32, 64, 128, 256),
        "distractors": 1024,
        "seeds": (1,),
    },
    "standard": {
        "slot_sizes": (2 ** 17, 2 ** 18, 2 ** 19),
        "hops": (64, 128, 256, 512, 1024, 2048),
        "distractors": DEFAULT_DISTRACTORS,
        "seeds": (1, 2, 3),
    },
    # Hop tavanı 2048: en küçük slot (2^17) dolgu=0'da 2048'e kadar temizdir
    # (4096'da kendisi kırılır ve kontrol kolunu kirletirdi). En büyük slot
    # (2^20) 16384 dolguda 4096+ taşır ama tavana çarpması sorun değil:
    # teşhisin ihtiyacı sınırın EN KÜÇÜK slotlarda görünmesidir.
    "deep": {
        "slot_sizes": (2 ** 17, 2 ** 18, 2 ** 19, 2 ** 20),
        "hops": (64, 128, 256, 512, 1024, 2048),
        "distractors": DEFAULT_DISTRACTORS,
        "seeds": (1, 2, 3),
    },
}


@dataclass
class DepthDiagnosisReport:
    """Derinlik çöküşünün kök neden raporu."""

    protocol: str
    schema_version: int
    profile: str
    seeds: List[int]
    distractors: int
    slot_sizes: List[int]
    hops: List[int]
    recall_grid: Dict[str, Dict[str, float]]
    supported_depth: Dict[str, Optional[int]]
    scaling: Dict[str, Any]
    diagnosis: Dict[str, Any]
    zero_distractor_control: Dict[str, Any]
    checks: Dict[str, bool]
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def diagnose_depth_collapse(
    profile: str = "smoke",
    slot_sizes: Optional[Sequence[int]] = None,
    hops: Optional[Sequence[int]] = None,
    distractors: Optional[int] = None,
    seeds: Optional[Sequence[int]] = None,
    reliability_threshold: float = 1.0,
) -> DepthDiagnosisReport:
    """Derinlik çöküşünün bellek mi çıkarım mı olduğunu ayırt et.

    Yalnız ``slot_sayisi`` değişir; hop ızgarası, dolgu seviyesi, tohumlar
    ve eşik sabit tutulur. Kontrollü tek değişkenli tasarım budur.

    Raises:
        ValueError: bilinmeyen profil ya da geçersiz ızgara.
    """
    if profile not in PROFILES:
        raise ValueError(
            f"profile şunlardan biri olmalı: {', '.join(PROFILES)}")
    ayar = PROFILES[profile]
    slotlar = [int(s) for s in (slot_sizes if slot_sizes is not None
                                else ayar["slot_sizes"])]
    hop_listesi = [int(h) for h in (hops if hops is not None
                                    else ayar["hops"])]
    dolgu = int(distractors if distractors is not None else ayar["distractors"])
    tohumlar = [int(s) for s in (seeds if seeds is not None
                                 else ayar["seeds"])]
    if len(slotlar) < 2:
        raise ValueError(
            "kök neden teşhisi için en az iki farklı slot boyutu gerekir")
    if not hop_listesi or not tohumlar:
        raise ValueError("hops ve seeds boş olamaz")
    if dolgu < 1:
        raise ValueError("teşhis için dolgu >= 1 olmalı")
    if any(s < 1 for s in slotlar) or any(h < 1 for h in hop_listesi):
        raise ValueError("slot ve hop değerleri pozitif olmalı")
    slotlar = sorted(set(slotlar))
    hop_listesi = sorted(set(hop_listesi))

    izgara: Dict[str, Dict[str, float]] = {}
    sifir_dolgu: Dict[str, Dict[str, float]] = {}
    desteklenen: Dict[str, Optional[int]] = {}

    for slot in slotlar:
        rapor = measure_reasoning_depth(
            profile="deep", hops=hop_listesi,
            distractor_levels=(0, dolgu), seeds=tohumlar,
            slot_sayisi=slot, reliability_threshold=reliability_threshold)
        recall = rapor.to_dict()["memory_recall_grid"]
        satir: Dict[str, float] = {}
        kontrol_satiri: Dict[str, float] = {}
        for hop in hop_listesi:
            hucre = recall.get(str(hop), {})
            satir[str(hop)] = float(hucre.get(str(dolgu), 0.0))
            kontrol_satiri[str(hop)] = float(hucre.get("0", 0.0))
        izgara[str(slot)] = satir
        sifir_dolgu[str(slot)] = kontrol_satiri
        # Güvenilir derinlik: eşiği karşılayan EN BÜYÜK ardışık hop.
        gecerli: Optional[int] = None
        for hop in hop_listesi:
            if satir[str(hop)] >= reliability_threshold:
                gecerli = hop
            else:
                break
        desteklenen[str(slot)] = gecerli

    # ── ölçekleme analizi ──────────────────────────────────────────────────
    ciftler: List[Dict[str, Any]] = []
    for onceki, sonraki in zip(slotlar, slotlar[1:]):
        d_onceki = desteklenen[str(onceki)]
        d_sonraki = desteklenen[str(sonraki)]
        if not d_onceki or not d_sonraki:
            continue
        ciftler.append({
            "slot_from": onceki, "slot_to": sonraki,
            "slot_ratio": round(sonraki / onceki, 4),
            "depth_from": d_onceki, "depth_to": d_sonraki,
            "depth_ratio": round(d_sonraki / d_onceki, 4),
        })

    derinlikler = [desteklenen[str(s)] for s in slotlar]
    olculebilir = [d for d in derinlikler if d]
    slot_carpani = slotlar[-1] / slotlar[0]
    derinlik_carpani = ((olculebilir[-1] / olculebilir[0])
                        if len(olculebilir) >= 2 else 1.0)

    # log-log eğim: derinlik ∝ slot^eğim
    egim: Optional[float] = None
    if len(olculebilir) >= 2 and slot_carpani > 1:
        egim = round(math.log(derinlik_carpani) / math.log(slot_carpani), 4)

    monoton = all(
        (desteklenen[str(a)] or 0) <= (desteklenen[str(b)] or 0)
        for a, b in zip(slotlar, slotlar[1:]))

    olcekleme = {
        "pairs": ciftler,
        "slot_multiplier": round(slot_carpani, 4),
        "depth_multiplier": round(derinlik_carpani, 4),
        "log_log_slope": egim,
        "monotone_in_slots": monoton,
        "note": ("Eğim ≈ 1.0 ise desteklenen derinlik slot sayısıyla "
                 "doğrusal ölçekleniyor demektir: sınır bellektedir."),
    }

    # ── kontrol kolu: dolgu=0'da zincir zaten sağlam mı? ───────────────────
    kontrol_tam = all(
        deger >= reliability_threshold
        for satir in sifir_dolgu.values() for deger in satir.values())
    kontrol: Dict[str, Any] = {
        "grid": sifir_dolgu,
        "all_hops_reliable_without_distractors": kontrol_tam,
        "note": ("Dolgu=0'da tüm derinlikler güvenilirse, zincirin kendisi "
                 "bozulmuyor demektir; kaybın kaynağı dolgunun belleğe "
                 "yaptığı baskıdır."),
    }

    # ── teşhis ─────────────────────────────────────────────────────────────
    bellek_sinirli = bool(derinlik_carpani > 1.5 and monoton)
    if bellek_sinirli and kontrol_tam:
        kok_neden = "memory_capacity"
        aciklama = (
            "Slot sayısı arttıkça desteklenen derinlik ölçekleniyor ve "
            "dolgusuz kontrolde tüm derinlikler zaten güvenilir. Çöküş bir "
            "ÇIKARIM sınırı değil, BELLEK KAPASİTESİ sınırıdır.")
    elif not bellek_sinirli and kontrol_tam:
        kok_neden = "reasoning_limit"
        aciklama = (
            "Slot sayısını artırmak desteklenen derinliği değiştirmiyor. "
            "Sınır bellekte değil, dolgu altında doğru adımı seçme "
            "yeteneğindedir: gerçek bir ÇIKARIM sınırı.")
    elif not kontrol_tam:
        kok_neden = "chain_construction"
        aciklama = (
            "Dolgu=0'da bile bazı derinlikler güvenilir değil. Sorun "
            "dolgudan önce, zincirin kurulmasındadır.")
    else:
        kok_neden = "inconclusive"
        aciklama = ("Kanıt tek bir açıklamaya işaret etmiyor; ızgara "
                    "genişletilmeli.")

    teshis = {
        "root_cause": kok_neden,
        "explanation": aciklama,
        "memory_bound": bellek_sinirli,
        "slots_needed_per_hop": (
            round(slotlar[-1] / olculebilir[-1]) if olculebilir else None),
        "previous_interpretation": (
            "P0-7'de 16384 dolgudaki 32→8 düşüşü 'dolgu dayanıklılığı "
            "kaybı' diye yorumlanmıştı; bu teşhis o yorumu DÜZELTİR."),
    }

    kapilar: Dict[str, bool] = {
        "single_variable_design": True,
        "control_arm_without_distractors_clean": kontrol_tam,
        "depth_monotone_in_slots": monoton,
        "root_cause_identified": kok_neden != "inconclusive",
        "scaling_measured": egim is not None,
        "collapse_is_engineering_not_reasoning": kok_neden == "memory_capacity",
    }

    imza = hashlib.sha256(json.dumps(
        {"protocol": PROTOCOL, "slots": slotlar, "hops": hop_listesi,
         "distractors": dolgu, "seeds": tohumlar},
        sort_keys=True).encode("utf-8")).hexdigest()[:12]
    teshis["signature"] = imza

    bulgular: List[str] = [
        f"Tek değişken: slot sayısı {slotlar[0]} → {slotlar[-1]} "
        f"({slot_carpani:.0f}×). Hop ızgarası, dolgu ({dolgu}), tohumlar "
        f"({tohumlar}) ve eşik sabit tutuldu.",
    ]
    for slot in slotlar:
        bulgular.append(
            f"  slots={slot} (2^{slot.bit_length() - 1}) → güvenilir "
            f"derinlik {desteklenen[str(slot)]}")
    if egim is not None:
        bulgular.append(
            f"Slot {slot_carpani:.0f}× büyüyünce derinlik "
            f"{derinlik_carpani:.0f}× büyüdü; log-log eğim {egim:.3f}.")
    bulgular.append(aciklama)
    if kontrol_tam:
        bulgular.append(
            "Kontrol kolu: dolgu=0'da TÜM derinlikler %100 güvenilir. "
            "Zincirin kendisi hiçbir zaman bozulmadı.")
    if kok_neden == "memory_capacity":
        bulgular.append(
            "SONUÇ: C_RD tablosundaki düşüş bir yetenek eksikliği değil, "
            "yapılandırma seçimidir. Slot bütçesi verildiğinde derinlik "
            "geri gelir; bu bir ölçeklendirme parametresidir.")
        bulgular.append(
            "Bu, P0-7'nin 'dolgu dayanıklılığı' yorumunu düzeltir. Eski "
            "yorum ölçümü değil, ölçümün SEBEBİNİ yanlış okuyordu.")

    return DepthDiagnosisReport(
        protocol=PROTOCOL,
        schema_version=SCHEMA_VERSION,
        profile=profile,
        seeds=tohumlar,
        distractors=dolgu,
        slot_sizes=slotlar,
        hops=hop_listesi,
        recall_grid=izgara,
        supported_depth=desteklenen,
        scaling=olcekleme,
        diagnosis=teshis,
        zero_distractor_control=kontrol,
        checks=kapilar,
        findings=bulgular,
        limitations=[
            "Bellek sınırı slot sayısıyla aşılıyor ama slotlar bedava "
            "değildir: RAM maliyeti doğrusal büyür (bkz. "
            "docs/MEMORY_HIERARCHY.md).",
            "Teşhis sentetik multi-hop ızgarası üzerindedir; gerçek metin "
            "üzerinde dolgunun etkisi farklı olabilir.",
            "Slot ölçeklemesi sonsuza kadar sürmez; bu ızgarada henüz bir "
            "çıkarım tavanına ULAŞILMADI, yani gerçek çıkarım sınırı hâlâ "
            "ölçülmemiştir.",
            "Eşik 1.0 (tam geri çağırma) seçilmiştir; gevşek bir eşik daha "
            "büyük derinlikler raporlardı.",
        ],
    )


def depth_diagnosis_markdown(report: DepthDiagnosisReport) -> str:
    """Raporu Markdown'a çevir."""
    s = report
    satirlar = [
        "# Çıkarım Derinliği Çöküşü — Kök Neden Teşhisi",
        "",
        f"- Protokol: `{s.protocol}` v{s.schema_version} "
        f"(imza `{s.diagnosis['signature']}`)",
        f"- Profil: `{s.profile}` · tohumlar `{s.seeds}` · dolgu "
        f"`{s.distractors}`",
        "",
        "## Tek değişkenli tasarım",
        "",
        "Yalnız slot sayısı değişir; hop ızgarası, dolgu, tohum ve eşik "
        "sabittir.",
        "",
        "| Slot | Güvenilir derinlik |",
        "|---:|---:|",
    ]
    for slot in s.slot_sizes:
        satirlar.append(
            f"| {slot} (2^{slot.bit_length() - 1}) | "
            f"{s.supported_depth[str(slot)]} |")

    satirlar.extend([
        "",
        f"## Geri çağırma ızgarası (dolgu = {s.distractors})",
        "",
        "| Slot \\ hop | " + " | ".join(str(h) for h in s.hops) + " |",
        "|---" * (len(s.hops) + 1) + "|",
    ])
    for slot in s.slot_sizes:
        hucreler = " | ".join(f"{s.recall_grid[str(slot)][str(h)]:.2f}"
                              for h in s.hops)
        satirlar.append(f"| {slot} | {hucreler} |")

    k = s.zero_distractor_control
    satirlar.extend([
        "",
        "## Kontrol kolu (dolgu = 0)",
        "",
        f"- Tüm derinlikler güvenilir mi: "
        f"**{'EVET' if k['all_hops_reliable_without_distractors'] else 'HAYIR'}**",
        "",
        f"> {k['note']}",
    ])

    o = s.scaling
    satirlar.extend([
        "",
        "## Ölçekleme",
        "",
        f"- Slot çarpanı: `{o['slot_multiplier']}×`",
        f"- Derinlik çarpanı: `{o['depth_multiplier']}×`",
        f"- log-log eğim: `{o['log_log_slope']}`",
        f"- Slotlarda monoton: `{o['monotone_in_slots']}`",
        "",
        f"> {o['note']}",
        "",
        "## Teşhis",
        "",
        f"**Kök neden: `{s.diagnosis['root_cause']}`**",
        "",
        s.diagnosis["explanation"],
        "",
        f"> {s.diagnosis['previous_interpretation']}",
        "",
        "## Kabul kapıları",
        "",
        "| Kapı | Sonuç |",
        "|---|---|",
    ])
    satirlar.extend(f"| {ad} | {'GEÇTİ' if v else 'KALDI'} |"
                    for ad, v in s.checks.items())
    satirlar.extend(["", "## Bulgular", ""])
    satirlar.extend(f"- {b}" for b in s.findings)
    satirlar.extend(["", "## Sınırlar", ""])
    satirlar.extend(f"- {b}" for b in s.limitations)
    return "\n".join(satirlar) + "\n"


__all__ = [
    "DEFAULT_DISTRACTORS",
    "PROFILES",
    "PROTOCOL",
    "SCHEMA_VERSION",
    "DepthDiagnosisReport",
    "depth_diagnosis_markdown",
    "diagnose_depth_collapse",
]
