# -*- coding: utf-8 -*-
"""
Memory Interference — Kasıtlı Çakışma ve Politika Karşılaştırması (Faz 15–17)
==============================================================================

`hga/memory/benchmark.py` collision'ı **istatistiksel** olarak ölçer: N context
yaz, kaçı çakıştı? Bu gerçekçi ama zayıf bir testtir, çünkü düşük yük
faktöründe çakışma neredeyse hiç olmaz ve "sparse memory çalışıyor" yanılsaması
üretir.

Bu modül tersini yapar: **çakışmayı kasıtlı olarak kurar.**

    A → slot 123
    B → slot 123          (adversarial olarak seçilmiş)
    Soru: A bilgisi B tarafından bozuluyor mu?

Milestone koşusu (`docs/MILESTONE_TABLOSU.md`) 100 döngüde recall'ın
`1.000 → 0.952`'ye düştüğünü, collision'ın `0 → 19`'a çıktığını gösterdi.
Yani bu artık teorik bir endişe değil, ölçülmüş bir çatlaktır. Burada o
çatlağın mekanizmasını izole ediyoruz.

Üç saklama politikası kontrollü olarak karşılaştırılır:

* ``FIRST_WINS``  — mevcut davranış: ilk yazan slotu tutar, sonraki kaybolur.
* ``LAST_WINS``   — sonraki yazan üzerine yazar, önceki kaybolur.
* ``DYNAMIC_KV``  — çakışma yok: anahtar → değer sözlüğü (Faz 17 karşılaştırma
  hedefi). Fiziksel maliyet girdi sayısıyla büyür.

Dürüstlük notu: DYNAMIC_KV kolu aktif Engine ile aynı saf-Python
`DynamicKVMemory` implementasyonudur. PyTorch `HashlenmisKureselTablo`
toplamsal vektör okuması yapar; oradaki bozulma "kimlik kaybı" değil "vektör
karışması" biçiminde görünür. Bu modül o iddiayı taşımaz;
`hga.memory.kopru` protokolüyle ayrı ölçülmelidir.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .dynamic_kv import DynamicKVMemory
from .sparse_memory import DeneyimSlotlari

Triple = Tuple[str, str, str]

FIRST_WINS = "FIRST_WINS"
LAST_WINS = "LAST_WINS"
DYNAMIC_KV = "DYNAMIC_KV"
POLITIKALAR = (FIRST_WINS, LAST_WINS, DYNAMIC_KV)


def _oran(pay: int, payda: int) -> float:
    return round(pay / payda, 8) if payda else 0.0


def carpisan_anahtar_bul(
    memory: Union[DeneyimSlotlari, DynamicKVMemory],
    hedef_anahtar: Triple,
    arama_limiti: int = 2_000_000,
    baslangic: int = 0,
) -> Optional[Triple]:
    """``hedef_anahtar`` ile AYNI slot(lar)a düşen farklı bir anahtar ara.

    Rastgele çakışma beklemek yerine adversarial örnek üretir. Bulunamazsa
    ``None`` döner (sessizce başarı varsaymaz).
    """
    hedef_adresler = memory.adresler(hedef_anahtar)
    for index in range(baslangic, baslangic + int(arama_limiti)):
        aday = (f"ADV:{index}", hedef_anahtar[1], f"ADVO:{index}")
        if aday == hedef_anahtar:
            continue
        if memory.adresler(aday) == hedef_adresler:
            return aday
    return None


@dataclass
class InterferenceCase:
    """Tek bir kasıtlı çakışma vakasının sonucu."""

    victim_id: str
    attacker_id: str
    shared_slots: List[int]
    victim_retained: bool          # kurban hâlâ okunabiliyor mu?
    attacker_retained: bool        # saldırgan yazılabildi mi?
    victim_corrupted: bool         # kurban bozuldu mu? (asıl soru)
    both_retained: bool            # ikisi birden korunabildi mi?

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InterferenceReport:
    policy: str
    slot_count: int
    table_count: int
    forced_collisions: int
    collisions_constructed: int
    victim_survival_rate: float
    attacker_survival_rate: float
    both_survival_rate: float
    corruption_rate: float
    collision_events: int
    physical_entries: int
    estimated_storage_bytes: int
    bytes_per_retained_item: float
    cases: List[InterferenceCase] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **{k: v for k, v in asdict(self).items() if k != "cases"},
            "cases": [c.to_dict() for c in self.cases],
        }


class _DinamikKV(DynamicKVMemory):
    """Benchmark'ın artık aktif Engine ile aynı Dynamic KV implementasyonu."""

    def __init__(self):
        super().__init__(max_entries=None)


class _SonYazanKazanir(DeneyimSlotlari):
    """LAST_WINS politikası: çakışmada yeni kayıt eskisinin üstüne yazar."""

    def yaz(self, experience_id: str, anahtar_bilesenleri) -> int:
        self._tik += 1
        self.yazma_sayisi += 1
        adresler = self.adresler(anahtar_bilesenleri)
        for t, adres in enumerate(adresler):
            mevcut = self._tablolar[t].get(adres)
            if mevcut is not None and mevcut != experience_id:
                self.cakisma_sayisi += 1
                self.cakisma_tablosu[t] += 1
                if len(self.cakismalar) < self.cakisma_ornek_limiti:
                    self.cakismalar.append({
                        "tablo": t, "slot": adres,
                        "onceki": mevcut, "yeni": experience_id,
                    })
            self._tablolar[t][adres] = experience_id      # üzerine yaz
            self._son_erisim[t][adres] = self._tik
            self._erisim_sayisi[t][adres] = self._erisim_sayisi[t].get(adres, 0) + 1
        return adresler[0]


def _depo_olustur(policy: str, slot_count: int, table_count: int):
    if policy == FIRST_WINS:
        return DeneyimSlotlari(slot_sayisi=slot_count, tablo_sayisi=table_count)
    if policy == LAST_WINS:
        return _SonYazanKazanir(slot_sayisi=slot_count, tablo_sayisi=table_count)
    if policy == DYNAMIC_KV:
        return _DinamikKV()
    raise ValueError(f"Bilinmeyen politika: {policy}")


def _depolama_bayt(depo) -> int:
    if isinstance(depo, _DinamikKV):
        return depo.depolama_bayt()
    toplam = sys.getsizeof(depo._tablolar)
    for tablo in depo._tablolar:
        toplam += sys.getsizeof(tablo)
        toplam += sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in tablo.items())
    return toplam


def run_interference_test(
    policy: str = FIRST_WINS,
    forced_collisions: int = 25,
    slot_count: int = 1024,
    table_count: int = 1,
    search_limit: int = 2_000_000,
) -> InterferenceReport:
    """Kasıtlı A/B çakışması kur ve kurbanın bozulup bozulmadığını ölç."""
    if forced_collisions < 1:
        raise ValueError("forced_collisions >= 1 olmalı")
    if policy not in POLITIKALAR:
        raise ValueError(f"policy {POLITIKALAR} içinde olmalı")

    depo = _depo_olustur(policy, slot_count, table_count)
    # Çakışan anahtar arama, adresleme davranışı aynı olan bir yardımcı
    # üzerinden yapılır; DYNAMIC_KV'de çakışma kavramı yoktur ama aynı
    # adversarial çiftler kullanılarak adil karşılaştırma sağlanır.
    adresleyici = DeneyimSlotlari(slot_sayisi=slot_count, tablo_sayisi=table_count)

    vakalar: List[InterferenceCase] = []
    arama_imleci = 0
    for index in range(int(forced_collisions)):
        kurban_anahtar = (f"VICTIM:{index}", "R_SHARED", f"VOBJ:{index}")
        kurban_id = f"VICTIM-{index}"
        saldirgan_anahtar = carpisan_anahtar_bul(
            adresleyici, kurban_anahtar, arama_limiti=search_limit,
            baslangic=arama_imleci,
        )
        if saldirgan_anahtar is None:
            continue
        arama_imleci = int(saldirgan_anahtar[0].split(":")[1]) + 1
        saldirgan_id = f"ATTACKER-{index}"

        depo.yaz(kurban_id, kurban_anahtar)
        kurban_once = depo.icerir(kurban_id, kurban_anahtar)
        depo.yaz(saldirgan_id, saldirgan_anahtar)
        kurban_sonra = depo.icerir(kurban_id, kurban_anahtar)
        saldirgan_sonra = depo.icerir(saldirgan_id, saldirgan_anahtar)

        vakalar.append(InterferenceCase(
            victim_id=kurban_id,
            attacker_id=saldirgan_id,
            shared_slots=list(adresleyici.adresler(kurban_anahtar)),
            victim_retained=bool(kurban_sonra),
            attacker_retained=bool(saldirgan_sonra),
            victim_corrupted=bool(kurban_once and not kurban_sonra),
            both_retained=bool(kurban_sonra and saldirgan_sonra),
        ))

    toplam = len(vakalar)
    kurban_yasayan = sum(v.victim_retained for v in vakalar)
    saldirgan_yasayan = sum(v.attacker_retained for v in vakalar)
    ikisi = sum(v.both_retained for v in vakalar)
    bozulan = sum(v.victim_corrupted for v in vakalar)
    korunan_toplam = kurban_yasayan + saldirgan_yasayan
    bayt = _depolama_bayt(depo)
    fiziksel = (len(depo) if isinstance(depo, _DinamikKV)
                else sum(len(t) for t in depo._tablolar))

    notlar = [
        "Çakışmalar kasıtlı olarak kurulmuştur; doğal çakışma oranı değildir.",
        "Dynamic kol aktif Engine KV'sidir; PyTorch toplamsal tablo davranışı ayrıdır.",
    ]
    if policy == DYNAMIC_KV:
        notlar.append(
            "DYNAMIC_KV çakışmayı ortadan kaldırır ama fiziksel maliyeti "
            "girdi sayısıyla büyütür (sabit tabloda maliyet slot sayısıyla sabittir).")

    return InterferenceReport(
        policy=policy,
        slot_count=int(slot_count),
        table_count=int(table_count),
        forced_collisions=int(forced_collisions),
        collisions_constructed=toplam,
        victim_survival_rate=_oran(kurban_yasayan, toplam),
        attacker_survival_rate=_oran(saldirgan_yasayan, toplam),
        both_survival_rate=_oran(ikisi, toplam),
        corruption_rate=_oran(bozulan, toplam),
        collision_events=int(getattr(depo, "cakisma_sayisi", 0)),
        physical_entries=fiziksel,
        estimated_storage_bytes=bayt,
        bytes_per_retained_item=(round(bayt / korunan_toplam, 4)
                                 if korunan_toplam else 0.0),
        cases=vakalar,
        notes=notlar,
    )


@dataclass
class PolicyComparisonReport:
    """FIRST_WINS / LAST_WINS / DYNAMIC_KV kontrollü karşılaştırması."""

    forced_collisions: int
    slot_count: int
    table_count: int
    reports: Dict[str, Dict[str, Any]]
    best_both_survival: str
    storage_ratio_dynamic_over_fixed: float
    findings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        satirlar = [
            "| Politika | Kurban yaşar | Saldırgan yaşar | İkisi birden | Bozulma | Fiziksel girdi | Bayt |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for ad in POLITIKALAR:
            r = self.reports[ad]
            satirlar.append(
                f"| `{ad}` | {r['victim_survival_rate']:.3f} | "
                f"{r['attacker_survival_rate']:.3f} | {r['both_survival_rate']:.3f} | "
                f"{r['corruption_rate']:.3f} | {r['physical_entries']:,} | "
                f"{r['estimated_storage_bytes']:,} |"
            )
        return "\n".join(satirlar)


def run_policy_comparison(
    forced_collisions: int = 25,
    slot_count: int = 1024,
    table_count: int = 1,
) -> PolicyComparisonReport:
    """Üç saklama politikasını aynı adversarial çakışma setinde karşılaştır."""
    raporlar = {
        ad: run_interference_test(
            policy=ad, forced_collisions=forced_collisions,
            slot_count=slot_count, table_count=table_count,
        ).to_dict()
        for ad in POLITIKALAR
    }
    en_iyi = max(POLITIKALAR, key=lambda ad: raporlar[ad]["both_survival_rate"])
    sabit_bayt = raporlar[FIRST_WINS]["estimated_storage_bytes"]
    dinamik_bayt = raporlar[DYNAMIC_KV]["estimated_storage_bytes"]

    bulgular = [
        f"FIRST_WINS: çakışmada kurban korunur, saldırgan kaybolur "
        f"(saldırgan yaşama={raporlar[FIRST_WINS]['attacker_survival_rate']}).",
        f"LAST_WINS: saldırgan kazanır, kurban BOZULUR "
        f"(bozulma={raporlar[LAST_WINS]['corruption_rate']}).",
        "İki sabit-tablo politikasında da çakışan iki bilgi AYNI ANDA "
        "korunamaz; bu bir ayar meselesi değil, sabit adres uzayının yapısal "
        "sonucudur.",
        f"DYNAMIC_KV her iki bilgiyi korur (both={raporlar[DYNAMIC_KV]['both_survival_rate']}) "
        "ama fiziksel maliyeti girdi sayısıyla büyür.",
    ]

    return PolicyComparisonReport(
        forced_collisions=int(forced_collisions),
        slot_count=int(slot_count),
        table_count=int(table_count),
        reports=raporlar,
        best_both_survival=en_iyi,
        storage_ratio_dynamic_over_fixed=(round(dinamik_bayt / sabit_bayt, 6)
                                          if sabit_bayt else 0.0),
        findings=bulgular,
    )


@dataclass
class ScalingPoint:
    context_count: int
    slot_count: int
    load_factor: float
    fixed_retrieval_accuracy: float
    fixed_collision_events: int
    fixed_storage_bytes: int
    dynamic_retrieval_accuracy: float
    dynamic_storage_bytes: int
    storage_ratio_dynamic_over_fixed: float
    # Gerçek PyTorch davranışı: nn.Embedding tabloyu BAŞTAN tahsis eder ve
    # boyutu context sayısından bağımsızdır. Saf Python dict prototipi bunu
    # yapmaz; karşılaştırmanın yanıltıcı olmaması için ikisi de raporlanır.
    preallocated_table_bytes: int = 0
    ratio_dynamic_over_preallocated: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScalingReport:
    """Faz 17: sabit tablo vs dinamik KV ölçekleme ödünleşimi."""

    slot_count: int
    points: List[ScalingPoint] = field(default_factory=list)
    crossover_context_count: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_count": self.slot_count,
            "points": [p.to_dict() for p in self.points],
            "crossover_context_count": self.crossover_context_count,
            "notes": list(self.notes),
        }

    def markdown(self) -> str:
        satirlar = [
            "| Context | Yük faktörü | Sabit recall | Collision | Dinamik recall | "
            "Önceden tahsisli tablo (bayt) | Dinamik KV (bayt) | Dinamik/Tahsisli |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for p in self.points:
            satirlar.append(
                f"| {p.context_count:,} | {p.load_factor:.4f} | "
                f"{p.fixed_retrieval_accuracy:.4f} | {p.fixed_collision_events:,} | "
                f"{p.dynamic_retrieval_accuracy:.4f} | "
                f"{p.preallocated_table_bytes:,} | {p.dynamic_storage_bytes:,} | "
                f"{p.ratio_dynamic_over_preallocated:.3f} |"
            )
        return "\n".join(satirlar)


def run_fixed_vs_dynamic_scaling(
    context_counts: Sequence[int] = (100, 1_000, 10_000, 100_000),
    slot_count: int = 4096,
    table_count: int = 1,
    seed: int = 42,
    vector_dim: int = 32,
    bytes_per_scalar: int = 4,
) -> ScalingReport:
    """Aynı context akışını sabit tabloya ve dinamik KV'ye yazıp karşılaştır.

    ``preallocated_table_bytes`` PyTorch ``nn.Embedding`` gerçeğini temsil
    eder: ``slot × tablo × boyut × 4 bayt``, context sayısından BAĞIMSIZ.
    """
    from .benchmark import run_memory_benchmark

    noktalar: List[ScalingPoint] = []
    crossover: Optional[int] = None
    for context_count in sorted(int(c) for c in context_counts):
        sabit = run_memory_benchmark(
            context_count=context_count, slot_count=slot_count,
            table_count=table_count, seed=seed,
        )
        dinamik = _DinamikKV()
        for index in range(context_count):
            dinamik.yaz(f"MEM-{seed}-{index}",
                        (f"S{seed}:E{index}", f"R{index % 97}",
                         f"O{index * 2654435761 % 4294967291}"))
        tutulan = sum(
            dinamik.icerir(f"MEM-{seed}-{index}",
                           (f"S{seed}:E{index}", f"R{index % 97}",
                            f"O{index * 2654435761 % 4294967291}"))
            for index in range(context_count)
        )
        dinamik_bayt = dinamik.depolama_bayt()
        onceden_tahsis = (int(slot_count) * int(table_count)
                          * int(vector_dim) * int(bytes_per_scalar))
        nokta = ScalingPoint(
            context_count=context_count,
            slot_count=slot_count,
            load_factor=sabit.load_factor,
            fixed_retrieval_accuracy=sabit.retrieval_accuracy,
            fixed_collision_events=sabit.collision_events,
            fixed_storage_bytes=sabit.estimated_storage_bytes,
            dynamic_retrieval_accuracy=_oran(tutulan, context_count),
            dynamic_storage_bytes=dinamik_bayt,
            storage_ratio_dynamic_over_fixed=(
                round(dinamik_bayt / sabit.estimated_storage_bytes, 6)
                if sabit.estimated_storage_bytes else 0.0),
            preallocated_table_bytes=onceden_tahsis,
            ratio_dynamic_over_preallocated=(round(dinamik_bayt / onceden_tahsis, 6)
                                             if onceden_tahsis else 0.0),
        )
        noktalar.append(nokta)
        if crossover is None and nokta.fixed_retrieval_accuracy < 1.0:
            crossover = context_count

    return ScalingReport(
        slot_count=int(slot_count),
        points=noktalar,
        crossover_context_count=crossover,
        notes=[
            "Dinamik KV recall'ı tanım gereği 1.0'dır; bedeli girdi başına "
            "büyüyen fiziksel depolamadır.",
            "crossover_context_count: sabit tablonun recall'ının ilk kez "
            "1.0'ın altına düştüğü ölçek.",
            "DİKKAT: saf Python 'sabit' tablo bir dict'tir ve önceden tahsis "
            "ETMEZ; bu yüzden fixed_storage_bytes gerçek nn.Embedding "
            "davranışını TEMSİL ETMEZ. Gerçek karşılaştırma "
            "preallocated_table_bytes sütunudur (context sayısından bağımsız "
            "sabit maliyet).",
            "Karar kuralı: dinamik KV yalnız ratio_dynamic_over_preallocated "
            "< 1 olduğu ölçeklerde bellek açısından da kazançlıdır; ötesinde "
            "ödünleşim recall lehine ama RAM aleyhinedir.",
        ],
    )


__all__ = [
    "FIRST_WINS", "LAST_WINS", "DYNAMIC_KV", "POLITIKALAR",
    "InterferenceCase", "InterferenceReport", "PolicyComparisonReport",
    "ScalingPoint", "ScalingReport",
    "carpisan_anahtar_bul", "run_interference_test", "run_policy_comparison",
    "run_fixed_vs_dynamic_scaling",
]
