# -*- coding: utf-8 -*-
"""
Araştırma Kuyruğu — CONFLICT → Exploration'ı Ölçekleme
========================================================
(rapor §11, §19 v0.5, §21)

`ConflictResolver` TEK bir çelişkiyi çözer; `ArastirmaKuyrugu` bunu toplu
çalıştırır: CONFLICT durumundaki deneyimleri biriktirir, deterministik
doğrulayıcıyla (örn. `AritmetikOrtam`) sınar ve kanıt üretebilenleri kesin
duruma (VALID/INVALID) indirir, kanıt üretemeyenleri kuyrukta bırakır.

Böylece "çelişkiyi çöpe atmak yerine araştırma sinyali olarak kullan" ilkesi
(rapor §11) sistemin aktif keşif mekanizmasına dönüşür ve ölçülebilir hâle
gelir: `ArastirmaRaporu` çözülen/açık kalan çelişkileri ve bilgi büyümesini
raporlar.
"""
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from ..knowledge.schemas import DeneyimDurumu, ExperienceCandidate
from .conflict import ConflictResolver
from .evaluator import ExperienceEvaluator


@dataclass
class ArastirmaRaporu:
    islenen: int = 0
    cozulen: int = 0
    acik: int = 0
    cozulenler: List[str] = field(default_factory=list)
    aciklar: List[str] = field(default_factory=list)
    bilgi_buyumesi: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


class ArastirmaKuyrugu:
    """CONFLICT ve UNCERTAIN deneyimleri kanıt araması için toplar."""

    def __init__(self, evaluator: Optional[ExperienceEvaluator] = None,
                 dogrulayici: Optional[Callable] = None):
        self.kuyruk: List[ExperienceCandidate] = []
        self.evaluator = evaluator or ExperienceEvaluator()
        self.dogrulayici = dogrulayici
        self.resolver = ConflictResolver(evaluator=self.evaluator,
                                         deterministik_test=dogrulayici)

    def ekle(self, aday: ExperienceCandidate) -> None:
        """CONFLICT/UNCERTAIN bir adayı kuyruğa al (tekrar eklenmez)."""
        if aday.state not in (DeneyimDurumu.CONFLICT, DeneyimDurumu.UNCERTAIN):
            return
        if any(a.experience_id == aday.experience_id for a in self.kuyruk):
            return
        self.kuyruk.append(aday)

    def besle(self, adaylar: List[ExperienceCandidate]) -> int:
        """Bir listedeki CONFLICT adayları kuyruğa ekle; eklenen sayıyı döner."""
        once = len(self.kuyruk)
        for a in adaylar:
            self.ekle(a)
        return len(self.kuyruk) - once

    def isle(self, store) -> ArastirmaRaporu:
        """Kuyruktaki her çelişkiyi çöz; kesinleşenleri çıkar.

        Döngü (rapor §11): neden → alternatif üret → deterministik test →
        güven güncelle → yeniden değerlendir. Kanıt üretilemeyenler CONFLICT
        kalır (güvenli varsayılan — yanlış yükseltme YAPILMAZ).
        """
        onceki = store.versiyon
        rapor = ArastirmaRaporu(islenen=len(self.kuyruk))
        kalan = []
        for a in self.kuyruk:
            self.resolver.coz(store, a)
            if a.state in (DeneyimDurumu.CONFLICT, DeneyimDurumu.UNCERTAIN):
                rapor.acik += 1
                rapor.aciklar.append(a.experience_id)
                kalan.append(a)
            else:
                rapor.cozulen += 1
                rapor.cozulenler.append(a.experience_id)
        self.kuyruk = kalan
        rapor.bilgi_buyumesi = store.versiyon - onceki
        return rapor

    def __len__(self) -> int:
        return len(self.kuyruk)
