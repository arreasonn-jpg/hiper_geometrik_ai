# -*- coding: utf-8 -*-
"""
Benchmark — Döngüyü Ground-Truth'a Karşı Ölç (v1.0)
=====================================================
(rapor §18, §19 v1.0, §20)

"Gerçek veri + generated experience + verification + continual learning
döngüsünü kontrollü benchmarklarla ölç" maddesinin somut karşılığı:

    * `aritmetik_etki_alani()` — deterministik ground-truth'u olan mini alan:
      eşitlik ifadeleri; `AritmetikOrtam` her üçlünün doğruluğunu bilir.
    * `kos(...)` — DeneyimDongusu'nu N adım çalıştırır.
    * `ozetle(...)` — adım raporlarını toplam BenchmarkOzeti'ne indirger.

Böylece rapor §20'deki metrikler (acceptance/conflict rate, false acceptance/
rejection, knowledge growth, replay efficiency) tek tabloda raporlanır.
"""
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

from ..knowledge import KnowledgeStore
from ..memory import BellekEntegrasyonu
from .consolidation import Consolidator
from .evaluator import ExperienceEvaluator
from .generator import ExperienceGenerator
from .loop import AdimRaporu, DeneyimDongusu
from .mini_env import AritmetikOrtam


@dataclass
class BenchmarkOzeti:
    adim_sayisi: int = 0
    toplam_uretilen: int = 0
    toplam_novel: int = 0
    toplam_valid: int = 0
    toplam_conflict: int = 0
    toplam_uncertain: int = 0
    toplam_invalid: int = 0
    toplam_verified: int = 0
    yanlis_kabul: Optional[int] = None      # false acceptance (doğrulayıcı varsa)
    yanlis_ret: Optional[int] = None        # false rejection  (doğrulayıcı varsa)
    bilgi_buyumesi: int = 0
    son_acceptance_rate: float = 0.0
    son_conflict_rate: float = 0.0
    son_replay_verimliligi: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def aritmetik_etki_alani(ifadeler: Optional[List[str]] = None) -> KnowledgeStore:
    """Deterministik ground-truth'lu mini alan: 'eşittir' ilişkisi."""
    ifadeler = ifadeler or ["1+1", "1+2", "2+2", "2", "3", "4"]
    k = KnowledgeStore()
    for i, t in enumerate(ifadeler, 1):
        k.varlik_ekle(t, entity_type="ifade", entity_id=f"E_{i:03d}")
    k.iliski_tanimla("eşittir", relation_id="R_001")
    return k


def _esittir_rid(store) -> str:
    for r in store.relations.iliskiler():
        if r.token.strip().lower() in ("eşittir", "esittir", "esit"):
            return r.relation_id
    raise KeyError("'eşittir' ilişkisi bulunamadı")


def kos(store, adimlar: int = 1, dogrulayici: Optional[Callable] = None,
        replay_n: int = 2, slot_sayisi: int = 64) -> List[AdimRaporu]:
    """Döngüyü N adım çalıştır; adım raporlarını döner."""
    dongu = DeneyimDongusu(
        store=store,
        generator=ExperienceGenerator(tip_filtresi=False),
        evaluator=ExperienceEvaluator(),
        consolidator=Consolidator(),
        bellek=BellekEntegrasyonu(slot_sayisi=slot_sayisi,
                                   replay_kapasitesi=32),
        dogrulayici=dogrulayici,
        replay_n=replay_n,
    )
    return dongu.calistir(int(adimlar), relation_ids=[_esittir_rid(store)])


def ozetle(raporlar: List[AdimRaporu]) -> BenchmarkOzeti:
    """Adım raporlarını toplam bir özete indirge."""
    o = BenchmarkOzeti()
    o.adim_sayisi = len(raporlar)
    for r in raporlar:
        o.toplam_uretilen += r.uretilen
        o.toplam_novel += r.novel
        o.toplam_valid += r.valid
        o.toplam_conflict += r.conflict
        o.toplam_uncertain += r.uncertain
        o.toplam_invalid += r.invalid
        o.toplam_verified += r.verified
        o.bilgi_buyumesi += r.knowledge_buyumesi
    if raporlar:
        son = raporlar[-1]
        o.son_acceptance_rate = son.acceptance_rate
        o.son_conflict_rate = son.conflict_rate
        o.son_replay_verimliligi = son.replay_verimliligi
        o.yanlis_kabul = son.false_acceptance
        o.yanlis_ret = son.false_rejection
    return o


def karsilastirma(adimlar: int = 1) -> dict:
    """Aynı alanda, ground-truth doğrulayıcılı tek koşunun ölçümünü yapar.

    Dönüş: {'store': ..., 'raporlar': ..., 'ozet': ...}. Doğrulayıcı
    `AritmetikOrtam.aday_dogrula`'dır; kural tabanlı değerlendirmenin
    gerçek yanlış kabul sayısını açığa çıkarır.
    """
    store = aritmetik_etki_alani()
    ortam = AritmetikOrtam()
    raporlar = kos(store, adimlar=adimlar, dogrulayici=ortam.aday_dogrula)
    return {"store": store, "raporlar": raporlar, "ozet": ozetle(raporlar)}
