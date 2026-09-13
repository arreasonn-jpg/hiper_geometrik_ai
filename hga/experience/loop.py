# -*- coding: utf-8 -*-
"""
DeneyimDongusu — Sürekli Öğrenme Döngüsü + Kontrollü Benchmark (v1.0)
======================================================================
(rapor §12, §19 v1.0, §20)

Uçtan uca döngüyü tek çağrıyla koşturur ve ölçer:

    Gerçek veri (bilgi tabanı) → Temsil → Üret → Değerlendir
    → VALID/UNCERTAIN/CONFLICT/INVALID → Konsolide et → Belleğe yaz → Replay
    → yeniden değerlendir → (döngü tekrar)

Her adımda rapor §20'deki metrikler toplanır:

    * novel experiences   — gerçekten yeni üçlü sayısı
    * acceptance rate     — kabul edilen / üretilen
    * conflict rate       — çelişkili / üretilen
    * false acceptance    — (doğrulayıcı varsa) yanlış kabul
    * false rejection     — (doğrulayıcı varsa) yanlış ret
    * knowledge growth    — kalıcı bilgi sürüm artışı
    * replay efficiency   — belleğin tekrar örnekleme verimliliği

`dogrulayici` (örn. `AritmetikOrtam.aday_dogrula`) verilirse false
acceptance/rejection ground-truth'a karşı ölçülür; verilmezse bu iki metrik
None kalır (dürüstçe ölçülmemiş olarak raporlanır).
"""
import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from ..knowledge.schemas import DeneyimDurumu


@dataclass
class AdimRaporu:
    adim: int
    uretilen: int
    valid: int
    conflict: int
    uncertain: int
    invalid: int
    verified: int
    novel: int
    acceptance_rate: float
    conflict_rate: float
    false_acceptance: Optional[int]
    false_rejection: Optional[int]
    knowledge_buyumesi: int
    replay_verimliligi: float

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)


class DeneyimDongusu:
    """Üret → değerlendir → konsolide et → belleğe yaz → replay döngüsü."""

    def __init__(self, store, generator, evaluator, consolidator,
                 bellek=None, text_gen=None,
                 dogrulayici: Optional[Callable] = None,
                 replay_n: int = 2):
        self.store = store
        self.generator = generator
        self.evaluator = evaluator
        self.consolidator = consolidator
        self.bellek = bellek
        self.text_gen = text_gen
        self.dogrulayici = dogrulayici
        self.replay_n = int(replay_n)

        self._adim = 0
        self._gorulen_ucluler: set = set()
        self._yanlis_kabul = 0
        self._yanlis_ret = 0

    # ── Tek adım ─────────────────────────────────────────────────────────
    def adim(self, relation_ids: Optional[List[str]] = None,
             max_aday: Optional[int] = None) -> AdimRaporu:
        self._adim += 1

        adaylar = self.generator.uret(self.store, relation_ids=relation_ids,
                                      max_aday=max_aday)
        for a in adaylar:
            self.evaluator.degerlendir(a, self.store)

        onceki_versiyon = self.store.versiyon
        k_rapor = self.consolidator.konsolide_et(self.store, adaylar)

        # yeni üçlü sayısı (novel experiences, §20)
        adim_novel = 0
        for a in adaylar:
            if a.uclusu not in self._gorulen_ucluler:
                self._gorulen_ucluler.add(a.uclusu)
                adim_novel += 1

        # bellek entegrasyonu: kabul edilenleri yaz (VALID/VERIFIED)
        if self.bellek is not None:
            for a in adaylar:
                if a.state in (DeneyimDurumu.VALID, DeneyimDurumu.VERIFIED):
                    self.bellek.yaz(a)

        # ground-truth doğrulayıcı ile false accept/reject (§20)
        if self.dogrulayici is not None:
            for a in adaylar:
                gercek = self.dogrulayici(self.store, a)
                if gercek is None:
                    continue
                kabul = a.state in (DeneyimDurumu.VALID, DeneyimDurumu.VERIFIED)
                ret = a.state == DeneyimDurumu.INVALID
                if kabul and gercek is False:
                    self._yanlis_kabul += 1
                if ret and gercek is True:
                    self._yanlis_ret += 1

        # replay: örnekle ve yeniden değerlendir (§12, §19 v0.4)
        if self.bellek is not None:
            for eski in self.bellek.ornek_oynat(self.replay_n):
                kopya = dataclasses.replace(eski, state=DeneyimDurumu.CANDIDATE)
                self.evaluator.degerlendir(kopya, self.store)

        replay_verimliligi = (self.bellek.replay.verimlilik()
                              if self.bellek is not None else 0.0)

        toplam = len(adaylar) or 1
        return AdimRaporu(
            adim=self._adim,
            uretilen=len(adaylar),
            valid=k_rapor.valid,
            conflict=k_rapor.conflict,
            uncertain=k_rapor.uncertain,
            invalid=k_rapor.invalid,
            verified=k_rapor.verified,
            novel=adim_novel,
            acceptance_rate=round(k_rapor.valid / toplam, 4),
            conflict_rate=round(k_rapor.conflict / toplam, 4),
            false_acceptance=(self._yanlis_kabul if self.dogrulayici is not None else None),
            false_rejection=(self._yanlis_ret if self.dogrulayici is not None else None),
            knowledge_buyumesi=self.store.versiyon - onceki_versiyon,
            replay_verimliligi=replay_verimliligi,
        )

    # ── N adımlık koşu ───────────────────────────────────────────────────
    def calistir(self, n: int, relation_ids: Optional[List[str]] = None,
                 max_aday: Optional[int] = None) -> List[AdimRaporu]:
        raporlar = []
        for _ in range(int(n)):
            raporlar.append(self.adim(relation_ids=relation_ids,
                                      max_aday=max_aday))
        return raporlar
