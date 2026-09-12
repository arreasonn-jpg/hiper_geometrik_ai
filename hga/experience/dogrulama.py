# -*- coding: utf-8 -*-
"""
Doğrulama Hattı — MODEL_GENERATED deneyimleri deterministik kanıtla doğrula
============================================================================
(rapor §11, §18, §21)

Benchmark'ın açığa çıkardığı sorunu kapatır: kural tabanlı değerlendirme
yapısal olarak uyumlu her adayı VALID kabul eder; oysa bunların çoğu yanlıştır
(rapor §18: genel dilde objektif environment yoktur, domain-specific
doğrulayıcılar gerekir). `DogrulamaHatti` bunu kurumsallaştırır:

    VALID (MODEL_GENERATED) aday
      → deterministik doğrulayıcı(store, aday) → True/False/None
      → True  : VERIFIED'a yükselt (kaynak EXTERNAL_VERIFIED olur; kanıt
                bağımsızdır, model üretimi DEĞİLDİR)
      → False : INVALID'e düşür (reddet + çelişki günlüğü)
      → None  : VALID kalır (doğrulanamadı — güvenli varsayılan)

GÜVENLİK (rapor §9/§21): yükseltme YALNIZCA bağımsız deterministik kanıtla
olur; doğrulayıcı "belirsiz" dönerse aday asla yükseltilmez. Kaynağın
MODEL_GENERATED'dan EXTERNAL_VERIFIED'a çevrilmesi bilinçlidir — kalıcı bilgiye
yazılan şey artık "modelin üretimi" değil, "doğrulayıcının onayladığı bilgi"dir
(kanıt zinciri `evidence` alanında saklanır).

Aritmetik alanında sonuç ölçülebilirdir: 30 eşitlik adayından 6'sı doğrulanır
(VERIFIED), 24'ü çürütülür (INVALID) → false acceptance 24'ten 0'a düşer.
"""
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional

from ..knowledge.schemas import (ExperienceCandidate, DeneyimDurumu,
                                 KaynakTuru)
from .evaluator import ExperienceEvaluator
from .consolidation import Consolidator


@dataclass
class DogrulamaRaporu:
    islenen: int = 0
    dogrulanan: int = 0            # → VERIFIED (kalıcı bilgiye yükseltildi)
    reddedilen: int = 0            # → INVALID (çürütüldü)
    belirsiz: int = 0              # → VALID kaldı (doğrulanamadı)
    bilgi_buyumesi: int = 0
    yanlis_kabul_oncesi: int = 0   # doğrulamadan ÖNCE yanlış kabul sayısı
    yanlis_kabul_sonrasi: int = 0  # doğrulamadan SONRA yanlış kabul sayısı
    dogrulananlar: List[str] = field(default_factory=list)
    reddedilenler: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class DogrulamaHatti:
    """Bağımsız deterministik doğrulayıcıyla deneyimleri doğrular."""

    def __init__(self, dogrulayici: Callable,
                 evaluator: Optional[ExperienceEvaluator] = None,
                 consolidator: Optional[Consolidator] = None,
                 dogrulayici_adi: str = "deterministik-dogrulayici"):
        self.dogrulayici = dogrulayici
        self.evaluator = evaluator or ExperienceEvaluator()
        self.consolidator = consolidator or Consolidator()
        self.dogrulayici_adi = dogrulayici_adi

    # ── Tek adayı doğrula ────────────────────────────────────────────────
    def _aday_dogrula(self, store, a: ExperienceCandidate) -> Optional[bool]:
        """Doğrulayıcıyı çağır; sonucu adayın durumuna yansıtır."""
        onceki = a.state
        sonuc = self.dogrulayici(store, a)
        if sonuc is True:
            a.state = DeneyimDurumu.VERIFIED
            a.source = KaynakTuru.EXTERNAL_VERIFIED   # bağımsız kanıt
            a.source_confidence = 1.0
            a.verified_by = self.dogrulayici_adi
            a.evidence.append(
                f"kaynak {onceki} → EXTERNAL_VERIFIED (deterministik onay)")
        elif sonuc is False:
            a.state = DeneyimDurumu.INVALID
            a.evidence.append("deterministik doğrulayıcı çürüttü")
            store.celiski_logla(a, "deterministik-curutme")
        # sonuc is None → durum değişmez (VALID kalır)
        return sonuc

    # ── Toplu doğrulama + konsolidasyon ──────────────────────────────────
    def isle(self, store, adaylar: List[ExperienceCandidate]) -> DogrulamaRaporu:
        """Adayları değerlendir, doğrula ve konsolide et.

        false acceptance "öncesi/sonrası", doğrulayıcının kendisi ground-truth
        kabul edilerek hesaplanır (aritmetik alanında doğrulayıcı kesindir).
        """
        for a in adaylar:
            if a.state == DeneyimDurumu.CANDIDATE:
                self.evaluator.degerlendir(a, store)

        rapor = DogrulamaRaporu(islenen=len(adaylar))
        for a in adaylar:
            if not (a.state == DeneyimDurumu.VALID and
                    a.source == KaynakTuru.MODEL_GENERATED):
                continue
            sonuc = self._aday_dogrula(store, a)
            if sonuc is False:
                # yanlış ama kabul edilmişti → doğrulama öncesi yanlış kabul
                rapor.yanlis_kabul_oncesi += 1
                rapor.reddedilen += 1
                rapor.reddedilenler.append(a.experience_id)
            elif sonuc is True:
                rapor.dogrulanan += 1
                rapor.dogrulananlar.append(a.experience_id)
            else:
                rapor.belirsiz += 1

        # doğrulama sonrası: hâlâ VALID/VERIFIED olup doğrulayıcının False dediği
        for a in adaylar:
            if a.state in (DeneyimDurumu.VALID, DeneyimDurumu.VERIFIED):
                if self.dogrulayici(store, a) is False:
                    rapor.yanlis_kabul_sonrasi += 1

        onceki_versiyon = store.versiyon
        self.consolidator.konsolide_et(store, adaylar)
        rapor.bilgi_buyumesi = store.versiyon - onceki_versiyon
        return rapor
