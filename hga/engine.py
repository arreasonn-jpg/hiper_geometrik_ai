# -*- coding: utf-8 -*-
"""
ExperienceEngine — Tüm Katmanın Tek Yüzden Orkestrasyonu
===========================================================
(v1.0 — rapor §12, §19)

Knowledge + Experience + Memory + Config parçalarını tek bir yapılandırılabilir
motor olarak birleştirir. Demoların ve CLI'nin tek dayanağıdır; bilimsel/üretim
kodları alt bileşenleri doğrudan kullanmaya devam edebilir.

Kurulum:
    engine = ExperienceEngine()                         # varsayılan config
    engine.gercek_veri(["Ali ataya bindi.", ...])        # REAL_DATA aktarımı
    adaylar = engine.uret(["R_001"])                     # kontrollü üretim
    engine.degerlendir(adaylar)                          # durum makinesi
    rapor = engine.konsolide(adaylar)                    # belleğe/redde
    dogrulama = engine.dogrula(adaylar)                  # deterministik kanıtla

Döngü (tek çağrı):
    raporlar = engine.dongu(n=3, relation_ids=["R_001"])
"""
from typing import Callable, Dict, List, Optional

from .config import yukle
from .knowledge import KnowledgeStore
from .memory import BellekEntegrasyonu
from .experience.scoring import Scoring
from .experience.generator import ExperienceGenerator
from .experience.evaluator import ExperienceEvaluator
from .experience.consolidation import Consolidator, ConsolidationReport
from .experience.arastirma import ArastirmaKuyrugu
from .experience.cumle_ayiklayici import CumleAyiklayici, cumlelerden_bilgi_aktar
from .experience.corpus import dosyadan_bilgi_aktar
from .experience.text_generator import TextGenerator
from .experience.dogrulama import DogrulamaHatti, DogrulamaRaporu
from .experience.loop import DeneyimDongusu, AdimRaporu


class ExperienceEngine:
    """Experience Engine'in tek yüzden kullanımı (config + bileşim)."""

    def __init__(self, store: Optional[KnowledgeStore] = None,
                 cfg: Optional[Dict] = None,
                 dogrulayici: Optional[Callable] = None,
                 replay_n: int = 2, slot_sayisi: int = 64):
        self.cfg = cfg or yukle()
        self.store = store or KnowledgeStore()
        self.replay_n = int(replay_n)

        agirliklar = self.cfg.get("agirliklar", {})
        esikler = self.cfg.get("esikler", {})
        gen_cfg = self.cfg.get("generator", {})

        self.scoring = Scoring(agirliklar)
        self.evaluator = ExperienceEvaluator(scoring=self.scoring, esikler=esikler)
        self.generator = ExperienceGenerator(**gen_cfg)
        self.consolidator = Consolidator()
        self.text_gen = TextGenerator()
        self.bellek = BellekEntegrasyonu(slot_sayisi=slot_sayisi,
                                         replay_kapasitesi=max(16, slot_sayisi))
        self.dogrulayici = dogrulayici
        self.dogrulama = (DogrulamaHatti(dogrulayici, evaluator=self.evaluator,
                                         consolidator=self.consolidator)
                          if dogrulayici else None)
        self.arastirma = ArastirmaKuyrugu(evaluator=self.evaluator,
                                          dogrulayici=dogrulayici)
        self.ayiklayici = CumleAyiklayici()

    # ── Gerçek veri girişi ───────────────────────────────────────────────
    def gercek_veri(self, cumleler: List[str],
                    iliski_kisitlari: Optional[Dict] = None):
        """Cümleleri üçlüye ayırıp REAL_DATA olarak aktarır."""
        return cumlelerden_bilgi_aktar(self.store, cumleler,
                                       ayiklayici=self.ayiklayici,
                                       iliski_kisitlari=iliski_kisitlari)

    def dosya_yukle(self, yol: str, iliski_kisitlari: Optional[Dict] = None):
        """Metin dosyasından cümleleri REAL_DATA olarak aktarır."""
        return dosyadan_bilgi_aktar(self.store, yol,
                                    ayiklayici=self.ayiklayici,
                                    iliski_kisitlari=iliski_kisitlari)

    # ── Üretim / değerlendirme / konsolidasyon ───────────────────────────
    def uret(self, relation_ids: Optional[List[str]] = None,
             max_aday: Optional[int] = None):
        return self.generator.uret(self.store, relation_ids=relation_ids,
                                   max_aday=max_aday)

    def degerlendir(self, adaylar):
        for a in adaylar:
            self.evaluator.degerlendir(a, self.store)
        return adaylar

    def konsolide(self, adaylar) -> ConsolidationReport:
        return self.consolidator.konsolide_et(self.store, adaylar)

    def dogrula(self, adaylar) -> Optional[DogrulamaRaporu]:
        """Deterministik doğrulayıcıyla deneyimleri doğrula (yoksa None)."""
        if self.dogrulama is None:
            return None
        return self.dogrulama.isle(self.store, adaylar)

    def metin(self, aday) -> str:
        return self.text_gen.cumle(self.store, aday)

    # ── Sürekli öğrenme döngüsü ──────────────────────────────────────────
    def dongu(self, n: int = 1, relation_ids: Optional[List[str]] = None
              ) -> List[AdimRaporu]:
        """Üret → değerlendir → konsolide → bellek → replay döngüsü (N adım)."""
        d = DeneyimDongusu(store=self.store, generator=self.generator,
                           evaluator=self.evaluator,
                           consolidator=self.consolidator,
                           bellek=self.bellek,
                           text_gen=self.text_gen,
                           dogrulayici=self.dogrulayici,
                           replay_n=self.replay_n)
        return d.calistir(int(n), relation_ids=relation_ids)

    # ── Özet ─────────────────────────────────────────────────────────────
    def ozet(self) -> Dict:
        return {
            "bilgi": self.store.ozet(),
            "bellek": self.bellek.rapor(),
            "arastirma_kuyrugu": len(self.arastirma),
        }
