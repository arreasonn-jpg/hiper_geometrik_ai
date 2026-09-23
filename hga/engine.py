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
from .experience.arastirma import ArastirmaKuyrugu
from .experience.consolidation import ConsolidationReport, Consolidator
from .experience.corpus import dosyadan_bilgi_aktar
from .experience.cumle_ayiklayici import CumleAyiklayici, cumlelerden_bilgi_aktar
from .experience.dogrulama import DogrulamaHatti, DogrulamaRaporu
from .experience.evaluator import ExperienceEvaluator
from .experience.exploration import ExplorationEngine, ExplorationMap
from .experience.generator import ExperienceGenerator
from .experience.graph import ExperienceGraph
from .experience.loop import AdimRaporu, DeneyimDongusu
from .experience.scoring import Scoring
from .experience.text_generator import TextGenerator
from .knowledge import KnowledgeStore
from .knowledge.schemas import DeneyimDurumu
from .memory import BellekEntegrasyonu


class ExperienceEngine:
    """Experience Engine'in tek yüzden kullanımı (config + bileşim)."""

    def __init__(self, store: Optional[KnowledgeStore] = None,
                 cfg: Optional[Dict] = None,
                 dogrulayici: Optional[Callable] = None,
                 replay_n: int = 2, slot_sayisi: int = 64,
                 memory_policy: Optional[str] = None,
                 eviction_policy: Optional[str] = None,
                 max_idle_ticks: Optional[int] = None,
                 ozellik_filtresi: bool = True):
        self.cfg = cfg or yukle()
        self.store = store or KnowledgeStore()
        self.replay_n = int(replay_n)

        agirliklar = self.cfg.get("agirliklar", {})
        esikler = self.cfg.get("esikler", {})
        gen_cfg = self.cfg.get("generator", {})
        memory_cfg = self.cfg.get("bellek", {})

        self.scoring = Scoring(agirliklar)
        self.evaluator = ExperienceEvaluator(scoring=self.scoring, esikler=esikler)
        gen_kwargs = dict(gen_cfg)
        gen_kwargs.setdefault("ozellik_filtresi", ozellik_filtresi)
        self.generator = ExperienceGenerator(**gen_kwargs)
        self.consolidator = Consolidator()
        self.text_gen = TextGenerator()
        configured_idle_ticks = memory_cfg.get("max_idle_ticks")
        self.bellek = BellekEntegrasyonu(
            slot_sayisi=slot_sayisi,
            replay_kapasitesi=max(16, slot_sayisi),
            politika=memory_policy or memory_cfg.get("policy", "DYNAMIC_KV"),
            eviction_policy=(
                eviction_policy or memory_cfg.get("eviction_policy", "lru")
            ),
            max_idle_ticks=(
                max_idle_ticks
                if max_idle_ticks is not None
                else configured_idle_ticks
            ),
        )
        self.dogrulayici = dogrulayici
        self.dogrulama = (DogrulamaHatti(dogrulayici, evaluator=self.evaluator,
                                         consolidator=self.consolidator)
                          if dogrulayici else None)
        self.arastirma = ArastirmaKuyrugu(evaluator=self.evaluator,
                                          dogrulayici=dogrulayici)
        self.ayiklayici = CumleAyiklayici()
        self.graph = ExperienceGraph()
        self.exploration = ExplorationEngine(scoring=self.scoring)

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
            self.graph.deneyim_kaydet(a)
        return adaylar

    def konsolide(self, adaylar) -> ConsolidationReport:
        rapor = self.consolidator.konsolide_et(self.store, adaylar)
        for aday in adaylar:
            if aday.state in (DeneyimDurumu.VALID, DeneyimDurumu.VERIFIED):
                self.bellek.yaz(aday)
        return rapor

    def dogrula(self, adaylar) -> Optional[DogrulamaRaporu]:
        """Deterministik doğrulayıcıyla deneyimleri doğrula (yoksa None)."""
        if self.dogrulama is None:
            return None
        rapor = self.dogrulama.isle(self.store, adaylar)
        for a in adaylar:
            self.graph.deneyim_kaydet(a)
            if a.state == DeneyimDurumu.VERIFIED:
                self.bellek.yaz(a)
        return rapor

    def metin(self, aday) -> str:
        return self.text_gen.cumle(self.store, aday)

    # ── Graf & Keşif Yardımcıları (Faz 23, 24, 25) ───────────────────────
    def uzay_haritasi(self, adaylar) -> ExplorationMap:
        return self.exploration.uzay_haritasi(self.store, adaylar)

    def aktif_ogrenme_sec(self, adaylar, k: int = 10):
        return self.exploration.aktif_ogrenme_sec(self.store, adaylar, k=k)

    def lineage(self, experience_id: str):
        return self.graph.lineage(experience_id)

    def aciklama(self, experience_id: str) -> str:
        return self.graph.aciklama(experience_id, store=self.store)

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

    # ── Aktif Dynamic KV yaşam döngüsü ──────────────────────────────────
    def bellekte_mi(self, aday) -> bool:
        """Aday etkin deneyim belleğinde hâlâ tutuluyor mu?"""
        return self.bellek.icerir(aday)

    def bellek_kaydet(self, yol, label: Optional[str] = None):
        return self.bellek.kaydet(yol, label=label)

    def bellek_yukle(self, yol):
        return self.bellek.yukle(yol)

    def bellek_snapshot(self, label: Optional[str] = None):
        return self.bellek.snapshot_olustur(label=label)

    def bellek_snapshot_geri_yukle(self, snapshot):
        return self.bellek.snapshot_geri_yukle(snapshot)

    def bellek_sikistir(self):
        return self.bellek.sikistir()

    def bellek_dynamic_kvye_migre_et(
        self, max_entries: Optional[int] = None, eviction_policy: str = "lru"
    ):
        return self.bellek.dynamic_kvye_migre_et(
            max_entries=max_entries, eviction_policy=eviction_policy
        )

    # ── Özet ─────────────────────────────────────────────────────────────
    def ozet(self) -> Dict:
        return {
            "bilgi": self.store.ozet(),
            "bellek": self.bellek.rapor(),
            "arastirma_kuyrugu": len(self.arastirma),
            "graf": self.graph.ozet(),
        }
