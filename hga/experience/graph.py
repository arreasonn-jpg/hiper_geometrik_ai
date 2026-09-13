# -*- coding: utf-8 -*-
"""
Experience Graph — Deneyim Grafı ve Soy Takibi (Lineage)
==========================================================
(Roadmap Faz 23 — P0-011, P0-021)

Sistem yalnızca statik bir Knowledge Graph değil, deneyimlerin nasıl üretildiğini,
hangi öncüllerden türetildiğini, neyle çeliştiğini ve nasıl doğrulandığını
kaydeden bir **Experience Graph** oluşturur.

Bağlantı Türleri:
  * ``DERIVED_FROM``  : Deneyimin türetildiği öncül deneyimler veya olgular
  * ``SUPPORTED_BY``  : Deneyimi destekleyen doğrulanmış kanıtlar
  * ``CONTRADICTS``   : Deneyimle çelişen olgu veya diğer deneyimler
  * ``INVERSE_OF``    : Ters yönlü ilişki eşleşmesi
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..knowledge.schemas import ExperienceCandidate, DeneyimDurumu, KaynakTuru


@dataclass
class GraphEdge:
    kaynak_id: str
    hedef_id: str
    iliski_tipi: str      # DERIVED_FROM, SUPPORTED_BY, CONTRADICTS, INVERSE_OF, SUB_PROP
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kaynak_id": self.kaynak_id,
            "hedef_id": self.hedef_id,
            "iliski_tipi": self.iliski_tipi,
            "metadata": self.metadata,
        }


class ExperienceGraph:
    """Deneyim adayları, olgular ve kavramlar arasındaki ilişki ve soy grafı."""

    def __init__(self):
        self.dugumler: Dict[str, Dict[str, Any]] = {}
        self.kenarlar: List[GraphEdge] = []
        self._girdi_kenarlari: Dict[str, List[GraphEdge]] = {}
        self._cikti_kenarlari: Dict[str, List[GraphEdge]] = {}

    def dugum_ekle(self, dugum_id: str, dugum_tipi: str, **ozellikler) -> None:
        """Graf düğümü ekle (entity, relation, fact, experience)."""
        self.dugumler[dugum_id] = {
            "id": dugum_id,
            "tip": dugum_tipi,
            **ozellikler,
        }
        if dugum_id not in self._girdi_kenarlari:
            self._girdi_kenarlari[dugum_id] = []
        if dugum_id not in self._cikti_kenarlari:
            self._cikti_kenarlari[dugum_id] = []

    def kenar_ekle(self, kaynak_id: str, hedef_id: str, iliski_tipi: str,
                   **metadata) -> GraphEdge:
        """İki düğüm arasına yönlü bağ ekle."""
        if kaynak_id not in self.dugumler:
            self.dugum_ekle(kaynak_id, "bilinmeyen")
        if hedef_id not in self.dugumler:
            self.dugum_ekle(hedef_id, "bilinmeyen")

        kenar = GraphEdge(kaynak_id=kaynak_id, hedef_id=hedef_id,
                          iliski_tipi=iliski_tipi, metadata=metadata)
        self.kenarlar.append(kenar)
        self._cikti_kenarlari[kaynak_id].append(kenar)
        self._girdi_kenarlari[hedef_id].append(kenar)
        return kenar

    def deneyim_kaydet(self, aday: ExperienceCandidate) -> None:
        """ExperienceCandidate'i grafa düğüm ve kenarlarıyla ekle."""
        self.dugum_ekle(
            dugum_id=aday.experience_id,
            dugum_tipi="experience",
            subject_id=aday.subject_id,
            relation_id=aday.relation_id,
            object_id=aday.object_id,
            state=aday.state.value,
            source=aday.source.value,
            scores=aday.scores,
            verified_by=aday.verified_by,
            generation_step=aday.generation_step,
        )

        # Temel üçlü bağları
        self.kenar_ekle(aday.experience_id, aday.subject_id, "HAS_SUBJECT")
        self.kenar_ekle(aday.experience_id, aday.relation_id, "HAS_RELATION")
        self.kenar_ekle(aday.experience_id, aday.object_id, "HAS_OBJECT")

        # Soy (Lineage) bağları
        for p_id in aday.parent_experiences:
            self.kenar_ekle(aday.experience_id, p_id, "DERIVED_FROM")

        for d_rule in aday.derived_from:
            self.kenar_ekle(aday.experience_id, d_rule, "DERIVED_BY_RULE")

        for supp_id in aday.supported_by:
            self.kenar_ekle(aday.experience_id, supp_id, "SUPPORTED_BY")

        for conf_id in aday.contradicts:
            self.kenar_ekle(aday.experience_id, conf_id, "CONTRADICTS")

    def lineage(self, experience_id: str, derinlik_siniri: int = 10) -> List[Dict[str, Any]]:
        """Bir deneyimin türetildiği öncül zincirini (lineage) geriye doğru izler."""
        zincir: List[Dict[str, Any]] = []
        ziyaret: Set[str] = set()

        def _dfs(dugum_id: str, d: int):
            if d > derinlik_siniri or dugum_id in ziyaret:
                return
            ziyaret.add(dugum_id)
            dugum = self.dugumler.get(dugum_id, {"id": dugum_id, "tip": "bilinmeyen"})
            zincir.append({"derinlik": d, "dugum": dugum})

            for k in self._cikti_kenarlari.get(dugum_id, []):
                if k.iliski_tipi in ("DERIVED_FROM", "SUPPORTED_BY", "DERIVED_BY_RULE"):
                    _dfs(k.hedef_id, d + 1)

        _dfs(experience_id, 0)
        return zincir

    def aciklama(self, experience_id: str, store: Optional[Any] = None) -> str:
        """Deneyimin neden VALID/INVALID/VERIFIED olduğunu açıklayan metin üretir."""
        if experience_id not in self.dugumler:
            return f"Deneyim bulunamadı: {experience_id}"

        d = self.dugumler[experience_id]
        s_id = d.get("subject_id", "?")
        r_id = d.get("relation_id", "?")
        o_id = d.get("object_id", "?")
        durum = d.get("state", "?")
        kaynak = d.get("source", "?")

        s_ad, r_ad, o_ad = s_id, r_id, o_id
        if store is not None:
            try:
                s_ad = store.entities.getir(s_id).token
                r_ad = store.relations.iliski_al(r_id).token
                o_ad = store.entities.getir(o_id).token
            except Exception:
                pass

        aciklamalar = [
            f"Deneyim [{experience_id}]: ({s_ad} --{r_ad}--> {o_ad})",
            f"  • Durum : {durum}",
            f"  • Kaynak: {kaynak}",
        ]

        if d.get("verified_by"):
            aciklamalar.append(f"  • Doğrulayan Hakem: {d['verified_by']}")

        # Bağlantılar
        girdi = self._cikti_kenarlari.get(experience_id, [])
        parents = [k.hedef_id for k in girdi if k.iliski_tipi == "DERIVED_FROM"]
        if parents:
            aciklamalar.append(f"  • Öncül Deneyimler : {', '.join(parents)}")

        contradictions = [k.hedef_id for k in girdi if k.iliski_tipi == "CONTRADICTS"]
        if contradictions:
            aciklamalar.append(f"  • Çeliştiği Kayıtlar : {', '.join(contradictions)}")

        return "\n".join(aciklamalar)

    def celiski_alt_grafi(self, experience_id: str) -> Dict[str, Any]:
        """Bir deneyim ve onunla çelişen düğümlerin alt grafını döner."""
        dugumler = [self.dugumler.get(experience_id, {})]
        kenarlar = []
        for k in self._cikti_kenarlari.get(experience_id, []):
            if k.iliski_tipi == "CONTRADICTS":
                kenarlar.append(k.to_dict())
                if k.hedef_id in self.dugumler:
                    dugumler.append(self.dugumler[k.hedef_id])
        return {"dugumler": dugumler, "kenarlar": kenarlar}

    def ozet(self) -> Dict[str, int]:
        return {
            "toplam_dugum": len(self.dugumler),
            "toplam_kenar": len(self.kenarlar),
            "deneyim_sayisi": sum(1 for d in self.dugumler.values() if d.get("tip") == "experience"),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dugumler": self.dugumler,
            "kenarlar": [k.to_dict() for k in self.kenarlar],
            "ozet": self.ozet(),
        }


__all__ = ["GraphEdge", "ExperienceGraph"]
