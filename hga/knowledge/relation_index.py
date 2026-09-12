# -*- coding: utf-8 -*-
"""
Relation Index / Relation Tensor — İlişki Dizini
=================================================
(v0.1 — rapor §6)

Salt kelime puanları yeterli değildir: "Ali + araba + bindi" örneğinde üç
bileşenin de uygun olması, İLİŞKİNİN kendisinin uygun olduğunu kanıtlamaz.
Bu yüzden özne–ilişki–nesne bağlantıları ayrıca temsil edilir (rapor §6).

İlk fiziksel uygulamada bu, dev bir tensör yerine SEYREK bir sözlük + kanıt
listesi ile başlar (§6: "seyrek sözlük veya graph yapısıyla başlatmak daha
doğrudur"). Her üçlü için birden çok kanıt (`RelationFact`) biriktirilir;
`olgu_agrega()` bunları kaynak güvenilirliğine göre birleştirir. Bu yapı
ileride mevcut Kronecker/geometrik katmanla ilişkilendirilebilir (§6).
"""
from typing import Dict, List, Optional, Tuple

from .schemas import Relation, RelationFact, KaynakTuru, KAYNAK_GUVENIRLIGI


class RelationIndex:
    """İlişki tanımları + seyrek (özne, ilişki, nesne) kanıt dizini."""

    def __init__(self):
        self._iliskiler: Dict[str, Relation] = {}
        self._olgular: List[RelationFact] = []
        self._sayac = 0

    # ── İlişki tanımı ────────────────────────────────────────────────────
    def iliski_ekle(self, token: str, relation_id: Optional[str] = None,
                    subject_types: Optional[List[str]] = None,
                    object_types: Optional[List[str]] = None,
                    requires_object_props: Optional[Dict[str, float]] = None,
                    requires_subject_props: Optional[Dict[str, float]] = None,
                    source: KaynakTuru = KaynakTuru.VERIFIED_RULE,
                    confidence: float = 1.0) -> Relation:
        """Yeni bir ilişki tanımla (ör. Binmek: nesne `rideable=1` gerektirir)."""
        if relation_id is None:
            self._sayac += 1
            relation_id = f"R_{self._sayac:03d}"
        if relation_id in self._iliskiler:
            raise ValueError(f"relation_id zaten kullanımda: {relation_id}")
        r = Relation(
            relation_id=relation_id,
            token=token.strip(),
            subject_types=list(subject_types or []),
            object_types=list(object_types or []),
            requires_object_props={k: float(v) for k, v in (requires_object_props or {}).items()},
            requires_subject_props={k: float(v) for k, v in (requires_subject_props or {}).items()},
            source=source,
            confidence=float(confidence),
        )
        self._iliskiler[relation_id] = r
        return r

    def iliski_al(self, relation_id: str) -> Relation:
        if relation_id not in self._iliskiler:
            raise KeyError(f"bilinmeyen relation_id: {relation_id}")
        return self._iliskiler[relation_id]

    def iliskiler(self) -> List[Relation]:
        return [self._iliskiler[k] for k in sorted(self._iliskiler)]

    def icerir(self, relation_id: str) -> bool:
        return relation_id in self._iliskiler

    # ── Kanıt (fact) ekleme ──────────────────────────────────────────────
    def olgu_ekle(self, subject_id: str, relation_id: str, object_id: str,
                  score: float, source: KaynakTuru = KaynakTuru.REAL_DATA,
                  confidence: float = 1.0) -> RelationFact:
        """(özne, ilişki, nesne) üçlüsüne dair bir kanıt kaydet."""
        score = float(score)
        if score < 0.0 or score > 1.0:
            raise ValueError(f"ilişki skoru [0,1] dışında: {score}")
        if relation_id not in self._iliskiler:
            raise KeyError(f"bilinmeyen relation_id: {relation_id}")
        f = RelationFact(subject_id=subject_id, relation_id=relation_id,
                         object_id=object_id, score=score, source=source,
                         confidence=float(confidence))
        self._olgular.append(f)
        return f

    # ── Sorgu ────────────────────────────────────────────────────────────
    def olgular(self, subject_id: Optional[str] = None,
                relation_id: Optional[str] = None,
                object_id: Optional[str] = None) -> List[RelationFact]:
        """Filtrelere uyan tüm kanıtlar (None = filtre yok)."""
        out = []
        for f in self._olgular:
            if subject_id is not None and f.subject_id != subject_id:
                continue
            if relation_id is not None and f.relation_id != relation_id:
                continue
            if object_id is not None and f.object_id != object_id:
                continue
            out.append(f)
        return out

    def olgu_agrega(self, subject_id: str, relation_id: str, object_id: str
                    ) -> Optional[Dict]:
        """Bir üçlünün birleşik kanıt özeti (yoksa None).

        Kaynak güvenilirliğiyle ağırlıklı ortalama skor + kanıt sayısı +
        kaynak çeşitliliği + skor yayılımı (anlaşmazlık tespiti için).
        """
        fs = self.olgular(subject_id, relation_id, object_id)
        if not fs:
            return None
        agir = sum(max(KAYNAK_GUVENIRLIGI.get(f.source, 0.5) * f.confidence, 1e-6)
                   for f in fs)
        skor = sum(f.score * max(KAYNAK_GUVENIRLIGI.get(f.source, 0.5) * f.confidence, 1e-6)
                   for f in fs) / agir
        guven = sum(f.confidence for f in fs) / len(fs)
        skorlar = sorted(f.score for f in fs)
        return {
            "score": round(skor, 4),
            "confidence": round(guven, 4),
            "count": len(fs),
            "min": skorlar[0],
            "max": skorlar[-1],
            "spread": round(skorlar[-1] - skorlar[0], 4),
            "sources": sorted({f.source.value for f in fs}),
        }

    # ── Serileştirme ─────────────────────────────────────────────────────
    def to_dict(self) -> Dict:
        return {
            "sayac": self._sayac,
            "iliskiler": [r.to_dict() for r in self.iliskiler()],
            "olgular": [f.to_dict() for f in self._olgular],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "RelationIndex":
        idx = cls()
        idx._sayac = int(d.get("sayac", 0))
        for rd in d.get("iliskiler", []):
            r = Relation.from_dict(rd)
            idx._iliskiler[r.relation_id] = r
        for fd in d.get("olgular", []):
            idx._olgular.append(RelationFact.from_dict(fd))
        return idx

    def __len__(self) -> int:
        return len(self._iliskiler)
