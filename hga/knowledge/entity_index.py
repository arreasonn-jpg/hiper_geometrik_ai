# -*- coding: utf-8 -*-
"""
Entity Index — Kavramsal Varlık Dizini
=======================================
(v0.1 — rapor §4, EK-A.4)

Her kavrama benzersiz bir `entity_id` atanır. Bu ID, tokenizer'ın token
ID'siyle aynı şey DEĞİLDİR (rapor §4):

  * Tokenizer → dilsel birimler (alt-kelime, kelime)
  * Entity Index → sistemin kavramsal dünyası (Ali, Ata, Araba, Gökyüzü, ...)

ID şeması deterministiktir: kullanıcı bir ID vermezse `E_001`, `E_002`, ...
biçiminde otomatik atanır (önceki sürümlerin tekrar üretilebilirliği için).
"""
from typing import Dict, List, Optional

from .schemas import Entity, KaynakTuru


def _norm(token: str) -> str:
    """Varlık adını arama için sadeleştir (Türkçe karakterler korunur)."""
    return (token or "").strip().lower()


class EntityIndex:
    """Kavramsal varlıkların benzersiz kimlikli dizini.

    entity_id → Entity kaydı eşlemesi tutar; ek olarak token (ad) ile
    arama yapar. Sürüm (`version`) her güncellemede artar — semantic drift
    izleme ve bilgi sürümleme için (rapor §21).
    """

    def __init__(self):
        self._varliklar: Dict[str, Entity] = {}
        self._token_idx: Dict[str, str] = {}   # norm(token) → entity_id
        self._sayac = 0

    # ── Ekleme ───────────────────────────────────────────────────────────
    def ekle(self, token: str, entity_type: str = "kavram",
             entity_id: Optional[str] = None,
             properties: Optional[Dict[str, float]] = None,
             relations: Optional[List[str]] = None,
             source: KaynakTuru = KaynakTuru.REAL_DATA,
             confidence: float = 1.0,
             context_tags: Optional[List[str]] = None) -> Entity:
        """Yeni bir kavramsal varlık ekle (veya var olanı döndür).

        `token` birden çok kez aynı adla eklenirse, mevcut varlık geri döner
        (idempotent) — aynı kavram için iki ayrı ID oluşmaz.
        """
        n = _norm(token)
        if n in self._token_idx:
            return self._varliklar[self._token_idx[n]]

        if entity_id is None:
            self._sayac += 1
            entity_id = f"E_{self._sayac:03d}"
        if entity_id in self._varliklar:
            raise ValueError(f"entity_id zaten kullanımda: {entity_id}")

        varlik = Entity(
            entity_id=entity_id,
            token=token.strip(),
            entity_type=entity_type,
            properties=dict(properties or {}),
            relations=list(relations or []),
            confidence=float(confidence),
            source=source,
            context_tags=list(context_tags or []),
        )
        self._varliklar[entity_id] = varlik
        self._token_idx[n] = entity_id
        return varlik

    # ── Okuma ────────────────────────────────────────────────────────────
    def getir(self, entity_id: str) -> Entity:
        """entity_id → Entity (yoksa KeyError)."""
        if entity_id not in self._varliklar:
            raise KeyError(f"bilinmeyen entity_id: {entity_id}")
        return self._varliklar[entity_id]

    def icerir(self, entity_id: str) -> bool:
        return entity_id in self._varliklar

    def ad_bul(self, token: str) -> Optional[str]:
        """Ad (token) → entity_id; yoksa None."""
        return self._token_idx.get(_norm(token))

    def hepsi(self) -> List[Entity]:
        """Tüm varlıklar (eklenme sırasına göre değil, ID sırasına göre)."""
        return [self._varliklar[k] for k in sorted(self._varliklar)]

    def tip_ile(self, entity_type: str) -> List[Entity]:
        """Belirli bir `entity_type`'a sahip tüm varlıklar."""
        return [e for e in self.hepsi() if e.entity_type == entity_type]

    # ── Güncelleme (sürüm artırımlı) ─────────────────────────────────────
    def guncelle(self, entity_id: str, *, entity_type: Optional[str] = None,
                 confidence: Optional[float] = None,
                 context_tags: Optional[List[str]] = None,
                 status: Optional[str] = None) -> Entity:
        """Var olan bir varlığın meta verisini güncelle; `version` artar."""
        v = self.getir(entity_id)
        if entity_type is not None:
            v.entity_type = entity_type
        if confidence is not None:
            v.confidence = float(confidence)
        if context_tags is not None:
            v.context_tags = list(context_tags)
        if status is not None:
            v.status = status
        v.version += 1
        return v

    # ── Serileştirme (bilgi sürümleme / kalıcılık) ───────────────────────
    def to_dict(self) -> Dict:
        return {"sayac": self._sayac,
                "varliklar": [v.to_dict() for v in self.hepsi()]}

    @classmethod
    def from_dict(cls, d: Dict) -> "EntityIndex":
        idx = cls()
        idx._sayac = int(d.get("sayac", 0))
        for vd in d.get("varliklar", []):
            v = Entity.from_dict(vd)
            idx._varliklar[v.entity_id] = v
            idx._token_idx[_norm(v.token)] = v.entity_id
        return idx

    def __len__(self) -> int:
        return len(self._varliklar)

    def __contains__(self, entity_id: str) -> bool:
        return self.icerir(entity_id)
