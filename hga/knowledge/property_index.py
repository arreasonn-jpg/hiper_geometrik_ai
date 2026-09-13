# -*- coding: utf-8 -*-
"""
Property Index — Varlık Özellik Dizini
=======================================
(v0.1 — rapor §5)

Bir varlığın özellikleri ayrı tutulur (Entity kaydına gömülmez; §5'in ana
fikri budur). Örnekler:

    Araba   → tasit=1, hareketli=1, binilebilir=1
    Gökyüzü → mekan=1,  tasit=0,      binilebilir≈0

İlk prototipte değerler 1/0 boolean'dır; kayıt `deger ∈ [0,1]` + `confidence`
taşır, böylece ileride güven/uygunluk değerlerine genişlemek veri modelini
bozmaz (rapor §5). Her özelliğin kendi kaynağı ve güveni vardır (§10).
"""
from typing import Dict, List, Optional, Set

from .schemas import KaynakTuru, PropertyValue


def _dogru_deger(deger) -> float:
    """bool/int/float → [0,1] aralığında float (1/0 boolean desteklenir)."""
    if isinstance(deger, bool):
        return 1.0 if deger else 0.0
    v = float(deger)
    if v < 0.0 or v > 1.0:
        raise ValueError(f"özellik değeri [0,1] dışında: {deger}")
    return v


class PropertyIndex:
    """entity_id → {ozellik_adı → PropertyValue} dizini.

    Aynı özellik farklı kaynaklardan tekrar yazılırsa değer güncellenir ve
    `version` artar (bilgi sürümleme, §21).
    """

    def __init__(self):
        self._ozellikler: Dict[str, Dict[str, PropertyValue]] = {}

    # ── Yazma ────────────────────────────────────────────────────────────
    def koy(self, entity_id: str, ad: str, deger,
            source: KaynakTuru = KaynakTuru.REAL_DATA,
            confidence: float = 1.0) -> PropertyValue:
        """Bir varlığa özellik yaz. Var olan özelliği güncellerse sürüm artar."""
        ad = (ad or "").strip().lower()
        if not ad:
            raise ValueError("özellik adı boş olamaz")
        v = _dogru_deger(deger)
        kume = self._ozellikler.setdefault(entity_id, {})
        onceki = kume.get(ad)
        yeni = PropertyValue(deger=v, confidence=float(confidence),
                             source=source,
                             version=(onceki.version + 1) if onceki else 1)
        kume[ad] = yeni
        return yeni

    # ── Okuma ────────────────────────────────────────────────────────────
    def al(self, entity_id: str, ad: str) -> Optional[PropertyValue]:
        """Özellik değeri; varlık/özellik yoksa None."""
        return self._ozellikler.get(entity_id, {}).get((ad or "").strip().lower())

    def hepsi(self, entity_id: str) -> Dict[str, PropertyValue]:
        """Bir varlığın tüm özellikleri (kopya)."""
        return dict(self._ozellikler.get(entity_id, {}))

    def adlar(self) -> List[str]:
        """Sistemde tanımlı tüm özellik adları (sıralı)."""
        s: set = set()
        for kume in self._ozellikler.values():
            s.update(kume.keys())
        return sorted(s)

    def sahip_olanlar(self, ad: str) -> List[str]:
        """Belirli bir özelliğe sahip varlık ID'leri (sıralı)."""
        return sorted(self.sahip_olanlar_kume(ad))

    def sahip_olanlar_kume(self, ad: str) -> Set[str]:
        """``sahip_olanlar`` ile aynı küme, SIRALAMASIZ.

        Sıcak yollarda (ör. ``Scoring.information_gain``) her çağrıda
        O(N log N) sıralama yapmamak için ayrılmıştır.
        """
        ad = (ad or "").strip().lower()
        return {e for e, kume in self._ozellikler.items() if ad in kume}

    # ── Serileştirme ─────────────────────────────────────────────────────
    def to_dict(self) -> Dict:
        return {e: {a: pv.to_dict() for a, pv in kume.items()}
                for e, kume in self._ozellikler.items()}

    @classmethod
    def from_dict(cls, d: Dict) -> "PropertyIndex":
        idx = cls()
        for e, kume in (d or {}).items():
            idx._ozellikler[e] = {a: PropertyValue.from_dict(pv)
                                  for a, pv in kume.items()}
        return idx

    def __len__(self) -> int:
        return sum(len(k) for k in self._ozellikler.values())
