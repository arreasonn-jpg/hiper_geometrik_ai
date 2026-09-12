# -*- coding: utf-8 -*-
"""
Conflict → Exploration — Çelişkiyi Araştırma Sinyaline Çevirme
===============================================================
(v0.1 — rapor §11, EK-A.11)

Çelişen bir deneyimi çöpe atmak yerine ARAŞTIRMA sinyali olarak kullanmak,
sistemin aktif keşif mekanizmasının başlangıcıdır (rapor §11). Döngü:

    CONFLICT
      ↓ Neden çelişiyor?
      ↓ Alternatif açıklamalar/deneyimler üret
      ↓ Kanıt veya deterministik test ara
      ↓ Bilgi güvenini güncelle
      ↓ İlk deneyimi yeniden değerlendir

v0.1'de "deterministik test" = kural tabanlı doğrulayıcılar (rapor §18:
genel dilde objektif environment yoktur; domain-specific doğrulayıcılar
gerekir). Bu sınıf, CONFLICT durumundaki bir aday için alternatifler üretir,
her alternatifi deterministik kurallarla sınar ve orijinal adayı yeniden
değerlendirir. MODEL_GENERATED adaylar burada da asla VERIFIED'a terfi etmez.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..knowledge.schemas import ExperienceCandidate, DeneyimDurumu, KaynakTuru
from .evaluator import ExperienceEvaluator


@dataclass
class ConflictResolution:
    aday: ExperienceCandidate
    nedenler: List[str] = field(default_factory=list)
    alternatifler: List[ExperienceCandidate] = field(default_factory=list)
    test_sonuclari: List[str] = field(default_factory=list)
    yeniden_degerlendirildi: bool = False
    son_durum: Optional[DeneyimDurumu] = None

    def to_dict(self) -> Dict:
        return {
            "experience_id": self.aday.experience_id,
            "nedenler": self.nedenler,
            "alternatifler": [a.uclusu for a in self.alternatifler],
            "test_sonuclari": self.test_sonuclari,
            "yeniden_degerlendirildi": self.yeniden_degerlendirildi,
            "son_durum": self.son_durum.value if self.son_durum else None,
        }


class ConflictResolver:
    """CONFLICT adaylarını araştırır: alternatif üret → deterministik test →
    güven güncelle → yeniden değerlendir."""

    def __init__(self, evaluator: Optional[ExperienceEvaluator] = None,
                 deterministik_test: Optional[Callable] = None):
        self.evaluator = evaluator or ExperienceEvaluator()
        # domain-specific doğrulayıcı: (store, aday) -> Optional[bool]
        # True = doğrulandı, False = çürütüldü, None = belirlenemedi
        self.deterministik_test = deterministik_test

    # ── Varsayılan deterministik test ─────────────────────────────────────
    @staticmethod
    def varsayilan_test(store, aday: ExperienceCandidate) -> Optional[bool]:
        """Kural tabanlı doğrulayıcı: ilişkinin özellik gereksinimini nesne
        üzerinde deterministik olarak sınar. Bilinmeyen → None."""
        try:
            r = store.relations.iliski_al(aday.relation_id)
        except KeyError:
            return None
        for ad, hedef in r.requires_object_props.items():
            pv = store.properties.al(aday.object_id, ad)
            if pv is None:
                return None
            if hedef >= 0.5:
                return pv.deger >= 0.5
            return pv.deger < 0.5
        return True

    # ── Alternatif açıklama/deneyim üretimi ───────────────────────────────
    def alternatif_uret(self, store, aday: ExperienceCandidate
                        ) -> List[ExperienceCandidate]:
        """Çelişen nesneyi, ilişkinin gerektirdiği özellikleri taşıyan diğer
        varlıklarla değiştirerek alternatif deneyimler üretir (keşif)."""
        try:
            r = store.relations.iliski_al(aday.relation_id)
            object_ = store.entities.getir(aday.object_id)
        except KeyError:
            return []

        alternatifler: List[ExperienceCandidate] = []
        for e in store.entities.hepsi():
            if e.entity_id == aday.object_id:
                continue
            # ilişkinin istediği özelliklere sahip olanlar adaydır
            uygun = True
            for ad, hedef in r.requires_object_props.items():
                pv = store.properties.al(e.entity_id, ad)
                if pv is None or (hedef >= 0.5) != (pv.deger >= 0.5):
                    uygun = False
                    break
            if uygun:
                alternatifler.append(ExperienceCandidate(
                    experience_id=f"{aday.experience_id}.alt{len(alternatifler) + 1}",
                    subject_id=aday.subject_id,
                    relation_id=aday.relation_id,
                    object_id=e.entity_id,
                    source=KaynakTuru.MODEL_GENERATED,
                    source_confidence=aday.source_confidence,
                ))
        return alternatifler

    # ── Güven güncelleme (deterministik test kanıt üretirse) ─────────────
    def guven_guncelle(self, store, aday: ExperienceCandidate, sonuc: Optional[bool]
                       ) -> str:
        """Test deterministik sonuç verdiyse, nesnenin ilgili özelliklerinin
        güvenini günceller (bilgi güveni güncelleme adımı, §11)."""
        if sonuc is None:
            return "Deterministik sonuç yok → güven güncellenmedi"
        try:
            r = store.relations.iliski_al(aday.relation_id)
        except KeyError:
            return "İlişki bulunamadı → güven güncellenmedi"
        for ad, hedef in r.requires_object_props.items():
            pv = store.properties.al(aday.object_id, ad)
            if pv is None:
                store.properties.koy(aday.object_id, ad,
                                     float(sonuc),
                                     source=KaynakTuru.VERIFIED_RULE,
                                     confidence=1.0)
                return f"'{ad}' özelliği kanıtla eklendi (={float(sonuc)})"
            store.properties.koy(aday.object_id, ad, float(sonuc),
                                 source=KaynakTuru.VERIFIED_RULE, confidence=1.0)
            return f"'{ad}' özelliği kanıtla güncellendi (={float(sonuc)})"
        return "Güncellenecek özellik yok"

    # ── Ana döngü ─────────────────────────────────────────────────────────
    def coz(self, store, aday: ExperienceCandidate) -> ConflictResolution:
        """CONFLICT bir adayı araştır ve yeniden değerlendir (rapor §11)."""
        nedenler = list(aday.rationale)
        test = self.deterministik_test or self.varsayilan_test

        # 1. Neden çelişiyor? → alternatifler üret
        alternatifler = self.alternatif_uret(store, aday)

        # 2. Kanıt / deterministik test
        sonuc = test(store, aday)
        test_notu = (f"deterministik test → {sonuc}"
                     if sonuc is not None else "deterministik test → belirsiz")
        test_sonuclari = [test_notu]
        for alt in alternatifler:
            r = test(store, alt)
            test_sonuclari.append(
                f"{alt.object_id}: deterministik test → {r}")

        # 3. Bilgi güvenini güncelle
        guncelleme = self.guven_guncelle(store, aday, sonuc)
        test_sonuclari.append(guncelleme)

        # 4. İlk deneyimi yeniden değerlendir
        self.evaluator.degerlendir(aday, store)

        return ConflictResolution(
            aday=aday,
            nedenler=nedenler,
            alternatifler=alternatifler,
            test_sonuclari=test_sonuclari,
            yeniden_degerlendirildi=True,
            son_durum=aday.state,
        )
