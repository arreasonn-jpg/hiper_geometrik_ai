# -*- coding: utf-8 -*-
"""
KnowledgeStore — Bilgi Deposu (birleşim kökü)
===============================================
(v0.1 — rapor §3)

EntityIndex + PropertyIndex + RelationIndex'i tek çatı altında toplar ve
bilginin "nereden geldiği / ne kadar güvenilir olduğu" bilgisini koruyarak
yazma işlemlerini yönetir (§10). Ayrıca:

  * `bellek_destegi()` — mevcut bilginin bir üçlüyü ne kadar desteklediği
    (Evaluator'ın `memory_support` sinyali, rapor §8).
  * `celiski_gunlugu` — çelişkilerin kaydı (semantic drift izleme, §21).
  * `versiyon` — bilgi tabanının genel sürümü; her kalıcı yazmada artar.
"""
from typing import Dict, List, Optional

from .entity_index import EntityIndex
from .property_index import PropertyIndex
from .relation_index import RelationIndex
from .schemas import (Entity, Relation, RelationFact, ExperienceCandidate,
                      KaynakTuru, DeneyimDurumu)


class KnowledgeStore:
    """Bilgi katmanının birleşim kökü: üç dizini birden yönetir."""

    def __init__(self):
        self.entities = EntityIndex()
        self.properties = PropertyIndex()
        self.relations = RelationIndex()
        self.celiski_gunlugu: List[Dict] = []   # çelişki kaydı (§21)
        self.versiyon = 1

    # ── Varlık + özellikler ──────────────────────────────────────────────
    def varlik_ekle(self, token: str, entity_type: str = "kavram",
                    properties: Optional[Dict[str, float]] = None,
                    entity_id: Optional[str] = None,
                    ozel_isim: bool = False,
                    source: KaynakTuru = KaynakTuru.REAL_DATA,
                    confidence: float = 1.0,
                    context_tags: Optional[List[str]] = None) -> Entity:
        """Varlık ekle; `properties` tohum değerlerini PropertyIndex'e yaz."""
        v = self.entities.ekle(
            token=token, entity_type=entity_type, entity_id=entity_id,
            is_ozel=ozel_isim, properties=properties, source=source,
            confidence=confidence, context_tags=context_tags)
        for ad, deger in (properties or {}).items():
            self.properties.koy(v.entity_id, ad, deger, source=source,
                                confidence=confidence)
        return v

    def ozellik_koy(self, entity_id: str, ad: str, deger,
                    source: KaynakTuru = KaynakTuru.REAL_DATA,
                    confidence: float = 1.0):
        """Bir varlığa özellik yaz (PropertyIndex yetkili)."""
        self.entities.getir(entity_id)  # varlık var mı?
        return self.properties.koy(entity_id, ad, deger, source=source,
                                   confidence=confidence)

    def ozellik_al(self, entity_id: str, ad: str):
        return self.properties.al(entity_id, ad)

    # ── İlişki tanımı + kanıt ────────────────────────────────────────────
    def iliski_tanimla(self, token: str, relation_id: Optional[str] = None,
                       subject_types: Optional[List[str]] = None,
                       object_types: Optional[List[str]] = None,
                       requires_object_props: Optional[Dict[str, float]] = None,
                       requires_subject_props: Optional[Dict[str, float]] = None,
                       source: KaynakTuru = KaynakTuru.VERIFIED_RULE,
                       confidence: float = 1.0) -> Relation:
        return self.relations.iliski_ekle(
            token=token, relation_id=relation_id,
            subject_types=subject_types, object_types=object_types,
            requires_object_props=requires_object_props,
            requires_subject_props=requires_subject_props,
            source=source, confidence=confidence)

    def olgu_kaydet(self, subject_id: str, relation_id: str, object_id: str,
                    score: float, source: KaynakTuru = KaynakTuru.REAL_DATA,
                    confidence: float = 1.0) -> RelationFact:
        return self.relations.olgu_ekle(subject_id, relation_id, object_id,
                                        score, source=source, confidence=confidence)

    # ── Bellek desteği (§8: memory_support sinyali) ──────────────────────
    def bellek_destegi(self, subject_id: str, relation_id: str,
                       object_id: str) -> float:
        """Mevcut bilgi, (özne, ilişki, nesne) üçlüsünü ne kadar destekliyor?

        1.0 = doğrudan kayıtlı kanıt var; 0.5 = yapısal olarak uyumlu ama
        doğrudan kanıt yok; 0.0 = hiç destek yok (rapor §8'deki sinyallerin
        girdisi olarak kullanılır).
        """
        agrega = self.relations.olgu_agrega(subject_id, relation_id, object_id)
        if agrega is not None:
            # kanıt skoru × kanıt güveni (0..1)
            return round(agrega["score"] * agrega["confidence"], 4)
        # doğrudan kanıt yok: ilişki kısıtları yapısal olarak uyuyorsa yarım destek
        try:
            r = self.relations.iliski_al(relation_id)
        except KeyError:
            return 0.0
        for ad, hedef in r.requires_object_props.items():
            pv = self.properties.al(object_id, ad)
            if pv is not None and abs(pv.deger - hedef) < 0.5:
                return 0.5
        return 0.0

    # ── Kalıcı bilgiye yükseltme (VERIFIED) ──────────────────────────────
    def dogrula(self, aday: ExperienceCandidate, score: float) -> RelationFact:
        """VERIFIED bir deneyimi kalıcı bilgiye yaz; bilgi sürümünü artır.

        İlişki kanıtı VERIFIED_RULE/EXTERNAL kaynakla yazılır ve gerekli
        özellikler pekiştirilir (ör. binilebilir=1). MODEL_GENERATED bu
        yola ASLA girmemelidir (§9/§21 — Evaluator bunu garanti eder).
        """
        self.relations.olgu_ekle(aday.subject_id, aday.relation_id,
                                 aday.object_id, score,
                                 source=KaynakTuru.EXTERNAL_VERIFIED,
                                 confidence=1.0)
        # ilişkinin gerektirdiği nesne özelliklerini pekiştir
        r = self.relations.iliski_al(aday.relation_id)
        for ad, hedef in r.requires_object_props.items():
            pv = self.properties.al(aday.object_id, ad)
            if pv is None or pv.confidence < 1.0:
                self.properties.koy(aday.object_id, ad, hedef,
                                    source=KaynakTuru.EXTERNAL_VERIFIED,
                                    confidence=1.0)
        self.versiyon += 1
        return self.relations.olgular(aday.subject_id, aday.relation_id,
                                      aday.object_id)[-1]

    # ── Çelişki kaydı (§21: contradiction log) ───────────────────────────
    def celiski_logla(self, aday: ExperienceCandidate, neden: str) -> None:
        self.celiski_gunlugu.append({
            "experience_id": aday.experience_id,
            "uclu": list(aday.uclusu),
            "neden": neden,
            "versiyon": self.versiyon,
        })

    # ── Serileştirme ─────────────────────────────────────────────────────
    def to_dict(self) -> Dict:
        return {
            "versiyon": self.versiyon,
            "entities": self.entities.to_dict(),
            "properties": self.properties.to_dict(),
            "relations": self.relations.to_dict(),
            "celiski_gunlugu": self.celiski_gunlugu,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "KnowledgeStore":
        k = cls()
        k.versiyon = int(d.get("versiyon", 1))
        k.entities = EntityIndex.from_dict(d.get("entities", {}))
        k.properties = PropertyIndex.from_dict(d.get("properties", {}))
        k.relations = RelationIndex.from_dict(d.get("relations", {}))
        k.celiski_gunlugu = list(d.get("celiski_gunlugu", []))
        return k

    # ── Rapor (deney metrikleri için özet) ────────────────────────────────
    def ozet(self) -> Dict:
        return {
            "varlik": len(self.entities),
            "ozellik": len(self.properties),
            "iliski": len(self.relations),
            "kanit": len(self.relations.olgular()),
            "versiyon": self.versiyon,
            "celiski": len(self.celiski_gunlugu),
        }
