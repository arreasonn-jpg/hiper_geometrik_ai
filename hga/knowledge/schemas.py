# -*- coding: utf-8 -*-
"""
Şemalar — Entity, Property, Relation ve Experience kayıtları
=============================================================
(v1.0 — Roadmap P0-005, P0-006, P0-007, P0-008, P0-011, P0-014)

Bu modül, sistemin kavramsal dünyasını oluşturan temel veri yapılarını tanımlar.
Tamamen saf Python'dur (torch gerektirmez); tüm yapılar `dataclass` olduğu için
hem okunabilir hem de `asdict` ile serileştirilebilir.

Önemli tasarım kararları (Roadmap v1.0 ile birebir):

  * Entity Index ile Tokenizer ayrıdır (P0-005): `entity_id`, tokenizer'ın token
    ID'si DEĞİLDİR. Tokenizer dilsel birimler, Entity Index kavramsal dünya içindir.
  * Property değerleri [0,1] aralığında sürekli `deger` + `confidence` taşır (P0-006, P0-009).
  * Relation tanımları kısıtları, RelationFact somut üçlü kanıtlarını ve ters çıkarımları (P0-008) taşır.
  * Bilgi yalnızca doğru/yanlış değil, `source` + `confidence` ile tutulur (P0-014):
    MODEL_GENERATED bilgi, REAL_DATA veya VERIFIED ile aynı epistemik seviyeye konmaz.
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────
# Kaynak türleri (P0-014) — bilginin nereden geldiği (Provenance)
# ──────────────────────────────────────────────────────────────────────────
class KaynakTuru(str, Enum):
    REAL_DATA = "REAL_DATA"                 # gerçek dış veri
    VERIFIED_RULE = "VERIFIED_RULE"         # deterministik / doğrulanmış kural
    DERIVED = "DERIVED"                     # mantıksal/ters çıkarımla türetilmiş bilgi (P0-014)
    EXTERNAL_VERIFIED = "EXTERNAL_VERIFIED" # dış doğrulayıcı onayı
    HUMAN_CONFIRMED = "HUMAN_CONFIRMED"     # insan onayı
    MODEL_GENERATED = "MODEL_GENERATED"     # modelin kendi üretimi
    FREE_GENERATION = "FREE_GENERATION"     # serbest/açık üretim (P0-014)


# Kaynak güvenilirliği (P0-014): aynı güven puanı farklı kaynaklarda farklı
# epistemik ağırlık taşır. MODEL_GENERATED bilinçli olarak 0.5'te tutulur:
# kendi üretimini asla dış veri veya doğrulanmış kural ile aynı seviyeye koymaz.
KAYNAK_GUVENIRLIGI: Dict[KaynakTuru, float] = {
    KaynakTuru.REAL_DATA: 1.0,
    KaynakTuru.VERIFIED_RULE: 1.0,
    KaynakTuru.HUMAN_CONFIRMED: 1.0,
    KaynakTuru.EXTERNAL_VERIFIED: 0.9,
    KaynakTuru.DERIVED: 0.85,
    KaynakTuru.MODEL_GENERATED: 0.5,
    KaynakTuru.FREE_GENERATION: 0.3,
}


# ──────────────────────────────────────────────────────────────────────────
# Deneyim durumları (P0-012) — güvenlik mekanizmasının durum makinesi
# ──────────────────────────────────────────────────────────────────────────
class DeneyimDurumu(str, Enum):
    CANDIDATE = "CANDIDATE"    # henüz değerlendirilmedi → Evaluator'a gönder
    EVALUATING = "EVALUATING"  # değerlendirme aşamasında (P0-012)
    VALID = "VALID"            # mevcut bilgiyle uyumlu → belleğe ADAY olarak ekle
    UNCERTAIN = "UNCERTAIN"    # karar için kanıt yetersiz; yanlış/çelişkili demek değildir
    CONFLICT = "CONFLICT"      # mevcut kanıtlar birbiriyle çelişiyor → araştırmaya gönder
    EXPLORE = "EXPLORE"        # çelişki araştırma modunda (CONFLICT sonrası)
    INVALID = "INVALID"        # kural/ilişki/özellik açısından uyumsuz → reddet
    REJECT = "REJECT"          # reddedilerek kapatıldı (INVALID sonrası terminal)
    VERIFYING = "VERIFYING"    # deterministik/dış doğrulama testinde (P0-012)
    VERIFIED = "VERIFIED"      # harici/deterministik doğrulama aldı → kalıcı bilgiye yükselt


# ──────────────────────────────────────────────────────────────────────────
# Entity kaydı (P0-005)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Entity:
    entity_id: str                       # benzersiz kavram kimliği (E_001, ...)
    token: str                           # canonical_name (normalize edilmiş ad)
    entity_type: str = "kavram"          # kavram tipi: insan, hayvan, tasit, mekan, ...
    is_ozel: bool = False                # özel isim mi? (morfoloji: Ali → Ali'ye)
    properties: Dict[str, float] = field(default_factory=dict)  # TOHUM değerler (PropertyIndex yetkili)
    relations: List[str] = field(default_factory=list)          # ilişkili relation_id'ler (tohum)
    confidence: float = 1.0              # varlığın kendisine duyulan güven [0,1]
    source: KaynakTuru = KaynakTuru.REAL_DATA
    context_tags: List[str] = field(default_factory=list)
    version: int = 1                     # sürüm (semantic drift izleme)
    status: str = "aktif"                # aktif / emekli / tartismali

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source"] = self.source.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Entity":
        d = dict(d)
        d["source"] = KaynakTuru(d.get("source", "REAL_DATA"))
        return cls(**d)


# ──────────────────────────────────────────────────────────────────────────
# Property değeri (P0-006, P0-009): [0,1] sürekli epistemik güven değeri
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class PropertyValue:
    deger: float                         # [0,1]; boolean 1/0 olarak da kullanılır
    confidence: float = 1.0              # bu özelliğe duyulan güven [0,1]
    source: KaynakTuru = KaynakTuru.REAL_DATA
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {"deger": self.deger, "confidence": self.confidence,
                "source": self.source.value, "version": self.version}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PropertyValue":
        return cls(deger=float(d["deger"]),
                   confidence=float(d.get("confidence", 1.0)),
                   source=KaynakTuru(d.get("source", "REAL_DATA")),
                   version=int(d.get("version", 1)))


# ──────────────────────────────────────────────────────────────────────────
# Relation (ilişki) tanımı (P0-007, P0-008): özne-ilişki-nesne kısıtları
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Relation:
    relation_id: str                     # R_001, ...
    token: str                           # canonical_name (örn. "binmek")
    subject_types: List[str] = field(default_factory=list)   # boş = serbest
    object_types: List[str] = field(default_factory=list)    # boş = serbest
    requires_object_props: Dict[str, float] = field(default_factory=dict)   # {"rideable": 1.0}
    requires_subject_props: Dict[str, float] = field(default_factory=dict)  # {"canli": 1.0}
    inverse_relation_id: Optional[str] = None                # ters ilişki kimliği (P0-008)
    symmetric: bool = False                                  # simetrik ilişki mi? (A-B => B-A)
    source: KaynakTuru = KaynakTuru.VERIFIED_RULE
    confidence: float = 1.0
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source"] = self.source.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Relation":
        d = dict(d)
        d["source"] = KaynakTuru(d.get("source", "VERIFIED_RULE"))
        return cls(**d)


# ──────────────────────────────────────────────────────────────────────────
# RelationFact (ilişki kanıtı, P0-007): (özne, ilişki, nesne) üçlüsüne dair tek kanıt
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class RelationFact:
    subject_id: str
    relation_id: str
    object_id: str
    score: float                         # üçlünün uygunluğu [0,1]
    source: KaynakTuru = KaynakTuru.REAL_DATA
    confidence: float = 1.0
    timestamp: Optional[float] = None
    version: int = 1

    @property
    def uclusu(self) -> Tuple[str, str, str]:
        return (self.subject_id, self.relation_id, self.object_id)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source"] = self.source.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RelationFact":
        d = dict(d)
        d["source"] = KaynakTuru(d.get("source", "REAL_DATA"))
        return cls(**d)


# ──────────────────────────────────────────────────────────────────────────
# ExperienceCandidate (P0-011, Faz 23): üretilen deneyim adayı ve graf soyu
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class ExperienceCandidate:
    experience_id: str
    subject_id: str
    relation_id: str
    object_id: str
    source: KaynakTuru = KaynakTuru.MODEL_GENERATED
    source_confidence: float = 0.5
    state: DeneyimDurumu = DeneyimDurumu.CANDIDATE
    scores: Dict[str, float] = field(default_factory=dict)
    rationale: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    parent_experiences: List[str] = field(default_factory=list)  # P0-011 / Faz 23: öncül deneyim kimlikleri
    generation_step: int = 0                                     # P0-011: döngü adım numarası
    derived_from: List[str] = field(default_factory=list)        # Faz 23: türetim kuralları/ilişkileri
    contradicts: List[str] = field(default_factory=list)         # Faz 23: çeliştiği olgu/deneyim kimlikleri
    supported_by: List[str] = field(default_factory=list)        # Faz 23: destekleyen olgu/deneyim kimlikleri
    verified_by: Optional[str] = None      # deterministik doğrulayıcı kimliği (VERIFIED ise)
    timestamp: Optional[float] = None
    version: int = 1

    @property
    def uclusu(self) -> Tuple[str, str, str]:
        return (self.subject_id, self.relation_id, self.object_id)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source"] = self.source.value
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperienceCandidate":
        d = dict(d)
        d["source"] = KaynakTuru(d.get("source", "MODEL_GENERATED"))
        d["state"] = DeneyimDurumu(d.get("state", "CANDIDATE"))
        return cls(**d)
