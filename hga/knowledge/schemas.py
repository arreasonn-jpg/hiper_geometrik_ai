# -*- coding: utf-8 -*-
"""
Şemalar — Entity, Property, Relation ve Experience kayıtları
=============================================================
(v0.1 — rapor §3, §4, §5, §6, §10, EK-A)

Bu modül, sistemin kavramsal dünyasını oluşturan temel veri yapılarını tanımlar.
Tamamen saf Python'dur (torch gerektirmez); tüm yapılar `dataclass` olduğu için
hem okunabilir hem de `asdict` ile serileştirilebilir.

Önemli tasarım kararları (raporla birebir):

  * Entity Index ile Tokenizer ayrıdır (§4): `entity_id`, tokenizer'ın token
    ID'si DEĞİLDİR. Tokenizer dilsel birimler, Entity Index kavramsal dünya içindir.
  * Property değerleri ilk prototipte 1/0 booleandır, ama kayıt [0,1] aralığında
    `deger` + `confidence` taşır (§5): böylece ileride güven/uygunluk değerlerine
    genişlemek için veri modeli değişmez.
  * Bilgi yalnızca doğru/yanlış değil, `source` + `confidence` ile tutulur (§10):
    MODEL_GENERATED bilgi, REAL_DATA ile aynı epistemik seviyeye konmaz.
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────
# Kaynak türleri (§10 / EK-A.10) — bilginin nereden geldiği
# ──────────────────────────────────────────────────────────────────────────
class KaynakTuru(str, Enum):
    REAL_DATA = "REAL_DATA"                 # gerçek dış veri
    VERIFIED_RULE = "VERIFIED_RULE"         # deterministik / doğrulanmış kural
    MODEL_GENERATED = "MODEL_GENERATED"     # modelin kendi üretimi
    EXTERNAL_VERIFIED = "EXTERNAL_VERIFIED" # dış doğrulayıcı onayı
    HUMAN_CONFIRMED = "HUMAN_CONFIRMED"     # insan onayı


# Kaynak güvenilirliği (§10): aynı güven puanı farklı kaynaklarda farklı
# epistemik ağırlık taşır. MODEL_GENERATED bilinçli olarak 0.5'te tutulur:
# kendi üretimini asla dış veriyle aynı seviyeye koymaz.
KAYNAK_GUVENIRLIGI: Dict[KaynakTuru, float] = {
    KaynakTuru.REAL_DATA: 1.0,
    KaynakTuru.VERIFIED_RULE: 1.0,
    KaynakTuru.HUMAN_CONFIRMED: 1.0,
    KaynakTuru.EXTERNAL_VERIFIED: 0.9,
    KaynakTuru.MODEL_GENERATED: 0.5,
}


# ──────────────────────────────────────────────────────────────────────────
# Deneyim durumları (EK-C) — güvenlik mekanizmasının kalbi (§9)
# ──────────────────────────────────────────────────────────────────────────
class DeneyimDurumu(str, Enum):
    CANDIDATE = "CANDIDATE"    # henüz değerlendirilmedi → Evaluator'a gönder
    VALID = "VALID"            # mevcut bilgiyle uyumlu → belleğe ADAY olarak ekle
    CONFLICT = "CONFLICT"      # mevcut bilgiyle çelişiyor → araştırma kuyruğuna gönder
    EXPLORE = "EXPLORE"        # çelişki araştırma modunda (CONFLICT sonrası)
    INVALID = "INVALID"        # kural/ilişki/özellik açısından uyumsuz → reddet
    REJECT = "REJECT"          # reddedilerek kapatıldı (INVALID sonrası terminal)
    VERIFIED = "VERIFIED"      # harici/deterministik doğrulama aldı → kalıcı bilgiye yükselt


# ──────────────────────────────────────────────────────────────────────────
# Entity kaydı (§3, EK-A.3)
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
    version: int = 1                     # sürüm (semantic drift izleme, §21)
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
# Property değeri (§5): 1/0 boolean'dan [0,1] güven değerine genişletilebilir
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
# Relation (ilişki) tanımı (§6): özne-ilişki-nesne kısıtları
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Relation:
    relation_id: str                     # R_001, ...
    token: str                           # canonical_name (örn. "binmek")
    subject_types: List[str] = field(default_factory=list)   # boş = serbest
    object_types: List[str] = field(default_factory=list)    # boş = serbest
    requires_object_props: Dict[str, float] = field(default_factory=dict)   # {"rideable": 1.0}
    requires_subject_props: Dict[str, float] = field(default_factory=dict)  # {"canli": 1.0}
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
# RelationFact (ilişki kanıtı): (özne, ilişki, nesne) üçlüsüne dair tek kanıt
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
# ExperienceCandidate (§7): üretilen deneyim adayı
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
