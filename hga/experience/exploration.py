# -*- coding: utf-8 -*-
r"""
Exploration Engine & Active Learning Selector
==============================================
(Roadmap Faz 24 & Faz 25 — P2-036)

Sistem "Ne biliyorum?" sorusundan sonra "Ne bilmiyorum?" ve "Bana en çok
bilgi kazandıracak deneyim hangisi?" sorularını sorar.

Uzay Ayrımı (Faz 24):
  * ``KNOWN``      : Yüksek güvenilirlikte (≥ 0.85) doğrulanmış/kayıtlı bilgi
  * ``UNKNOWN``    : Henüz denenmemiş veya kütüphanede yer almayan kombinasyonlar
  * ``CONFLICTED`` : Çelişki kuyruğundaki veya zıt kanıt barındıran üçlüler
  * ``UNCERTAIN``  : Yetersiz kanıt veya düşük güven taşıyan alanlar

Aktif Öğrenme (Faz 25):
  $$E^* = \arg\max_{E} \left[ w_1 \cdot \text{InfoGain}(E) + w_2 \cdot \text{Novelty}(E) + w_3 \cdot \text{Uncertainty}(E) - w_4 \cdot \text{ConflictPenalty}(E) \right]$$
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..knowledge.schemas import DeneyimDurumu, ExperienceCandidate
from .scoring import Scoring


class EpistemicSpace(str, Enum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    CONFLICTED = "CONFLICTED"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class ExplorationMap:
    known: List[Tuple[str, str, str]] = field(default_factory=list)
    unknown: List[Tuple[str, str, str]] = field(default_factory=list)
    conflicted: List[Tuple[str, str, str]] = field(default_factory=list)
    uncertain: List[Tuple[str, str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "known_count": len(self.known),
            "unknown_count": len(self.unknown),
            "conflicted_count": len(self.conflicted),
            "uncertain_count": len(self.uncertain),
            "toplam_uzay": len(self.known) + len(self.unknown) + len(self.conflicted) + len(self.uncertain),
        }


# Faz 25: Priority(E) varsayılan ağırlıkları TEK KAYNAKTA ve açıkça tanımlı.
# Bu sözlük hem varsayılan değerleri hem de her terimin ne ölçtüğünü taşır;
# "sihirli sabit" bırakmamak içindir. Pozitif terimler toplanır, ceza çıkarılır.
VARSAYILAN_PRIORITY_AGIRLIKLARI: Dict[str, float] = {
    "w_gain": 0.40,               # bilgi kazancı: aday ne kadar ayırt edici?
    "w_novelty": 0.35,            # yenilik: daha önce görülmemişlik
    "w_uncertainty": 0.25,        # belirsizlik: kayıtlı kanıtın azlığı
    "w_conflict_penalty": 0.20,   # ceza: çelişkili adayı geri it
}

PRIORITY_AGIRLIK_ACIKLAMALARI: Dict[str, str] = {
    "w_gain": "information_gain — adayın bilgi tabanındaki diğer varlıklardan "
              "ne kadar farklı olduğu",
    "w_novelty": "novelty — üçlünün daha önce görülmemiş olması",
    "w_uncertainty": "1 - confidence — kayıtlı kanıtın zayıflığı",
    "w_conflict_penalty": "CONFLICT durumundaki adaya uygulanan ceza "
                          "(çıkarılır, eklenmez)",
}


class ExplorationEngine:
    """Bilinmeyen kavram uzayını haritalandırır ve en bilgilendirici deneyimleri seçer.

    Faz 25 — ``Priority(E)`` ağırlıkları:

        Priority(E) = w_gain·InfoGain + w_novelty·Novelty
                      + w_uncertainty·Uncertainty − w_conflict_penalty·Conflict

    Ağırlıklar yapıcıda **açıkça ayarlanabilir** ve doğrulanır: negatif ağırlık
    kabul edilmez (bir sinyali "kötü" saymak istiyorsanız ağırlığı 0 yapın;
    ceza terimi zaten ayrı ve çıkarılır). ``agirliklar()`` ile o anki
    konfigürasyon raporlanabilir — ablasyon deneylerinin girdisi budur.
    """

    def __init__(self, scoring: Optional[Scoring] = None,
                 w_gain: Optional[float] = None,
                 w_novelty: Optional[float] = None,
                 w_uncertainty: Optional[float] = None,
                 w_conflict_penalty: Optional[float] = None,
                 normalize: bool = False):
        self.scoring = scoring or Scoring()
        varsayilan = VARSAYILAN_PRIORITY_AGIRLIKLARI
        secilen = {
            "w_gain": varsayilan["w_gain"] if w_gain is None else w_gain,
            "w_novelty": (varsayilan["w_novelty"]
                          if w_novelty is None else w_novelty),
            "w_uncertainty": (varsayilan["w_uncertainty"]
                              if w_uncertainty is None else w_uncertainty),
            "w_conflict_penalty": (varsayilan["w_conflict_penalty"]
                                   if w_conflict_penalty is None
                                   else w_conflict_penalty),
        }
        for ad, deger in secilen.items():
            sayi = float(deger)
            if sayi < 0.0:
                raise ValueError(
                    f"{ad} negatif olamaz ({sayi}); bir sinyali devre dışı "
                    f"bırakmak için 0.0 kullanın.")
            secilen[ad] = sayi

        # İsteğe bağlı: pozitif terimleri 1'e normalize et. Böylece Priority
        # değerleri farklı ağırlık setleri arasında karşılaştırılabilir olur.
        if normalize:
            pozitif = (secilen["w_gain"] + secilen["w_novelty"]
                       + secilen["w_uncertainty"])
            if pozitif <= 0.0:
                raise ValueError(
                    "normalize=True iken pozitif ağırlıkların toplamı > 0 olmalı")
            for ad in ("w_gain", "w_novelty", "w_uncertainty"):
                secilen[ad] = secilen[ad] / pozitif

        self.w_gain = secilen["w_gain"]
        self.w_novelty = secilen["w_novelty"]
        self.w_uncertainty = secilen["w_uncertainty"]
        self.w_conflict_penalty = secilen["w_conflict_penalty"]
        self.normalize = bool(normalize)

    def agirliklar(self) -> Dict[str, Any]:
        """O anki Priority(E) konfigürasyonu (deney raporlarına yazmak için)."""
        return {
            "w_gain": self.w_gain,
            "w_novelty": self.w_novelty,
            "w_uncertainty": self.w_uncertainty,
            "w_conflict_penalty": self.w_conflict_penalty,
            "normalized": self.normalize,
            "positive_weight_sum": round(
                self.w_gain + self.w_novelty + self.w_uncertainty, 6),
            "descriptions": dict(PRIORITY_AGIRLIK_ACIKLAMALARI),
        }

    def uzay_haritasi(self, store, adaylar: List[ExperienceCandidate]) -> ExplorationMap:
        """Deneyim adaylarını epistemik uzay kategorilerine (KNOWN/UNKNOWN/CONFLICTED/UNCERTAIN) ayırır."""
        harita = ExplorationMap()
        for aday in adaylar:
            uclu = aday.uclusu
            if aday.state == DeneyimDurumu.CONFLICT or len(aday.contradicts) > 0:
                harita.conflicted.append(uclu)
                continue

            agrega = store.relations.olgu_agrega(aday.subject_id, aday.relation_id, aday.object_id)
            if agrega is not None and agrega["confidence"] >= 0.85:
                harita.known.append(uclu)
            elif agrega is not None and agrega["confidence"] < 0.50:
                harita.uncertain.append(uclu)
            else:
                harita.unknown.append(uclu)

        return harita

    def priority_dokumu(self, store, aday: ExperienceCandidate) -> Dict[str, Any]:
        """Priority(E)'nin terim terim dökümü — skorun NEDEN o olduğu görünür.

        Tek bir sayı döndürmek ağırlık ablasyonunu denetlenemez kılar; bu
        yüzden her terim, ağırlığı ve katkısı ayrı raporlanır.
        """
        # Bilinmeyen varlık/ilişki bir HATA değil, epistemik bir durumdur:
        # kanıt yokluğu en yüksek belirsizlik demektir. Ama bunu geniş bir
        # `except Exception` ile yutmak gerçek hataları da gizler; bu yüzden
        # yalnız KeyError yakalanır.
        cozulemedi = False
        try:
            subject = store.entities.getir(aday.subject_id)
            relation = store.relations.iliski_al(aday.relation_id)
            object_ = store.entities.getir(aday.object_id)
        except KeyError:
            cozulemedi = True

        if cozulemedi:
            # Kayıt yoksa aday tamamen yenidir: gain ve novelty azami.
            gain, novelty = 1.0, 1.0
        else:
            score_breakdown = self.scoring.skorla(
                store, aday, subject, relation, object_,
                celiski=(aday.state == DeneyimDurumu.CONFLICT)
            )
            gain = score_breakdown.information_gain
            novelty = score_breakdown.novelty

        agrega = store.relations.olgu_agrega(aday.subject_id, aday.relation_id,
                                             aday.object_id)
        uncertainty = 1.0 if agrega is None else max(0.0, 1.0 - agrega["confidence"])
        conflict_penalty = 1.0 if aday.state == DeneyimDurumu.CONFLICT else 0.0

        terimler = {
            "information_gain": {"value": round(float(gain), 6),
                                 "weight": self.w_gain,
                                 "contribution": round(self.w_gain * gain, 6)},
            "novelty": {"value": round(float(novelty), 6),
                        "weight": self.w_novelty,
                        "contribution": round(self.w_novelty * novelty, 6)},
            "uncertainty": {"value": round(float(uncertainty), 6),
                            "weight": self.w_uncertainty,
                            "contribution": round(self.w_uncertainty * uncertainty, 6)},
            "conflict_penalty": {"value": round(float(conflict_penalty), 6),
                                 "weight": self.w_conflict_penalty,
                                 "contribution": round(
                                     -self.w_conflict_penalty * conflict_penalty, 6)},
        }
        ham = sum(t["contribution"] for t in terimler.values())
        return {
            "experience_id": aday.experience_id,
            "terms": terimler,
            "raw_priority": round(ham, 6),
            "priority": max(0.0, min(1.0, round(ham, 4))),
            "resolved": not cozulemedi,
            "clipped": not (0.0 <= round(ham, 4) <= 1.0),
        }

    def bilgi_kazanci_skoru(self, store, aday: ExperienceCandidate) -> float:
        """Deneyimin Priority(E) skorunu (Faz 25) hesaplar."""
        return self.priority_dokumu(store, aday)["priority"]

    def aktif_ogrenme_sec(self, store, adaylar: List[ExperienceCandidate],
                          k: int = 10) -> List[Tuple[ExperienceCandidate, float]]:
        """Adaylar arasından en fazla bilgi kazandıracak en iyi K tanesini seçer (argmax InfoGain)."""
        puanli = []
        for a in adaylar:
            puan = self.bilgi_kazanci_skoru(store, a)
            puanli.append((a, puan))

        puanli.sort(key=lambda item: item[1], reverse=True)
        return puanli[:max(1, int(k))]


@dataclass
class AgirlikAblasyonu:
    """Faz 25: Priority(E) ağırlıklarının seçime etkisi ölçülebilir mi?

    Her terim tek tek kapatılır (ağırlık 0) ve seçilen ilk-K kümesinin ne kadar
    değiştiği raporlanır. Bir terimi kapatmak seçimi HİÇ değiştirmiyorsa, o
    terim o veri üzerinde işlevsizdir — ağırlığı "ayarlanabilir" olsa bile
    pratikte ölü koddur.
    """

    baseline_weights: Dict[str, Any]
    k: int
    variants: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    findings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        satirlar = ["| Kapatılan terim | Seçim örtüşmesi | Değişen seçim | Etkili mi |",
                    "|---|---:|---:|:--:|"]
        for ad in sorted(self.variants):
            v = self.variants[ad]
            satirlar.append(
                f"| `{ad}` | {v['overlap_ratio']:.3f} | {v['changed']} | "
                f"{'evet' if v['effective'] else 'HAYIR'} |")
        return "\n".join(satirlar)


def agirlik_ablasyonu(store, adaylar: List[ExperienceCandidate],
                      k: int = 10,
                      scoring: Optional[Scoring] = None) -> AgirlikAblasyonu:
    """Her Priority(E) terimini sırayla kapatıp seçim değişimini ölç."""
    if not adaylar:
        raise ValueError("adaylar boş olamaz")
    k = max(1, int(k))
    temel = ExplorationEngine(scoring=scoring)
    temel_secim = [a.experience_id
                   for a, _ in temel.aktif_ogrenme_sec(store, adaylar, k=k)]
    temel_kume = set(temel_secim)

    varyantlar: Dict[str, Dict[str, Any]] = {}
    for terim in ("w_gain", "w_novelty", "w_uncertainty", "w_conflict_penalty"):
        motor = ExplorationEngine(scoring=scoring, **{terim: 0.0})
        secim = [a.experience_id
                 for a, _ in motor.aktif_ogrenme_sec(store, adaylar, k=k)]
        ortak = len(temel_kume & set(secim))
        ortusme = round(ortak / len(temel_kume), 6) if temel_kume else 0.0
        varyantlar[terim] = {
            "weights": motor.agirliklar(),
            "selection": secim,
            "overlap_ratio": ortusme,
            "changed": len(temel_kume) - ortak,
            "effective": ortusme < 1.0,
        }

    etkisiz = [ad for ad, v in varyantlar.items() if not v["effective"]]
    bulgular = [
        f"Temel seçim (ilk {k}) referans alındı; her terim tek tek kapatıldı.",
    ]
    if etkisiz:
        bulgular.append(
            f"UYARI: {etkisiz} terimlerini kapatmak seçimi HİÇ değiştirmedi — "
            f"bu veri üzerinde işlevsizler. Ağırlığın ayarlanabilir olması "
            f"etkili olduğu anlamına gelmez.")
    else:
        bulgular.append(
            "Dört terimin tamamı seçimi değiştiriyor: Priority(E) bileşenleri "
            "bu veri üzerinde gerçekten çalışıyor.")
    return AgirlikAblasyonu(baseline_weights=temel.agirliklar(), k=k,
                            variants=varyantlar, findings=bulgular)


__all__ = ["EpistemicSpace", "ExplorationMap", "ExplorationEngine",
           "AgirlikAblasyonu", "agirlik_ablasyonu",
           "VARSAYILAN_PRIORITY_AGIRLIKLARI",
           "PRIORITY_AGIRLIK_ACIKLAMALARI"]
