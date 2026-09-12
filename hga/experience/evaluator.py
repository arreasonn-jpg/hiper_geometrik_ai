# -*- coding: utf-8 -*-
"""
Experience Evaluator — Değerlendirici + VALID/CONFLICT/INVALID durum makinesi
==============================================================================
(v0.1 — rapor §8, §9, §15, EK-C)

Yeni deneyim doğrudan belleğe YAZILMAZ. Evaluator birden fazla sinyal kullanır
(rapor §8): özellik uyumu, ilişki uyumu, bağlam tutarlılığı, bellek desteği,
yenilik, kaynak güveni ve çelişki. Karar, puanı geçip yalnızca kısıtlara
bakmaz — puan "kanıt", kısıtlar "kural"dır (rapor §16).

Karar ağacı (EK-C ile birebir):

    1. İlişki/varlık bilinmiyor          → INVALID
    2. Özne/nesne tipi izinli değil      → INVALID   (bağlam ihlali)
    3. Gerekli özellik bilinen ve YANLIŞ → INVALID   (deterministik kural ihlali)
    4. Gerekli özellik BİLİNMİYOR        → CONFLICT  (yetersiz kanıt → araştırma)
    5. Kayıtlı kanıt yapısal tahminle ÇELİŞİYOR → CONFLICT
    6. MODEL_GENERATED                   → VALID     (bellek adayı, ASLA VERIFIED değil)
    7. Harici/deterministik kaynak       → puan ≥ verify_eşik → VERIFIED
                                             puan ≥ valid_eşik  → VALID
                                             değilse            → CONFLICT

En kritik güvenlik kuralı (rapor §9, §21): MODEL_GENERATED kaynaklı bir deneyim
OTOMATİK olarak VERIFIED kabul EDİLMEZ — en fazla VALID (bellek adayı) olur.
"""

from typing import Dict, Optional

from ..knowledge.schemas import (Entity, Relation, ExperienceCandidate,
                                 KaynakTuru, DeneyimDurumu)
from .scoring import Scoring

# Varsayılan eşikler (experience_config.yaml ile örtüşür)
VARSAYILAN_ESIKLER = {
    "valid_esik": 0.5,            # bu puanın üstü → VALID (harici kaynak için)
    "verify_esik": 0.7,           # bu puanın üstü + harici kaynak → VERIFIED
    "belirsiz_guven_esik": 0.5,   # özellik güveni bunun altındaysa "yetersiz kanıt"
    "celiski_kanit_esik": 0.6,    # kayıtlı kanıt bu güvenin üstündeyse çelişkiye dikkate alınır
    "celiski_skor_ayrimi": 0.4,   # kanıt skoru ile yapısal tahmin arasındaki açıklık
}


class ExperienceEvaluator:
    """Deneyim adayını değerlendirir; durumunu karar ağacıyla belirler."""

    def __init__(self, scoring: Optional[Scoring] = None,
                 esikler: Optional[Dict[str, float]] = None):
        self.scoring = scoring or Scoring()
        self.esikler = dict(VARSAYILAN_ESIKLER)
        if esikler:
            self.esikler.update(esikler)

    # ── Yardımcılar ──────────────────────────────────────────────────────
    def _coz(self, store, aday):
        """Varlık ve ilişki kayıtlarını çöz; eksikse (None, None, None, hata)."""
        hata = None
        try:
            subject = store.entities.getir(aday.subject_id)
        except KeyError as e:
            subject, hata = None, str(e)
        try:
            object_ = store.entities.getir(aday.object_id)
        except KeyError as e:
            object_, hata = None, str(e)
        try:
            relation = store.relations.iliski_al(aday.relation_id)
        except KeyError as e:
            relation, hata = None, str(e)
        return subject, relation, object_, hata

    @staticmethod
    def _ozellik_yanlis_mi(deger: Optional[float], hedef: float) -> bool:
        """Bilinen özellik hedefle ÇELİŞİYOR mu? (hedef=1 → deger<0.5 yanlış)."""
        if deger is None:
            return False
        if hedef >= 0.5:
            return deger < 0.5
        return deger >= 0.5

    # ── Çelişki tespiti (kayıtlı kanıt vs. yapısal tahmin) ───────────────
    def _kanit_celisiyor_mu(self, store, aday, subject, relation, object_,
                            pc: float) -> bool:
        agrega = store.relations.olgu_agrega(aday.subject_id,
                                             aday.relation_id, aday.object_id)
        if agrega is None or agrega["confidence"] < self.esikler["celiski_kanit_esik"]:
            return False
        yapisal_olumlu = pc >= 0.5
        kanit_olumlu = agrega["score"] >= 0.5
        # kayıtlı kanıt, yapısal kuralların tahminiyle zıt yönde ise çelişki
        return yapisal_olumlu != kanit_olumlu

    # ── Ana değerlendirme ────────────────────────────────────────────────
    def degerlendir(self, aday: ExperienceCandidate, store) -> ExperienceCandidate:
        """Adayı değerlendir; durumunu + skorunu + gerekçelerini YAZAR ve döner.

        Aynı nesne döndürülür (in-place güncellenir) — test ve boru hatları
        için uygundur.
        """
        aday.rationale = []
        aday.evidence = []

        subject, relation, object_, hata = self._coz(store, aday)
        if hata:
            aday.rationale.append(f"Çözülemedi: {hata}")
            aday.state = DeneyimDurumu.INVALID
            aday.scores = {}
            return aday

        pc = self.scoring.property_uyumluluk(store, subject, relation, object_)
        cc = self.scoring.baglam_tutarliligi(subject, relation, object_)

        # ── 1/2. Tip (bağlam) kısıtları ───────────────────────────────────
        if relation.subject_types and subject.entity_type not in relation.subject_types:
            aday.rationale.append(
                f"Özne tipi '{subject.entity_type}', ilişkinin izin verdikleri "
                f"{relation.subject_types} içinde değil → INVALID")
            aday.state = DeneyimDurumu.INVALID
            self._skorla(store, aday, subject, relation, object_, celiski=False)
            return aday

        if relation.object_types and object_.entity_type not in relation.object_types:
            aday.rationale.append(
                f"Nesne tipi '{object_.entity_type}', ilişkinin izin verdikleri "
                f"{relation.object_types} içinde değil → INVALID")
            aday.state = DeneyimDurumu.INVALID
            self._skorla(store, aday, subject, relation, object_, celiski=False)
            return aday

        # ── 3/4. Özellik kısıtları (deterministik kural) ──────────────────
        yetersiz_kanit = False
        for ad, hedef in {**relation.requires_object_props,
                          **relation.requires_subject_props}.items():
            # hedef öznenin mi nesnenin mi olduğunu ayır
            if ad in relation.requires_object_props:
                pv = store.properties.al(object_.entity_id, ad)
                taraf = "nesne"
            else:
                pv = store.properties.al(subject.entity_id, ad)
                taraf = "özne"
            if pv is None:
                yetersiz_kanit = True
                aday.rationale.append(
                    f"'{ad}' özelliği {taraf} için BİLİNMİYOR → araştırma gerekli")
                continue
            if pv.confidence < self.esikler["belirsiz_guven_esik"]:
                yetersiz_kanit = True
                aday.rationale.append(
                    f"'{ad}' özelliğinin güveni {pv.confidence} < belirsizlik eşiği "
                    f"→ araştırma gerekli")
                continue
            if self._ozellik_yanlis_mi(pv.deger, hedef):
                aday.rationale.append(
                    f"Kural ihlali: ilişki '{ad}={hedef}' ister; {taraf} "
                    f"'{object_.token if ad in relation.requires_object_props else subject.token}' "
                    f"için '{ad}={pv.deger}' (kaynak={pv.source.value}) → INVALID")
                aday.state = DeneyimDurumu.INVALID
                self._skorla(store, aday, subject, relation, object_, celiski=False)
                return aday

        # ── 5. Kayıtlı kanıtla çelişki ────────────────────────────────────
        celiski = self._kanit_celisiyor_mu(store, aday, subject, relation, object_, pc)
        if celiski:
            aday.rationale.append(
                "Kayıtlı kanıt, yapısal kuralların tahminiyle ÇELİŞİYOR → CONFLICT")
            store.celiski_logla(aday, "kanit-vs-yapisal-tahmin")
            aday.state = DeneyimDurumu.CONFLICT
            self._skorla(store, aday, subject, relation, object_, celiski=True)
            return aday

        # ── 4b. Yetersiz kanıt → araştırma kuyruğu ────────────────────────
        if yetersiz_kanit:
            aday.rationale.append("Gerekli özellik için yetersiz kanıt → CONFLICT (araştır)")
            store.celiski_logla(aday, "yetersiz-kanit")
            aday.state = DeneyimDurumu.CONFLICT
            self._skorla(store, aday, subject, relation, object_, celiski=False)
            return aday

        # ── 6/7. Kaynak temelli karar ─────────────────────────────────────
        br = self._skorla(store, aday, subject, relation, object_, celiski=False)
        puan = br.weighted

        if aday.source == KaynakTuru.MODEL_GENERATED:
            # Güvenlik kuralı (§9/§21): model üretimi ASLA otomatik VERIFIED olamaz
            aday.rationale.append(
                "Uyumlu ama kaynak MODEL_GENERATED → yalnızca VALID (bellek adayı), "
                "ASLA otomatik VERIFIED değil")
            aday.state = DeneyimDurumu.VALID
            return aday

        if puan >= self.esikler["verify_esik"]:
            aday.rationale.append(
                f"Harici/deterministik kaynak ({aday.source.value}) ve puan "
                f"{puan} ≥ verify eşiği → VERIFIED")
            aday.state = DeneyimDurumu.VERIFIED
        elif puan >= self.esikler["valid_esik"]:
            aday.rationale.append(
                f"Harici kaynak ve puan {puan} ≥ valid eşiği → VALID")
            aday.state = DeneyimDurumu.VALID
        else:
            aday.rationale.append(
                f"Puan {puan} eşiklerin altında → CONFLICT (araştır)")
            aday.state = DeneyimDurumu.CONFLICT
        return aday

    # ── Skoru hesaplayıp adaya yaz (erken dönüş yolları için) ────────────
    def _skorla(self, store, aday, subject, relation, object_, celiski: bool):
        br = self.scoring.skorla(store, aday, subject, relation, object_,
                                 celiski=celiski)
        aday.scores = br.to_dict()
        aday.evidence.append(f"skor={br.weighted}")
        return br
