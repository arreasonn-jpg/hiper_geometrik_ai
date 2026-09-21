# -*- coding: utf-8 -*-
"""
Experience Generator — Deneyim Üretici
=======================================
(v0.1 — rapor §7)

Generator, modelin mevcut temsil ve belleğinden yeni deneyim ADAYLARI üretir.
İlk sürümde hedef serbest/sınırsız hayal üretimi DEĞİL, KONTROLLÜ kombinasyon
üretimidir (rapor §7):

    Ali + Araba + Binmek kavramlarının birlikte bulunması → yeni aday

Kontrollü üretimin iki modu vardır:

  * `tip_filtresi=True`  (varsayılan): yalnızca ilişkinin izin verdiği
    özne/nesne tiplerine uyan kombinasyonlar üretilir. Böylece sistem
    imkânsız kombinasyonları hayal etmez (combinatorial explosion'a karşı
    ilk kısıt, rapor §21).
  * `tip_filtresi=False` (keşif/diyagnostik): tüm çapraz çarpım üretilir —
    Evaluator'ın INVALID/CONFLICT tespitini test etmek için.

Üretilen her aday `source=MODEL_GENERATED` olarak işaretlenir ve durumu
`CANDIDATE`'tir — Evaluator onayı olmadan belleğe YAZILAMAZ (rapor §9).
"""
from typing import List, Optional

from ..knowledge.schemas import DeneyimDurumu, Entity, ExperienceCandidate, KaynakTuru


class ExperienceGenerator:
    """Mevcut bilgiden kontrollü deneyim adayları üretir."""

    def __init__(self, source_confidence: float = 0.5,
                 tip_filtresi: bool = True):
        self.source_confidence = float(source_confidence)
        self.tip_filtresi = bool(tip_filtresi)
        self._sayac = 0

    # ── Özne/nesne havuzu ────────────────────────────────────────────────
    def _havuz(self, entities: List[Entity], izinli_tipler: List[str],
               gerekli_ozellikler: Optional[dict] = None, store=None
               ) -> List[Entity]:
        """İzinli tipler + zorunlu özellikler (property) filtresi."""
        if izinli_tipler:
            izinli = set(izinli_tipler)
            adaylar = [e for e in entities if e.entity_type in izinli]
        else:
            adaylar = list(entities)
        if gerekli_ozellikler and store is not None:
            filtreli = []
            for e in adaylar:
                uygun = True
                for ad, hedef in gerekli_ozellikler.items():
                    pv = store.properties.al(e.entity_id, ad)
                    if pv is None or abs(pv.deger - hedef) > 1e-6:
                        uygun = False
                        break
                if uygun:
                    filtreli.append(e)
            adaylar = filtreli
        return adaylar
    def _aday(self, subject: Entity, relation_id: str, object_: Entity
              ) -> ExperienceCandidate:
        self._sayac += 1
        return ExperienceCandidate(
            experience_id=f"X_{self._sayac:04d}",
            subject_id=subject.entity_id,
            relation_id=relation_id,
            object_id=object_.entity_id,
            source=KaynakTuru.MODEL_GENERATED,
            source_confidence=self.source_confidence,
            state=DeneyimDurumu.CANDIDATE,
        )

    # ── Üretim ───────────────────────────────────────────────────────────
    def uret(self, store, relation_ids: Optional[List[str]] = None,
             max_aday: Optional[int] = None) -> List[ExperienceCandidate]:
        """Belirtilen ilişkiler (yoksa tümü) için kontrollü adaylar üret.

        Her aday CANDIDATE durumunda ve MODEL_GENERATED kaynaklıdır; bellek
        desteği veya değerlendirme sonucu YAZILMAZ.
        """
        iliskiler = store.relations.iliskiler()
        if relation_ids is not None:
            secilen = set(relation_ids)
            iliskiler = [r for r in iliskiler if r.relation_id in secilen]

        varliklar = store.entities.hepsi()
        adaylar: List[ExperienceCandidate] = []

        for r in iliskiler:
            o_havuzu = self._havuz(varliklar, r.object_types, r.requires_object_props, store) if self.tip_filtresi \
                else varliklar
            s_havuzu = self._havuz(varliklar, r.subject_types, r.requires_subject_props, store) if self.tip_filtresi \
                else varliklar
            for s in s_havuzu:
                for o in o_havuzu:
                    if s.entity_id == o.entity_id:
                        continue  # özne == nesne anlamsız aday
                    adaylar.append(self._aday(s, r.relation_id, o))
                    if max_aday and len(adaylar) >= max_aday:
                        return adaylar
        return adaylar
