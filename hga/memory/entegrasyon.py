# -*- coding: utf-8 -*-
"""
Bellek Entegrasyonu — Replay + Consolidation ↔ Seyrek Bellek
=============================================================
(v0.4 — rapor §19, §22 commit 6)

Deneyim tekrarını (Experience Replay) ve konsolidasyonu seyrek bellekle
birleştirir: değerlendirilen deneyimler

  1. seyrek slotlara hash'lenerek yazılır (`DeneyimSlotlari` — v0.6'da
     `mimari/seyrek_tablo.py`'deki torch tablosuna köprülenecek),
  2. tekrar tamponuna itilir (`DeneyimTekrari`),
  3. gerektiğinde örneklenip yeniden değerlendirmeye sokulur.

Böylece "eski deneyimler yeni bilgi üretimine katkı sağlar" döngüsü
(rapor §12, §20 "Replay efficiency") somut bir boru hattına kavuşur.
"""
from typing import Any, Dict, List, Optional

from .sparse_memory import DeneyimSlotlari
from .replay import DeneyimTekrari


class BellekEntegrasyonu:
    """Seyrek slot deposu + deneyim tekrarı tek çatı altında."""

    def __init__(self, slot_sayisi: int = 4096, replay_kapasitesi: int = 1000,
                 tohum: Optional[int] = None):
        self.slotlar = DeneyimSlotlari(slot_sayisi=slot_sayisi)
        self.replay = DeneyimTekrari(kapasite=replay_kapasitesi, tohum=tohum)

    # ── Yazma ────────────────────────────────────────────────────────────
    def yaz(self, aday) -> int:
        """Deneyimi hem seyrek slotlara hem tekrar tamponuna yaz."""
        adres = self.slotlar.yaz(aday.experience_id, aday.uclusu)
        self.replay.it(aday)
        return adres

    def coklu_yaz(self, adaylar: List[Any]) -> int:
        """Birden çok deneyimi yaz; yazılan aday sayısını döner."""
        for a in adaylar:
            self.yaz(a)
        return len(adaylar)

    # ── Okuma / tekrar ───────────────────────────────────────────────────
    def ornek_oynat(self, n: int = 1) -> List[Any]:
        """Tekrar tamponundan örnekle (yeniden değerlendirme için)."""
        return self.replay.ornekle(n)

    def icerir(self, aday) -> bool:
        return self.slotlar.icerir(aday.experience_id, aday.uclusu)

    # ── Metrikler (rapor §20, §21) ───────────────────────────────────────
    def rapor(self) -> Dict:
        slot = self.slotlar.kapasite()
        rp = self.replay.rapor()
        return {
            "slot_dolu": slot["dolu_slot"],
            "slot_toplam": slot["toplam_slot"],
            "cakisma": slot["cakisma"],
            "cakisma_orani": slot["cakisma_orani"],
            "replay_dolu": rp["dolu"],
            "replay_verimliligi": rp["replay_verimliligi"],
        }
