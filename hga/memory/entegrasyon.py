# -*- coding: utf-8 -*-
"""Aktif Engine belleği: Dynamic KV + tutarlı replay entegrasyonu.

Varsayılan politika ``DYNAMIC_KV``'dir. Eski ``FIRST_WINS`` slot deposu açık
bir legacy seçeneği olarak korunur ve kayıp/çakışma muhasebesiyle migrate
edilebilir.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .dynamic_kv import (
    CompactionReport,
    DynamicKVMemory,
    DynamicKVSnapshot,
    MigrationReport,
)
from .replay import DeneyimTekrari
from .sparse_memory import DeneyimSlotlari

DYNAMIC_KV = "DYNAMIC_KV"
FIRST_WINS = "FIRST_WINS"


class BellekEntegrasyonu:
    """Dynamic KV/legacy slot deposu + experience replay tek çatısı."""

    def __init__(
        self,
        slot_sayisi: int = 4096,
        replay_kapasitesi: int = 1000,
        tohum: Optional[int] = None,
        politika: str = DYNAMIC_KV,
        eviction_policy: str = "lru",
        max_idle_ticks: Optional[int] = None,
    ):
        self.politika = str(politika).upper()
        self.replay = DeneyimTekrari(kapasite=replay_kapasitesi, tohum=tohum)
        if self.politika == DYNAMIC_KV:
            self.slotlar = DynamicKVMemory(
                max_entries=int(slot_sayisi),
                eviction_policy=eviction_policy,
                max_idle_ticks=max_idle_ticks,
            )
        elif self.politika == FIRST_WINS:
            self.slotlar = DeneyimSlotlari(slot_sayisi=slot_sayisi)
        else:
            raise ValueError("bellek politikası DYNAMIC_KV veya FIRST_WINS olmalı")

    @property
    def dynamic_kv(self) -> DynamicKVMemory:
        if not isinstance(self.slotlar, DynamicKVMemory):
            raise RuntimeError("Aktif bellek politikası DYNAMIC_KV değil")
        return self.slotlar

    def yaz(self, aday) -> int:
        """Deneyimi aktif depoya ve stale kayıt bırakmadan replay'e yaz."""
        if isinstance(self.slotlar, DynamicKVMemory):
            payload = aday.to_dict() if hasattr(aday, "to_dict") else None
            adres = self.slotlar.yaz(
                aday.experience_id, aday.uclusu, payload=payload
            )
            stale_keys = [aday.uclusu]
            stale_keys.extend(event.key for event in self.slotlar.last_evictions)
            self.replay.anahtar_sil(stale_keys)
            self.replay.it(aday)
            return adres
        adres = self.slotlar.yaz(aday.experience_id, aday.uclusu)
        self.replay.it(aday)
        return adres

    def coklu_yaz(self, adaylar: List[Any]) -> int:
        for aday in adaylar:
            self.yaz(aday)
        return len(adaylar)

    def ornek_oynat(self, n: int = 1) -> List[Any]:
        return self.replay.ornekle(n)

    def icerir(self, aday) -> bool:
        return self.slotlar.icerir(aday.experience_id, aday.uclusu)

    def kaydet(self, yol, label: Optional[str] = None) -> DynamicKVSnapshot:
        return self.dynamic_kv.kaydet(yol, label=label)

    def yukle(self, yol) -> DynamicKVSnapshot:
        """Kalıcı snapshot'ı doğrula, etkin belleği ve replay'i atomikçe değiştir."""
        memory = DynamicKVMemory.yukle(yol)
        snapshot = memory.last_snapshot
        if snapshot is None:  # defensive: yukle her zaman snapshot kurmalıdır
            raise ValueError("Yüklenen Dynamic KV snapshot metadata'sı eksik")
        self.slotlar = memory
        self.politika = DYNAMIC_KV
        self._replayi_kayitlardan_yenile()
        return snapshot

    def snapshot_olustur(self, label: Optional[str] = None) -> DynamicKVSnapshot:
        return self.dynamic_kv.snapshot_olustur(label=label)

    def snapshot_geri_yukle(self, snapshot: DynamicKVSnapshot) -> int:
        version = self.dynamic_kv.snapshot_geri_yukle(snapshot)
        self._replayi_kayitlardan_yenile()
        return version

    def sikistir(self) -> CompactionReport:
        return self.dynamic_kv.sikistir()

    def dynamic_kvye_migre_et(
        self,
        max_entries: Optional[int] = None,
        eviction_policy: str = "lru",
    ) -> MigrationReport:
        """Aktif FIRST_WINS slot/replay durumunu kayıp muhasebesiyle migrate et."""
        if isinstance(self.slotlar, DynamicKVMemory):
            raise RuntimeError("Bellek zaten DYNAMIC_KV politikasında")
        candidates = self.replay.icerik()
        target_capacity = max_entries
        if target_capacity is None:
            target_capacity = max(self.slotlar.slot_sayisi, len(candidates), 1)
        memory, report = DynamicKVMemory.legacy_slotlardan_migre_et(
            self.slotlar,
            candidates,
            max_entries=target_capacity,
            eviction_policy=eviction_policy,
        )
        self.slotlar = memory
        self.politika = DYNAMIC_KV
        self._replayi_kayitlardan_yenile()
        return report

    def _replayi_kayitlardan_yenile(self) -> None:
        from ..knowledge.schemas import ExperienceCandidate

        candidates = []
        for record in self.dynamic_kv.kayitlar():
            if record.payload is not None:
                candidates.append(ExperienceCandidate.from_dict(record.payload))
        self.replay.yenile(candidates)

    def rapor(self) -> Dict:
        slot = self.slotlar.kapasite()
        replay = self.replay.rapor()
        report = {
            "politika": self.politika,
            "slot_dolu": slot["dolu_slot"],
            "slot_toplam": slot["toplam_slot"],
            "cakisma": slot["cakisma"],
            "cakisma_orani": slot["cakisma_orani"],
            "replay_dolu": replay["dolu"],
            "replay_verimliligi": replay["replay_verimliligi"],
        }
        if isinstance(self.slotlar, DynamicKVMemory):
            report.update({
                "store_version": slot["store_version"],
                "journal_events": slot["journal_events"],
                "eviction_policy": slot["eviction_policy"],
                "evictions": slot["statistics"]["evictions"],
                "ttl_evictions": slot["statistics"]["ttl_evictions"],
                "estimated_storage_bytes": slot["estimated_storage_bytes"],
            })
        return report


__all__ = ["BellekEntegrasyonu", "DYNAMIC_KV", "FIRST_WINS"]
