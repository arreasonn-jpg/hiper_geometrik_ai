# -*- coding: utf-8 -*-
"""
Knowledge Versioning & Rollback — K₀ → K₁ → K₂ … ve geri alma (Faz 23)
=======================================================================

Self-learning döngüsünde bilgi tabanı kendi ürettiği deneyimlerle büyür.
Bir döngü yanlış bir olguyu kalıcı bilgiye yazarsa (verifier hatası, bozuk
veri kaynağı, hatalı konsolidasyon) bunun **geri alınabilir** olması lüks
değil sigortadır.

Bu modül `KnowledgeStore` üzerinde içerik-adresli, değişmez (immutable)
snapshot zinciri kurar::

    K0 ──commit──▶ K1 ──commit──▶ K2 ──commit──▶ K3
                                   ▲
                                   └── rollback("K2") → yeni K4 = K2 içeriği

Tasarım kararları:

* **Snapshot içerik-adreslidir.** Her sürümün `state_hash` değeri
  `KnowledgeStore.to_dict()` üzerinden kanonik SHA-256 ile hesaplanır. Aynı
  içerik her zaman aynı hash'i verir; "bu iki sürüm gerçekten aynı mı?"
  sorusu deterministik olarak cevaplanır.
* **Geçmiş asla silinmez.** `rollback` bir sürümü yok etmez; hedef sürümün
  içeriğini taşıyan YENİ bir sürüm ekler (git `revert` semantiği,
  `reset --hard` değil). Böylece hatalı K3 de denetim için kalır.
* **Store'a bağımlı değildir.** Snapshot'lar saf JSON'dur; diske yazılabilir
  (`kaydet`/`yukle`) ve deney manifestlerine gömülebilir.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .knowledge_store import KnowledgeStore


def kanonik_hash(deger: Any) -> str:
    """Sıralı anahtarlı kanonik JSON üzerinden SHA-256."""
    ham = json.dumps(
        deger, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(ham).hexdigest()


def _icerik_hash(durum: Dict[str, Any]) -> str:
    """Sürüm sayacından bağımsız içerik hash'i.

    Rollback, hedef sürümün İÇERİĞİNİ geri getirir ama sürüm sayacı ileri
    sayar (bilgi tabanı zamanda geriye gitmez). Bu yüzden "aynı içerik mi?"
    sorusu `versiyon` alanı dışlanarak cevaplanır.
    """
    kopya = dict(durum)
    kopya.pop("versiyon", None)
    return kanonik_hash(kopya)


def _uclu_kumesi(durum: Dict[str, Any]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    olgular = durum.get("relations", {}).get("olgular", [])
    sonuc: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for olgu in olgular:
        anahtar = (str(olgu.get("subject_id")), str(olgu.get("relation_id")),
                   str(olgu.get("object_id")))
        sonuc[anahtar] = olgu
    return sonuc


@dataclass
class KnowledgeVersion:
    """Tek bir bilgi sürümünün değişmez kaydı."""

    version_id: str                 # K0, K1, K2 ...
    index: int                      # 0, 1, 2 ...
    parent_id: Optional[str]        # bir önceki sürüm (K0 için None)
    state_hash: str                 # tam durum SHA-256'sı (versiyon sayacı dâhil)
    created_at_utc: str
    content_hash: str = ""          # versiyon sayacı hariç içerik SHA-256'sı
    label: str = ""                 # insan okunur açıklama
    kind: str = "COMMIT"            # COMMIT | ROLLBACK
    rolled_back_to: Optional[str] = None   # kind=ROLLBACK ise hedef sürüm
    store_version: int = 1          # KnowledgeStore.versiyon sayacı
    ozet: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KnowledgeVersion":
        return cls(
            version_id=str(d["version_id"]),
            index=int(d["index"]),
            parent_id=d.get("parent_id"),
            state_hash=str(d["state_hash"]),
            content_hash=str(d.get("content_hash", "")),
            created_at_utc=str(d.get("created_at_utc", "")),
            label=str(d.get("label", "")),
            kind=str(d.get("kind", "COMMIT")),
            rolled_back_to=d.get("rolled_back_to"),
            store_version=int(d.get("store_version", 1)),
            ozet=dict(d.get("ozet", {})),
            state=dict(d.get("state", {})),
        )


@dataclass
class KnowledgeDiff:
    """İki sürüm arasındaki olgu düzeyinde fark."""

    kaynak_id: str
    hedef_id: str
    ayni_mi: bool
    eklenen_olgular: List[List[str]] = field(default_factory=list)
    silinen_olgular: List[List[str]] = field(default_factory=list)
    degisen_olgular: List[List[str]] = field(default_factory=list)
    varlik_farki: int = 0
    ozellik_farki: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KnowledgeVersionStore:
    """Append-only bilgi sürümü zinciri: commit / rollback / diff / checkout."""

    def __init__(self, store: Optional[KnowledgeStore] = None,
                 ilk_etiket: str = "K0 başlangıç bilgisi"):
        self.store = store if store is not None else KnowledgeStore()
        self.versions: List[KnowledgeVersion] = []
        self._commit(self.store, label=ilk_etiket, kind="COMMIT",
                     rolled_back_to=None)

    # ── İç yardımcılar ───────────────────────────────────────────────────
    def _commit(self, store: KnowledgeStore, label: str, kind: str,
                rolled_back_to: Optional[str]) -> KnowledgeVersion:
        # Derin kopya şart: KnowledgeStore.to_dict() bazı alanlarda (ör.
        # celiski_gunlugu) canlı referans döndürür; snapshot değişmez olmalı.
        durum = copy.deepcopy(store.to_dict())
        index = len(self.versions)
        kayit = KnowledgeVersion(
            version_id=f"K{index}",
            index=index,
            parent_id=self.versions[-1].version_id if self.versions else None,
            state_hash=kanonik_hash(durum),
            content_hash=_icerik_hash(durum),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            label=label,
            kind=kind,
            rolled_back_to=rolled_back_to,
            store_version=int(getattr(store, "versiyon", 1)),
            ozet=store.ozet(),
            state=durum,
        )
        self.versions.append(kayit)
        return kayit

    def _getir(self, version_id: str) -> KnowledgeVersion:
        for kayit in self.versions:
            if kayit.version_id == version_id:
                return kayit
        raise KeyError(f"Bilinmeyen bilgi sürümü: {version_id}")

    # ── Genel API ────────────────────────────────────────────────────────
    @property
    def current(self) -> KnowledgeVersion:
        return self.versions[-1]

    @property
    def current_id(self) -> str:
        return self.versions[-1].version_id

    def commit(self, label: str = "", store: Optional[KnowledgeStore] = None
               ) -> KnowledgeVersion:
        """Mevcut (veya verilen) bilgi durumunu yeni bir sürüm olarak mühürle."""
        if store is not None:
            self.store = store
        return self._commit(self.store, label=label, kind="COMMIT",
                            rolled_back_to=None)

    def commit_gerekli_mi(self) -> bool:
        """Son sürümden bu yana içerik değişti mi?"""
        return kanonik_hash(self.store.to_dict()) != self.current.state_hash

    def checkout(self, version_id: str) -> KnowledgeStore:
        """Bir sürümün içeriğinden BAĞIMSIZ bir KnowledgeStore kur (yan etkisiz)."""
        kayit = self._getir(version_id)
        return KnowledgeStore.from_dict(kayit.state)

    def rollback(self, version_id: str, label: str = "") -> KnowledgeVersion:
        """Hedef sürümün içeriğini yeni bir sürüm olarak geri yükle.

        Geçmiş silinmez: hatalı sürüm zincirde kalır, üstüne ROLLBACK kaydı
        eklenir. Aktif `self.store` hedef içerikle değiştirilir.
        """
        hedef = self._getir(version_id)
        if hedef.version_id == self.current_id and not self.commit_gerekli_mi():
            raise ValueError(
                f"{version_id} zaten aktif içerik; rollback anlamsız")
        yeni_store = KnowledgeStore.from_dict(hedef.state)
        # Sürüm sayacı geriye gitmez: bilgi tabanı ileriye doğru sayar.
        yeni_store.versiyon = max(int(getattr(self.store, "versiyon", 1)) + 1,
                                  hedef.store_version + 1)
        self.store = yeni_store
        return self._commit(
            self.store,
            label=label or f"{version_id} içeriğine geri dönüldü",
            kind="ROLLBACK",
            rolled_back_to=hedef.version_id,
        )

    def diff(self, kaynak_id: str, hedef_id: str) -> KnowledgeDiff:
        """İki sürüm arasındaki olgu/varlık/özellik farkı."""
        kaynak = self._getir(kaynak_id)
        hedef = self._getir(hedef_id)
        a = _uclu_kumesi(kaynak.state)
        b = _uclu_kumesi(hedef.state)
        eklenen = sorted(set(b) - set(a))
        silinen = sorted(set(a) - set(b))
        degisen = sorted(
            anahtar for anahtar in set(a) & set(b)
            if kanonik_hash(a[anahtar]) != kanonik_hash(b[anahtar])
        )
        return KnowledgeDiff(
            kaynak_id=kaynak.version_id,
            hedef_id=hedef.version_id,
            ayni_mi=kaynak.content_hash == hedef.content_hash,
            eklenen_olgular=[list(x) for x in eklenen],
            silinen_olgular=[list(x) for x in silinen],
            degisen_olgular=[list(x) for x in degisen],
            varlik_farki=(hedef.ozet.get("varlik", 0) - kaynak.ozet.get("varlik", 0)),
            ozellik_farki=(hedef.ozet.get("ozellik", 0) - kaynak.ozet.get("ozellik", 0)),
        )

    def gecmis(self) -> List[Dict[str, Any]]:
        """State gövdesi olmadan, okunabilir sürüm geçmişi."""
        return [
            {
                "version_id": k.version_id,
                "parent_id": k.parent_id,
                "kind": k.kind,
                "rolled_back_to": k.rolled_back_to,
                "state_hash": k.state_hash,
                "content_hash": k.content_hash,
                "label": k.label,
                "created_at_utc": k.created_at_utc,
                "ozet": k.ozet,
            }
            for k in self.versions
        ]

    def zincir_dogrula(self) -> bool:
        """Sürüm zincirinin yapısal tutarlılığı (parent bağları + hash'ler)."""
        for beklenen_index, kayit in enumerate(self.versions):
            if kayit.index != beklenen_index:
                return False
            if kayit.version_id != f"K{beklenen_index}":
                return False
            beklenen_parent = (self.versions[beklenen_index - 1].version_id
                               if beklenen_index else None)
            if kayit.parent_id != beklenen_parent:
                return False
            if kanonik_hash(kayit.state) != kayit.state_hash:
                return False
            if _icerik_hash(kayit.state) != kayit.content_hash:
                return False
            if kayit.kind == "ROLLBACK":
                hedef = self._getir(str(kayit.rolled_back_to))
                # Sürüm sayacı ileri sayar; içerik hedefle birebir aynı olmalı.
                if hedef.content_hash != kayit.content_hash:
                    return False
        return True

    # ── Kalıcılık ────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "current": self.current_id,
            "versions": [k.to_dict() for k in self.versions],
        }

    def kaydet(self, yol: str) -> str:
        """Sürüm zincirini atomik JSON olarak diske yaz."""
        yol = os.path.abspath(yol)
        dizin = os.path.dirname(yol) or "."
        os.makedirs(dizin, exist_ok=True)
        veri = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        fd, gecici = tempfile.mkstemp(dir=dizin, prefix=".hga_kv_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(veri)
            os.replace(gecici, yol)
        finally:
            if os.path.exists(gecici):
                os.remove(gecici)
        return yol

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KnowledgeVersionStore":
        kayitlar = [KnowledgeVersion.from_dict(x) for x in d.get("versions", [])]
        if not kayitlar:
            raise ValueError("Sürüm zinciri boş olamaz")
        nesne = cls.__new__(cls)
        nesne.versions = kayitlar
        nesne.store = KnowledgeStore.from_dict(kayitlar[-1].state)
        return nesne

    @classmethod
    def yukle(cls, yol: str) -> "KnowledgeVersionStore":
        with open(yol, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


__all__ = [
    "KnowledgeVersion",
    "KnowledgeVersionStore",
    "KnowledgeDiff",
    "kanonik_hash",
]
