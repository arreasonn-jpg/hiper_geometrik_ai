# -*- coding: utf-8 -*-
"""
Immutable Experience Ledger — Silinmeyen Deneyim Defteri (Faz 24)
==================================================================

Reddedilen, çürütülen veya belirsiz kalan bir deneyim de bilimsel bilgidir:
"model geçmişte nerede hata yaptı?" sorusu ancak kayıt tutulursa cevaplanır.
Bu modül append-only, hash-zincirli (tamper-evident) bir defter uygular.

Her kayıt şu alanları taşır::

    {
      "seq": 12,
      "experience": "E_001|R_001|E_004",
      "experience_id": "SL-1-0003-0007",
      "status": "INVALID",
      "reason": "bağımsız doğrulayıcı çürüttü",
      "evaluator": "ExperienceEvaluator",
      "verifier": "arithmetic-env-v1",
      "source": "MODEL_GENERATED",
      "knowledge_version": "K12",
      "timestamp": "2026-09-13T10:00:00+00:00",
      "scores": {...},
      "prev_hash": "…",
      "entry_hash": "…"
    }

Tamper-evidence: `entry_hash = SHA256(prev_hash + kanonik(kayıt gövdesi))`.
Geçmiş bir kayıt sonradan değiştirilirse `zincir_dogrula()` False döner.
Defter **yalnızca ekleme** kabul eder; `sil`/`guncelle` API'si bilinçli olarak
yoktur.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..knowledge.schemas import DeneyimDurumu, ExperienceCandidate

GENESIS_HASH = "0" * 64

# Gövde alanları: entry_hash hesabına giren alanlar (sıra önemli değil,
# kanonik JSON sıralı anahtar kullanır).
_GOVDE_ALANLARI = (
    "seq", "experience", "experience_id", "status", "reason", "evaluator",
    "verifier", "source", "knowledge_version", "timestamp", "scores", "cycle",
)


def _kanonik(deger: Any) -> bytes:
    return json.dumps(
        deger, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


@dataclass
class LedgerEntry:
    seq: int
    experience: str
    experience_id: str
    status: str
    reason: str
    evaluator: str
    verifier: Optional[str]
    source: str
    knowledge_version: str
    timestamp: str
    scores: Dict[str, float] = field(default_factory=dict)
    cycle: int = 0
    prev_hash: str = GENESIS_HASH
    entry_hash: str = ""

    def govde(self) -> Dict[str, Any]:
        tam = asdict(self)
        return {alan: tam[alan] for alan in _GOVDE_ALANLARI}

    def hash_hesapla(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.prev_hash.encode("utf-8"))
        digest.update(_kanonik(self.govde()))
        return digest.hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LedgerEntry":
        return cls(
            seq=int(d["seq"]),
            experience=str(d["experience"]),
            experience_id=str(d["experience_id"]),
            status=str(d["status"]),
            reason=str(d.get("reason", "")),
            evaluator=str(d.get("evaluator", "")),
            verifier=d.get("verifier"),
            source=str(d.get("source", "")),
            knowledge_version=str(d.get("knowledge_version", "")),
            timestamp=str(d.get("timestamp", "")),
            scores={k: float(v) for k, v in dict(d.get("scores", {})).items()},
            cycle=int(d.get("cycle", 0)),
            prev_hash=str(d.get("prev_hash", GENESIS_HASH)),
            entry_hash=str(d.get("entry_hash", "")),
        )


class ExperienceLedger:
    """Append-only, hash-zincirli deneyim defteri."""

    def __init__(self, entries: Optional[Iterable[LedgerEntry]] = None):
        self._entries: List[LedgerEntry] = list(entries or [])

    # ── Yazma (yalnız ekleme) ────────────────────────────────────────────
    def kaydet(
        self,
        aday: ExperienceCandidate,
        knowledge_version: str = "K0",
        reason: str = "",
        evaluator: str = "ExperienceEvaluator",
        verifier: Optional[str] = None,
        cycle: int = 0,
        timestamp: Optional[str] = None,
    ) -> LedgerEntry:
        """Bir deneyimin o anki akıbetini deftere ekle (durum ne olursa olsun)."""
        prev_hash = self._entries[-1].entry_hash if self._entries else GENESIS_HASH
        gerekce = reason or (aday.evidence[-1] if aday.evidence else "")
        kayit = LedgerEntry(
            seq=len(self._entries),
            experience="|".join(aday.uclusu),
            experience_id=aday.experience_id,
            status=aday.state.value if isinstance(aday.state, DeneyimDurumu)
            else str(aday.state),
            reason=gerekce,
            evaluator=evaluator,
            verifier=verifier if verifier is not None else aday.verified_by,
            source=aday.source.value if hasattr(aday.source, "value") else str(aday.source),
            knowledge_version=str(knowledge_version),
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            scores={k: round(float(v), 8) for k, v in dict(aday.scores).items()},
            cycle=int(cycle),
            prev_hash=prev_hash,
        )
        kayit.entry_hash = kayit.hash_hesapla()
        self._entries.append(kayit)
        return kayit

    def toplu_kaydet(self, adaylar: Iterable[ExperienceCandidate], **kwargs
                     ) -> List[LedgerEntry]:
        return [self.kaydet(aday, **kwargs) for aday in adaylar]

    # ── Okuma ────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    @property
    def entries(self) -> List[LedgerEntry]:
        """Kayıtların kopyası (dıştan mutasyona kapalı)."""
        return list(self._entries)

    @property
    def head_hash(self) -> str:
        return self._entries[-1].entry_hash if self._entries else GENESIS_HASH

    def gecmis(self, experience_id: str) -> List[LedgerEntry]:
        """Tek bir deneyimin tüm yaşam öyküsü (durum geçişleri dâhil)."""
        return [k for k in self._entries if k.experience_id == experience_id]

    def duruma_gore(self, status: str) -> List[LedgerEntry]:
        return [k for k in self._entries if k.status == status]

    def ozet(self) -> Dict[str, Any]:
        sayim: Dict[str, int] = {}
        for kayit in self._entries:
            sayim[kayit.status] = sayim.get(kayit.status, 0) + 1
        sürümler = sorted({k.knowledge_version for k in self._entries})
        return {
            "kayit": len(self._entries),
            "durumlar": dict(sorted(sayim.items())),
            "benzersiz_deneyim": len({k.experience for k in self._entries}),
            "bilgi_surumleri": sürümler,
            "head_hash": self.head_hash,
            "zincir_gecerli": self.zincir_dogrula(),
        }

    # ── Bütünlük ─────────────────────────────────────────────────────────
    def zincir_dogrula(self) -> bool:
        """Hash zinciri kopmuş mu / kayıt sonradan değiştirilmiş mi?"""
        onceki = GENESIS_HASH
        for beklenen_seq, kayit in enumerate(self._entries):
            if kayit.seq != beklenen_seq or kayit.prev_hash != onceki:
                return False
            if kayit.entry_hash != kayit.hash_hesapla():
                return False
            onceki = kayit.entry_hash
        return True

    # ── Kalıcılık (JSONL: append-only dosya biçimi) ──────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "head_hash": self.head_hash,
            "entries": [k.to_dict() for k in self._entries],
        }

    def kaydet_jsonl(self, yol: str) -> str:
        """Defteri JSONL olarak yaz (her satır bir kayıt)."""
        yol = os.path.abspath(yol)
        os.makedirs(os.path.dirname(yol) or ".", exist_ok=True)
        with open(yol, "w", encoding="utf-8") as f:
            for kayit in self._entries:
                f.write(json.dumps(kayit.to_dict(), ensure_ascii=False,
                                   sort_keys=True) + "\n")
        return yol

    def ekle_jsonl(self, yol: str, kayit: LedgerEntry) -> str:
        """Tek kaydı mevcut JSONL dosyasının sonuna ekle (append-only)."""
        yol = os.path.abspath(yol)
        os.makedirs(os.path.dirname(yol) or ".", exist_ok=True)
        with open(yol, "a", encoding="utf-8") as f:
            f.write(json.dumps(kayit.to_dict(), ensure_ascii=False,
                               sort_keys=True) + "\n")
        return yol

    @classmethod
    def yukle_jsonl(cls, yol: str) -> "ExperienceLedger":
        entries: List[LedgerEntry] = []
        with open(yol, "r", encoding="utf-8") as f:
            for satir in f:
                satir = satir.strip()
                if satir:
                    entries.append(LedgerEntry.from_dict(json.loads(satir)))
        return cls(entries)


__all__ = ["ExperienceLedger", "LedgerEntry", "GENESIS_HASH"]
