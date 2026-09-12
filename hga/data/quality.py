# -*- coding: utf-8 -*-
"""Türkçe veri kalite filtresi.

Basit ama ölçülebilir filtreler: tekrar/duplicate, bozuk encoding, spam benzeri
çok düşük harf oranı ve aşırı karakter tekrarı. Büyük veri borularına eklenmek
üzere bağımlılıksız tasarlanmıştır.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Tuple


@dataclass
class VeriKaliteRaporu:
    toplam: int = 0
    kabul: int = 0
    bos: int = 0
    duplicate: int = 0
    kisa: int = 0
    bozuk_encoding: int = 0
    dusuk_harf_orani: int = 0
    asiri_tekrar: int = 0
    nedenler: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["kabul_orani"] = self.kabul / self.toplam if self.toplam else 0.0
        return d


_TR_HARF_RE = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]")
_URL_RE = re.compile(r"https?://|www\.", re.I)


def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _harf_orani(s: str) -> float:
    if not s:
        return 0.0
    harf = len(_TR_HARF_RE.findall(s))
    return harf / max(1, len(s))


def _max_tekrar_orani(s: str) -> float:
    if not s:
        return 0.0
    en, simdiki, onceki = 1, 0, None
    for ch in s:
        if ch == onceki:
            simdiki += 1
        else:
            onceki, simdiki = ch, 1
        en = max(en, simdiki)
    return en / len(s)


def kalite_skoru(cumle: str) -> float:
    """0..1 kaba kalite skoru (yüksek daha iyi)."""
    s = str(cumle).strip()
    if not s:
        return 0.0
    skor = 1.0
    if "�" in s:
        skor -= 0.5
    if _URL_RE.search(s):
        skor -= 0.2
    skor -= max(0.0, 0.55 - _harf_orani(s))
    skor -= max(0.0, _max_tekrar_orani(s) - 0.25)
    return max(0.0, min(1.0, skor))


def temizle_cumleler(cumleler: Iterable[str], min_karakter: int = 8,
                     min_harf_orani: float = 0.45,
                     max_tekrar_orani: float = 0.35) -> Tuple[List[str], VeriKaliteRaporu]:
    """Cümleleri temizle ve neden bazlı rapor döndür."""
    rapor = VeriKaliteRaporu()
    temiz: List[str] = []
    gorulen = set()
    for cumle in cumleler:
        rapor.toplam += 1
        s = re.sub(r"\s+", " ", str(cumle).strip())
        n = _norm(s)
        neden = None
        if not s:
            neden = "bos"
            rapor.bos += 1
        elif n in gorulen:
            neden = "duplicate"
            rapor.duplicate += 1
        elif len(s) < min_karakter:
            neden = "kisa"
            rapor.kisa += 1
        elif "�" in s:
            neden = "bozuk_encoding"
            rapor.bozuk_encoding += 1
        elif _harf_orani(s) < min_harf_orani:
            neden = "dusuk_harf_orani"
            rapor.dusuk_harf_orani += 1
        elif _max_tekrar_orani(s) > max_tekrar_orani:
            neden = "asiri_tekrar"
            rapor.asiri_tekrar += 1
        if neden:
            rapor.nedenler[neden] = rapor.nedenler.get(neden, 0) + 1
            continue
        gorulen.add(n)
        temiz.append(s)
        rapor.kabul += 1
    return temiz, rapor


__all__ = ["VeriKaliteRaporu", "temizle_cumleler", "kalite_skoru"]
