# -*- coding: utf-8 -*-
"""Bellek doluluk haritası üretimi."""
from __future__ import annotations

from typing import Dict, List


def _slotlar_dolu_mu(slotlar) -> List[int]:
    """DeneyimSlotlari benzeri nesneden 0/1 doluluk vektörü çıkar."""
    if hasattr(slotlar, "_tablolar") and hasattr(slotlar, "slot_sayisi"):
        out: List[int] = []
        for tablo in slotlar._tablolar:  # noqa: SLF001 - gözlemlenebilirlik introspection
            out.extend(1 if i in tablo else 0 for i in range(slotlar.slot_sayisi))
        return out
    raise TypeError("Desteklenen bellek tipi değil: DeneyimSlotlari bekleniyor")


def bellek_doluluk_haritasi(slotlar, genislik: int = 16) -> Dict:
    """Bellek slotlarını grid olarak raporla.

    Hücre değeri: 1 = dolu, 0 = boş. Şimdilik saf Python ``DeneyimSlotlari``
    için tasarlandı; torch tabanlı tablo için ayrı/gpu-dostu tarama önerilir.
    """
    genislik = max(1, int(genislik))
    v = _slotlar_dolu_mu(slotlar)
    grid = [v[i:i + genislik] for i in range(0, len(v), genislik)]
    dolu = sum(v)
    return {
        "dolu": dolu,
        "toplam": len(v),
        "doluluk_orani": dolu / len(v) if v else 0.0,
        "genislik": genislik,
        "grid": grid,
    }


def ascii_bellek_haritasi(slotlar, genislik: int = 16,
                          dolu: str = "█", bos: str = "·") -> str:
    """Doluluk haritasını terminal dostu ASCII/Unicode metne çevir."""
    h = bellek_doluluk_haritasi(slotlar, genislik=genislik)
    satirlar = ["".join(dolu if c else bos for c in satir) for satir in h["grid"]]
    return "\n".join(satirlar)


__all__ = ["bellek_doluluk_haritasi", "ascii_bellek_haritasi"]
