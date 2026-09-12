# -*- coding: utf-8 -*-
"""Türkçe mini benchmark koşucusu.

OSCAR/Wikipedia gibi büyük kümeler için aynı arayüz kullanılabilir; bu dosyadaki
mini set eğitim gerektirmeden tokenizer/metric smoke testi sağlar ve Türkçe
karakter/morfoloji kapsamını garanti eder.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional


MINI_TURKCE_CUMLELER = [
    "İstanbul'da yağmur yağarken öğrenciler kütüphaneye gitti.",
    "Iğdır ovasında çiftçiler ürünlerini özenle topladı.",
    "Şeker, çay, ğ harfi, ı harfi, ö ve ü Türkçe metinlerde korunmalıdır.",
    "Ali ataya bindi, Ayşe arabaya bindi, gökyüzüne bakıldı.",
    "Küçük model büyük fikri test eder; ölçmediğin şeyi iyileştiremezsin.",
]


@dataclass
class TurkceBenchmarkRaporu:
    cumle_sayisi: int
    token_sayisi: int
    loss: Optional[float] = None
    perplexity: Optional[float] = None
    tokenizer_ok: bool = True
    notlar: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def mini_turkce_corpus() -> List[str]:
    return list(MINI_TURKCE_CUMLELER)


def tokenizer_kapsami(tokenizer, cumleler: Optional[Iterable[str]] = None) -> Dict:
    """Türkçe karakterler ve UNK oranı için hafif tokenizer benchmark'ı."""
    cumleler = list(cumleler or MINI_TURKCE_CUMLELER)
    metin = "\n".join(cumleler)
    ids = tokenizer.encode(metin) if hasattr(tokenizer, "encode") else []
    unk_id = getattr(tokenizer, "UNK_ID", 1)
    unk = sum(1 for i in ids if int(i) == unk_id)
    chars = set("şçğıöüİ")
    decoded = tokenizer.decode(ids) if hasattr(tokenizer, "decode") else ""
    karakter_kapsami = {c: (c.lower().replace("i̇", "i") in decoded or c in decoded)
                         for c in chars}
    return {
        "cumle_sayisi": len(cumleler),
        "token_sayisi": len(ids),
        "unk": unk,
        "unk_orani": unk / len(ids) if ids else 0.0,
        "turkce_karakter_kapsami": karakter_kapsami,
        "turkce_karakter_tam": all(karakter_kapsami.values()),
        "karakter_kapsama": sum(1 for v in karakter_kapsami.values() if v) / len(karakter_kapsami),
    }


def perplexity_benchmark(model, tokenizer, cumleler: Optional[Iterable[str]] = None,
                         batch_size: int = 64) -> TurkceBenchmarkRaporu:
    """Mini Türkçe corpus üzerinde perplexity ölç."""
    cumleler = list(cumleler or MINI_TURKCE_CUMLELER)
    ids = tokenizer.encode("\n".join(cumleler))
    from egitim.degerlendirme import perplexity
    loss, ppl = perplexity(model, ids, batch_size=batch_size)
    kaps = tokenizer_kapsami(tokenizer, cumleler)
    return TurkceBenchmarkRaporu(cumle_sayisi=len(cumleler), token_sayisi=len(ids),
                                 loss=loss, perplexity=ppl,
                                 tokenizer_ok=kaps["unk_orani"] < 0.05,
                                 notlar=[f"unk_orani={kaps['unk_orani']:.4f}"])


__all__ = [
    "MINI_TURKCE_CUMLELER",
    "TurkceBenchmarkRaporu",
    "mini_turkce_corpus",
    "tokenizer_kapsami",
    "perplexity_benchmark",
]
