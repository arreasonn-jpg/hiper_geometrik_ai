# -*- coding: utf-8 -*-
"""
Experience Replay — Deneyim Tekrarı (v0.4 için çekirdek)
==========================================================
(v0.1 — rapor §19 v0.4, §20 "replay efficiency")

Deneyimlerin sonradan yeniden örneklenebilmesi için sınırlı kapasiteli bir
tekrar tamponu. v0.1'de yalnızca depolama + örnekleme + `verimlilik()` metriği
sağlar; sinir ağı eğitimiyle gerçek birleştirme (rapor §19 v0.4) sonraki
aşamadır.

`verimlilik()` metriği: tekrar oynatılan deneyimlerin ne kadarının YENİ bilgi
üretimine katkı sağladığı — burada "benzersiz üçlü oranı" ile yaklaşık ölçülür
(rapor §20 "Replay efficiency").
"""
from typing import Any, Dict, List, Optional
import random


class DeneyimTekrari:
    """Sınırlı kapasiteli deneyim tekrar tamponu (FIFO + tek tip örnekleme)."""

    def __init__(self, kapasite: int = 1000, tohum: Optional[int] = None):
        self.kapasite = int(kapasite)
        self._tampon: List[Any] = []
        self._rng = random.Random(tohum)
        self._ornek_sayisi = 0
        self._benzersiz_ornekler: set = set()

    def it(self, deneyim) -> None:
        """Tampona bir deneyim it (doluysa en eskiyi düşür — FIFO)."""
        self._tampon.append(deneyim)
        if len(self._tampon) > self.kapasite:
            self._tampon.pop(0)

    def ornekle(self, n: int = 1) -> List[Any]:
        """Tek tip rastgele `n` deneyim örnekle (en fazla tampon boyutu)."""
        n = max(0, min(int(n), len(self._tampon)))
        ornekler = self._rng.sample(self._tampon, n)
        for o in ornekler:
            self._ornek_sayisi += 1
            anahtar = getattr(o, "uclusu", None)
            if callable(anahtar):
                anahtar = anahtar()
            self._benzersiz_ornekler.add(anahtar if anahtar is not None else repr(o))
        return ornekler

    def verimlilik(self) -> float:
        """Replay efficiency ≈ benzersiz örnek / toplam örnek (0..1).

        Boş tampon → 0.0. Tekrar oynatılan örnekler çeşitli olduğunda
        verimlilik 1.0'a yaklaşır; aynı deneyimin tekrarı ise düşürür.
        """
        if self._ornek_sayisi == 0:
            return 0.0
        return round(len(self._benzersiz_ornekler) / self._ornek_sayisi, 4)

    def __len__(self) -> int:
        return len(self._tampon)

    def rapor(self) -> Dict:
        return {
            "kapasite": self.kapasite,
            "dolu": len(self._tampon),
            "ornek_sayisi": self._ornek_sayisi,
            "benzersiz_ornek": len(self._benzersiz_ornekler),
            "replay_verimliligi": self.verimlilik(),
        }
