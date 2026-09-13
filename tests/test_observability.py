# -*- coding: utf-8 -*-
"""Gözlemlenebilirlik modülleri testleri."""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.knowledge import DeneyimDurumu, ExperienceCandidate, KaynakTuru  # noqa: E402
from hga.memory import DeneyimSlotlari  # noqa: E402
from hga.observability import (  # noqa: E402
    ascii_bellek_haritasi,
    bellek_doluluk_haritasi,
    deneyim_akisi,
    head_diversity,
    liste_katman_benzerligi,
)


def test_deneyim_akisi_oranlar():
    adaylar = [
        ExperienceCandidate("X1", "E1", "R1", "E2", state=DeneyimDurumu.VALID),
        ExperienceCandidate("X2", "E1", "R1", "E3", state=DeneyimDurumu.INVALID),
        ExperienceCandidate("X3", "E1", "R1", "E4", state=DeneyimDurumu.CONFLICT,
                            source=KaynakTuru.REAL_DATA),
    ]
    r = deneyim_akisi(adaylar)
    assert r["toplam"] == 3
    assert r["accepted"] == 1
    assert r["rejected"] == 1
    assert r["conflict"] == 1
    assert abs(r["acceptance_rate"] - 1 / 3) < 1e-9


def test_bellek_doluluk_haritasi():
    s = DeneyimSlotlari(slot_sayisi=8)
    s.yaz("X1", ("E1", "R1", "E2"))
    h = bellek_doluluk_haritasi(s, genislik=4)
    assert h["toplam"] == 8
    assert h["dolu"] == 1
    assert len(h["grid"]) == 2
    metin = ascii_bellek_haritasi(s, genislik=4)
    assert "█" in metin and "·" in metin


def test_liste_katman_benzerligi():
    ciftler = [([1, 0], [1, 0]), ([0, 1], [1, 0]), ([1, 0], [1, 0])]
    r = liste_katman_benzerligi(ciftler)
    op = r["kronecker_operator_cosine"]
    assert op[0][0] == 1.0
    assert op[0][1] == 0.0
    assert op[0][2] == 1.0
    assert r["katman_sayisi"] == 3


def test_head_diversity():
    weights = [[
        [[1.0, 0.0], [1.0, 0.0]],
        [[0.0, 1.0], [0.0, 1.0]],
    ]]
    r = head_diversity(weights)
    assert r["head_sayisi"] == 2
    assert r["pair_count"] == 1
    assert r["diversity"] > 0.9


if __name__ == "__main__":
    testler = [(ad, fn) for ad, fn in sorted(globals().items())
               if ad.startswith("test_") and callable(fn)]
    basarisiz = 0
    for ad, fn in testler:
        try:
            fn()
            print(f"  ✅ {ad}")
        except Exception as e:  # noqa: BLE001
            basarisiz += 1
            print(f"  ❌ {ad}: {type(e).__name__}: {e}")
    print("\nTÜM TESTLER GEÇTİ" if basarisiz == 0 else f"{basarisiz} test başarısız")
    sys.exit(1 if basarisiz else 0)
