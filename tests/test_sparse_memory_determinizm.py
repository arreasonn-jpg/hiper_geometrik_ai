# -*- coding: utf-8 -*-
"""Saf Python deneyim slot hash'i süreç/seed bağımsız deterministik olmalı."""
import os
import subprocess
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.memory.sparse_memory import DeneyimSlotlari, parmak_izi  # noqa: E402


def test_parmak_izi_kararli():
    uclu = ("E_001", "R_001", "E_002")
    assert parmak_izi(uclu) == parmak_izi(uclu)
    kod = (
        "import sys; sys.path.insert(0, %r); "
        "from hga.memory.sparse_memory import parmak_izi; "
        "print(parmak_izi(('E_001','R_001','E_002')))"
    ) % KOK
    env1 = dict(os.environ, PYTHONHASHSEED="1")
    env2 = dict(os.environ, PYTHONHASHSEED="2")
    a = subprocess.check_output([sys.executable, "-c", kod], env=env1, text=True).strip()
    b = subprocess.check_output([sys.executable, "-c", kod], env=env2, text=True).strip()
    assert a == b == str(parmak_izi(uclu))


def test_deneyim_slotlari_collision_yok_kucuk_senaryo():
    s = DeneyimSlotlari(slot_sayisi=64)
    s.yaz("X1", ("E_001", "R_001", "E_002"))
    s.yaz("X2", ("E_001", "R_001", "E_003"))
    dolu, toplam = s.doluluk_orani()
    assert dolu == 2 and toplam == 64
    assert s.cakisma_orani() == 0.0


def test_deneyim_slotlari_lru_ve_okuma_yazma():
    s = DeneyimSlotlari(slot_sayisi=64)
    s.yaz("X1", ("E_001", "R_001", "E_002"))
    s.yaz("X2", ("E_001", "R_001", "E_003"))
    assert s.icerir("X2", ("E_001", "R_001", "E_003")) is True
    rapor = s.okuma_yazma_raporu()
    assert rapor["okuma"] == 1 and rapor["yazma"] == 2
    temiz = s.lru_temizle(max_yas=1)
    assert temiz == 1
    assert s.icerir("X1", ("E_001", "R_001", "E_002")) is False
    assert s.icerir("X2", ("E_001", "R_001", "E_003")) is True


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
