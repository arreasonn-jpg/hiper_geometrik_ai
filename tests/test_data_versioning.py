# -*- coding: utf-8 -*-
"""Veri manifest/versiyonlama yardımcıları testleri."""
import hashlib
import os
import sys
import tempfile

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.data import dosya_hashle, manifest_kaydet, manifest_yukle  # noqa: E402


def test_dosya_hashle():
    fd, yol = tempfile.mkstemp()
    try:
        veri = "Ali ataya bindi.\nAyşe arabaya bindi.\n".encode("utf-8")
        with os.fdopen(fd, "wb") as f:
            f.write(veri)
        m = dosya_hashle(yol)
        assert m.bytes == len(veri)
        assert m.lines == 2
        assert m.sha256 == hashlib.sha256(veri).hexdigest()
    finally:
        os.remove(yol)


def test_manifest_kaydet_yukle():
    fd, yol = tempfile.mkstemp()
    out = yol + ".json"
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("Türkçe veri\n")
        m = manifest_kaydet([yol], out)
        y = manifest_yukle(out)
        assert y == m
        assert y["dosyalar"][0]["path"] == yol
    finally:
        if os.path.exists(yol):
            os.remove(yol)
        if os.path.exists(out):
            os.remove(out)


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
