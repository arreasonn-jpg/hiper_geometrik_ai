# -*- coding: utf-8 -*-
"""
Türkçe morfoloji testleri — yönelme eki + sesli uyumu
======================================================
Çalıştırma:
    python tests/test_turkce.py
    pytest tests/test_turkce.py -q
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.experience import yonelme_eki, son_unlu, kucult  # noqa: E402


def test_son_unlu():
    assert son_unlu("Ata") == "a"
    assert son_unlu("Gökyüzü") == "ü"
    assert son_unlu("kitap") == "a"
    assert son_unlu("Ali") == "i"
    assert son_unlu("Atatürk") == "ü"
    assert son_unlu("") == ""


def test_yonelme_cins_isim():
    assert yonelme_eki("ata") == "ataya"          # kalın, sesliyle biten → 'y' + a
    assert yonelme_eki("araba") == "arabaya"
    assert yonelme_eki("gökyüzü") == "gökyüzüne"  # iyelik ekli → istisna ('n' kaynaştırma)
    assert yonelme_eki("ev") == "eve"             # ince, sessizle biten
    # NOT: "kitap → kitaba" ünsüz yumuşaması (p→b) gerektirir; kapsam dışıdır
    # ve bu modül bilinçli olarak uygulamaz (bkz. turkce.py docstring).


def test_yonelme_ozel_isim():
    assert yonelme_eki("Ali", ozel_isim=True) == "Ali'ye"
    assert yonelme_eki("Ata", ozel_isim=True) == "Ata'ya"
    assert yonelme_eki("Atatürk", ozel_isim=True) == "Atatürk'e"
    assert yonelme_eki("İzmir", ozel_isim=True) == "İzmir'e"


def test_kucult():
    assert kucult("İzmir") == "izmir"     # İ → i
    assert kucult("Araba") == "araba"
    assert kucult("GÖKYÜZÜ") == "gökyüzü"


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
