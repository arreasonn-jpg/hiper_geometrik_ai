# -*- coding: utf-8 -*-
"""
Türkçe morfoloji testleri — ek uyumu + ünsüz yumuşaması
=========================================================
Çalıştırma:
    python tests/test_turkce.py
    pytest tests/test_turkce.py -q
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.experience import (yonelme_eki, belirtme_eki, bulunma_eki,  # noqa: E402
                            ayrilma_eki, cogul_eki, unsuz_yumusat,
                            hece_sayisi, son_unlu, kucult)


def test_son_unlu():
    assert son_unlu("Ata") == "a"
    assert son_unlu("Gökyüzü") == "ü"
    assert son_unlu("kitap") == "a"
    assert son_unlu("Ali") == "i"
    assert son_unlu("") == ""


def test_hece_sayisi():
    assert hece_sayisi("ata") == 2
    assert hece_sayisi("kitap") == 2
    assert hece_sayisi("ev") == 1
    assert hece_sayisi("top") == 1
    assert hece_sayisi("gökyüzü") == 3


def test_unsuz_yumusat():
    assert unsuz_yumusat("kitap") == "kitab"      # çok heceli p→b
    assert unsuz_yumusat("ağaç") == "ağac"        # ç→c
    assert unsuz_yumusat("yurt") == "yurd"        # tek heceli istisna t→d
    assert unsuz_yumusat("renk") == "reng"        # tek heceli istisna k→ğ
    assert unsuz_yumusat("top") == "top"          # tek heceli yumuşamaz
    assert unsuz_yumusat("at") == "at"
    assert unsuz_yumusat("saat") == "saat"        # alıntı istisna
    assert unsuz_yumusat("Ata", ozel_isim=True) == "Ata"  # özel isim yumuşamaz


def test_yonelme_cins_isim():
    assert yonelme_eki("ata") == "ataya"
    assert yonelme_eki("araba") == "arabaya"
    assert yonelme_eki("gökyüzü") == "gökyüzüne"   # iyelikli istisna
    assert yonelme_eki("ev") == "eve"
    assert yonelme_eki("kitap") == "kitaba"        # ünsüz yumuşaması
    assert yonelme_eki("ağaç") == "ağaca"
    assert yonelme_eki("yurt") == "yurda"
    assert yonelme_eki("top") == "topa"            # tek heceli yumuşamaz
    assert yonelme_eki("saat") == "saate"          # alıntı istisna


def test_yonelme_ozel_isim():
    assert yonelme_eki("Ali", ozel_isim=True) == "Ali'ye"
    assert yonelme_eki("Ata", ozel_isim=True) == "Ata'ya"
    assert yonelme_eki("Atatürk", ozel_isim=True) == "Atatürk'e"


def test_belirtme_eki():
    assert belirtme_eki("araba") == "arabayı"
    assert belirtme_eki("kitap") == "kitabı"       # yumuşama + belirtme
    assert belirtme_eki("ev") == "evi"
    assert belirtme_eki("okul") == "okulu"
    assert belirtme_eki("gökyüzü") == "gökyüzünü"  # iyelikli istisna
    assert belirtme_eki("Ata", ozel_isim=True) == "Ata'yı"


def test_bulunma_eki():
    assert bulunma_eki("ev") == "evde"
    assert bulunma_eki("kitap") == "kitapta"       # ünsüz benzeşmesi -ta
    assert bulunma_eki("ağaç") == "ağaçta"
    assert bulunma_eki("okul") == "okulda"
    assert bulunma_eki("araba") == "arabada"
    assert bulunma_eki("gökyüzü") == "gökyüzünde"  # iyelikli istisna


def test_ayrilma_eki():
    assert ayrilma_eki("ev") == "evden"
    assert ayrilma_eki("kitap") == "kitaptan"      # ünsüz benzeşmesi -tan
    assert ayrilma_eki("araba") == "arabadan"
    assert ayrilma_eki("okul") == "okuldan"
    assert ayrilma_eki("gökyüzü") == "gökyüzünden"


def test_cogul_eki():
    assert cogul_eki("araba") == "arabalar"
    assert cogul_eki("ev") == "evler"
    assert cogul_eki("Ali", ozel_isim=True) == "Ali'ler"


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
