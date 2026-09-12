# -*- coding: utf-8 -*-
"""
Sözlük büyütme testleri — gerçek korpustan yeni varlık desenleri
=================================================================
Çalıştırma:
    python tests/test_sozluk_buyutme.py
    pytest tests/test_sozluk_buyutme.py -q
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.experience import sozlugu_buyut, VARSAYILAN_SOZLUK  # noqa: E402

KORPUS = [
    "Ali ataya bindi.",
    "Mehmet kamyona bindi.",
    "Zeynep dağa baktı.",
    "Fatma yıldıza baktı.",
]


def test_buyutme_yeni_ozne_nesne():
    sozluk, rapor, ucluler = sozlugu_buyut(VARSAYILAN_SOZLUK, KORPUS)
    # yeni özneler (büyük harfli özel isimler → insan)
    assert "mehmet" in sozluk["binmek"]["ozneler"]
    assert sozluk["binmek"]["ozneler"]["mehmet"]["token"] == "Mehmet"
    assert sozluk["binmek"]["ozneler"]["mehmet"]["tip"] == "insan"
    assert "zeynep" in sozluk["bakmak"]["ozneler"]
    # yeni nesneler (durum ekli, kök çıkarıldı)
    assert sozluk["binmek"]["nesneler"]["kamyona"]["token"] == "Kamyon"
    assert sozluk["bakmak"]["nesneler"]["daga"]["token"] == "Dağ"      # Türkçe korundu
    assert sozluk["bakmak"]["nesneler"]["yildiza"]["token"] == "Yıldız"
    assert rapor.yeni_ozne == 3   # Mehmet, Zeynep, Fatma
    assert rapor.yeni_nesne == 3  # Kamyon, Dağ, Yıldız
    assert rapor.yeni_iliski == 0
    # üçlüler yeni varlık token'larıyla döner
    tokenler = {(o, i, n) for o, i, n in ucluler}
    assert ("Mehmet", "Binmek", "Kamyon") in tokenler
    assert ("Zeynep", "Bakmak", "Dağ") in tokenler


def test_buyutme_bilinmeyen_atlanir():
    sozluk, rapor, ucluler = sozlugu_buyut(VARSAYILAN_SOZLUK,
                                           ["kuantum bilgisayar nedir"])
    assert rapor.taranan_cumle == 1
    assert rapor.eslesen_cumle == 0
    assert rapor.atlanan_cumle == 1
    assert rapor.yeni_ozne == 0 and rapor.yeni_nesne == 0
    assert ucluler == []


def test_buyutme_iliski_uydurulmaz():
    sozluk, rapor, _ = sozlugu_buyut(VARSAYILAN_SOZLUK, KORPUS)
    assert set(sozluk.keys()) == set(VARSAYILAN_SOZLUK.keys())
    assert rapor.yeni_iliski == 0


def test_buyutme_durum_eki_sarti():
    """Nominatif yeni nesne ('kamyon') yönelme bekleyen binmek'te EKLENMEZ."""
    sozluk, rapor, _ = sozlugu_buyut(VARSAYILAN_SOZLUK, ["Ali kamyon bindi."])
    assert "kamyon" not in sozluk["binmek"]["nesneler"]
    assert rapor.yeni_nesne == 0
    assert rapor.eslesen_cumle == 0


def test_buyutme_orijinal_sozluk_degismez():
    onceki = set(VARSAYILAN_SOZLUK["binmek"]["ozneler"].keys())
    sozlugu_buyut(VARSAYILAN_SOZLUK, KORPUS)
    assert set(VARSAYILAN_SOZLUK["binmek"]["ozneler"].keys()) == onceki
    assert "mehmet" not in VARSAYILAN_SOZLUK["binmek"]["ozneler"]


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
