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
                            unlu_dusmesi, iyelik_eki, iyelik_li_durum,
                            gecmis_zaman_3tekil,
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


# ── Ünlü düşmesi (vowel elision) ─────────────────────────────────────────────
def test_unlu_dusmesi_govde():
    assert unlu_dusmesi("burun") == "burn"
    assert unlu_dusmesi("şehir") == "şehr"
    assert unlu_dusmesi("isim") == "ism"
    assert unlu_dusmesi("kayıp") == "kayb"        # elision + yumuşama
    assert unlu_dusmesi("fesat") == "fesad"
    assert unlu_dusmesi("top") == "top"           # listede yok → değişmez
    assert unlu_dusmesi("Ali", ozel_isim=True) == "Ali"


def test_unlu_dusmesi_yonelme_belirtme():
    assert yonelme_eki("burun") == "burna"        # burun → burna
    assert belirtme_eki("burun") == "burnu"
    assert yonelme_eki("şehir") == "şehre"
    assert belirtme_eki("şehir") == "şehri"
    assert yonelme_eki("isim") == "isme"
    assert belirtme_eki("kayıp") == "kaybı"       # kayıp → kaybı
    assert belirtme_eki("vakit") == "vakti"       # ince uyumlu alıntı
    assert yonelme_eki("vakit") == "vakte"
    assert yonelme_eki("fesat") == "fesada"


def test_unlu_dusmesi_bulunma_ayrilma_yok():
    # Ünsüzle başlayan eklerde düşme YOKTUR: burunda (burna değil), vakit→vakitte.
    assert bulunma_eki("burun") == "burunda"
    assert ayrilma_eki("burun") == "burundan"
    assert bulunma_eki("şehir") == "şehirde"
    assert bulunma_eki("vakit") == "vakitte"      # ince uyum + benzeşme


# ── İyelik (sahiplik) ekleri ─────────────────────────────────────────────────
def test_iyelik_eki():
    assert iyelik_eki("ev", "1t") == "evim"
    assert iyelik_eki("ev", "2t") == "evin"
    assert iyelik_eki("ev", "3t") == "evi"
    assert iyelik_eki("ev", "1c") == "evimiz"
    assert iyelik_eki("ev", "2c") == "eviniz"
    assert iyelik_eki("ev", "3c") == "evleri"
    # sesliyle biten gövde + 3t 's' kaynaştırma
    assert iyelik_eki("araba", "3t") == "arabası"
    assert iyelik_eki("araba", "1c") == "arabamız"
    # yuvarlak ünlü uyumu
    assert iyelik_eki("okul", "3t") == "okulu"
    assert iyelik_eki("göz", "3t") == "gözü"
    assert iyelik_eki("göz", "1c") == "gözümüz"


def test_iyelik_yumusama_ve_dusme():
    assert iyelik_eki("kitap", "1t") == "kitabım"   # yumuşama
    assert iyelik_eki("kitap", "3c") == "kitapları"  # 3c yumuşamaz
    assert iyelik_eki("ağaç", "3t") == "ağacı"
    assert iyelik_eki("burun", "1t") == "burnum"     # ünlü düşmesi
    assert iyelik_eki("şehir", "3t") == "şehri"
    assert iyelik_eki("şehir", "3c") == "şehirleri"  # 3c düşmez


def test_iyelik_gecersiz_kisi():
    hata = None
    try:
        iyelik_eki("ev", "4t")
    except ValueError as e:
        hata = e
    assert hata is not None


# ── İyelik + durum zinciri ───────────────────────────────────────────────────
def test_iyelik_li_durum():
    # 3. tekil iyelikten sonra durum eki 'n' ara harfi alır.
    assert iyelik_li_durum("ev", "3t", "yonelme") == "evine"
    assert iyelik_li_durum("ev", "3t", "belirtme") == "evini"
    assert iyelik_li_durum("ev", "3t", "bulunma") == "evinde"
    assert iyelik_li_durum("ev", "3t", "ayrilma") == "evinden"
    assert iyelik_li_durum("araba", "3t", "yonelme") == "arabasına"
    assert iyelik_li_durum("araba", "3t", "bulunma") == "arabasında"
    # diğer kişilerde ara harf YOK.
    assert iyelik_li_durum("ev", "1t", "yonelme") == "evime"
    assert iyelik_li_durum("araba", "2t", "bulunma") == "arabanda"
    assert iyelik_li_durum("ev", "1c", "ayrilma") == "evimizden"


# ── Görülen geçmiş zaman 3. tekil ────────────────────────────────────────────
def test_gecmis_zaman_3tekil():
    assert gecmis_zaman_3tekil("bin") == "bindi"
    assert gecmis_zaman_3tekil("bak") == "baktı"     # ötümsüz → -tı
    assert gecmis_zaman_3tekil("git") == "gitti"
    assert gecmis_zaman_3tekil("sev") == "sevdi"
    assert gecmis_zaman_3tekil("gel") == "geldi"
    assert gecmis_zaman_3tekil("gör") == "gördü"     # yuvarlak → -dü
    assert gecmis_zaman_3tekil("dur") == "durdu"
    assert gecmis_zaman_3tekil("oku") == "okudu"     # sesli biten kök
    assert gecmis_zaman_3tekil("izle") == "izledi"
    # mastar kökünden otomatik düşürme
    assert gecmis_zaman_3tekil("bakmak") == "baktı"
    assert gecmis_zaman_3tekil("gelmek") == "geldi"
    assert gecmis_zaman_3tekil("") == ""


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
