# -*- coding: utf-8 -*-
"""
Korpus borusu testleri — veri toplayıcı çıktısı → sözlük büyütme → REAL_DATA
=============================================================================
Çalıştırma:
    python tests/test_korpus_boru.py
    pytest tests/test_korpus_boru.py -q
"""
import os
import sys
import tempfile

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.experience import (  # noqa: E402
    korpus_borusu,
    korpus_dosyasindan,
    veri_toplayici_ciktisindan,
)
from hga.knowledge import KnowledgeStore  # noqa: E402

METIN = ("Ali ataya bindi.\n"
         "Mehmet kamyona bindi.\n"
         "Zeynep dağa baktı.\n")


def test_korpus_borusu_aktarir():
    k = KnowledgeStore()
    rapor = korpus_borusu(k, METIN)
    assert rapor.cumle_sayisi == 3
    assert rapor.aktarilan_uclu == 3
    assert rapor.buyutme.yeni_ozne == 2      # Mehmet, Zeynep
    assert rapor.buyutme.yeni_nesne == 2     # Kamyon, Dağ
    ozet = k.ozet()
    assert ozet["kanit"] == 3
    # yeni varlıklar REAL_DATA kaynaklı kuruldu
    assert k.entities.getir(k.entities.ad_bul("Kamyon")).source.value == "REAL_DATA"
    assert k.entities.getir(k.entities.ad_bul("Mehmet")).entity_type == "insan"
    # binmek kanıtı yeni nesneyle yazıldı
    rid = next(r.relation_id for r in k.relations.iliskiler()
               if r.token == "Binmek")
    assert k.relations.olgu_agrega(k.entities.ad_bul("Mehmet"), rid,
                                   k.entities.ad_bul("Kamyon"))["count"] == 1


def test_korpus_dosyasindan():
    k = KnowledgeStore()
    with tempfile.TemporaryDirectory() as d:
        yol = os.path.join(d, "korpus.txt")
        with open(yol, "w", encoding="utf-8") as f:
            f.write(METIN)
        rapor = korpus_dosyasindan(k, yol)
    assert rapor.aktarilan_uclu == 3
    assert k.ozet()["kanit"] == 3


def test_veri_toplayici_ciktisindan():
    """veri_toplayici.py'nin metni_kaydet sözleşmesi (turkce_metin.txt)."""
    k = KnowledgeStore()
    with tempfile.TemporaryDirectory() as d:
        yol = os.path.join(d, "turkce_metin.txt")
        with open(yol, "w", encoding="utf-8") as f:
            f.write(METIN)
        rapor = veri_toplayici_ciktisindan(k, d)
    assert rapor.aktarilan_uclu == 3


def test_veri_toplayici_dosya_yoksa_hata():
    k = KnowledgeStore()
    with tempfile.TemporaryDirectory() as d:
        hata = None
        try:
            veri_toplayici_ciktisindan(k, d)
        except FileNotFoundError as e:
            hata = e
        assert hata is not None
    assert k.ozet()["kanit"] == 0


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
