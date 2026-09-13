# -*- coding: utf-8 -*-
"""
Korpus yükleyici testleri — dosyadan cümle → üçlü → REAL_DATA
===============================================================
Çalıştırma:
    python tests/test_corpus.py
    pytest tests/test_corpus.py -q
"""
import os
import sys
import tempfile

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.experience import cumlelere_bol, dosyadan_bilgi_aktar  # noqa: E402
from hga.knowledge import KnowledgeStore  # noqa: E402


def test_cumlelere_bol():
    cumleler = cumlelere_bol("Ali ataya bindi. Ayşe arabaya bindi! Ali yürüdü?")
    assert len(cumleler) == 3
    assert cumleler[0] == "Ali ataya bindi."
    assert cumleler[1] == "Ayşe arabaya bindi!"


def test_dosyadan_bilgi_aktar():
    k = KnowledgeStore()
    metin = ("Ali ataya bindi.\n"
             "Ali arabaya bindi.\n"
             "Ayşe gökyüzüne baktı.\n")
    with tempfile.TemporaryDirectory() as d:
        yol = os.path.join(d, "korpus.txt")
        with open(yol, "w", encoding="utf-8") as f:
            f.write(metin)
        aktarilan = dosyadan_bilgi_aktar(
            k, yol,
            iliski_kisitlari={
                "Binmek": {"subject_types": ["insan"],
                           "requires_object_props": {"binilebilir": 1.0}},
            })
    assert len(aktarilan) == 3
    # varlıklar REAL_DATA kaynaklı kuruldu
    ayse = k.entities.getir(k.entities.ad_bul("Ayşe"))
    assert ayse.entity_type == "insan"
    assert k.entities.getir(k.entities.ad_bul("Gökyüzü")).token == "Gökyüzü"
    # binmek kanıtı yazıldı
    ali = k.entities.ad_bul("Ali")
    ata = k.entities.ad_bul("Ata")
    rid = next(r.relation_id for r in k.relations.iliskiler() if r.token == "Binmek")
    assert k.relations.olgu_agrega(ali, rid, ata)["count"] == 1


def test_dosyadan_bilgi_aktar_bilinmeyen_atlanir():
    """Sözlük dışı cümleler sessizce atlanır; uydurma yapılmaz."""
    k = KnowledgeStore()
    metin = "Ali ataya bindi.\nkuantum bilgisayar nedir\n"
    with tempfile.TemporaryDirectory() as d:
        yol = os.path.join(d, "korpus.txt")
        with open(yol, "w", encoding="utf-8") as f:
            f.write(metin)
        aktarilan = dosyadan_bilgi_aktar(k, yol)
    assert len(aktarilan) == 1
    assert k.ozet()["varlik"] == 2  # yalnız Ali ve Ata


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
