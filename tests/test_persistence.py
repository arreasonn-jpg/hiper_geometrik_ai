# -*- coding: utf-8 -*-
"""
Kalıcılık testleri — KnowledgeStore kaydet/yükle
==================================================
Çalıştırma:
    python tests/test_persistence.py
    pytest tests/test_persistence.py -q
"""
import os
import sys
import tempfile

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.knowledge import KaynakTuru, KnowledgeStore, kaydet, yukle  # noqa: E402


def _store():
    k = KnowledgeStore()
    k.varlik_ekle("Ali", entity_type="insan", properties={"canli": 1},
                  entity_id="E_001", ozel_isim=True)
    k.varlik_ekle("Gökyüzü", entity_type="mekan", properties={"binilebilir": 0},
                  entity_id="E_002")
    k.iliski_tanimla("Binmek", relation_id="R_001",
                     subject_types=["insan"],
                     requires_object_props={"binilebilir": 1.0})
    return k


def test_roundtrip():
    k = _store()
    with tempfile.TemporaryDirectory() as d:
        yol = os.path.join(d, "bilgi.json")
        kaydet(k, yol)
        k2 = yukle(yol)
    assert k2.ozet() == k.ozet()
    # Türkçe karakter ve özel isim bilgisi korundu
    gok = k2.entities.getir("E_002")
    assert gok.token == "Gökyüzü"
    assert k2.properties.al("E_002", "binilebilir").deger == 0.0
    assert k2.entities.getir("E_001").is_ozel is True
    # kaynak enum'u doğru geri geldi
    assert k2.relations.iliski_al("R_001").source == KaynakTuru.VERIFIED_RULE


def test_method_kaydet_yukle():
    k = _store()
    with tempfile.TemporaryDirectory() as d:
        yol = os.path.join(d, "bilgi.json")
        k.kaydet(yol)
        k2 = KnowledgeStore.yukle(yol)
    assert k2.ozet()["varlik"] == 2


def test_turkce_karakter_ascii_degil():
    """JSON'da Türkçe karakterler kaçış dizisi olarak DEĞİL, doğrudan yazılır."""
    k = _store()
    with tempfile.TemporaryDirectory() as d:
        yol = os.path.join(d, "bilgi.json")
        kaydet(k, yol)
        with open(yol, "r", encoding="utf-8") as f:
            ham = f.read()
    assert "Gökyüzü" in ham
    assert "\\u00f6" not in ham


def test_atomik_yazma_tmp_birakmaz():
    k = _store()
    with tempfile.TemporaryDirectory() as d:
        yol = os.path.join(d, "bilgi.json")
        kaydet(k, yol)
        kalintilar = [f for f in os.listdir(d) if f.endswith(".tmp")]
        assert kalintilar == []


def test_yukle_eksik_dosya_hata():
    with tempfile.TemporaryDirectory() as d:
        hata = None
        try:
            yukle(os.path.join(d, "yok.json"))
        except FileNotFoundError as e:
            hata = e
        assert hata is not None


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
