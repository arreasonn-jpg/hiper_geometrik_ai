# -*- coding: utf-8 -*-
"""
EntityIndex testleri
=====================
(v0.1 — rapor §4, §14 madde 1)

Kapsam:
  * benzersiz entity_id atama (E_001, E_002, ...) ve idempotent ekleme
  * token (ad) ile arama, tip ile filtreleme
  * sürüm artırımlı güncelleme
  * serileştirme (to_dict/from_dict) round-trip

Çalıştırma:
    python tests/test_entity_index.py
    pytest tests/test_entity_index.py -q
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.knowledge import EntityIndex, KaynakTuru  # noqa: E402


def test_benzersiz_id_atama():
    idx = EntityIndex()
    ali = idx.ekle("Ali", entity_type="insan")
    ata = idx.ekle("Ata", entity_type="hayvan")
    assert ali.entity_id == "E_001"
    assert ata.entity_id == "E_002"
    assert ali.entity_id != ata.entity_id
    assert len(idx) == 2


def test_idempotent_ekleme():
    """Aynı ad ikinci kez eklenirse aynı varlık döner (yeni ID üretilmez)."""
    idx = EntityIndex()
    a1 = idx.ekle("Araba", entity_type="tasit")
    a2 = idx.ekle("araba", entity_type="tasit")   # küçük/büyük harf farkı
    assert a1.entity_id == a2.entity_id
    assert len(idx) == 1


def test_ad_ile_bulma():
    idx = EntityIndex()
    idx.ekle("Gökyüzü", entity_type="mekan")
    assert idx.ad_bul("gökyüzü") == "E_001"
    assert idx.ad_bul("gökyüzü ") == "E_001"      # boşluk toleransı
    assert idx.ad_bul("deniz") is None


def test_tip_ile_filtreleme():
    idx = EntityIndex()
    idx.ekle("Ali", entity_type="insan")
    idx.ekle("Ata", entity_type="hayvan")
    idx.ekle("Araba", entity_type="tasit")
    insanlar = idx.tip_ile("insan")
    assert [e.token for e in insanlar] == ["Ali"]


def test_guncelleme_surum_artirir():
    idx = EntityIndex()
    v = idx.ekle("Ali", entity_type="insan")
    assert v.version == 1
    idx.guncelle(v.entity_id, confidence=0.9, context_tags=["test"])
    assert v.version == 2
    assert v.confidence == 0.9
    assert v.context_tags == ["test"]


def test_serilestirme_roundtrip():
    idx = EntityIndex()
    idx.ekle("Ali", entity_type="insan", source=KaynakTuru.REAL_DATA)
    idx.ekle("Ata", entity_type="hayvan")
    geri = EntityIndex.from_dict(idx.to_dict())
    assert len(geri) == 2
    assert geri.getir("E_001").token == "Ali"
    assert geri.getir("E_002").entity_type == "hayvan"
    # kaynak enum'u doğru geri geldi
    assert geri.getir("E_001").source == KaynakTuru.REAL_DATA


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
