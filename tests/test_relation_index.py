# -*- coding: utf-8 -*-
"""
PropertyIndex + RelationIndex testleri
=======================================
(v0.1 — rapor §5, §6, §14 madde 2-3)

Kapsam:
  * 1/0 boolean özellikler + [0,1] genişletilebilirlik + sürüm/güven
  * ilişki tanımı (kısıtlarla) ve özne-ilişki-nesne kanıtları
  * olgu_agrega: kaynak güvenilirliğiyle ağırlıklı birleştirme

Çalıştırma:
    python tests/test_relation_index.py
    pytest tests/test_relation_index.py -q
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.knowledge import PropertyIndex, RelationIndex, KaynakTuru  # noqa: E402


def test_property_boolean_ve_guven():
    p = PropertyIndex()
    p.koy("E_001", "binilebilir", 1, source=KaynakTuru.REAL_DATA)
    p.koy("E_002", "binilebilir", 0, source=KaynakTuru.REAL_DATA)
    assert p.al("E_001", "binilebilir").deger == 1.0
    assert p.al("E_002", "binilebilir").deger == 0.0
    assert p.al("E_001", "binilebilir").confidence == 1.0
    assert p.al("E_999", "binilebilir") is None


def test_property_aralik_kontrolu():
    p = PropertyIndex()
    hata = None
    try:
        p.koy("E_001", "x", 1.5)
    except ValueError as e:
        hata = e
    assert hata is not None, "[0,1] dışı değer reddedilmeli"


def test_property_gunceleme_surum():
    p = PropertyIndex()
    p.koy("E_001", "binilebilir", 1)
    v2 = p.koy("E_001", "binilebilir", 0.7, source=KaynakTuru.HUMAN_CONFIRMED,
               confidence=0.8)
    assert v2.version == 2
    assert v2.deger == 0.7
    assert v2.source == KaynakTuru.HUMAN_CONFIRMED


def test_iliski_tanimlama():
    r = RelationIndex()
    binmek = r.iliski_ekle("Binmek", relation_id="R_001",
                           subject_types=["insan"],
                           requires_object_props={"binilebilir": 1.0})
    assert binmek.relation_id == "R_001"
    assert r.iliski_al("R_001").requires_object_props == {"binilebilir": 1.0}
    assert "insan" in r.iliski_al("R_001").subject_types


def test_olgu_ekleme_ve_sorgu():
    r = RelationIndex()
    r.iliski_ekle("Binmek", relation_id="R_001")
    r.olgu_ekle("E_001", "R_001", "E_002", 0.95, source=KaynakTuru.REAL_DATA)
    r.olgu_ekle("E_001", "R_001", "E_003", 0.9, source=KaynakTuru.REAL_DATA)
    assert len(r.olgular()) == 2
    assert len(r.olgular(subject_id="E_001")) == 2
    assert len(r.olgular(object_id="E_003")) == 1
    assert r.olgular(object_id="E_999") == []


def test_olgu_agrega_kaynak_agirligi():
    """MODEL_GENERATED kanıt, REAL_DATA kanıttan daha az ağırlık taşır (§10)."""
    r = RelationIndex()
    r.iliski_ekle("Binmek", relation_id="R_001")
    r.olgu_ekle("E_001", "R_001", "E_002", 1.0, source=KaynakTuru.MODEL_GENERATED,
                confidence=0.5)
    r.olgu_ekle("E_001", "R_001", "E_002", 0.0, source=KaynakTuru.REAL_DATA,
                confidence=1.0)
    agrega = r.olgu_agrega("E_001", "R_001", "E_002")
    # gerçek veri ağırlıklı ortalama → 0.0'a yakın, 0.5'ten küçük olmalı
    assert agrega["score"] < 0.5
    assert agrega["count"] == 2
    assert "REAL_DATA" in agrega["sources"]


def test_ters_iliski_ve_cift_yonlu_turetim():
    """P0-008: Relation yönü ve ters/simetrik ilişki türetimi."""
    r = RelationIndex()
    r.iliski_ekle("Binmek", relation_id="R_001", inverse_relation_id="R_001_INV")
    r.iliski_ekle("Binilmek", relation_id="R_001_INV", inverse_relation_id="R_001")

    assert r.ters_iliski_al("R_001") == "R_001_INV"
    assert r.ters_iliski_al("R_001_INV") == "R_001"

    # Otomatik ters olgu türetimi
    f = r.olgu_ekle("E_001", "R_001", "E_002", score=0.98,
                     source=KaynakTuru.REAL_DATA, confidence=1.0,
                     otomatik_ters=True)
    assert f.uclusu == ("E_001", "R_001", "E_002")

    olgular = r.olgular()
    assert len(olgular) == 2
    ters_olgu = r.olgular(subject_id="E_002", relation_id="R_001_INV")[0]
    assert ters_olgu.object_id == "E_001"
    assert ters_olgu.source == KaynakTuru.DERIVED
    assert ters_olgu.score == 0.98


def test_simetrik_iliski():
    """Simetrik ilişki (ör. Komşu, Eş, Benzer)."""
    r = RelationIndex()
    r.iliski_ekle("Benzer", relation_id="R_SYM", symmetric=True)
    assert r.ters_iliski_al("R_SYM") == "R_SYM"


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
