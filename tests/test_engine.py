# -*- coding: utf-8 -*-
"""
ExperienceEngine entegrasyon testleri
========================================
Çalıştırma:
    python tests/test_engine.py
    pytest tests/test_engine.py -q
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.engine import ExperienceEngine  # noqa: E402
from hga.experience import AritmetikOrtam  # noqa: E402


def _bilgi_engine():
    e = ExperienceEngine()
    e.store.varlik_ekle("Ali", entity_type="insan", properties={"canli": 1},
                        entity_id="E_001", ozel_isim=True)
    e.store.varlik_ekle("Ata", entity_type="hayvan", properties={"binilebilir": 1},
                        entity_id="E_002")
    e.store.varlik_ekle("Araba", entity_type="tasit", properties={"binilebilir": 1},
                        entity_id="E_003")
    e.store.varlik_ekle("Gökyüzü", entity_type="mekan", properties={"binilebilir": 0},
                        entity_id="E_004")
    e.store.iliski_tanimla("Binmek", relation_id="R_001",
                           subject_types=["insan"],
                           requires_object_props={"binilebilir": 1.0})
    return e


def test_engine_bilgi_uret_degerlendir():
    e = _bilgi_engine()
    adaylar = e.uret(["R_001"])
    assert len(adaylar) == 3
    e.degerlendir(adaylar)
    sonuc = {a.object_id: a.state.value for a in adaylar}
    assert sonuc["E_002"] == "VALID"       # Ata
    assert sonuc["E_003"] == "VALID"       # Araba
    assert sonuc["E_004"] == "INVALID"     # Gökyüzü


def test_engine_metin():
    e = _bilgi_engine()
    adaylar = e.uret(["R_001"])
    e.degerlendir(adaylar)
    ata = next(a for a in adaylar if a.object_id == "E_002")
    assert e.metin(ata) == "Ali ataya bindi."


def test_engine_gercek_veri():
    e = ExperienceEngine()
    aktarilan = e.gercek_veri(
        ["Ali ataya bindi.", "Ali gökyüzüne baktı."],
        iliski_kisitlari={"Binmek": {"subject_types": ["insan"],
                                     "requires_object_props": {"binilebilir": 1.0}}})
    assert len(aktarilan) == 2
    assert e.store.ozet()["varlik"] == 3   # Ali, Ata, Gökyüzü


def test_engine_dogrulama_ve_dongu():
    from hga.experience import aritmetik_etki_alani
    store = aritmetik_etki_alani()
    e = ExperienceEngine(store=store, dogrulayici=AritmetikOrtam().aday_dogrula)
    adaylar = e.uret()
    e.degerlendir(adaylar)
    rapor = e.dogrula(adaylar)
    assert rapor.dogrulanan == 6
    assert rapor.reddedilen == 24
    # döngü çalışır ve bilgi büyümesi raporlanır
    raporlar = e.dongu(2)
    assert len(raporlar) == 2


def test_engine_dogrulayicisiz_dogrula_none():
    e = _bilgi_engine()
    adaylar = e.uret(["R_001"])
    e.degerlendir(adaylar)
    assert e.dogrula(adaylar) is None   # doğrulayıcı yok → dürüstçe None


def test_engine_ozet():
    e = _bilgi_engine()
    ozet = e.ozet()
    assert ozet["bilgi"]["varlik"] == 4
    assert "bellek" in ozet and "arastirma_kuyrugu" in ozet


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
