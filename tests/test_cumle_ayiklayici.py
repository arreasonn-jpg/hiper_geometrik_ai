# -*- coding: utf-8 -*-
"""
Cümle ayıklayıcı + REAL_DATA aktarımı testleri
================================================
Çalıştırma:
    python tests/test_cumle_ayiklayici.py
    pytest tests/test_cumle_ayiklayici.py -q
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.experience import (  # noqa: E402
    CumleAyiklayici,
    ExperienceEvaluator,
    cumlelerden_bilgi_aktar,
)
from hga.knowledge import DeneyimDurumu, KaynakTuru, KnowledgeStore  # noqa: E402


def test_ayikla_binmek():
    ay = CumleAyiklayici()
    u = ay.ayikla("Ali ataya bindi.")
    assert u is not None
    assert (u.ozne, u.iliski, u.nesne) == ("Ali", "Binmek", "Ata")


def test_ayikla_turkce_karakterler():
    ay = CumleAyiklayici()
    u = ay.ayikla("Ali gökyüzüne baktı.")   # ğ/ö/ü/ş/ı/ç karakterleri
    assert u is not None
    assert (u.ozne, u.iliski, u.nesne) == ("Ali", "Bakmak", "Gökyüzü")


def test_ayikla_eslesmezse_none():
    ay = CumleAyiklayici()
    assert ay.ayikla("kuantum bilgisayar nedir") is None  # sözlük dışı
    assert ay.ayikla("") is None


def test_ayikla_genis_sozluk():
    """Genişletilmiş sözlük: gitmek/gelmek (yönelme) + okumak/yazmak/sevmek (belirtme)."""
    ay = CumleAyiklayici()
    beklenen = [
        ("Ali okula gitti.", "Gitmek", "Okul"),
        ("Ayşe eve geldi.", "Gelmek", "Ev"),
        ("Ali kitabı okudu.", "Okumak", "Kitap"),
        ("Ayşe mektubu yazdı.", "Yazmak", "Mektup"),
        ("Ali kediyi sevdi.", "Sevmek", "Kedi"),
    ]
    for cumle, iliski, nesne in beklenen:
        u = ay.ayikla(cumle)
        assert u is not None, cumle
        assert u.iliski == iliski, cumle
        assert u.nesne == nesne, cumle


def test_ayikla_genis_sozluk_belirtme_ekli():
    """Belirtme ekli (yumuşamalı) nesneler doğru kökle çıkarılır."""
    ay = CumleAyiklayici()
    u = ay.ayikla("Ali gazeteyi okudu.")
    assert u is not None and u.nesne == "Gazete"
    u = ay.ayikla("Ali şiiri yazdı.")
    assert u is not None and u.nesne == "Şiir"


def test_bilgi_aktar_real_data():
    """Cümleler → REAL_DATA varlık + kanıt (döngünün 'Gerçek veri' aşaması)."""
    k = KnowledgeStore()
    aktarilan = cumlelerden_bilgi_aktar(
        k,
        ["Ali ataya bindi.", "Ali arabaya bindi."],
        iliski_kisitlari={
            "Binmek": {"subject_types": ["insan"],
                       "requires_object_props": {"binilebilir": 1.0}},
        })
    assert len(aktarilan) == 2
    # varlıklar REAL_DATA kaynaklı ve özellikleriyle kuruldu
    ata = k.entities.getir(k.entities.ad_bul("Ata"))
    assert ata.entity_type == "hayvan"
    assert k.properties.al(ata.entity_id, "binilebilir").deger == 1.0
    assert k.properties.al(ata.entity_id, "binilebilir").source == KaynakTuru.REAL_DATA
    # kanıtlar REAL_DATA olarak yazıldı
    ali = k.entities.ad_bul("Ali")
    araba = k.entities.ad_bul("Araba")
    rid = next(r.relation_id for r in k.relations.iliskiler() if r.token == "Binmek")
    agrega = k.relations.olgu_agrega(ali, rid, araba)
    assert agrega is not None and agrega["count"] == 1
    assert "REAL_DATA" in agrega["sources"]


def test_bilgi_aktar_sonra_degerlendirme():
    """Gerçek veriyle kurulan bilgi, değerlendirmeyi besler."""
    k = KnowledgeStore()
    cumlelerden_bilgi_aktar(
        k,
        ["Ali ataya bindi.", "Ali gökyüzüne bindi."],
        iliski_kisitlari={
            "Binmek": {"subject_types": ["insan"],
                       "requires_object_props": {"binilebilir": 1.0}},
        })
    # Gökyüzü gerçek veriden binilebilir=0 olarak geldi → aday INVALID
    from hga.knowledge import ExperienceCandidate
    ali = k.entities.ad_bul("Ali")
    gok = k.entities.ad_bul("Gökyüzü")
    rid = next(r.relation_id for r in k.relations.iliskiler() if r.token == "Binmek")
    ev = ExperienceEvaluator()
    a = ExperienceCandidate(experience_id="X", subject_id=ali,
                            relation_id=rid, object_id=gok)
    ev.degerlendir(a, k)
    assert a.state == DeneyimDurumu.INVALID


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
