# -*- coding: utf-8 -*-
"""
v0.2 (metin/olay üretimi) + v0.3 (information gain) testleri
==============================================================
Çalıştırma:
    python tests/test_v02_v03.py
    pytest tests/test_v02_v03.py -q
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.knowledge import KnowledgeStore  # noqa: E402
from hga.experience import TextGenerator, Scoring  # noqa: E402


def _store():
    k = KnowledgeStore()
    k.varlik_ekle("Ali", entity_type="insan", properties={"canli": 1}, entity_id="E_001")
    k.varlik_ekle("Ata", entity_type="hayvan", properties={"binilebilir": 1}, entity_id="E_002")
    k.varlik_ekle("Araba", entity_type="tasit", properties={"binilebilir": 1}, entity_id="E_003")
    k.varlik_ekle("Gökyüzü", entity_type="mekan", properties={"binilebilir": 0}, entity_id="E_004")
    k.iliski_tanimla("Binmek", relation_id="R_001",
                     subject_types=["insan"],
                     requires_object_props={"binilebilir": 1.0})
    return k


def test_text_generator_binmek():
    k = _store()
    tg = TextGenerator()
    from hga.knowledge import ExperienceCandidate
    a = ExperienceCandidate(experience_id="X", subject_id="E_001",
                            relation_id="R_001", object_id="E_002")
    assert tg.cumle(k, a) == "Ali ataya bindi."


def test_text_generator_olay():
    k = _store()
    tg = TextGenerator()
    from hga.knowledge import ExperienceCandidate
    a = ExperienceCandidate(experience_id="X", subject_id="E_001",
                            relation_id="R_001", object_id="E_003")
    olay = tg.olay(k, a)
    assert olay["ozne"] == "Ali"
    assert olay["iliski"] == "Binmek"
    assert olay["nesne"] == "Araba"
    assert olay["cumle"] == "Ali arabaya bindi."


def test_text_generator_ozel_isim_kesme_isareti():
    """Özel isim nesne → kesme işareti (Ata'ya) ve virgül."""
    k = KnowledgeStore()
    k.varlik_ekle("Ali", entity_type="insan", entity_id="E_001", ozel_isim=True)
    k.varlik_ekle("Ata", entity_type="insan", entity_id="E_002", ozel_isim=True)
    k.iliski_tanimla("Binmek", relation_id="R_001")
    tg = TextGenerator()
    from hga.knowledge import ExperienceCandidate
    a = ExperienceCandidate(experience_id="X", subject_id="E_001",
                            relation_id="R_001", object_id="E_002")
    assert tg.cumle(k, a) == "Ali, Ata'ya bindi."


def test_text_generator_ozel_uretec():
    """İlişkiye özel üreteç değiştirilebilir (pluggable)."""
    k = _store()
    tg = TextGenerator()
    tg.kayit_ekle("Binmek", lambda s, o: f"{s} {o} üzerinde deneyim yaşadı.")
    from hga.knowledge import ExperienceCandidate
    a = ExperienceCandidate(experience_id="X", subject_id="E_001",
                            relation_id="R_001", object_id="E_002")
    assert tg.cumle(k, a) == "Ali Ata üzerinde deneyim yaşadı."


def test_text_generator_durum_ekleri():
    """sevmek (belirtme), durmak (bulunma), gelmek (ayrılma) şablonları."""
    k = KnowledgeStore()
    k.varlik_ekle("Ali", entity_type="insan", entity_id="E_001", ozel_isim=True)
    k.varlik_ekle("Araba", entity_type="tasit", entity_id="E_002")
    k.varlik_ekle("Ev", entity_type="mekan", entity_id="E_003")
    k.iliski_tanimla("Sevmek", relation_id="R_001")
    k.iliski_tanimla("Durmak", relation_id="R_002")
    k.iliski_tanimla("Gelmek", relation_id="R_003")
    tg = TextGenerator()
    from hga.knowledge import ExperienceCandidate
    a1 = ExperienceCandidate(experience_id="X1", subject_id="E_001",
                             relation_id="R_001", object_id="E_002")
    a2 = ExperienceCandidate(experience_id="X2", subject_id="E_001",
                             relation_id="R_002", object_id="E_003")
    a3 = ExperienceCandidate(experience_id="X3", subject_id="E_001",
                             relation_id="R_003", object_id="E_003")
    assert tg.cumle(k, a1) == "Ali arabayı seviyor."   # belirtme
    assert tg.cumle(k, a2) == "Ali evde duruyor."      # bulunma
    assert tg.cumle(k, a3) == "Ali evden geliyor."     # ayrılma


def test_information_gain_ayirt_edicilik():
    """Birebir aynı özelliklere sahip nesne → düşük kazanç; benzersiz → yüksek."""
    k = _store()
    sc = Scoring()
    ata = k.entities.getir("E_002")     # {binilebilir:1} — Araba ile aynı
    ali = k.entities.getir("E_001")     # {canli:1} — kimseyle örtüşmez
    ig_ata = sc.information_gain(k, ata)
    ig_ali = sc.information_gain(k, ali)
    assert ig_ata == 0.0                # en yakın komşu (Araba) ile özdeş
    assert ig_ali == 1.0                # hiçbir varlıkla örtüşmüyor


def test_skor_breakdown_ig_icerecek():
    k = _store()
    sc = Scoring()
    from hga.knowledge import ExperienceCandidate
    a = ExperienceCandidate(experience_id="X", subject_id="E_001",
                            relation_id="R_001", object_id="E_002")
    br = sc.skorla(k, a, k.entities.getir("E_001"),
                   k.relations.iliski_al("R_001"), k.entities.getir("E_002"))
    assert "information_gain" in br.to_dict()
    assert 0.0 <= br.information_gain <= 1.0


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
