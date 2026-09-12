# -*- coding: utf-8 -*-
"""
Mini-environment (v0.5) testleri — deterministik aritmetik doğrulayıcı
========================================================================
Çalıştırma:
    python tests/test_mini_env.py
    pytest tests/test_mini_env.py -q
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.knowledge import KnowledgeStore  # noqa: E402
from hga.experience import AritmetikOrtam  # noqa: E402


def test_iddia_dogrula():
    ortam = AritmetikOrtam()
    assert ortam.iddia_dogrula("2+3=5") is True
    assert ortam.iddia_dogrula("2+3=6") is False
    assert ortam.iddia_dogrula("2+3") is None      # '=' yok → belirsiz
    assert ortam.iddia_dogrula("4/0=1") is None    # sıfıra bölme → belirsiz


def test_deger_guvenli():
    ortam = AritmetikOrtam()
    assert ortam.deger("2+3*2") == 8.0             # işlem önceliği
    assert ortam.deger("(2+3)*2") == 10.0
    assert ortam.deger("2-5") == -3.0
    assert ortam.deger("import os") is None        # eval yok, isim reddedilir
    assert ortam.deger("__import__('os')") is None


def test_aday_dogrula():
    k = KnowledgeStore()
    k.varlik_ekle("2+3", entity_type="ifade", entity_id="E_001")
    k.varlik_ekle("5", entity_type="sayi", entity_id="E_002")
    k.varlik_ekle("6", entity_type="sayi", entity_id="E_003")
    k.iliski_tanimla("eşittir", relation_id="R_001")

    ortam = AritmetikOrtam()
    from hga.knowledge import ExperienceCandidate
    dogru = ExperienceCandidate(experience_id="X1", subject_id="E_001",
                                relation_id="R_001", object_id="E_002")
    yanlis = ExperienceCandidate(experience_id="X2", subject_id="E_001",
                                 relation_id="R_001", object_id="E_003")
    assert ortam.aday_dogrula(k, dogru) is True
    assert ortam.aday_dogrula(k, yanlis) is False


def test_aday_dogrula_ilgisiz_iliski():
    """İlişki eşittir değilse belirsiz (None) döner — yanlış yargılamaz."""
    k = KnowledgeStore()
    k.varlik_ekle("Ali", entity_type="insan", entity_id="E_001")
    k.varlik_ekle("Ata", entity_type="hayvan", entity_id="E_002")
    k.iliski_tanimla("Binmek", relation_id="R_001")
    ortam = AritmetikOrtam()
    from hga.knowledge import ExperienceCandidate
    a = ExperienceCandidate(experience_id="X", subject_id="E_001",
                            relation_id="R_001", object_id="E_002")
    assert ortam.aday_dogrula(k, a) is None


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
