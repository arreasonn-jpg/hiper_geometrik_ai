# -*- coding: utf-8 -*-
"""
Araştırma kuyruğu testleri — CONFLICT → deterministik kanıt → kesin durum
===========================================================================
Çalıştırma:
    python tests/test_arastirma.py
    pytest tests/test_arastirma.py -q
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.knowledge import KnowledgeStore, DeneyimDurumu, KaynakTuru  # noqa: E402
from hga.experience import (ExperienceEvaluator, ArastirmaKuyrugu,  # noqa: E402
                            ConflictResolver)


def _store():
    k = KnowledgeStore()
    k.varlik_ekle("Ali", entity_type="insan", properties={"canli": 1},
                  entity_id="E_001", ozel_isim=True)
    k.varlik_ekle("Halı", entity_type="esya", entity_id="E_002")  # özellik bilinmiyor
    k.varlik_ekle("Ata", entity_type="hayvan", properties={"binilebilir": 1},
                  entity_id="E_003")
    k.iliski_tanimla("Binmek", relation_id="R_001",
                     subject_types=["insan"],
                     requires_object_props={"binilebilir": 1.0})
    return k


def _belirsiz_aday():
    from hga.knowledge import ExperienceCandidate
    return ExperienceCandidate(experience_id="X_PROBE", subject_id="E_001",
                               relation_id="R_001", object_id="E_002")


def test_kuyruk_kanitla_invalid_cozer():
    """Kanıt 'binilebilir=0' derse çelişki INVALID'e iner, bilgi güncellenir."""
    k = _store()
    ev = ExperienceEvaluator()
    a = _belirsiz_aday()
    ev.degerlendir(a, k)
    assert a.state == DeneyimDurumu.UNCERTAIN

    # deterministik doğrulayıcı: Halı binilebilir DEĞİLDİR
    kuyruk = ArastirmaKuyrugu(evaluator=ev,
                              dogrulayici=lambda store, aday: False)
    kuyruk.ekle(a)
    rapor = kuyruk.isle(k)

    assert rapor.islenen == 1
    assert rapor.cozulen == 1 and rapor.acik == 0
    assert a.state == DeneyimDurumu.INVALID
    # kanıt, bilgi tabanına yazıldı (bilgi büyümesi)
    assert rapor.bilgi_buyumesi >= 1
    assert k.properties.al("E_002", "binilebilir").deger == 0.0
    assert k.properties.al("E_002", "binilebilir").source == KaynakTuru.VERIFIED_RULE


def test_kuyruk_kanitla_valid_cozer():
    """Kanıt 'binilebilir=1' derse çelişki VALID'e iner (yine VERIFIED değil)."""
    k = _store()
    ev = ExperienceEvaluator()
    a = _belirsiz_aday()
    ev.degerlendir(a, k)
    assert a.state == DeneyimDurumu.UNCERTAIN

    kuyruk = ArastirmaKuyrugu(evaluator=ev,
                              dogrulayici=lambda store, aday: True)
    kuyruk.ekle(a)
    rapor = kuyruk.isle(k)

    assert rapor.cozulen == 1
    assert a.state == DeneyimDurumu.VALID
    assert a.state != DeneyimDurumu.VERIFIED          # MODEL_GENERATED güvenliği


def test_kuyruk_kanitsiz_acik_kalir():
    """Doğrulayıcı belirsiz (None) dönerse belirsizlik AÇIK kalır (yanlış yükseltme yok)."""
    k = _store()
    ev = ExperienceEvaluator()
    a = _belirsiz_aday()
    ev.degerlendir(a, k)
    assert a.state == DeneyimDurumu.UNCERTAIN

    kuyruk = ArastirmaKuyrugu(evaluator=ev,
                              dogrulayici=lambda store, aday: None)
    kuyruk.ekle(a)
    rapor = kuyruk.isle(k)

    assert rapor.cozulen == 0 and rapor.acik == 1
    assert a.state == DeneyimDurumu.UNCERTAIN
    assert len(kuyruk) == 1                          # kuyrukta kaldı


def test_kuyruk_besleme():
    k = _store()
    ev = ExperienceEvaluator()
    from hga.knowledge import ExperienceCandidate
    a1 = _belirsiz_aday()
    a2 = ExperienceCandidate(experience_id="X_OK", subject_id="E_001",
                             relation_id="R_001", object_id="E_003")
    ev.degerlendir(a1, k)
    ev.degerlendir(a2, k)
    kuyruk = ArastirmaKuyrugu(evaluator=ev)
    eklenen = kuyruk.besle([a1, a2])                # yalnızca araştırılması gereken durum eklenir
    assert eklenen == 1 and len(kuyruk) == 1
    assert kuyruk.kuyruk[0].experience_id == "X_PROBE"


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
