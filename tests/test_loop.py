# -*- coding: utf-8 -*-
"""
DeneyimDongusu (v1.0) testleri — sürekli öğrenme döngüsü + metrikler
=====================================================================
Çalıştırma:
    python tests/test_loop.py
    pytest tests/test_loop.py -q
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.experience import (  # noqa: E402
    AritmetikOrtam,
    Consolidator,
    DeneyimDongusu,
    ExperienceEvaluator,
    ExperienceGenerator,
)
from hga.knowledge import KnowledgeStore  # noqa: E402
from hga.memory import BellekEntegrasyonu  # noqa: E402


def test_dongu_bilgi_temelli():
    """Bilgi tabanlı tek adım: 3 aday, 2 VALID, 1 INVALID, novel=3."""
    k = KnowledgeStore()
    k.varlik_ekle("Ali", entity_type="insan", properties={"canli": 1}, entity_id="E_001")
    k.varlik_ekle("Ata", entity_type="hayvan", properties={"binilebilir": 1}, entity_id="E_002")
    k.varlik_ekle("Araba", entity_type="tasit", properties={"binilebilir": 1}, entity_id="E_003")
    k.varlik_ekle("Gökyüzü", entity_type="mekan", properties={"binilebilir": 0}, entity_id="E_004")
    k.iliski_tanimla("Binmek", relation_id="R_001",
                     subject_types=["insan"],
                     requires_object_props={"binilebilir": 1.0})

    dongu = DeneyimDongusu(
        store=k,
        generator=ExperienceGenerator(),
        evaluator=ExperienceEvaluator(),
        consolidator=Consolidator(),
        bellek=BellekEntegrasyonu(slot_sayisi=64, replay_kapasitesi=16),
    )
    rapor = dongu.adim(relation_ids=["R_001"])
    assert rapor.uretilen == 3
    assert rapor.valid == 2
    assert rapor.invalid == 1
    assert rapor.conflict == 0
    assert rapor.verified == 0               # MODEL_GENERATED asla VERIFIED değil
    assert rapor.novel == 3
    assert rapor.knowledge_buyumesi == 0     # kalıcı bilgi yazılmadı (yalnız aday)
    assert rapor.false_acceptance is None    # doğrulayıcı yok → dürüstçe None


def test_dongu_aritmetik_ortamla():
    """Deterministik mini-env ile false acceptance ölçülür (v0.5 + v1.0)."""
    k = KnowledgeStore()
    k.varlik_ekle("1+1", entity_type="ifade", entity_id="E_001")
    k.varlik_ekle("2", entity_type="sayi", entity_id="E_002")
    k.varlik_ekle("3", entity_type="sayi", entity_id="E_003")
    k.iliski_tanimla("eşittir", relation_id="R_001")

    ortam = AritmetikOrtam()
    gen = ExperienceGenerator(tip_filtresi=False)   # tüm kombinasyonlar
    dongu = DeneyimDongusu(
        store=k,
        generator=gen,
        evaluator=ExperienceEvaluator(),
        consolidator=Consolidator(),
        bellek=BellekEntegrasyonu(slot_sayisi=64, replay_kapasitesi=16),
        dogrulayici=ortam.aday_dogrula,
    )
    # kombinasyonlar (özne≠nesne): 3 varlık → 3×3-3 = 6 üçlü
    # ground-truth: yalnızca "1+1=2" ve "2=1+1" doğru; kalan 4'ü yanlış
    rapor = dongu.adim(relation_ids=["R_001"])
    assert rapor.uretilen == 6
    assert rapor.false_acceptance == 4         # 4 yanlış eşitlik VALID kabul edildi
    assert rapor.false_rejection == 0          # hiçbir doğru ifade reddedilmedi


def test_dongu_coklu_adim_replay():
    """Birden çok adımda replay verimliliği ve bilgi büyümesi takip edilir."""
    k = KnowledgeStore()
    k.varlik_ekle("Ali", entity_type="insan", properties={"canli": 1}, entity_id="E_001")
    k.varlik_ekle("Ata", entity_type="hayvan", properties={"binilebilir": 1}, entity_id="E_002")
    k.varlik_ekle("Araba", entity_type="tasit", properties={"binilebilir": 1}, entity_id="E_003")
    k.varlik_ekle("Gökyüzü", entity_type="mekan", properties={"binilebilir": 0}, entity_id="E_004")
    k.iliski_tanimla("Binmek", relation_id="R_001",
                     subject_types=["insan"],
                     requires_object_props={"binilebilir": 1.0})
    dongu = DeneyimDongusu(
        store=k,
        generator=ExperienceGenerator(),
        evaluator=ExperienceEvaluator(),
        consolidator=Consolidator(),
        bellek=BellekEntegrasyonu(slot_sayisi=64, replay_kapasitesi=16),
        replay_n=2,
    )
    raporlar = dongu.calistir(3, relation_ids=["R_001"])
    assert len(raporlar) == 3
    # ilk adımdan sonra üçlüler artık yeni değil (novel azalır)
    assert raporlar[0].novel == 3
    assert raporlar[1].novel == 0
    assert raporlar[2].novel == 0
    # bellek dolu ve replay verimliliği [0,1] aralığında
    assert 0.0 <= raporlar[-1].replay_verimliligi <= 1.0


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
