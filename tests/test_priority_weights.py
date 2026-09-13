# -*- coding: utf-8 -*-
"""Faz 25: Priority(E) ağırlıklarının açıklığı, doğrulanması ve ablasyonu."""
import random

import pytest

from hga.experience.exploration import (
    PRIORITY_AGIRLIK_ACIKLAMALARI,
    VARSAYILAN_PRIORITY_AGIRLIKLARI,
    ExplorationEngine,
    agirlik_ablasyonu,
)
from hga.knowledge import (
    DeneyimDurumu,
    ExperienceCandidate,
    KaynakTuru,
    KnowledgeStore,
)


def _ornek_veri(n=40, seed=1):
    store = KnowledgeStore()
    rng = random.Random(seed)
    store.iliski_tanimla("iliski", relation_id="R_P")
    adaylar = []
    for index in range(n):
        ozne, nesne = f"PS{index}", f"PO{index}"
        store.varlik_ekle(ozne, entity_id=ozne)
        store.varlik_ekle(nesne, entity_id=nesne)
        for j in range(3):
            store.ozellik_koy(nesne, f"p{j}", rng.choice([0.0, 1.0]),
                              source=KaynakTuru.REAL_DATA, confidence=0.9)
        if index % 3 == 0:
            store.olgu_kaydet(ozne, "R_P", nesne, 1.0,
                              confidence=rng.uniform(0.2, 0.99))
        aday = ExperienceCandidate(f"PE{index}", ozne, "R_P", nesne)
        if index % 7 == 0:
            aday.state = DeneyimDurumu.CONFLICT
        adaylar.append(aday)
    return store, adaylar


def test_varsayilan_agirliklar_tek_kaynakta_ve_aciklamali():
    assert set(VARSAYILAN_PRIORITY_AGIRLIKLARI) == set(PRIORITY_AGIRLIK_ACIKLAMALARI)
    motor = ExplorationEngine()
    for ad, deger in VARSAYILAN_PRIORITY_AGIRLIKLARI.items():
        assert getattr(motor, ad) == deger
    assert motor.agirliklar()["positive_weight_sum"] == 1.0


def test_agirliklar_ayarlanabilir():
    motor = ExplorationEngine(w_gain=0.7, w_novelty=0.1,
                              w_uncertainty=0.2, w_conflict_penalty=0.5)
    a = motor.agirliklar()
    assert (a["w_gain"], a["w_novelty"], a["w_uncertainty"],
            a["w_conflict_penalty"]) == (0.7, 0.1, 0.2, 0.5)


def test_negatif_agirlik_reddedilir():
    """Bir sinyali 'kötü' saymak istiyorsan 0 kullan — negatif anlamsız."""
    for ad in ("w_gain", "w_novelty", "w_uncertainty", "w_conflict_penalty"):
        with pytest.raises(ValueError, match="negatif olamaz"):
            ExplorationEngine(**{ad: -0.1})


def test_normalize_pozitif_agirliklari_bire_olceklendirir():
    motor = ExplorationEngine(w_gain=1.0, w_novelty=1.0, w_uncertainty=2.0,
                              normalize=True)
    a = motor.agirliklar()
    assert a["positive_weight_sum"] == 1.0
    assert a["w_uncertainty"] == 0.5
    assert a["normalized"] is True
    with pytest.raises(ValueError):
        ExplorationEngine(w_gain=0.0, w_novelty=0.0, w_uncertainty=0.0,
                          normalize=True)


def test_priority_dokumu_terim_terim_seffaf():
    store, adaylar = _ornek_veri(n=10)
    motor = ExplorationEngine()
    dokum = motor.priority_dokumu(store, adaylar[0])
    assert set(dokum["terms"]) == {"information_gain", "novelty",
                                   "uncertainty", "conflict_penalty"}
    toplam = sum(t["contribution"] for t in dokum["terms"].values())
    assert dokum["raw_priority"] == pytest.approx(toplam, abs=1e-6)
    # Ceza terimi NEGATİF katkı vermeli.
    assert dokum["terms"]["conflict_penalty"]["contribution"] <= 0.0
    assert dokum["priority"] == motor.bilgi_kazanci_skoru(store, adaylar[0])


def test_bilinmeyen_varlik_hata_degil_azami_yenilik():
    """Kanıt yokluğu bir çökme değil, epistemik durumdur."""
    store = KnowledgeStore()
    store.iliski_tanimla("r", relation_id="R")
    aday = ExperienceCandidate("E_YOK", "OLMAYAN_S", "R", "OLMAYAN_O")
    dokum = ExplorationEngine().priority_dokumu(store, aday)
    assert dokum["resolved"] is False
    assert dokum["terms"]["novelty"]["value"] == 1.0
    assert dokum["terms"]["uncertainty"]["value"] == 1.0


def test_agirlik_degisimi_secimi_degistirebilir():
    store, adaylar = _ornek_veri()
    varsayilan = [a.experience_id for a, _ in
                  ExplorationEngine().aktif_ogrenme_sec(store, adaylar, k=10)]
    farkli = [a.experience_id for a, _ in
              ExplorationEngine(w_gain=0.0, w_novelty=0.0, w_uncertainty=1.0)
              .aktif_ogrenme_sec(store, adaylar, k=10)]
    assert varsayilan != farkli


def test_ablasyon_isleyen_ve_islemeyen_terimleri_ayirir():
    """Faz 25'in asıl testi: ayarlanabilir olmak etkili olmak değildir."""
    store, adaylar = _ornek_veri()
    rapor = agirlik_ablasyonu(store, adaylar, k=10)
    assert set(rapor.variants) == {"w_gain", "w_novelty", "w_uncertainty",
                                   "w_conflict_penalty"}
    for varyant in rapor.variants.values():
        assert 0.0 <= varyant["overlap_ratio"] <= 1.0
        assert varyant["effective"] == (varyant["overlap_ratio"] < 1.0)
    # w_gain bu veri üzerinde seçimi değiştirmeli.
    assert rapor.variants["w_gain"]["effective"]
    assert "|" in rapor.markdown()
    assert rapor.findings


def test_ablasyon_bos_aday_acik_hata():
    store, _ = _ornek_veri(n=5)
    with pytest.raises(ValueError):
        agirlik_ablasyonu(store, [], k=5)
