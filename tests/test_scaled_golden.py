# -*- coding: utf-8 -*-
"""Faz 3/6: ölçeklendirilmiş golden benchmark testleri."""
import pytest

from hga.evaluation.scaled_golden import (
    SINIFLAR,
    run_scaled_golden,
    run_scaled_golden_sweep,
    veri_seti_uret,
)
from hga.experience.evaluator import ExperienceEvaluator
from hga.knowledge import DeneyimDurumu, ExperienceCandidate


def test_veri_seti_dort_sinifa_dengeli_dagilir():
    veri = veri_seti_uret(size=400, seed=1)
    assert sum(veri.class_counts.values()) == 400
    for ad in SINIFLAR:
        assert veri.class_counts[ad] == 100, veri.class_counts


def test_ayni_seed_ayni_dataset_hash():
    a = veri_seti_uret(size=200, seed=7)
    b = veri_seti_uret(size=200, seed=7)
    assert a.dataset_hash() == b.dataset_hash()


def test_test_ucluleri_bilgi_tabanina_olgu_olarak_yazilmaz():
    """Sızıntı sözleşmesi: test üçlüsü CONFLICT kanıtı dışında olgu olmamalı."""
    veri = veri_seti_uret(size=100, seed=1)
    for kayit in veri.test_records:
        if kayit.expected_state == "CONFLICT":
            continue  # CONFLICT tanımı gereği kayıtlı karşı-kanıt ister
        agrega = veri.store.relations.olgu_agrega(
            kayit.subject_id, kayit.relation_id, kayit.object_id)
        assert agrega is None, f"{kayit.experience_id} sızmış"


def test_sizinti_denetimi_temiz():
    rapor = run_scaled_golden(size=200, seed=1)
    assert rapor.leakage_clean


def test_100_olcekte_dort_sinif_da_dogru_siniflaniyor():
    rapor = run_scaled_golden(size=100, seed=1)
    assert rapor.metrics["accuracy"] == 1.0
    for ad in SINIFLAR:
        assert rapor.per_class_accuracy[ad] == 1.0, (ad, rapor.misclassifications)


def test_zor_mod_sinir_vakalari_ekler_ve_hepsini_gecer():
    """Zor mod gerçekten farklı vakalar üretmeli (ve evaluator geçmeli)."""
    kolay = veri_seti_uret(size=200, seed=1, hard=False)
    zor = veri_seti_uret(size=200, seed=1, hard=True)
    assert kolay.dataset_hash() != zor.dataset_hash()
    zor_insalar = [k.construction for k in zor.test_records if "[zor:" in k.construction]
    assert len(zor_insalar) >= 90, "zor vakalar üretilmemiş"
    assert len({i.split("]")[0] for i in zor_insalar}) == 4, "4 sınır vakası bekleniyor"
    rapor = run_scaled_golden(size=200, seed=1, hard=True)
    assert rapor.metrics["accuracy"] == 1.0, rapor.misclassifications


def test_zor_vaka_kisit_onceligi_paylasilan_iliskiyi_kirletmez():
    """Regresyon: zor vaka özel ilişki kullanmazsa sonraki örnekler bozulur."""
    veri = veri_seti_uret(size=400, seed=1, hard=True)
    evaluator = ExperienceEvaluator()
    for kayit in veri.test_records:
        if "[zor:" in kayit.construction:
            continue
        aday = ExperienceCandidate(kayit.experience_id, kayit.subject_id,
                                   kayit.relation_id, kayit.object_id)
        evaluator.degerlendir(aday, veri.store)
        assert aday.state.value == kayit.expected_state, (
            f"{kayit.experience_id} temel vaka bozulmuş: {aday.rationale}")


def test_uncertain_gercekten_kanit_yoklugundan_geliyor():
    veri = veri_seti_uret(size=100, seed=3)
    evaluator = ExperienceEvaluator()
    for kayit in veri.test_records:
        if kayit.expected_state != "UNCERTAIN":
            continue
        aday = ExperienceCandidate(kayit.experience_id, kayit.subject_id,
                                   kayit.relation_id, kayit.object_id)
        evaluator.degerlendir(aday, veri.store)
        assert aday.state is DeneyimDurumu.UNCERTAIN
        assert any("BİLİNMİYOR" in r or "güveni" in r for r in aday.rationale)


def test_sweep_olcekle_std_dusmeli_ve_maliyet_raporlanmali():
    rapor = run_scaled_golden_sweep(sizes=(100, 400), seeds=(1, 2, 3))
    assert rapor.sizes == [100, 400]
    for size in ("100", "400"):
        assert rapor.summary[size]["accuracy"]["mean"] == 1.0
        assert rapor.cost[size]["per_example_ms"] > 0
    assert rapor.leakage_clean
    md = rapor.markdown()
    assert "Accuracy" in md and "Örnek başına" in md
    assert any("MALİYET" in f for f in rapor.findings)
    assert any("tautolojik" in f for f in rapor.findings)


def test_kucuk_size_acik_hata():
    with pytest.raises(ValueError):
        veri_seti_uret(size=2)
