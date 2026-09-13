# -*- coding: utf-8 -*-
"""Faz 21: neural-only vs symbolic-only vs hybrid kontrollü ablasyon testleri."""
import pytest

from hga.evaluation.paradigma import (
    KOLLAR,
    gorev_uret,
    hibrit_tahmin,
    run_paradigm_ablation,
    run_paradigm_sweep,
    sembolik_tahmin,
)

torch = pytest.importorskip("torch", reason="Nöral kol için torch gerekli")


def test_gorev_dengeli_ve_gorulmemis_varlik_iceriyor():
    gorev = gorev_uret(entity_count=60, train_size=300, test_size=200, seed=1)
    etiketler = [o.label for o in gorev.test]
    oran = sum(etiketler) / len(etiketler)
    assert 0.4 <= oran <= 0.6, f"test seti dengesiz: {oran}"
    assert any(o.unseen_entity for o in gorev.test)
    # Görülmemiş varlıklar eğitimde GERÇEKTEN geçmemeli (sızıntı testi).
    egitim_varliklari = {o.subject_id for o in gorev.train} | \
        {o.object_id for o in gorev.train}
    for ornek in gorev.test:
        if ornek.unseen_entity:
            assert ornek.subject_id not in egitim_varliklari
            assert ornek.object_id not in egitim_varliklari


def test_sembolik_kol_bilinmeyende_cekimser_kalir():
    """Kanıt yoksa tahmin uydurmamalı: None döndürmeli."""
    gorev = gorev_uret(entity_count=60, train_size=300, test_size=200,
                       hidden_feature_ratio=0.5, seed=2)
    tahminler = sembolik_tahmin(gorev, gorev.test)
    assert any(t is None for t in tahminler), "çekimserlik yok"
    # Cevapladıklarında kural gereği HATASIZ olmalı.
    for ornek, tahmin in zip(gorev.test, tahminler):
        if tahmin is not None:
            assert tahmin == ornek.label


def test_sembolik_cekimserlik_gizli_ozellikten_kaynaklanir():
    az = gorev_uret(entity_count=60, train_size=200, test_size=200,
                    hidden_feature_ratio=0.0, seed=3)
    cok = gorev_uret(entity_count=60, train_size=200, test_size=200,
                     hidden_feature_ratio=0.6, seed=3)
    az_kapsam = sum(t is not None for t in sembolik_tahmin(az, az.test))
    cok_kapsam = sum(t is not None for t in sembolik_tahmin(cok, cok.test))
    assert az_kapsam > cok_kapsam
    assert az_kapsam == len(az.test), "gizli özellik yokken kapsam tam olmalı"


def test_hibrit_sembolik_vetoyu_korur_bosluklari_doldurur():
    sembolik = [1, None, 0, None]
    noral = [0, 1, 1, 0]
    assert hibrit_tahmin(sembolik, noral) == [1, 1, 0, 0]


def test_ablasyon_uc_kolu_da_raporlar():
    rapor = run_paradigm_ablation(entity_count=60, train_size=300,
                                  test_size=150, seed=1, epochs=20)
    assert set(rapor.arms) == set(KOLLAR)
    for kol in KOLLAR:
        m = rapor.arms[kol]
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["far"] <= 1.0 and 0.0 <= m["frr"] <= 1.0
    assert rapor.arms["neural"]["coverage"] == 1.0
    assert rapor.arms["hybrid"]["coverage"] == 1.0


def test_hibrit_her_iki_saf_koldan_iyi():
    """Faz 21'in ana hipotezi: hibrit > sembolik ve hibrit > nöral."""
    rapor = run_paradigm_sweep(seeds=(1, 2, 3), entity_count=100,
                               train_size=800, test_size=300, epochs=40)
    sym = rapor.summary["symbolic"]["accuracy"]["mean"]
    neu = rapor.summary["neural"]["accuracy"]["mean"]
    hyb = rapor.summary["hybrid"]["accuracy"]["mean"]
    assert hyb > sym, f"hibrit ({hyb}) sembolikten ({sym}) iyi olmalı"
    assert hyb > neu, f"hibrit ({hyb}) nöralden ({neu}) iyi olmalı"


def test_noral_kol_gorulmemis_varlikta_cokuyor():
    """Ezberi ölçen asıl test: cold-start'ta şans seviyesine yaklaşmalı."""
    rapor = run_paradigm_sweep(seeds=(1, 2, 3), entity_count=100,
                               train_size=800, test_size=300, epochs=40)
    gorulen = rapor.summary["neural"]["accuracy_seen"]["mean"]
    gorulmemis = rapor.summary["neural"]["accuracy_unseen"]["mean"]
    assert gorulen > gorulmemis + 0.15, "cold-start düşüşü görünmüyor"
    assert gorulmemis < 0.70, "nöral kol kuralı genellememeli (ezber bekleniyor)"
    # Genelleme açığı: eğitimde ezberler, testte düşer.
    assert rapor.neural_train_accuracy["mean"] > rapor.summary["neural"]["accuracy"]["mean"]


def test_sweep_5_seed_mean_std_ve_markdown():
    rapor = run_paradigm_sweep(seeds=(1, 2, 3, 4, 5), entity_count=60,
                               train_size=300, test_size=150, epochs=20)
    assert rapor.seeds == [1, 2, 3, 4, 5]
    assert len(rapor.reports) == 5
    for kol in KOLLAR:
        istat = rapor.summary[kol]["accuracy"]
        assert set(istat) == {"mean", "std", "min", "max"}
        assert istat["min"] <= istat["mean"] <= istat["max"]
    md = rapor.markdown()
    assert "±" in md and "symbolic" in md and "hybrid" in md
    assert len(rapor.findings) >= 5


def test_gecersiz_gorev_parametreleri_acik_hata():
    with pytest.raises(ValueError):
        gorev_uret(hidden_feature_ratio=1.5)
    with pytest.raises(ValueError):
        gorev_uret(unseen_entity_ratio=1.0)
    with pytest.raises(ValueError):
        gorev_uret(entity_count=3)
