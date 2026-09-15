# -*- coding: utf-8 -*-
"""Priority(E) ağırlık optimizasyonu + held-out doğrulama testleri."""
from __future__ import annotations

import pytest

from hga.evaluation.capability_vector import build_scorecard
from hga.evaluation.priority_ablation import TERIMLER
from hga.evaluation.priority_optimization import (
    OBJECTIVE,
    PROFILES,
    PROTOCOL,
    optimize_priority_weights,
    priority_optimization_markdown,
)


@pytest.fixture(scope="module")
def rapor():
    """Held-out kazancının görünür olduğu gerçek koşum (hızlı alt küme)."""
    return optimize_priority_weights(
        grid=(0.0, 0.2, 0.4),
        train_seeds=(1, 2, 3, 4, 5),
        test_seeds=tuple(range(101, 111)),
        candidate_count=80)


def test_protokol_kimligi(rapor):
    assert rapor.protocol == PROTOCOL
    assert rapor.schema_version == 1
    assert rapor.objective == OBJECTIVE


def test_egitim_test_tohumlari_ayrik(rapor):
    """Held-out doğrulamanın ön koşulu."""
    assert set(rapor.train_seeds) & set(rapor.test_seeds) == set()
    assert rapor.checks["train_test_seed_isolation"] is True


def test_sizinti_reddediliyor():
    """Kesişen tohum kümeleri kabul edilmemeli."""
    with pytest.raises(ValueError, match="sızıntı"):
        optimize_priority_weights(
            train_seeds=(1, 2, 3), test_seeds=(3, 4, 5))


def test_arama_uzayi_sifiri_iceriyor(rapor):
    """Ablasyon bir terimi kapatmanın iyi olabileceğini gösterdi."""
    assert 0.0 in rapor.grid
    assert rapor.checks["search_space_includes_zero"] is True


def test_tum_kombinasyonlar_degerlendirildi(rapor):
    # 3^4 = 81, hepsi-sıfır kombinasyonu hariç 80
    assert rapor.candidates_evaluated == 80


def test_arama_varsayilandan_iyi_agirlik_buluyor(rapor):
    assert rapor.train["best_mean"] >= rapor.train["default_mean"]
    assert set(rapor.best_weights) == set(TERIMLER)
    assert all(v >= 0.0 for v in rapor.best_weights.values())
    assert any(v > 0.0 for v in rapor.best_weights.values())


def test_elle_secilmis_agirliklar_optimal_degil(rapor):
    """Asıl bulgu: 0.40/0.35/0.25/0.20 savunulabilir değil."""
    assert rapor.best_weights != rapor.default_weights
    sifirlanan = [t for t, v in rapor.best_weights.items() if v == 0.0]
    assert sifirlanan, "arama hiçbir terimi kapatmadı"
    assert any("TAMAMEN KAPATTI" in b for b in rapor.findings)


def test_kazanc_held_out_ta_korunuyor(rapor):
    """Eğitimde kazanç bulmak kolaydır; held-out'ta korumak zordur."""
    assert rapor.overfitting["holdout_gain"] > 0.0
    assert rapor.checks["gain_survives_holdout"] is True
    assert rapor.checks["optimized_at_least_matches_default_on_holdout"]


def test_asiri_uydurma_olculuyor(rapor):
    o = rapor.overfitting
    assert o["train_gain"] is not None
    assert o["generalization_ratio"] is not None
    assert rapor.checks["overfitting_measured"] is True
    # Kazancın anlamlı bir kısmı korunmalı, yoksa arama ezberlemiştir.
    assert o["generalization_ratio"] > 0.5


def test_held_out_karsilastirma_eslesmis_ve_tam(rapor):
    k = rapor.comparison
    assert len(rapor.holdout["optimized_per_seed"]) == len(rapor.test_seeds)
    assert len(rapor.holdout["default_per_seed"]) == len(rapor.test_seeds)
    assert rapor.checks["paired_design"] is True
    # Çıplak p-değeri yasak: CI + etki büyüklüğü + test zorunlu.
    assert k["difference_ci"]["lower"] <= k["difference_ci"]["upper"]
    assert "value" in k["effect_size"]
    assert "p_value" in k["permutation_test"]


def test_kazanc_istatistiksel_olarak_ayrisiyor(rapor):
    """10 held-out tohumla fark gerçekten ayrışmalı."""
    assert rapor.checks["gain_statistically_distinguishable"] is True
    ci = rapor.comparison["difference_ci"]
    assert not (ci["lower"] <= 0.0 <= ci["upper"])


def test_duyarlilik_olcum_artefakti_uretmiyor(rapor):
    """Tek aktif terimde sabit eğri 'ölü ağırlık' sayılmamalı.

    Tek terim kaldığında ölçek sıralamayı değiştirmez (monoton dönüşüm);
    eğri düzdür ama terim belirleyicidir.
    """
    assert rapor.checks["sensitivity_measured"] is True
    for terim, veri in rapor.weight_sensitivity.items():
        assert veri["points_evaluated"] >= 1
        if veri["scale_invariant"]:
            assert veri["matters"] is True, (
                f"{terim} ölçek-değişmez ama ölü sayıldı")
            assert "GELMEZ" in veri["note"]
        else:
            assert veri["matters"] == (veri["holdout_range"] > 1e-9)


def test_determinizm():
    a = optimize_priority_weights(profile="smoke")
    b = optimize_priority_weights(profile="smoke")
    assert a.best_weights == b.best_weights
    assert a.holdout["optimized_per_seed"] == b.holdout["optimized_per_seed"]
    assert a.holdout["signature"] == b.holdout["signature"]


def test_smoke_profilinde_anlamlilik_kapisi_durustce_kaliyor():
    """3 tohumla p<0.05 ulaşılamaz; kapı GEÇMİŞ görünmemeli."""
    r = optimize_priority_weights(profile="smoke")
    assert len(r.test_seeds) == 3
    assert r.checks["gain_statistically_distinguishable"] is False


def test_gecersiz_girdiler():
    with pytest.raises(ValueError):
        optimize_priority_weights(profile="yok-boyle")
    with pytest.raises(ValueError):
        optimize_priority_weights(grid=())
    with pytest.raises(ValueError):
        optimize_priority_weights(grid=(-0.1, 0.2))
    with pytest.raises(ValueError):
        optimize_priority_weights(train_seeds=(1,))
    with pytest.raises(ValueError):
        optimize_priority_weights(k=0)


def test_profiller_tanimli():
    for ad, ayar in PROFILES.items():
        assert set(ayar["train_seeds"]) & set(ayar["test_seeds"]) == set(), ad
        assert 0.0 in ayar["grid"], ad
        assert len(ayar["test_seeds"]) >= 2


def test_sinirlar_izgara_kabaligini_itiraf_ediyor(rapor):
    metin = " ".join(rapor.limitations)
    assert "küresel optimum değildir" in metin
    assert "ALAN genellemesi değildir" in metin


def test_markdown_uretimi(rapor):
    md = priority_optimization_markdown(rapor)
    assert "Held-Out Doğrulama" in md
    assert "HİÇ görülmedi" in md
    assert "Terim duyarlılığı" in md
    assert rapor.holdout["signature"] in md


def test_karne_verification_bolumune_baglandi(rapor):
    karne = build_scorecard(priority_optimization=rapor.to_dict())
    bolum = karne.sections["verification"]
    assert bolum["score"] is not None
    assert bolum["inputs"]["weights_validated_on_holdout"] is True
    assert bolum["inputs"]["holdout_gain"] > 0.0
    assert "priority_optimization" in karne.provenance["reports_supplied"]


def test_verification_skoru_dusmuyor_ve_holdout_baglandi(rapor):
    """Held-out doğrulama eklemek skoru asla düşürmemeli ve girdi olarak
    bağlanmalı.

    Not: Ablasyon havuzu düzeltildikten sonra (kayıtsız-köşe degenerasyonu
    giderildi) ablasyonun 6/6 kapısı geçiyor ve bölüm skoru tek başına
    tavanda (10.0). Bu yüzden eski "skor kesin YÜKSELMELİ" iddiası artık
    matematiksel olarak imkânsız; doğru sözleşme, optimizasyonun skoru
    düşürmemesi ve kapı/kanıt kümesini genişletmesidir.
    """
    from hga.evaluation.priority_ablation import run_priority_weight_ablation
    ablasyon = run_priority_weight_ablation(k=10, seeds=(1, 2, 3)).to_dict()
    once = build_scorecard(
        priority_ablation=ablasyon).sections["verification"]
    sonra = build_scorecard(
        priority_ablation=ablasyon,
        priority_optimization=rapor.to_dict()).sections["verification"]
    assert sonra["score"] >= once["score"]
    assert sonra["inputs"]["weights_validated_on_holdout"] is True
    assert sonra["inputs"]["holdout_gain"] > 0.0
    # Optimizasyon kapıları bölüm kanıtına gerçekten eklenmiş olmalı.
    assert any("weight_optimization" in kanit for kanit in sonra["evidence"])
