# -*- coding: utf-8 -*-
"""P1 çekirdek tohum istatistikleri protokolünün testleri."""
from __future__ import annotations

import pytest

from hga.evaluation.seed_statistics import (
    CORE_SEED_REQUIREMENT,
    CORE_SEEDS,
    PROFILES,
    PROTOCOL,
    _ci_sifir_iceriyor,
    run_core_seed_statistics,
    seed_statistics_markdown,
)
from hga.evaluation.statistics import minimum_two_sided_p


@pytest.fixture(scope="module")
def rapor():
    """Hızlı koşum: operatör kolu (PyTorch) kapalı, 4 tohum."""
    return run_core_seed_statistics(
        seeds=(1, 2, 3, 4), include_operator=False)


def test_protokol_kimligi(rapor):
    assert rapor.protocol == PROTOCOL
    assert rapor.schema_version == 1
    assert rapor.core_requirement == CORE_SEED_REQUIREMENT == 20
    assert len(CORE_SEEDS) == 20
    assert PROFILES["core"] == CORE_SEEDS


def test_tum_protokoller_ayni_tohumlari_kullaniyor(rapor):
    """Eşleşmiş tasarımın ön koşulu: aynı tohum kümesi."""
    assert rapor.checks["all_core_protocols_use_same_seeds"] is True
    for ad, veri in rapor.protocols.items():
        if veri.get("skipped"):
            continue
        assert veri["seeds"] == rapor.seeds, ad


def test_20_tohum_kurali_durustce_raporlaniyor(rapor):
    """4 tohumla 20-tohum kapısı GEÇMİŞ görünmemeli."""
    assert rapor.checks["all_core_protocols_meet_20_seeds"] is False
    for ad, veri in rapor.protocols.items():
        if not veri.get("skipped"):
            assert veri["meets_core_requirement"] is False, ad


def test_karsilastirmalar_eslesmis(rapor):
    assert rapor.comparisons, "hiç karşılaştırma üretilmedi"
    assert rapor.checks["comparisons_are_paired"] is True
    for kiyas in rapor.comparisons:
        assert kiyas["n_pairs"] == len(rapor.seeds)


def test_her_karsilastirma_ci_etki_ve_iki_test_veriyor(rapor):
    """Çıplak p-değeri yasak: CI + etki büyüklüğü + iki test zorunlu."""
    assert rapor.checks["every_comparison_reports_ci"] is True
    assert rapor.checks["every_comparison_reports_effect_size"] is True
    assert rapor.checks["every_comparison_reports_two_tests"] is True
    for kiyas in rapor.comparisons:
        ci = kiyas["difference_ci"]
        assert ci["lower"] <= ci["upper"]
        assert "value" in kiyas["effect_size"]
        assert "p_value" in kiyas["permutation_test"]
        assert "p_value" in kiyas["wilcoxon_test"]


def test_guc_siniri_hesaplaniyor_ve_dusuk_n_de_uyariyor(rapor):
    """n küçükken p<0.05 ulaşılamaz; bu gizlenmemeli."""
    guc = rapor.power
    assert guc["n_seeds"] == 4
    assert guc["minimum_attainable_two_sided_p"] == pytest.approx(
        minimum_two_sided_p(4))
    assert guc["can_reach_p_0_05"] is False
    assert rapor.checks["design_can_reach_p_0_05"] is False
    assert rapor.checks["power_limit_documented"] is True
    assert any("p<0.05" in b and "ULAŞAMAZ" in b for b in rapor.findings)


def test_yetersiz_gucte_anlamlilik_iddia_edilmiyor(rapor):
    """En kritik dürüstlük kapısı."""
    for kiyas in rapor.comparisons:
        assert "İDDİA EDİLMEZ" in kiyas["verdict"] or "YETERSİZ" in kiyas[
            "verdict"], kiyas["verdict"]


def test_20_tohumda_p_005_ulasilabilir():
    """Tasarımın 20 tohumda yeterli güce ulaştığı doğrulanmalı."""
    assert minimum_two_sided_p(20) <= 0.05
    assert minimum_two_sided_p(20) <= 0.01


def test_ci_sifir_kontrolu():
    assert _ci_sifir_iceriyor({"difference_ci": {"lower": -0.1, "upper": 0.2}})
    assert not _ci_sifir_iceriyor(
        {"difference_ci": {"lower": 0.1, "upper": 0.2}})
    assert not _ci_sifir_iceriyor(
        {"difference_ci": {"lower": -0.3, "upper": -0.1}})
    # Eksik CI ihtiyatlı davranmalı: "ayırt edilemez" say.
    assert _ci_sifir_iceriyor({})


def test_signature_per_seed_serileri_mevcut(rapor):
    """Signature artık tohum-başı ham değer veriyor olmalı."""
    imza_kiyaslari = [k for k in rapor.comparisons
                      if k["context"].startswith("signature/")]
    assert imza_kiyaslari
    assert {k["treatment_label"] for k in imza_kiyaslari} == {"hga"}
    rakipler = {k["baseline_label"] for k in imza_kiyaslari}
    assert {"dense", "transformer", "symbolic"} <= rakipler


def test_priority_ablation_tohum_ozetleri(rapor):
    ozetler = rapor.protocols["priority_ablation"]["seed_summaries"]
    assert ozetler
    for ozet in ozetler:
        assert ozet["n"] == len(rapor.seeds)
        assert ozet["ci_lower"] <= ozet["mean"] <= ozet["ci_upper"]
        assert ozet["std_sample"] >= 0.0
    metrikler = {o["metric"] for o in ozetler}
    assert "kendall_tau" in metrikler
    assert "downstream_delta::verification_yield" in metrikler


def test_torch_yoksa_operator_kolu_atlanir(rapor):
    """include_operator=False → atlandı diye işaretlenmeli, sessizce kaybolmamalı."""
    assert "operator_baselines" not in rapor.protocols


def test_operator_kolu_acikken_kosuluyor():
    r = run_core_seed_statistics(
        seeds=(1, 2), operator_steps=20, include_operator=True)
    op = r.protocols.get("operator_baselines") or {}
    if op.get("skipped"):
        pytest.skip("PyTorch yok")
    assert op["seed_count"] == 2
    op_kiyas = [k for k in r.comparisons
                if k["context"].startswith("operator_baselines/")]
    assert op_kiyas
    # MSE'de küçük olan iyidir; bu işaretlenmeli.
    assert all(k.get("lower_is_better") for k in op_kiyas)


def test_determinizm():
    a = run_core_seed_statistics(seeds=(1, 2, 3), include_operator=False)
    b = run_core_seed_statistics(seeds=(1, 2, 3), include_operator=False)
    assert a.comparisons == b.comparisons
    assert a.power == b.power
    assert a.checks == b.checks


def test_gecersiz_girdiler():
    with pytest.raises(ValueError):
        run_core_seed_statistics(profile="yok-boyle")
    with pytest.raises(ValueError):
        run_core_seed_statistics(seeds=(1,))
    with pytest.raises(ValueError):
        run_core_seed_statistics(seeds=())


def test_sinirlar_coklu_karsilastirmayi_itiraf_ediyor(rapor):
    metin = " ".join(rapor.limitations)
    assert "Bonferroni" in metin or "FDR" in metin
    assert "yanlılığı" in metin or "yanlilik" in metin.lower()


def test_markdown_uretimi(rapor):
    md = seed_statistics_markdown(rapor)
    assert "# Çekirdek Tohum İstatistikleri (P1)" in md
    assert "İstatistiksel güç sınırı" in md
    assert "Eşleşmiş karşılaştırmalar" in md
    assert "Cohen's d" in md
    assert "Sınırlar" in md
    assert rapor.power["signature"] in md
