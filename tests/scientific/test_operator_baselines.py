# -*- coding: utf-8 -*-
"""P0-2: Kronecker vs gerçek Dense baseline ailesi testleri."""
import pytest

torch = pytest.importorskip("torch")

from hga.evaluation.operator_baselines import (  # noqa: E402
    STRUCTURED_TEACHERS,
    UNSTRUCTURED_TEACHERS,
    arm_budget,
    budget_table,
    operator_baseline_markdown,
    run_operator_baseline_benchmark,
)


def test_butce_muhasebesi_analitik_dogru():
    n = 8
    d = n * n
    assert arm_budget("kronecker", n) == {"parameters": 2 * d, "flops": 2 * n ** 3}
    assert arm_budget("rank1", n) == {"parameters": 2 * d, "flops": 2 * d}
    assert arm_budget("full_dense", n) == {"parameters": d * d, "flops": d * d}
    # Eşit-FLOP dense kolu Kronecker'ın FLOP'una eşit olmalı.
    assert (arm_budget("low_rank_flop_matched", n)["flops"]
            == arm_budget("kronecker", n)["flops"])


def test_full_dense_kasitli_olarak_butce_disi():
    tablo = budget_table(8)
    assert tablo["full_dense"]["parameter_ratio_to_kronecker"] > 1.0
    assert not tablo["full_dense"]["parameter_matched"]
    # rank1 eşit parametrelidir ama eşit FLOP değildir.
    assert tablo["rank1"]["parameter_matched"]
    assert not tablo["rank1"]["flop_matched"]


def test_bilinmeyen_kol_acik_hata():
    with pytest.raises(ValueError):
        arm_budget("yok_boyle_bir_kol", 8)
    with pytest.raises(ValueError):
        run_operator_baseline_benchmark(arms=["uydurma"])
    with pytest.raises(ValueError):
        run_operator_baseline_benchmark(teachers=["uydurma"])


def test_gercek_parametre_sayimi_analitikle_ayni():
    """Adalet iddiası buna dayanır: analitik bütçe gerçek modelle tutmalı."""
    rapor = run_operator_baseline_benchmark(
        n=6, steps=20, seeds=(1,), teachers=("kronecker_teacher",))
    for arm in rapor.arms:
        sonuc = rapor.results["kronecker_teacher"][arm]
        assert sonuc["parameters"] == sonuc["analytic_parameters"]
    assert rapor.checks["parameter_accounting_exact"]


def test_kronecker_yapili_gorevde_rank1i_yener():
    rapor = run_operator_baseline_benchmark(
        n=8, steps=200, seeds=(1, 2), teachers=STRUCTURED_TEACHERS)
    assert rapor.checks["kronecker_beats_rank1_on_structured"]
    assert rapor.checks["kronecker_beats_flop_matched_dense_on_structured"]


def test_kronecker_yapisiz_dense_ogretmende_geri_kalir():
    """2n² parametre ile n⁴ serbestlik öğrenilemez; aksi sonuç ölçümü şüpheli kılar."""
    rapor = run_operator_baseline_benchmark(
        n=8, steps=200, seeds=(1, 2), teachers=UNSTRUCTURED_TEACHERS)
    assert rapor.checks["kronecker_loses_to_dense_on_unstructured"]
    fark = rapor.ceiling_gap["full_dense_teacher"]
    assert fark > 0.0, "Kronecker tavanı geçemez; pozitif mesafe beklenir"


def test_tavan_farki_her_ogretmen_icin_raporlanir():
    rapor = run_operator_baseline_benchmark(n=6, steps=50, seeds=(1,))
    assert set(rapor.ceiling_gap) == set(rapor.teachers)


def test_siralama_ve_esit_parametre_siralamasi_ayri():
    rapor = run_operator_baseline_benchmark(n=6, steps=50, seeds=(1,))
    for task in rapor.teachers:
        assert set(rapor.rankings[task]) == set(rapor.arms)
        # Eşit-parametre sıralaması full_dense'i İÇERMEMELİ (bütçe dışı).
        assert "full_dense" not in rapor.parameter_matched_rankings[task]


def test_optimizasyon_kararli():
    rapor = run_operator_baseline_benchmark(n=6, steps=50, seeds=(1,))
    assert rapor.checks["all_arms_optimization_stable"]


def test_markdown_butce_ve_sinirlari_icerir():
    rapor = run_operator_baseline_benchmark(n=4, steps=10, seeds=(1,),
                                            teachers=("kronecker_teacher",))
    md = operator_baseline_markdown(rapor)
    assert "Bütçe muhasebesi" in md
    assert "rakip değil tavandır" in md
    assert "Kabul kapıları" in md


def test_ayni_tohum_ayni_sonuc():
    a = run_operator_baseline_benchmark(n=4, steps=10, seeds=(1,),
                                        teachers=("kronecker_teacher",))
    b = run_operator_baseline_benchmark(n=4, steps=10, seeds=(1,),
                                        teachers=("kronecker_teacher",))
    assert (a.results["kronecker_teacher"]["kronecker"]["test_normalized_mse_mean"]
            == b.results["kronecker_teacher"]["kronecker"]["test_normalized_mse_mean"])
    assert a.dataset_hash == b.dataset_hash
