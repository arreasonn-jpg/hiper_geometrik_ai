# -*- coding: utf-8 -*-
"""Faz 19/20: Kronecker effective rank ve zincir çöküşü testleri."""
import pytest

from hga.evaluation.kronecker_rank import (
    measure_chain_collapse,
    measure_single_layer_rank,
    run_nk_rank_sweep,
    theoretical_contract,
)

torch = pytest.importorskip("torch", reason="rank ölçümü için torch gerekli")


def test_teorik_sozlesme_ust_sinirin_erisilebilir_olmadigini_soyler():
    s = theoretical_contract(n=16, k=4)
    assert s["interaction_space_upper_bound"] == 16 ** 8
    assert s["trainable_parameters"] == 4 * 2 * 16 * 16
    assert s["upper_bound_is_reachable"] is False
    assert s["collapses_without_activation"] is True
    assert s["single_layer_kronecker_manifold_dimension"] == 2 * 16 * 16 - 1
    assert s["fixed_feature_affine_readout_vc_dimension_upper_bound"] == 16 * 16 + 1
    assert s["full_chain_vc_dimension"] is None
    assert s["full_chain_vc_dimension_status"] == "NOT_ESTABLISHED"


def test_kronecker_rank_ozdesligi():
    """rank(Bᵀ ⊗ A) = rank(A)·rank(B) — ölçüm bu özdeşliği doğrulamalı."""
    olcum = measure_single_layer_rank(n=8, seed=1)
    assert olcum.measured["numerical_rank"] == 64      # 8·8
    assert olcum.rank_utilization == 1.0
    # Tam rank ama serbestlik derecesi yalnız 2n²=128.
    assert olcum.trainable_parameters == 128


def test_rank_tam_olsa_da_etkin_boyut_cok_daha_kucuk():
    """Asıl nokta: tam rank ≠ tüm yönler enerji taşıyor."""
    olcum = measure_single_layer_rank(n=16, seed=1)
    assert olcum.measured["numerical_rank"] == 256
    assert olcum.measured["entropy_effective_dimension"] < 256 * 0.6


def test_aktivasyonsuz_zincir_tek_katmana_cokuyor():
    """Faz 19/20'nin ana iddiası: K katkısı aktivasyonsuz SIFIR."""
    for k in (2, 3, 4):
        c = measure_chain_collapse(n=6, k=k, seed=1, activation=None)
        assert c.collapsed, f"K={k} çökmeliydi (kalıntı={c.collapse_residual})"
        assert c.collapse_residual < 1e-6


def test_aktivasyon_cokusu_kiriyor():
    for act in ("silu", "tanh", "relu"):
        c = measure_chain_collapse(n=6, k=3, seed=1, activation=act)
        assert not c.collapsed, act
        assert c.collapse_residual > 1e-3, act


def test_k1_zinciri_tanim_geregi_tek_katman():
    c = measure_chain_collapse(n=6, k=1, seed=1, activation=None)
    assert c.collapsed
    assert c.composite_rank == c.single_layer_equivalent_rank


def test_nk_taramasi_tum_konfigurasyonlarda_cokus_raporlar():
    rapor = run_nk_rank_sweep(n_values=(4, 8), k_values=(1, 2), seed=1)
    assert len(rapor.rows) == 4
    for satir in rapor.rows:
        assert satir.collapsed_without_activation
        assert satir.collapse_residual_with_activation > 1e-3
        assert satir.rank_utilization == 1.0
        assert satir.measured_rank == satir.max_operator_rank
    md = rapor.markdown()
    assert "ÇÖKTÜ" in md and "n^(2K)" in md
    assert any("AKTİVASYONDAN" in f for f in rapor.findings)


def test_ust_sinir_parametreden_cok_daha_hizli_buyuyor():
    """n^(2K) ile K·2n² arasındaki uçurum ölçekle açılmalı."""
    rapor = run_nk_rank_sweep(n_values=(8,), k_values=(1, 2, 4), seed=1)
    oranlar = [r.interaction_space_upper_bound / r.trainable_parameters
               for r in rapor.rows]
    assert oranlar == sorted(oranlar)
    assert oranlar[-1] > oranlar[0] * 1000


def test_gecersiz_parametreler_acik_hata():
    with pytest.raises(ValueError):
        theoretical_contract(n=1, k=1)
    with pytest.raises(ValueError):
        theoretical_contract(n=4, k=0)
    with pytest.raises(ValueError):
        measure_chain_collapse(n=4, k=0)
