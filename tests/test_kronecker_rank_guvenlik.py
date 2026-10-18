"""P0-8 regresyon testi: n^(2K) taşma koruması (bkz. KRONECKER_RANK_BUG.md)."""
from __future__ import annotations

import pytest


def test_guvenli_mi_tam_sinir_kabul():
    from hga.experiments.kronecker_rank import guvenli_mi
    ok, sebep = guvenli_mi(64, 8)
    assert ok, f"n=64, K=8 sinirda kabul edilmeli: {sebep}"


@pytest.mark.parametrize("n,k", [
    (0, 1), (1, 1), (16, 0), (128, 1), (64, 16), (256, 32), (128, 16),
])
def test_guvenli_mi_reddeder(n, k):
    from hga.experiments.kronecker_rank import guvenli_mi
    ok, sebep = guvenli_mi(n, k)
    assert not ok, f"n={n}, K={k} guvensiz olmali"
    assert sebep


def test_run_nk_rank_sweep_guvensiz_konfigurasyonu_reddeder():
    from hga.experiments.kronecker_rank import run_nk_rank_sweep
    with pytest.raises(ValueError, match="[Gg]uvensiz"):
        run_nk_rank_sweep(n_values=[256], k_values=[32])


def test_run_nk_rank_sweep_guvenli_konfigurasyon_calisir():
    """n=4, K=1 güvenli; çökmeden dönmeli."""
    from hga.experiments.kronecker_rank import run_nk_rank_sweep
    rapor = run_nk_rank_sweep(n_values=[4], k_values=[1], samples=32)
    assert rapor is not None


def test_ornek_hata_senaryosu_reddedilir():
    """KRONECKER_RANK_BUG.md'deki orijinal hata: n=128, K=16 (10^67 element)."""
    from hga.experiments.kronecker_rank import run_nk_rank_sweep
    with pytest.raises(ValueError):
        run_nk_rank_sweep(n_values=[128], k_values=[16])
