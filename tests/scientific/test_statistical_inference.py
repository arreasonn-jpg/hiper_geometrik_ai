# -*- coding: utf-8 -*-
"""İstatistiksel çıkarım katmanının sözleşme testleri.

Buradaki testler "kod çalışıyor mu" değil, **sayılar doğru mu** sorusunu
sorar: referans değerler bağımsız olarak (tam sayım / elle hesap) türetilir.
"""
from __future__ import annotations

import math
import statistics as st
from itertools import product

import pytest

from hga.evaluation.statistics import (
    EXACT_PERMUTATION_LIMIT,
    bootstrap_ci,
    cohens_d_paired,
    compare_paired,
    minimum_two_sided_p,
    paired_permutation_test,
    summarize_seed_metric,
    wilcoxon_signed_rank,
)

# Wikipedia'nın Wilcoxon signed-rank örneği; W=18, etkin n=9 (bir çift sıfır).
WILCOXON_TREATMENT = [125, 115, 130, 140, 140, 115, 140, 125, 140, 135]
WILCOXON_BASELINE = [110, 122, 125, 120, 140, 124, 123, 137, 135, 145]


# ── Güç sınırı: 5 seed ile p<0.05 imkânsızdır ───────────────────────────────
@pytest.mark.parametrize(
    "n,beklenen",
    [(5, 2 / 32), (6, 2 / 64), (8, 2 / 256), (10, 2 / 1024)],
)
def test_minimum_p_degeri_permutasyon_sayisiyla_uyumlu(n, beklenen):
    assert minimum_two_sided_p(n) == pytest.approx(beklenen)


def test_bes_seed_tasarim_geregi_yetersiz_guclu_olarak_isaretlenir():
    """n=5'te en küçük p 0.0625'tir; sonuç ne olursa olsun anlamlılık iddia edilemez."""
    sonuc = paired_permutation_test([1.0] * 5, [0.0] * 5)
    assert sonuc.sample_size == 5
    assert sonuc.p_value == pytest.approx(2 / 32)
    assert sonuc.underpowered_by_design is True
    assert sonuc.significant_at_0_05 is False
    assert any("YETERSİZ GÜÇ" in satir for satir in sonuc.limitations)


def test_alti_seed_anlamlilik_esigine_ulasabilir():
    """n=6'da minimum p 0.03125'tir: artık p<0.05 mümkündür."""
    sonuc = paired_permutation_test([1.0] * 6, [0.0] * 6)
    assert sonuc.p_value == pytest.approx(2 / 64)
    assert sonuc.underpowered_by_design is False
    assert sonuc.significant_at_0_05 is True


# ── Permütasyon testi tam sayıma eşit olmalı ────────────────────────────────
def test_permutasyon_p_degeri_bagimsiz_tam_sayimla_ayni():
    islem = [0.90, 0.85, 0.88, 0.91, 0.87, 0.93]
    kontrol = [0.60, 0.62, 0.58, 0.61, 0.59, 0.64]
    farklar = [a - b for a, b in zip(islem, kontrol)]
    gozlenen = st.fmean(farklar)

    toplam = uc = 0
    for isaretler in product((1.0, -1.0), repeat=len(farklar)):
        aday = st.fmean([s * f for s, f in zip(isaretler, farklar)])
        toplam += 1
        if abs(aday) >= abs(gozlenen) - 1e-15:
            uc += 1

    sonuc = paired_permutation_test(islem, kontrol)
    assert sonuc.exact is True
    assert sonuc.permutations == toplam == 2 ** len(farklar)
    assert sonuc.p_value == pytest.approx(uc / toplam)


def test_tek_yonlu_permutasyon_en_uc_durumda_bir_bolu_iki_uzeri_n():
    sonuc = paired_permutation_test([2, 3, 4, 5, 6], [1, 1, 1, 1, 1],
                                    alternative="greater")
    assert sonuc.p_value == pytest.approx(1 / 32)


def test_buyuk_n_monte_carloya_duser_ve_bunu_gizlemez():
    n = EXACT_PERMUTATION_LIMIT + 2
    islem = [1.0 + 0.01 * i for i in range(n)]
    kontrol = [0.0] * n
    sonuc = paired_permutation_test(islem, kontrol, monte_carlo_samples=5000)
    assert sonuc.exact is False
    assert sonuc.p_value > 0.0, "Monte Carlo'da p asla tam 0 raporlanmamalı"
    assert any("Monte Carlo" in satir for satir in sonuc.limitations)


# ── Wilcoxon ────────────────────────────────────────────────────────────────
def test_wilcoxon_referans_ornekle_dogrulanir():
    sonuc = wilcoxon_signed_rank(WILCOXON_TREATMENT, WILCOXON_BASELINE)
    assert sonuc.statistic == pytest.approx(18.0)
    assert sonuc.sample_size == 9, "Sıfır fark atılmalı, etkin n 9 olmalı"
    assert sonuc.exact is True
    assert sonuc.p_value == pytest.approx(0.6328125)
    assert any("atıldı" in satir for satir in sonuc.limitations)


def test_wilcoxon_tum_farklar_sifirsa_tanimsizligi_bildirir():
    sonuc = wilcoxon_signed_rank([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert sonuc.sample_size == 0
    assert sonuc.p_value == 1.0
    assert sonuc.significant_at_0_05 is False


def test_wilcoxon_esit_mutlak_farklara_ortalama_sira_verir():
    """Tie'lar ortalama sıra almalı; sıra toplamı n(n+1)/2 olmalı."""
    islem = [1.0, -1.0, 2.0, -2.0]
    kontrol = [0.0, 0.0, 0.0, 0.0]
    sonuc = wilcoxon_signed_rank(islem, kontrol)
    n = sonuc.sample_size
    assert n == 4
    # W+ + W- her zaman n(n+1)/2'dir; simetrik veride ikisi de yarısıdır.
    assert sonuc.statistic == pytest.approx(n * (n + 1) / 4)


# ── Etki büyüklüğü ──────────────────────────────────────────────────────────
def test_cohens_d_z_elle_hesapla_ayni():
    islem = WILCOXON_TREATMENT
    kontrol = WILCOXON_BASELINE
    farklar = [a - b for a, b in zip(islem, kontrol)]
    beklenen = st.fmean(farklar) / st.stdev(farklar)   # stdev = ddof 1
    sonuc = cohens_d_paired(islem, kontrol)
    assert sonuc.value == pytest.approx(beklenen)
    assert abs(sonuc.hedges_g) < abs(sonuc.value), "Hedges g, d'yi küçültmeli"


def test_hedges_duzeltmesi_kucuk_orneklemde_daha_guclu():
    kucuk = cohens_d_paired([1, 2, 3, 4], [0, 0, 0, 1])
    buyuk = cohens_d_paired([1, 2, 3, 4] * 8, [0, 0, 0, 1] * 8)
    kucuk_oran = abs(kucuk.hedges_g / kucuk.value)
    buyuk_oran = abs(buyuk.hedges_g / buyuk.value)
    assert kucuk_oran < buyuk_oran < 1.0


def test_sifir_varyansli_fark_tanimsiz_olarak_raporlanir():
    sonuc = cohens_d_paired([1.0] * 5, [0.0] * 5)
    assert math.isinf(sonuc.value)
    assert sonuc.magnitude == "undefined"
    assert "tanımsız" in sonuc.interpretation


def test_etki_buyuklugu_esikleri():
    assert cohens_d_paired([0, 0, 0, 0], [0, 0, 0, 0]).magnitude == "negligible"


# ── Bootstrap CI ────────────────────────────────────────────────────────────
def test_bootstrap_ci_deterministik_ve_nokta_tahmini_icerir():
    degerler = [0.90, 0.85, 0.88, 0.91, 0.87]
    birinci = bootstrap_ci(degerler, seed=7)
    ikinci = bootstrap_ci(degerler, seed=7)
    assert birinci.to_dict() == ikinci.to_dict()
    assert birinci.lower <= birinci.point_estimate <= birinci.upper
    assert birinci.point_estimate == pytest.approx(st.fmean(degerler))


def test_bootstrap_ci_farkli_seedle_degisir_ama_yakin_kalir():
    degerler = [0.90, 0.85, 0.88, 0.91, 0.87]
    a = bootstrap_ci(degerler, seed=1)
    b = bootstrap_ci(degerler, seed=2)
    assert a.point_estimate == b.point_estimate
    assert abs(a.lower - b.lower) < 0.01


def test_sabit_dizide_ci_genisligi_sifir():
    sonuc = bootstrap_ci([5.0] * 6)
    assert sonuc.lower == sonuc.upper == 5.0


def test_kucuk_ornekte_ci_darlik_uyarisi_verilir():
    sonuc = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert any("n=5 küçük" in satir for satir in sonuc.limitations)


def test_gecersiz_girdiler_reddedilir():
    with pytest.raises(ValueError):
        bootstrap_ci([1.0])                       # tek gözlem
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, float("nan")])         # NaN
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], resamples=10)    # yetersiz yeniden örnekleme
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], confidence_level=1.5)
    with pytest.raises(ValueError):
        paired_permutation_test([1.0, 2.0], [1.0])   # eşleşmeyen uzunluk


# ── Birleşik karşılaştırma ──────────────────────────────────────────────────
def test_compare_paired_yetersiz_gucu_karara_yansitir():
    """Fark devasa olsa bile 5 seed'te karar 'YETERSİZ GÜÇ' olmalı."""
    rapor = compare_paired("accuracy", [0.99] * 5, [0.10] * 5)
    assert "YETERSİZ GÜÇ" in rapor.verdict
    assert rapor.permutation_test["underpowered_by_design"] is True
    assert rapor.difference_ci["lower"] > 0.0


def test_compare_paired_ayrisma_tespit_eder():
    islem = [0.90, 0.92, 0.88, 0.91, 0.89, 0.93, 0.90, 0.92]
    kontrol = [0.60, 0.61, 0.59, 0.62, 0.58, 0.63, 0.60, 0.61]
    rapor = compare_paired("accuracy", islem, kontrol)
    assert rapor.permutation_test["significant_at_0_05"] is True
    assert "AYRIŞMA" in rapor.verdict
    assert rapor.difference_ci["lower"] > 0.0


def test_compare_paired_ayrim_yoksa_bunu_soyler():
    islem = [0.50, 0.52, 0.48, 0.51, 0.49, 0.53, 0.47, 0.50]
    kontrol = [0.51, 0.49, 0.50, 0.48, 0.52, 0.47, 0.53, 0.50]
    rapor = compare_paired("accuracy", islem, kontrol)
    assert "AYRIM YOK" in rapor.verdict
    assert rapor.difference_ci["lower"] <= 0.0 <= rapor.difference_ci["upper"]


def test_compare_paired_deterministik():
    islem = [0.90, 0.85, 0.88, 0.91, 0.87]
    kontrol = [0.60, 0.62, 0.58, 0.61, 0.59]
    assert compare_paired("m", islem, kontrol).to_dict() == \
        compare_paired("m", islem, kontrol).to_dict()


def test_compare_paired_coklu_test_uyarisini_tasir():
    rapor = compare_paired("m", [1, 2, 3, 4, 5], [0, 1, 2, 3, 4])
    assert any("çoklu test düzeltmesi" in satir for satir in rapor.limitations)
    assert any("gerçek dünya genellemesi" in satir for satir in rapor.limitations)


def test_summarize_seed_metric_alanlari():
    ozet = summarize_seed_metric([0.90, 0.85, 0.88, 0.91, 0.87])
    assert ozet["n"] == 5
    assert ozet["ci_lower"] <= ozet["mean"] <= ozet["ci_upper"]
    assert ozet["std_sample"] == pytest.approx(st.stdev([0.90, 0.85, 0.88, 0.91, 0.87]))
    assert ozet["min"] == 0.85 and ozet["max"] == 0.91
