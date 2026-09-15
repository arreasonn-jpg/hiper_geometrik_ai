# -*- coding: utf-8 -*-
"""Uzun bağlam LM protokolünün (long_context_lm_v1) bilimsel testleri."""
import math

import pytest

torch = pytest.importorskip("torch")

from hga.evaluation.long_context import (  # noqa: E402
    PROFILES,
    _sample_positions,
    _windows_at,
    run_long_context_benchmark,
)


# ── Pencere kurulumu (hızlı, koşumsuz) ──────────────────────────────────────
def test_pozisyonlar_deterministik():
    docs = [[1, 2, 3, 4, 5], [6, 7, 8]]
    a = _sample_positions(torch, docs, 4, seed=99)
    b = _sample_positions(torch, docs, 4, seed=99)
    assert a == b
    c = _sample_positions(torch, docs, 4, seed=100)
    assert a != c  # tohum kimliğin parçası


def test_pencere_belge_sinirini_asmaz():
    """İlk tokenın bağlamı tamamen PAD olmalı; komşu belge sızmamalı."""
    docs = [[11, 12, 13], [21, 22]]
    poz = [(0, 0), (1, 0), (1, 1)]
    xs, ys = _windows_at(torch, docs, poz, context=4)
    assert xs[0].tolist() == [0, 0, 0, 0]      # belge başı: yalnız PAD
    assert xs[1].tolist() == [0, 0, 0, 0]      # ikinci belgenin başı da öyle
    assert xs[2].tolist() == [0, 0, 0, 21]     # yalnız KENDİ belgesinden
    assert ys.tolist() == [11, 21, 22]


def test_ayni_pozisyon_farkli_baglam_ayni_hedef():
    """Eşleşmiş tasarımın özü: bağlam değişir, hedef token değişmez."""
    docs = [[1, 2, 3, 4, 5, 6, 7, 8]]
    poz = _sample_positions(torch, docs, 5, seed=7)
    _, y_kisa = _windows_at(torch, docs, poz, context=2)
    _, y_uzun = _windows_at(torch, docs, poz, context=6)
    assert y_kisa.tolist() == y_uzun.tolist()


def test_gecersiz_girdiler_acik_hata():
    with pytest.raises(ValueError):
        run_long_context_benchmark(profile="yok_boyle")
    with pytest.raises(ValueError):
        run_long_context_benchmark(profile="smoke", seeds=(1,))
    with pytest.raises(ValueError):
        run_long_context_benchmark(profile="smoke",
                                   overrides={"contexts": (16,)})


# ── Koşum (yavaş; smoke profil) ─────────────────────────────────────────────
@pytest.fixture(scope="module")
def rapor():
    return run_long_context_benchmark(profile="smoke")


def test_tum_hucreler_tamamlanir(rapor):
    assert rapor.checks["all_cells_completed"]
    assert rapor.checks["finite_metrics_all_cells"]
    for arm in ("dense", "transformer", "hga"):
        for c in rapor.config["contexts"]:
            hucre = rapor.results[arm][str(c)]
            assert len(hucre["per_seed"]) == len(rapor.seeds)
            assert math.isfinite(hucre["test_ppl_mean"])


def test_parametre_butcesi_her_baglamda_esli(rapor):
    assert rapor.checks["parameter_budget_leq_1_05_every_context"]
    for c, b in rapor.budget_by_context.items():
        assert b["max_to_min_ratio"] <= 1.05, c


def test_baglam_etkisi_yonu_raporlanir(rapor):
    assert rapor.checks["context_effect_direction_reported"]
    for arm, e in rapor.context_effect["per_arm"].items():
        assert e["direction"] in ("improves", "degrades", "flat")
        assert any(arm in b for b in rapor.findings)


def test_ngram_zemini_ayni_pencerelerde(rapor):
    for c in rapor.config["contexts"]:
        ng = rapor.ngram_by_context[str(c)]
        assert ng["unigram_test_ppl"] > 1.0
        assert ng["bigram_test_ppl"] > 1.0
        # bigram bağlamı kullanır; unigramdan kötü olmamalı
        assert ng["bigram_test_ppl"] <= ng["unigram_test_ppl"] * 1.05


def test_ngram_zemini_kapi_degil_rapor(rapor):
    """N-gram zemini KAPI değildir ama her bağlamda raporlanmak zorundadır.

    Mutlak LM kalitesi (n-gramı geçmek) ana ``turkish_lm`` protokolünün
    kapısıdır (full: 4000 adım, PASS). Bu kısa eşit-bütçe taramasının
    iddiası bağlam ETKİSİDİR; neural kollar bu bütçede n-gramı geçmiyorsa
    bu bulgularda AÇIKÇA yazılır, kapı arkasına saklanmaz.
    """
    assert rapor.checks["ngram_floor_reported_every_context"]
    en_iyi = min(
        rapor.results[a][str(rapor.context_effect["max_context"])]
        ["test_ppl_mean"] for a in ("dense", "transformer", "hga"))
    bigram = rapor.ngram_by_context[
        str(rapor.context_effect["max_context"])]["bigram_test_ppl"]
    if en_iyi >= bigram:
        assert any("AÇIK SINIR" in b for b in rapor.findings)


def test_rapor_kimligi_ve_serilestirme(rapor):
    d = rapor.to_dict()
    assert d["protocol"] == "long_context_lm_v1"
    assert len(d["dataset_hash"]) == 64
    assert d["positions"]["signature"]
    md = rapor.markdown()
    assert "Uzun Bağlam" in md and "Kapılar" in md
    assert "Sınırlar" in md


def test_profiller_en_az_iki_baglam_ve_tohum():
    for ad, p in PROFILES.items():
        assert len(p["contexts"]) >= 2, ad
        assert len(p["seeds"]) >= 2, ad
