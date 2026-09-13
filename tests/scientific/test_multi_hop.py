"""Çok adımlı çıkarım + uzun bağlam benchmarkının bilimsel kapıları.

Buradaki testlerin çoğu "koştu mu" değil, **metrik ölü mü** sorusunu sorar.
İlk taslakta zincir takibi Python sözlüğünden yapılıyordu; doğruluk her
koşulda 1.0 çıkıyor, bellek katmanı süs kalıyordu. Aşağıdaki testler o
tuzağın geri gelmesini engeller.
"""
from __future__ import annotations

import pytest

from hga.evaluation.multi_hop import (
    DEFAULT_DISTRACTORS,
    DEFAULT_HOPS,
    multi_hop_markdown,
    run_multi_hop_benchmark,
)


def test_bol_bellekte_zincir_takibi_calisir():
    """Bellek bolken çok adımlı çıkarım tam doğru olmalı."""
    r = run_multi_hop_benchmark(hops=(1, 2, 3), distractor_levels=(0, 16), seeds=(1, 2))
    assert r.overall_accuracy == 1.0
    assert r.inference_accuracy == 1.0
    assert r.checks["multi_hop_inference_works"]
    assert r.checks["beats_degenerate"]
    assert r.checks["rejects_broken_chains"]


def test_kopuk_zincir_reddediliyor():
    """Negatif vakalarda yol kopuktur; sistem 'evet' dememeli."""
    r = run_multi_hop_benchmark(hops=(2, 3), distractor_levels=(0,), seeds=(1, 2))
    negatifler = [v for v in r.cases if not v.expected]
    assert negatifler, "negatif vaka üretilmemiş"
    for vaka in negatifler:
        assert vaka.predicted is False, (
            f"kopuk zincirde pozitif tahmin: hop={vaka.hop}"
        )


def test_metrik_olu_degil_bellek_daralinca_dogruluk_dusuyor():
    """EN ÖNEMLİ KAPI: bellek baskısı doğruluğa YANSIMALI.

    Zincir takibi bellekten geçmezse bu test kırılır — çünkü doğruluk
    slot sayısından bağımsız olarak 1.0 kalır.
    """
    bol = run_multi_hop_benchmark(
        hops=(2, 4), distractor_levels=(0, 256), seeds=(1,), slot_sayisi=4096
    )
    dar = run_multi_hop_benchmark(
        hops=(2, 4), distractor_levels=(0, 256), seeds=(1,), slot_sayisi=16
    )
    assert bol.overall_accuracy == 1.0
    assert dar.overall_accuracy < bol.overall_accuracy, (
        "Bellek 4096→16 slota indirildiği hâlde doğruluk düşmedi: "
        "zincir takibi bellekten geçmiyor, metrik ÖLÜ."
    )


def test_uzun_baglam_baskisi_gercekten_etki_ediyor():
    """Dar bellekte dolgu olgu sayısı artınca doğruluk bozulmalı."""
    r = run_multi_hop_benchmark(
        hops=(4,), distractor_levels=(0, 256), seeds=(1,), slot_sayisi=64
    )
    temiz = next(c for c in r.cells if c.distractors == 0)
    yuklu = next(c for c in r.cells if c.distractors == 256)
    assert yuklu.accuracy < temiz.accuracy, (
        "Araya 256 alakasız olgu girdiği hâlde doğruluk değişmedi; "
        "uzun bağlam ekseni ölçmüyor."
    )


def test_derin_zincir_sig_zincirden_once_kirilir():
    """Bellek baskısı altında hop=4, hop=2'den daha kırılgan olmalı."""
    r = run_multi_hop_benchmark(
        hops=(2, 4), distractor_levels=(256,), seeds=(1,), slot_sayisi=256
    )
    sig = next(c for c in r.cells if c.hop == 2)
    derin = next(c for c in r.cells if c.hop == 4)
    assert derin.accuracy <= sig.accuracy, (
        "Derin zincir sığdan daha dayanıklı çıktı; zincir uzunluğu ekseni şüpheli."
    )


def test_dejenere_kollar_yenilir_ve_raporlanir():
    """always_yes / always_no kolları %50'de kalmalı, motor onları yenmeli."""
    r = run_multi_hop_benchmark(hops=(1, 2, 3), distractor_levels=(0,), seeds=(1, 2))
    kollar = {k.arm: k for k in r.degenerate_arms}
    assert set(kollar) == {"always_yes", "always_no"}
    for kol in kollar.values():
        assert kol.accuracy == 0.5, (
            "Veri kümesi dengesiz: sabit cevap %50'den farklı alıyor"
        )
    assert r.overall_accuracy > max(k.accuracy for k in kollar.values())


def test_hop1_geri_cagirma_olarak_ayri_raporlaniyor():
    """hop=1 çıkarım değildir; ayrı metrik olarak görünmeli."""
    r = run_multi_hop_benchmark(hops=(1, 2), distractor_levels=(0,), seeds=(1,))
    assert r.recall_accuracy == 1.0
    assert r.inference_accuracy == 1.0
    # Çıkarım metriği yalnız hop>=2 vakalarından hesaplanmalı.
    assert all(v.hop >= 2 for v in r.cases if v.hop >= 2)


def test_determinizm_ayni_tohum_ayni_sonuc():
    a = run_multi_hop_benchmark(hops=(2, 3), distractor_levels=(0, 16), seeds=(1, 2))
    b = run_multi_hop_benchmark(hops=(2, 3), distractor_levels=(0, 16), seeds=(1, 2))
    assert a.to_dict() == b.to_dict()
    assert a.dataset_hash == b.dataset_hash


def test_rapor_sinirlari_dil_modeli_iddiasi_yapmiyor():
    """Dürüstlük kapısı: bu bir 'context window' testi değildir."""
    r = run_multi_hop_benchmark(hops=(1, 2), distractor_levels=(0,), seeds=(1,))
    metin = " ".join(r.limitations).lower()
    assert "context window" in metin or "dil modeli" in metin
    assert "dar" in metin, "Çıkarımın darlığı sınırlarda belirtilmeli"
    assert any("hop=1" in s for s in r.limitations), (
        "hop=1'in geri çağırma olduğu sınırlarda yazmalı"
    )


def test_markdown_izgara_ve_kapilari_iceriyor():
    r = run_multi_hop_benchmark(hops=(1, 2), distractor_levels=(0, 16), seeds=(1,))
    md = multi_hop_markdown(r)
    assert "# Çok Adımlı Çıkarım" in md
    assert "geri çağırma" in md
    assert "Dejenere kontrol kolları" in md
    assert "Kabul kapıları" in md
    for ad in r.checks:
        assert ad in md


@pytest.mark.parametrize("hop", DEFAULT_HOPS)
def test_varsayilan_hop_seviyeleri_kosulabilir(hop):
    r = run_multi_hop_benchmark(hops=(hop,), distractor_levels=(0,), seeds=(1,))
    assert r.cells and r.cells[0].total > 0


def test_varsayilan_dolgu_seviyeleri_artan_sirada():
    assert list(DEFAULT_DISTRACTORS) == sorted(DEFAULT_DISTRACTORS)
    assert DEFAULT_DISTRACTORS[0] == 0, "Dolgusuz taban çizgisi şart"
