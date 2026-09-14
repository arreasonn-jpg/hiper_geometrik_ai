# -*- coding: utf-8 -*-
"""P0-1: Priority(E) nedensel zincir ablasyonu testleri."""
import pytest

from hga.evaluation.priority_ablation import (
    TERIMLER,
    havuz_uret,
    kendall_tau,
    priority_ablation_markdown,
    run_priority_weight_ablation,
    spearman_rho,
)


def test_kendall_tau_ozdes_diziler_bire_esittir():
    a = [3.0, 1.0, 2.0, 5.0]
    assert kendall_tau(a, a) == pytest.approx(1.0)
    assert spearman_rho(a, a) == pytest.approx(1.0)


def test_kendall_tau_ters_dizide_eksi_bir():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [4.0, 3.0, 2.0, 1.0]
    assert kendall_tau(a, b) == pytest.approx(-1.0)
    assert spearman_rho(a, b) == pytest.approx(-1.0)


def test_beraberlikler_hataya_yol_acmaz():
    """Tümü eşit skor: tau-b tanımsız paydaya düşmemeli."""
    a = [1.0, 1.0, 1.0]
    assert kendall_tau(a, a) == pytest.approx(1.0)
    assert spearman_rho(a, a) == pytest.approx(1.0)


def test_farkli_uzunluk_acik_hata():
    with pytest.raises(ValueError):
        kendall_tau([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        spearman_rho([1.0, 2.0], [1.0])


def test_havuz_heterojen_ve_dogrulayici_bagimsiz():
    """Havuz doğru/yanlış/bilinmeyen karışımı olmalı; aksi halde ablasyon ölü."""
    havuz = havuz_uret(candidate_count=60, seed=3)
    ozet = havuz.ozet()
    assert ozet["candidates"] == 60
    assert ozet["verifier_true"] > 0
    assert ozet["verifier_false"] > 0
    # Doğrulayıcı kararı Priority'den bağımsız aritmetik ortamdan gelir.
    assert ozet["verifier_true"] + ozet["verifier_false"] <= 60


def test_havuz_deterministik():
    a = havuz_uret(candidate_count=40, seed=7).ozet()
    b = havuz_uret(candidate_count=40, seed=7).ozet()
    assert a == b


def test_zincir_dort_halkanin_hepsini_raporlar():
    rapor = run_priority_weight_ablation(k=10, seeds=(1, 2), candidate_count=80)
    assert set(rapor.variants) == set(TERIMLER)
    for terim in TERIMLER:
        zincir = rapor.variants[terim]["chain"]
        assert set(zincir) == {"score_changed", "ranking_changed",
                               "selection_changed", "downstream_changed",
                               "chain_complete"}


def test_her_terim_en_azindan_skoru_degistirir():
    """Ağırlığı sıfırlamak skoru hiç değiştirmiyorsa terim ölü koddur."""
    rapor = run_priority_weight_ablation(k=10, seeds=(1, 2), candidate_count=80)
    for terim in TERIMLER:
        assert rapor.variants[terim]["chain"]["score_changed"], (
            f"{terim} skoru hiç değiştirmiyor — ölü ağırlık")


def test_w_gain_zinciri_uctan_uca_tasir():
    """En az bir terimin downstream'e kadar etkisi ölçülebilmeli."""
    rapor = run_priority_weight_ablation(k=10, seeds=(1, 2, 3),
                                         candidate_count=120)
    assert rapor.checks["at_least_one_full_causal_chain"]
    assert rapor.variants["w_gain"]["chain"]["chain_complete"]


def test_downstream_dis_dogrulayiciya_dayanir():
    """Baseline downstream metrikleri 0..1 aralığında ve tutarlı olmalı."""
    rapor = run_priority_weight_ablation(k=10, seeds=(1,), candidate_count=60)
    d = rapor.baseline_downstream
    for ad in ("verification_yield", "novel_knowledge_yield", "decidability",
               "false_selection_rate", "subject_diversity"):
        assert 0.0 <= d[ad] <= 1.0
    # Yeni bilgi verimi, doğrulama veriminden büyük olamaz.
    assert d["novel_knowledge_yield"] <= d["verification_yield"] + 1e-9


def test_etkisiz_terimler_bulgu_olarak_yazilir():
    """Zinciri taşımayan terim gizlenmemeli, bulgu metnine girmeli."""
    rapor = run_priority_weight_ablation(k=10, seeds=(1, 2), candidate_count=80)
    kopuk = [t for t in TERIMLER
             if not rapor.variants[t]["chain"]["chain_complete"]]
    metin = " ".join(rapor.findings)
    for terim in kopuk:
        assert terim in metin


def test_markdown_tablo_ve_kapilari_icerir():
    rapor = run_priority_weight_ablation(k=5, seeds=(1,), candidate_count=40)
    md = priority_ablation_markdown(rapor)
    assert "Kendall" in md and "Spearman" in md
    assert "Kabul kapıları" in md
    assert "Sınırlar" in md


def test_gecersiz_parametreler_acik_hata():
    with pytest.raises(ValueError):
        run_priority_weight_ablation(k=0)
    with pytest.raises(ValueError):
        run_priority_weight_ablation(seeds=())
    with pytest.raises(ValueError):
        havuz_uret(candidate_count=2)
    with pytest.raises(ValueError):
        havuz_uret(known_fact_ratio=1.5)


def test_rapor_serilestirilebilir():
    import json
    rapor = run_priority_weight_ablation(k=5, seeds=(1,), candidate_count=40)
    assert json.loads(json.dumps(rapor.to_dict()))["protocol"]
