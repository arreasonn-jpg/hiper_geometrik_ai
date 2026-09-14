# -*- coding: utf-8 -*-
"""P0-3: TWT gerçek sonuç tablosu, FLOP muhasebesi ve çoklu tohum testleri."""
import pytest

from hga.evaluation.twt_results import (
    CALIBRATION_METRICS,
    HEADLINE_METRICS,
    analytic_forward_flops,
    flop_fairness,
)

torch = pytest.importorskip("torch")

from hga.evaluation.twt_baselines import MODEL_ORDER  # noqa: E402
from hga.evaluation.twt_results import (  # noqa: E402
    measured_forward_flops,
    reconcile_flops,
    results_markdown,
    run_twt_results,
)


# ── FLOP muhasebesi ─────────────────────────────────────────────────────────
def test_her_mimari_icin_flop_turetilir():
    for model in MODEL_ORDER:
        sonuc = analytic_forward_flops(model)
        assert sonuc["forward_flops_per_example"] > 0
        assert sonuc["breakdown"]


def test_bilinmeyen_model_acik_hata():
    with pytest.raises(ValueError):
        analytic_forward_flops("yok_boyle")
    with pytest.raises(ValueError):
        measured_forward_flops("yok_boyle")


def test_flop_dokumu_toplami_tutar():
    for model in MODEL_ORDER:
        sonuc = analytic_forward_flops(model)
        assert sum(sonuc["breakdown"].values()) == sonuc["forward_flops_per_example"]


def test_analitik_flop_olculenle_uyusur():
    """Formül yanlışsa bu test yakalar — yer gerçeği PyTorch sayacıdır."""
    uzlasma = reconcile_flops()
    assert uzlasma["all_agree"], uzlasma["per_model"]


def test_dikkatsiz_mimarilerde_sapma_sifir():
    """SDPA füzyonu yoksa analitik form ölçümle TAM eşleşmeli."""
    for model in ("dense", "kronecker"):
        satir = reconcile_flops([model])["per_model"][model]
        assert satir["relative_error"] == 0.0


def test_kronecker_flopu_n4_degil_2n3():
    """Kronecker'ın hesap avantajı iddiası tam olarak burada yatıyor."""
    import math

    from hga.evaluation.twt_baselines import ARCHITECTURE_CONFIG
    n = int(ARCHITECTURE_CONFIG["kronecker_n"])
    katman = int(ARCHITECTURE_CONFIG["kronecker_layers"])
    dokum = analytic_forward_flops("kronecker")["breakdown"]
    assert dokum["kronecker_chain"] == katman * 2 * n ** 3
    assert dokum["kronecker_chain"] < katman * n ** 4
    assert not math.isclose(dokum["kronecker_chain"], katman * n ** 4)


def test_flop_adilligi_orani_hesaplar():
    adillik = flop_fairness()
    assert adillik["flop_max_to_min_ratio"] >= 1.0
    assert adillik["cheapest_model"] in MODEL_ORDER
    assert adillik["most_expensive_model"] in MODEL_ORDER


def test_esit_parametre_esit_flop_demek_degil():
    """Bu projenin adillik iddiasının bilinen sınırı; sessizce geçilmemeli."""
    adillik = flop_fairness()
    assert adillik["flop_max_to_min_ratio"] > 1.0
    assert not adillik["within_tolerance"], (
        "Oran kapıya girdiyse bu testin beklentisi güncellenmeli")


def test_gecersiz_tolerans_acik_hata():
    with pytest.raises(ValueError):
        flop_fairness(tolerance=0.5)


# ── Koşum ───────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def rapor():
    return run_twt_results(seeds=(1, 2), profile="smoke")


def test_gecersiz_girdiler_acik_hata():
    with pytest.raises(ValueError):
        run_twt_results(seeds=())
    with pytest.raises(ValueError):
        run_twt_results(seeds=(1, 1))
    with pytest.raises(ValueError):
        run_twt_results(seeds=(1,), models=("yok",))
    with pytest.raises(ValueError):
        run_twt_results(seeds=(1,), dimensions=("yok",))


def test_tum_ana_metrikler_her_dilimde(rapor):
    for model in rapor.results:
        for boyut in rapor.results[model]:
            for ad, _ in HEADLINE_METRICS:
                assert rapor.results[model][boyut][ad]["n"] == len(rapor.seeds)


def test_kalibrasyon_metrikleri_tum_dilimlerde(rapor):
    for model in rapor.calibration:
        for boyut in rapor.calibration[model]:
            for ad, _ in CALIBRATION_METRICS:
                assert rapor.calibration[model][boyut][ad]["n"] == len(rapor.seeds)


def test_sentence_disjoint_ayri_raporlanir(rapor):
    """Kullanıcının açıkça istediği dilim; toplamda gizlenemez."""
    assert "sentence_disjoint" in rapor.results["hga"]
    assert rapor.checks["sentence_disjoint_reported"]


def test_tek_tohumda_ci_uretilmez():
    """Tek tohumla std/CI TANIMSIZ; sahte kesinlik üretilmemeli."""
    tek = run_twt_results(seeds=(1,), profile="smoke",
                          dimensions=("all",))
    ozet = tek.results["hga"]["all"]["f1"]
    assert ozet["n"] == 1
    assert ozet["std_sample"] is None
    assert ozet["ci_lower"] is None
    assert "TANIMSIZ" in ozet["note"]


def test_coklu_tohumda_ci_uretilir(rapor):
    ozet = rapor.results["hga"]["all"]["f1"]
    assert ozet["std_sample"] is not None
    assert ozet["ci_lower"] <= ozet["mean"] <= ozet["ci_upper"]


def test_yirmi_tohumdan_azi_yetersiz_isaretlenir(rapor):
    assert not rapor.checks["seed_count_sufficient_for_inference"]
    assert any("20" in b for b in rapor.findings)


def test_maliyet_tablosu_flop_icerir(rapor):
    for model in rapor.cost:
        assert rapor.cost[model]["forward_flops_per_example"] > 0
        assert rapor.cost[model]["physical_parameters"] > 0
    assert rapor.checks["flops_reported_for_all_models"]


def test_parametre_kapisi_gecer_flop_kapisi_kalir(rapor):
    """Dürüst rapor: bir kapı geçip diğeri kalabilir ve bu gizlenmez."""
    assert rapor.checks["parameter_budget_within_one_percent"]
    assert not rapor.checks["flop_budget_within_tolerance"]
    assert any("FLOP" in b for b in rapor.findings)


def test_eslesmis_karsilastirma_uretilir(rapor):
    assert rapor.comparison
    for boyut, k in rapor.comparison.items():
        assert k["treatment_label"] == "hga"
        assert k["baseline_label"] != "hga"
        assert "verdict" in k


def test_gorev_dil_modelleme_degil(rapor):
    """TWT bir arc doğrulama görevidir; PPL iddiası taşımamalı."""
    assert rapor.task_summary["is_language_modeling"] is False
    assert any("dil modelleme" in s for s in rapor.limitations)


def test_alt_adillik_kapilari_devralinir(rapor):
    assert rapor.checks["underlying_fairness_gates_pass"]


def test_markdown_tum_bolumleri_icerir(rapor):
    md = results_markdown(rapor)
    assert "Maliyet bütçesi" in md
    assert "Kalibrasyon" in md
    assert "sentence_disjoint" in md
    assert "Kabul kapıları" in md
    assert "dil modelleme değildir" in md


def test_rapor_serilestirilebilir(rapor):
    import json
    assert json.loads(json.dumps(rapor.to_dict()))["protocol"]
