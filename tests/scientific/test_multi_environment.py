# -*- coding: utf-8 -*-
"""P1 çoklu ortam + cross-domain verifier izolasyonu testleri."""
from __future__ import annotations

import pytest

from hga.evaluation.capability_vector import build_scorecard
from hga.evaluation.multi_environment import (
    ENVIRONMENTS,
    GENERATORS,
    PROTOCOL,
    VERIFIERS,
    build_adversarial_probes,
    multi_environment_markdown,
    run_multi_environment_benchmark,
)


@pytest.fixture(scope="module")
def rapor():
    return run_multi_environment_benchmark(
        seeds=(1, 2), claims_per_environment=40)


def test_protokol_ve_ortamlar(rapor):
    assert rapor.protocol == PROTOCOL
    assert rapor.schema_version == 1
    assert set(rapor.environments) == set(ENVIRONMENTS)
    assert len(ENVIRONMENTS) == 5
    assert set(GENERATORS) == set(ENVIRONMENTS)
    assert set(VERIFIERS) == set(ENVIRONMENTS)


def test_her_dogrulayici_kendi_alaninda_yetkin(rapor):
    """Köşegen: doğrulayıcı kendi ortamında karar vermeli ve doğru olmalı."""
    for ortam in ENVIRONMENTS:
        kendi = rapor.own_domain[ortam]
        assert kendi["coverage"] == pytest.approx(1.0)
        assert kendi["accuracy"] == pytest.approx(1.0)
        assert kendi["abstain"] == 0
        assert rapor.cross_matrix[ortam][ortam]["abstain_rate"] == 0.0


def test_kosegen_disi_tam_cekimserlik(rapor):
    """Asıl izolasyon iddiası: yabancı alanda ASLA karar verme."""
    for uretici in ENVIRONMENTS:
        for dogrulayici in ENVIRONMENTS:
            if uretici == dogrulayici:
                continue
            hucre = rapor.cross_matrix[uretici][dogrulayici]
            assert hucre["abstain_rate"] == pytest.approx(1.0), (
                f"{dogrulayici} doğrulayıcısı {uretici} alanında karar verdi")
            assert hucre["spoke_rate"] == pytest.approx(0.0)
            assert hucre["contaminated"] is False


def test_kontaminasyon_yok(rapor):
    kirlilik = rapor.contamination
    assert kirlilik["isolation_rate"] == pytest.approx(1.0)
    assert kirlilik["cross_domain_decisions"] == 0
    assert kirlilik["contamination_rate"] == pytest.approx(0.0)
    assert kirlilik["contaminated_pairs"] == []


def test_cekismeli_sondalar_kandirilamiyor(rapor):
    """Tek ipucu (sembol ya da ilişki) yetmemeli; ikisi birden gerekmeli."""
    cekismeli = rapor.adversarial
    assert cekismeli["probes"] > 0
    assert cekismeli["abstain_rate"] == pytest.approx(1.0)
    assert cekismeli["spoke"] == 0
    assert cekismeli["violations"] == []
    for tur, veri in cekismeli["by_kind"].items():
        assert veri["abstain"] == veri["probes"], f"{tur} tuzağına düşüldü"


def test_cekismeli_sonda_turleri_kapsanmis():
    sondalar = build_adversarial_probes(ENVIRONMENTS)
    turler = {p["kind"] for p in sondalar}
    assert turler == {
        "own_symbols_foreign_relation",
        "foreign_symbols_own_relation",
        "shared_surface",
    }
    assert len(sondalar) >= 3 * len(ENVIRONMENTS)


def test_kendi_sembolu_yabanci_iliski_reddediliyor():
    """En zor tuzak: sembol tanıdık ama ilişki başka alanın."""
    for sonda in build_adversarial_probes(ENVIRONMENTS):
        if sonda["kind"] != "own_symbols_foreign_relation":
            continue
        karar = VERIFIERS[sonda["verifier"]](sonda["claim"])
        assert karar == "ABSTAIN", (
            f"{sonda['verifier']} tanıdık sembole kanıp yabancı ilişkiye "
            f"karar verdi: {sonda['claim']}")


def test_yanlis_kabul_yok(rapor):
    for ortam in ENVIRONMENTS:
        assert rapor.own_domain[ortam]["false_acceptance_rate"] == 0.0
        assert rapor.own_domain[ortam]["false_positive"] == 0


def test_tum_kapilar_geciyor(rapor):
    kaldi = [ad for ad, v in rapor.checks.items() if not v]
    assert kaldi == [], f"başarısız kapılar: {kaldi}"
    assert "verifiers_abstain_on_adversarial_probes" in rapor.checks


def test_determinizm():
    a = run_multi_environment_benchmark(seeds=(3,), claims_per_environment=20)
    b = run_multi_environment_benchmark(seeds=(3,), claims_per_environment=20)
    assert a.cross_matrix == b.cross_matrix
    assert a.contamination == b.contamination
    assert a.adversarial == b.adversarial


def test_tohum_sayisi_orantili_iddia(rapor):
    for ortam in ENVIRONMENTS:
        # 2 tohum × 40 iddia
        assert rapor.own_domain[ortam]["claims"] == 80


def test_sembol_ortusmesi_seffaf_raporlaniyor(rapor):
    """Ayrık sembol uzayı bir kolaylıktır; rapor bunu gizlememeli."""
    assert "note" in rapor.symbol_overlap
    assert rapor.limitations
    assert any("sembol" in s.lower() for s in rapor.limitations)


def test_ortam_alt_kumesi_kosulabilir():
    r = run_multi_environment_benchmark(
        seeds=(1,), claims_per_environment=10,
        environments=("physics", "causal"))
    assert set(r.environments) == {"physics", "causal"}
    assert set(r.cross_matrix) == {"physics", "causal"}


def test_gecersiz_girdiler():
    with pytest.raises(ValueError):
        run_multi_environment_benchmark(seeds=())
    with pytest.raises(ValueError):
        run_multi_environment_benchmark(claims_per_environment=0)
    with pytest.raises(ValueError):
        run_multi_environment_benchmark(environments=("yok-boyle",))


def test_markdown_uretimi(rapor):
    md = multi_environment_markdown(rapor)
    assert "physics" in md and "language" in md
    assert "ABSTAIN" in md or "çekimser" in md.lower()
    assert "Sınırlar" in md


def test_karne_dogrulama_bolumune_baglandi(rapor):
    karne = build_scorecard(multi_environment=rapor.to_dict())
    bolum = karne.sections["verification"]
    assert bolum["score"] is not None
    assert bolum["inputs"]["verifier_isolation_rate"] == pytest.approx(1.0)
    assert bolum["inputs"]["adversarial_abstain_rate"] == pytest.approx(1.0)
    assert "multi_environment" in karne.provenance["reports_supplied"]
