# -*- coding: utf-8 -*-
"""P1 mühendislik sözleşmesi + yeniden üretilebilirlik denetimi testleri."""
from __future__ import annotations

import pytest

from hga.evaluation.capability_vector import build_scorecard
from hga.evaluation.engineering_audit import (
    CORE_SEED_REQUIREMENT,
    PROTOCOL_ENGINEERING,
    PROTOCOL_REPRODUCIBILITY,
    audit_engineering,
    audit_reproducibility,
    engineering_markdown,
    reproducibility_markdown,
)


@pytest.fixture(scope="module")
def muhendislik():
    return audit_engineering()


@pytest.fixture(scope="module")
def yeniden_uretim():
    return audit_reproducibility({})


# ── mühendislik ────────────────────────────────────────────────────────────

def test_protokol_kimligi(muhendislik):
    assert muhendislik.protocol == PROTOCOL_ENGINEERING
    assert muhendislik.schema_version == 1


def test_ci_matrisi_coklu_python(muhendislik):
    ci = muhendislik.ci
    assert ci["exists"] is True
    assert ci["matrix_size"] >= 3
    assert "3.10" in ci["python_matrix"]


def test_ci_temel_kapilari_var(muhendislik):
    for kapi in ("lint", "type_check", "tests", "package_build"):
        assert muhendislik.ci["gates_present"][kapi] is True, kapi


def test_yeni_p1_komutlari_ci_smoke_de(muhendislik):
    """Yeni komut eklenip CI'a konmazsa denetim bunu yakalamalı."""
    komutlar = set(muhendislik.ci["cli_commands_smoke_tested"])
    for komut in ("cok-ortam", "muhendislik", "yeniden-uretilebilirlik",
                  "championship-benchmark"):
        assert komut in komutlar, f"{komut} CI smoke'ta yok"


def test_paketleme_sozlesmesi(muhendislik):
    paket = muhendislik.packaging
    assert paket["pyproject_parsed"] is True
    assert paket["missing_fields"] == []
    assert paket["requires_python"].startswith(">=3.")


def test_test_paketi_bilimsel_dosyalar(muhendislik):
    testler = muhendislik.test_suite
    assert testler["pytest_configured"] is True
    assert testler["scientific_test_files"] >= 15
    assert testler["test_files"] >= testler["scientific_test_files"]


def test_muhendislik_kapilari(muhendislik):
    kaldi = [ad for ad, v in muhendislik.checks.items() if not v]
    assert kaldi == [], f"başarısız kapılar: {kaldi}"


def test_muhendislik_determinizm():
    a, b = audit_engineering(), audit_engineering()
    assert a.checks == b.checks
    assert a.ci == b.ci
    assert a.packaging == b.packaging


def test_muhendislik_markdown(muhendislik):
    md = engineering_markdown(muhendislik)
    assert "CI" in md
    assert "Kabul kapıları" in md or "kapı" in md.lower()


# ── yeniden üretilebilirlik ────────────────────────────────────────────────

def test_yeniden_uretim_protokolu(yeniden_uretim):
    assert yeniden_uretim.protocol == PROTOCOL_REPRODUCIBILITY
    assert CORE_SEED_REQUIREMENT == 20


def test_manifest_tam_kayit_tutuyor(yeniden_uretim):
    m = yeniden_uretim.manifest
    assert m["can_create_run"] is True
    for alan in ("has_config_hash", "has_dataset_hash", "has_seed",
                 "has_git_metadata", "has_runtime_metadata"):
        assert m[alan] is True, alan
    # Ortam alanları düz kayıtlıdır (iç içe 'runtime' anahtarı YOK).
    assert "python_version" in m["recorded_environment_fields"]
    assert "platform" in m["recorded_environment_fields"]


def test_determinizm_gercekten_olculuyor(yeniden_uretim):
    """İddia edilmiyor, iki kez koşulup bayt karşılaştırılıyor."""
    det = yeniden_uretim.determinism
    assert det, "determinizm hiç ölçülmemiş"
    for protokol, veri in det.items():
        assert veri["byte_identical"] is True, protokol
        assert veri["digest"]


def test_tohum_disiplini_20_seed_esigi(yeniden_uretim):
    tohum = yeniden_uretim.seed_discipline
    assert tohum["core_requirement"] == CORE_SEED_REQUIREMENT


def test_rapor_verilince_tohum_disiplini_degerlendiriliyor():
    """Boş girdiyle 0 protokol; rapor verilince gerçekten sayılmalı."""
    sahte = {
        "yeterli": {"protocol": "p_yeterli", "seeds": list(range(1, 21))},
        "yetersiz": {"protocol": "p_yetersiz", "seeds": [1, 2, 3]},
    }
    rapor = audit_reproducibility(sahte)
    tohum = rapor.seed_discipline
    assert tohum["protocols_examined"] == 2
    assert tohum["protocols_meeting_core_requirement"] == 1
    # 3 tohumlu protokol "yeterli" sayılmamalı.
    assert tohum["per_protocol"]["yetersiz"]["meets_core_requirement"] is False
    assert tohum["per_protocol"]["yetersiz"]["tier"] == "engineering"
    assert tohum["per_protocol"]["yeterli"]["meets_core_requirement"] is True
    assert tohum["per_protocol"]["yeterli"]["tier"] == "core"


def test_bos_girdide_tohum_kapilari_durustce_kaliyor(yeniden_uretim):
    """Hiç rapor verilmeden 20-tohum kapısı GEÇMİŞ görünmemeli."""
    assert yeniden_uretim.checks["at_least_one_protocol_meets_20_seeds"] is False
    assert yeniden_uretim.checks["all_protocols_carry_data_signature"] is False
    assert yeniden_uretim.seed_discipline["protocols_examined"] == 0
    # Rapordan bağımsız kapılar (manifest/determinizm) yine de geçmeli.
    assert yeniden_uretim.checks["deterministic_protocols_are_byte_identical"] is True


def test_gercek_raporlarla_tum_kapilar_geciyor():
    """20 tohumlu gerçek protokoller verilince tüm kapılar geçmeli."""
    from hga.evaluation.multi_environment import (
        run_multi_environment_benchmark,
    )
    rapor = audit_reproducibility({
        "multi_environment": run_multi_environment_benchmark(
            seeds=tuple(range(1, 21)), claims_per_environment=10).to_dict(),
    })
    kaldi = [ad for ad, v in rapor.checks.items() if not v]
    assert kaldi == [], f"başarısız kapılar: {kaldi}"


def test_lockfile_eksigi_durustce_raporlaniyor(yeniden_uretim):
    """Bilinen eksik gizlenmemeli."""
    metin = " ".join(yeniden_uretim.findings + yeniden_uretim.limitations)
    assert "lock" in metin.lower()


def test_yeniden_uretim_markdown(yeniden_uretim):
    md = reproducibility_markdown(yeniden_uretim)
    assert "determinizm" in md.lower() or "Determinizm" in md
    assert "manifest" in md.lower()


# ── karne bağlantısı ───────────────────────────────────────────────────────

def test_karneye_baglandi(muhendislik, yeniden_uretim):
    karne = build_scorecard(engineering=muhendislik.to_dict(),
                            reproducibility=yeniden_uretim.to_dict())
    muh = karne.sections["engineering"]
    yen = karne.sections["reproducibility"]
    assert muh["score"] == pytest.approx(10.0)
    assert muh["inputs"]["python_matrix"]
    assert yen["score"] is not None
    assert 0.0 <= yen["score"] <= 10.0
    assert {"engineering", "reproducibility"} <= set(
        karne.provenance["reports_supplied"])


def test_karne_kanitsizken_puan_vermiyor():
    karne = build_scorecard()
    assert karne.sections["engineering"]["score"] is None
    assert karne.sections["reproducibility"]["score"] is None
