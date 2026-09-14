# -*- coding: utf-8 -*-
"""P1 self-learning ölçeklendirme protokolünün testleri."""
from __future__ import annotations

import pytest

from hga.evaluation.capability_vector import SCORECARD_SECTIONS, build_scorecard
from hga.evaluation.self_learning_scaling import (
    PROFILES,
    PROTOCOL,
    SATURATION_THRESHOLD,
    run_self_learning_scaling,
    self_learning_scaling_markdown,
)

HIZLI = ((20, 15), (60, 15))


@pytest.fixture(scope="module")
def rapor():
    return run_self_learning_scaling(seeds=(1,), grid=HIZLI)


def test_protokol_kimligi(rapor):
    assert rapor.protocol == PROTOCOL
    assert rapor.schema_version == 1
    assert rapor.profile == "custom"
    assert rapor.seeds == [1]


def test_her_izgara_noktasi_kosuldu(rapor):
    assert len(rapor.points) == len(HIZLI)
    for nokta, (cycles, alan) in zip(rapor.points, HIZLI):
        assert nokta["cycles"] == cycles
        assert nokta["operands_max"] == alan
        # İstenen cycle sayısı gerçekten tamamlanmalı; kısa kesilmemeli.
        assert nokta["cycles_completed"] == cycles


def test_dogrulayici_yanlis_bilgi_biriktirmiyor(rapor):
    """CLOSED_VERIFIED rejiminin temel iddiası: sürüklenme yok."""
    for nokta in rapor.points:
        assert nokta["incorrect_knowledge"] == 0
        assert nokta["far"] == 0.0
        assert nokta["correct_knowledge"] == nokta["final_knowledge"]
    assert rapor.drift["total_incorrect_knowledge"] == 0
    assert rapor.checks["no_incorrect_knowledge_accumulated"] is True
    assert rapor.checks["long_run_stable"] is True


def test_dogrulayici_oncesi_far_sifirdan_buyuk(rapor):
    """Doğrulayıcı gerçekten iş yapıyor olmalı.

    Eğer ham üretim zaten hatasız olsaydı FAR=0 sonucu doğrulayıcıyı değil
    üreticiyi övürdü. Önce/sonra farkı kanıtın kendisidir.
    """
    assert rapor.drift["max_far_before_verifier"] > 0.0
    assert rapor.drift["max_far"] == 0.0


def test_egitim_test_izolasyonu_temiz(rapor):
    assert all(n["isolation_clean"] for n in rapor.points)
    assert rapor.checks["train_test_isolation_clean"] is True


def test_doygunluk_olculuyor(rapor):
    """Sabit alanda cycle artışı bilgiyi doğrusal artırmamalı."""
    eksen = rapor.cycle_axis["operands_max=15"]
    assert eksen["cycle_multiplier"] == pytest.approx(3.0)
    # Doygunluk: cycle 3× arttı ama bilgi neredeyse sabit kaldı.
    assert eksen["knowledge_multiplier"] < 1.10
    assert eksen["saturated"] is True
    assert eksen["scaling_efficiency"] < 1.0
    assert rapor.checks["saturation_measured"] is True


def test_doygunluk_cycle_i_makul_aralikta(rapor):
    for satir in rapor.cycle_axis["operands_max=15"]["rows"]:
        doygunluk = satir["saturation_cycle"]
        assert doygunluk is not None
        assert 1 <= doygunluk <= satir["cycles"]


def test_alan_ekseni_cycle_ekseninden_daha_verimli():
    """Asıl bulgu: ölçekleme cycle'dan değil alandan gelir."""
    rapor = run_self_learning_scaling(
        seeds=(1,), grid=((60, 15), (60, 31)))
    eksen = rapor.domain_axis["cycles=60"]
    assert eksen["domain_multiplier"] == pytest.approx(2.0, abs=0.1)
    # Alanı 2× büyütmek, cycle'ı 3× büyütmekten çok daha fazla kazandırır.
    assert eksen["knowledge_multiplier"] > 1.5
    assert rapor.checks["domain_axis_measured"] is True


def test_uzun_cycle_kapilari_durustce_kaliyor(rapor):
    """20/60 cycle ile 1000/3000 kapıları GEÇMİŞ görünmemeli."""
    assert rapor.checks["reached_1000_cycles"] is False
    assert rapor.checks["reached_3000_cycles"] is False


def test_determinizm():
    """Aynı tohum + aynı ızgara → bayt-eş sonuç."""
    a = run_self_learning_scaling(seeds=(7,), grid=((30, 15),)).to_dict()
    b = run_self_learning_scaling(seeds=(7,), grid=((30, 15),)).to_dict()
    for d in (a, b):
        for nokta in d["points"]:
            nokta.pop("wall_seconds")
    assert a["points"] == b["points"]
    assert a["cycle_axis"] == b["cycle_axis"]
    assert a["drift"]["signature"] == b["drift"]["signature"]


def test_farkli_tohum_farkli_kosu_ayni_kapilar():
    """Tohum değişince sonuç değişebilir ama doğruluk garantisi değişmez."""
    for tohum in (1, 2, 3):
        r = run_self_learning_scaling(seeds=(tohum,), grid=((30, 15),))
        assert r.drift["total_incorrect_knowledge"] == 0
        assert r.points[0]["far"] == 0.0


def test_cok_tohumlu_ortalama():
    r = run_self_learning_scaling(seeds=(1, 2), grid=((30, 15),))
    assert len(r.points) == 2
    assert {n["seed"] for n in r.points} == {1, 2}


def test_deneyim_verimi_olcekle_bozulmuyor(rapor):
    verimler = [n["experience_yield"] for n in rapor.points]
    assert all(0.0 < v < 1.0 for v in verimler)
    # Üretim dağılımı sabit olduğu için EY ölçekten bağımsız kalmalı.
    assert max(verimler) - min(verimler) < 0.05


def test_gecersiz_girdiler():
    with pytest.raises(ValueError):
        run_self_learning_scaling(profile="yok-boyle")
    with pytest.raises(ValueError):
        run_self_learning_scaling(seeds=())
    with pytest.raises(ValueError):
        run_self_learning_scaling(grid=())
    with pytest.raises(ValueError):
        run_self_learning_scaling(grid=((0, 15),))


def test_profiller_100_1000_3000_eksenini_kapsiyor():
    standart = [c for c, _ in PROFILES["standard"]]
    derin = [c for c, _ in PROFILES["deep"]]
    assert 100 in standart and 1000 in standart
    assert {100, 1000, 3000} <= set(derin)


def test_sinirlar_ve_bulgular_dolu(rapor):
    assert len(rapor.limitations) >= 3
    assert len(rapor.findings) >= 3
    assert any("DOYGUNLUK" in b for b in rapor.findings)
    assert 0.0 < SATURATION_THRESHOLD <= 1.0


def test_markdown_uretimi(rapor):
    md = self_learning_scaling_markdown(rapor)
    assert "# Self-Learning Ölçeklendirme (P1)" in md
    assert "Sürüklenme" in md
    assert "Kabul kapıları" in md
    assert "Sınırlar" in md
    assert rapor.drift["signature"] in md


def test_karneye_baglandi(rapor):
    assert "self_learning" in SCORECARD_SECTIONS
    karne = build_scorecard(self_learning_scaling=rapor.to_dict())
    bolum = karne.sections["self_learning"]
    assert bolum["score"] is not None
    assert 0.0 <= bolum["score"] <= 10.0
    assert bolum["inputs"]["total_incorrect_knowledge"] == 0
    assert bolum["inputs"]["saturation_detected"] is True
    assert "self_learning_scaling" in karne.provenance["reports_supplied"]


def test_karne_kanitsizken_puan_vermiyor():
    karne = build_scorecard()
    assert karne.sections["self_learning"]["score"] is None
