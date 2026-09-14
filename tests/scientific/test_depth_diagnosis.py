# -*- coding: utf-8 -*-
"""Çıkarım derinliği çöküşünün kök neden teşhisi testleri."""
from __future__ import annotations

import pytest

from hga.evaluation.capability_vector import build_scorecard
from hga.evaluation.depth_diagnosis import (
    PROFILES,
    PROTOCOL,
    depth_diagnosis_markdown,
    diagnose_depth_collapse,
)


@pytest.fixture(scope="module")
def rapor():
    """Bellek sınırının görünür olduğu gerçek ızgara (hızlı alt küme)."""
    return diagnose_depth_collapse(
        slot_sizes=(2 ** 18, 2 ** 19, 2 ** 20),
        hops=(8, 16, 32), distractors=16384, seeds=(1, 2))


def test_protokol_kimligi(rapor):
    assert rapor.protocol == PROTOCOL
    assert rapor.schema_version == 1
    assert rapor.distractors == 16384


def test_tek_degiskenli_tasarim(rapor):
    """Slot dışında hiçbir şey değişmemeli."""
    assert rapor.checks["single_variable_design"] is True
    assert len(rapor.slot_sizes) >= 2
    assert rapor.slot_sizes == sorted(rapor.slot_sizes)
    # Aynı hop ızgarası her slot için ölçülmüş olmalı.
    for slot in rapor.slot_sizes:
        assert set(rapor.recall_grid[str(slot)]) == {
            str(h) for h in rapor.hops}


def test_kontrol_kolu_dolgusuz_temiz(rapor):
    """Dolgu=0'da zincir bozulmuyorsa kayıp dolgunun belleğe baskısıdır."""
    assert rapor.checks["control_arm_without_distractors_clean"] is True
    for satir in rapor.zero_distractor_control["grid"].values():
        for deger in satir.values():
            assert deger == pytest.approx(1.0)


def test_derinlik_slotla_monoton_artiyor(rapor):
    assert rapor.checks["depth_monotone_in_slots"] is True
    derinlikler = [rapor.supported_depth[str(s)] for s in rapor.slot_sizes]
    assert all(a <= b for a, b in zip(derinlikler, derinlikler[1:]))
    # İlk ve son arasında gerçek bir artış olmalı, yoksa teşhis boş olurdu.
    assert derinlikler[-1] > derinlikler[0]


def test_slot_ikiye_katlaninca_derinlik_ikiye_katlaniyor(rapor):
    """Kritik nicel bulgu: log-log eğim ≈ 1."""
    for cift in rapor.scaling["pairs"]:
        assert cift["depth_ratio"] == pytest.approx(
            cift["slot_ratio"], rel=0.35), cift
    assert rapor.scaling["log_log_slope"] == pytest.approx(1.0, abs=0.2)
    assert rapor.checks["scaling_measured"] is True


def test_kok_neden_bellek_kapasitesi(rapor):
    """Çöküş çıkarım sınırı DEĞİL."""
    assert rapor.diagnosis["root_cause"] == "memory_capacity"
    assert rapor.diagnosis["memory_bound"] is True
    assert rapor.checks["root_cause_identified"] is True
    assert rapor.checks["collapse_is_engineering_not_reasoning"] is True
    assert "ÇIKARIM sınırı değil" in rapor.diagnosis["explanation"]


def test_eski_yorumun_duzeltildigi_kayitli(rapor):
    """P0-7'nin yanlış yorumu açıkça düzeltilmeli, sessizce değiştirilmemeli."""
    assert "P0-7" in rapor.diagnosis["previous_interpretation"]
    assert any("düzeltir" in b.lower() for b in rapor.findings)


def test_yetersiz_slot_ile_derinlik_dusuk(rapor):
    """En küçük slotta derinlik gerçekten kısıtlı olmalı (bulgu gerçek)."""
    en_kucuk = rapor.slot_sizes[0]
    assert rapor.supported_depth[str(en_kucuk)] <= 8


def test_smoke_profili_farkli_teshis_verebilir():
    """Küçük ızgarada bellek sınırı görünmez; teşhis bunu uydurmamalı."""
    r = diagnose_depth_collapse(profile="smoke")
    assert r.diagnosis["root_cause"] in (
        "memory_capacity", "reasoning_limit", "chain_construction",
        "inconclusive")
    # Bellek sınırı görünmüyorsa 'mühendislik' kapısı GEÇMİŞ görünmemeli.
    if r.diagnosis["root_cause"] != "memory_capacity":
        assert r.checks["collapse_is_engineering_not_reasoning"] is False


def test_determinizm():
    a = diagnose_depth_collapse(
        slot_sizes=(2 ** 18, 2 ** 19), hops=(8, 16), seeds=(1,))
    b = diagnose_depth_collapse(
        slot_sizes=(2 ** 18, 2 ** 19), hops=(8, 16), seeds=(1,))
    assert a.recall_grid == b.recall_grid
    assert a.supported_depth == b.supported_depth
    assert a.diagnosis["signature"] == b.diagnosis["signature"]


def test_gecersiz_girdiler():
    with pytest.raises(ValueError):
        diagnose_depth_collapse(profile="yok-boyle")
    with pytest.raises(ValueError):
        # tek slot ile kök neden ayırt edilemez
        diagnose_depth_collapse(slot_sizes=(2 ** 18,))
    with pytest.raises(ValueError):
        diagnose_depth_collapse(slot_sizes=(2 ** 18, 2 ** 19), hops=())
    with pytest.raises(ValueError):
        diagnose_depth_collapse(slot_sizes=(2 ** 18, 2 ** 19), distractors=0)
    with pytest.raises(ValueError):
        diagnose_depth_collapse(slot_sizes=(2 ** 18, -1))


def test_profiller_tanimli():
    for ad, ayar in PROFILES.items():
        assert len(ayar["slot_sizes"]) >= 2, ad
        assert ayar["hops"] and ayar["seeds"]
        assert ayar["distractors"] >= 1


def test_sinirlar_slot_maliyetini_itiraf_ediyor(rapor):
    metin = " ".join(rapor.limitations)
    assert "RAM" in metin
    # Gerçek çıkarım tavanının hâlâ bilinmediği söylenmeli.
    assert "ULAŞILMADI" in metin or "ölçülmemiştir" in metin


def test_markdown_uretimi(rapor):
    md = depth_diagnosis_markdown(rapor)
    assert "Kök Neden Teşhisi" in md
    assert "Tek değişkenli tasarım" in md
    assert "Kontrol kolu" in md
    assert "memory_capacity" in md
    assert rapor.diagnosis["signature"] in md


def test_karne_reasoning_bolumune_baglandi(rapor):
    from hga.evaluation.reasoning_depth import measure_reasoning_depth
    derinlik = measure_reasoning_depth(profile="smoke").to_dict()
    karne = build_scorecard(reasoning_depth=derinlik,
                            depth_diagnosis=rapor.to_dict())
    bolum = karne.sections["reasoning"]
    assert bolum["score"] is not None
    assert bolum["inputs"]["collapse_root_cause"] == "memory_capacity"
    assert bolum["inputs"]["depth_per_slot_log_log_slope"] is not None
    # Teşhis kapıları reasoning bölümüne karışmış olmalı.
    assert "depth_diagnosis" in karne.provenance["reports_supplied"]


def test_teshis_reasoning_skorunu_yukseltiyor(rapor):
    """Kök nedeni bulmak ölçülebilir bir iyileşme olmalı."""
    from hga.evaluation.reasoning_depth import measure_reasoning_depth
    derinlik = measure_reasoning_depth(profile="smoke").to_dict()
    once = build_scorecard(reasoning_depth=derinlik).sections["reasoning"]
    sonra = build_scorecard(
        reasoning_depth=derinlik,
        depth_diagnosis=rapor.to_dict()).sections["reasoning"]
    assert sonra["score"] > once["score"]
