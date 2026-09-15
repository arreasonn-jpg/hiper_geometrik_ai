# -*- coding: utf-8 -*-
"""P0-6 (C_G v2) ve P3 (Capability Vector + otomatik karne) testleri."""
import json

import pytest

from hga.evaluation.capability_vector import (
    SCORECARD_SECTIONS,
    TERMINOLOGY,
    build_capability_vector,
    build_scorecard,
    save_scorecard,
    scorecard_markdown,
)
from hga.evaluation.compositional_v2 import (
    AXES,
    HARD_CASES,
    compositional_v2_markdown,
    run_compositional_v2_benchmark,
)


# ── C_G v2 ──────────────────────────────────────────────────────────────────
def test_sema_onceden_verilmez():
    rapor = run_compositional_v2_benchmark()
    assert rapor.schema_leakage["schema_provided_in_advance"] is False
    assert rapor.schema_leakage["ontology_provided_in_advance"] is False
    assert rapor.schema_leakage["clean"]


def test_kesif_egitim_korpusundan_turetilir():
    rapor = run_compositional_v2_benchmark()
    # Eğitimde geçen varlıklar keşfedilmiş olmalı; geçmeyenler olmamalı.
    assert "at" in rapor.discovered_train_entities
    assert "araba" in rapor.discovered_train_entities
    assert "bisiklet" not in rapor.discovered_train_entities
    assert set(rapor.discovered_train_relations) == {"binmek", "gitmek"}


def test_zor_alt_kume_varsayilan_olarak_dahil():
    """Kolay set 1.0 verirse sınır görünmez; zor set varsayılan olmalı."""
    rapor = run_compositional_v2_benchmark()
    assert rapor.checks["hard_subset_included"]
    cumleler = ({v["sentence"] for v in rapor.per_case}
                | {v["sentence"] for v in rapor.abstention_cases})
    for zor in HARD_CASES:
        assert zor.sentence in cumleler


def test_sozluk_disi_fiil_morfolojiden_induklenir():
    """Kanıt-tabanlı mastar indüksiyonu: sözlükte olmayan fiil, kanonik
    SOV yapısı varsa DÜŞÜK güvenle indüklenir. Eski davranış (koşulsuz
    çekimserlik) 'unseen_relation' eksenini yapısal olarak 0.5'e
    sabitliyordu; yeni sınır çatı ekli fiillerdir (ayrı test)."""
    rapor = run_compositional_v2_benchmark()
    vakalar = {v["sentence"]: v for v in rapor.per_case}
    v = vakalar["Ali kitabı inceledi."]
    assert v["composition_correct"], "incelemek indüklenmeliydi"
    assert rapor.axes["unseen_relation"]["composition_accuracy"] >= 1.0
    assert rapor.checks["unseen_relation_induction_works"]


def test_induksiyon_cati_ekinde_cekimser_kalir():
    """İndüksiyonun SINIRI: ettirgen/edilgen çatıda üye yapısı yüzey
    durumlardan çıkarılamaz; ilişki üretmek YANLIŞ bilgi olurdu. Bu kapı
    indüksiyonun 'her fiile mastar tak' dejenerasyonuna kaymadığını kilitler."""
    rapor = run_compositional_v2_benchmark()
    assert rapor.checks["induction_abstains_on_ambiguous_voice"]
    assert rapor.abstention_cases, "çekimserlik vakaları koşulmalı"
    for v in rapor.abstention_cases:
        assert v["abstained"], f"ihlal: {v['sentence']} → {v['predicted_relations']}"


def test_sozluk_disi_varlikta_tip_dusuk_kompozisyon_yuksek():
    """Tip bilinmese de kompozisyon kurulabilmeli — ikisi ayrı ölçülür."""
    rapor = run_compositional_v2_benchmark()
    eksen = rapor.axes["unseen_entity"]
    assert eksen["composition_accuracy"] >= 1.0
    assert eksen["type_accuracy"] < 1.0


def test_zor_set_haric_tutulabilir():
    kolay = run_compositional_v2_benchmark(include_hard=False)
    assert not kolay.checks["hard_subset_included"]
    assert kolay.overall_c_g_v2 >= run_compositional_v2_benchmark().overall_c_g_v2


def test_eksenler_ayri_raporlanir():
    rapor = run_compositional_v2_benchmark()
    assert set(rapor.axes) <= set(AXES)
    assert {"seen", "unseen_entity", "unseen_relation", "unseen_both",
            "unseen_wording"} <= set(rapor.axes)


def test_c_g_v2_kompozisyon_carpi_kesif():
    rapor = run_compositional_v2_benchmark()
    for m in rapor.axes.values():
        beklenen = round(m["composition_accuracy"]
                         * m["entity_discovery_accuracy"], 6)
        assert m["c_g_v2"] == pytest.approx(beklenen, abs=1e-6)


def test_bos_test_seti_acik_hata():
    with pytest.raises(ValueError):
        run_compositional_v2_benchmark(test_cases=[])


def test_markdown_sizinti_bolumunu_icerir():
    md = compositional_v2_markdown(run_compositional_v2_benchmark())
    assert "Şema sızıntısı denetimi" in md
    assert "önceden verilmedi" in md


# ── Capability Vector ───────────────────────────────────────────────────────
def test_terminoloji_ust_sinirlari_isaretler():
    assert TERMINOLOGY["C_I^UB"]["kind"] == "upper_bound"
    assert TERMINOLOGY["C_M^UB"]["kind"] == "upper_bound"
    assert TERMINOLOGY["C_R"]["kind"] == "measured"
    assert "PARAMETRE DEĞİLDİR" in TERMINOLOGY["C_I^UB"]["note"]


def test_kanit_yoksa_deger_none():
    vektor = build_capability_vector()
    for girdi in vektor:
        assert girdi.value is None
        assert girdi.evidence is None


def test_vektor_raporlardan_beslenir():
    v2 = run_compositional_v2_benchmark().to_dict()
    vektor = {e.symbol: e for e in build_capability_vector(compositional_v2=v2)}
    assert vektor["C_G"].value == pytest.approx(v2["overall_c_g_v2"])
    assert v2["dataset_hash"] in vektor["C_G"].evidence


# ── Otomatik karne ──────────────────────────────────────────────────────────
def test_kanitsiz_bolum_skor_uretmez():
    karne = build_scorecard()
    assert karne.overall is None
    assert karne.scored_sections == 0
    assert set(karne.unscored_sections) == set(SCORECARD_SECTIONS)
    assert karne.warnings


def test_skor_kapi_oranindan_hesaplanir():
    """Elle yazılan puan yok: skor = geçen kapı / toplam kapı × 10."""
    sahte = {"protocol": "x", "dataset_hash": "h",
             "checks": {"a": True, "b": True, "c": False, "d": False}}
    karne = build_scorecard(operator_baselines=sahte)
    assert karne.sections["architecture"]["score"] == pytest.approx(5.0)


def test_tum_kapilar_gecerse_on_uzerinden_on():
    sahte = {"protocol": "x", "dataset_hash": "h", "checks": {"a": True}}
    karne = build_scorecard(operator_baselines=sahte)
    assert karne.sections["architecture"]["score"] == pytest.approx(10.0)


def test_kanit_protokol_ve_imza_tasir():
    v2 = run_compositional_v2_benchmark().to_dict()
    karne = build_scorecard(compositional_v2=v2)
    kanit = karne.sections["generalization"]["evidence"][0]
    assert v2["protocol"] in kanit and v2["dataset_hash"] in kanit


def test_genel_skor_yalniz_kanitli_bolumlerden():
    sahte = {"protocol": "x", "dataset_hash": "h", "checks": {"a": True}}
    karne = build_scorecard(operator_baselines=sahte)
    assert karne.overall is not None
    assert karne.scored_sections < len(SCORECARD_SECTIONS)
    assert any("kanıtsız" in u for u in karne.warnings)


def test_markdown_elle_yazim_yasagini_belirtir():
    md = scorecard_markdown(build_scorecard())
    assert "otomatik" in md
    assert "n/a" in md
    assert "HGA Capability Vector" in md


def test_karne_diske_yazilir(tmp_path):
    karne = build_scorecard(compositional_v2=run_compositional_v2_benchmark().to_dict())
    yollar = save_scorecard(karne, json_path=str(tmp_path / "k.json"),
                            markdown_path=str(tmp_path / "k.md"))
    assert set(yollar) == {"json", "markdown"}
    veri = json.loads((tmp_path / "k.json").read_text(encoding="utf-8"))
    assert veri["provenance"]["reports_supplied"] == ["compositional_v2"]
    assert (tmp_path / "k.md").read_text(encoding="utf-8").startswith(
        "# HGA RESEARCH SCORECARD")
