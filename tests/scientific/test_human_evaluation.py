# -*- coding: utf-8 -*-
"""P1 insan değerlendirme protokolü + Krippendorff α testleri."""
from __future__ import annotations

import csv
import json

import pytest

from hga.evaluation.capability_vector import SCORECARD_SECTIONS, build_scorecard
from hga.evaluation.human_eval_fill import (
    build_human_evaluation_from_csv,
    collect_ratings_from_csv,
    fill_and_export_packages,
    human_rating_import_markdown,
    human_rating_import_report,
)
from hga.evaluation.human_evaluation import (
    ALPHA_ACCEPTABLE,
    ALPHA_TENTATIVE,
    ARMS,
    DIMENSIONS,
    MAX_PROMPTS,
    MAX_RATERS,
    MIN_PROMPTS,
    MIN_RATERS,
    PROTOCOL,
    analyze_ratings,
    build_evaluation_sheets,
    build_human_evaluation_protocol,
    human_evaluation_markdown,
    krippendorff_alpha,
)

# Krippendorff'un kanonik örneği: 3 kodlayıcı × 15 birim, eksik veri dahil.
_A = [None, None, None, None, None, 3, 4, 1, 2, 1, 1, 3, 3, None, 3]
_B = [1, None, 2, 1, 3, 3, 4, 3, None, None, None, None, None, None, None]
_C = [None, None, 2, 1, 3, 4, 4, None, 2, 1, 1, 3, 3, None, 4]
CANONICAL = {i: [_A[i], _B[i], _C[i]] for i in range(15)}


# ── α doğruluğu: literatür referansına karşı ───────────────────────────────

@pytest.mark.parametrize("level,expected", [
    ("nominal", 0.691),
    ("ordinal", 0.807),
    ("interval", 0.811),
])
def test_alpha_kanonik_referans_degerlerini_veriyor(level, expected):
    """Araç 'yazıldı' değil, DOĞRULUĞU kanıtlanmış olmalı."""
    sonuc = krippendorff_alpha(CANONICAL, level=level)
    assert sonuc["alpha"] == pytest.approx(expected, abs=0.001)
    assert sonuc["level"] == level


def test_alpha_eksik_veriyi_dusuruyor():
    """Birim 1 ve 13 hiç değerlendirilmemiş; 14'ü tek kodlayıcı görmüş."""
    sonuc = krippendorff_alpha(CANONICAL, level="nominal")
    assert sonuc["units_used"] == 12
    assert sonuc["pairable_values"] == pytest.approx(26.0)


def test_alpha_tam_uyumda_bir():
    veri = {i: [3, 3, 3] for i in range(10)}
    veri[0] = [1, 1, 1]  # varyans olmalı yoksa tanımsız
    assert krippendorff_alpha(veri, "nominal")["alpha"] == pytest.approx(1.0)


def test_alpha_sans_duzeyinde_sifira_yakin():
    """Rastgele etiketleme α≈0 vermeli."""
    import random
    rastgele = random.Random(7)
    veri = {i: [rastgele.choice([0, 1]) for _ in range(5)]
            for i in range(400)}
    alfa = krippendorff_alpha(veri, "nominal")["alpha"]
    assert abs(alfa) < 0.12


def test_alpha_sistematik_uyusmazlikta_negatif():
    veri = {i: [0, 1] if i % 2 == 0 else [1, 0] for i in range(20)}
    assert krippendorff_alpha(veri, "nominal")["alpha"] < 0.0


def test_alpha_varyans_yoksa_tanimsiz():
    """Herkes aynı etikete basarsa α hesaplanamaz — 1.0 DEĞİL."""
    sonuc = krippendorff_alpha({i: [4, 4, 4] for i in range(10)}, "nominal")
    assert sonuc["alpha"] is None
    assert "TANIMSIZ" in sonuc["verdict"]


def test_alpha_yuzde_uyumdan_farkli():
    """Yüzde uyum %100 iken bile α bilgi üretmeyebilir.

    Bu, protokolde neden yüzde uyum kullanılmadığının kanıtıdır.
    """
    hepsi_ayni = {i: [1, 1] for i in range(50)}
    assert krippendorff_alpha(hepsi_ayni, "nominal")["alpha"] is None


def test_alpha_ordinal_komsu_uyusmazligi_odullendiriyor():
    """Ordinal metrik, yakın değerleri uzak değerlerden ayırmalı."""
    yakin = {i: [3, 4] for i in range(20)}
    yakin[0] = [1, 1]
    uzak = {i: [1, 5] for i in range(20)}
    uzak[0] = [1, 1]
    a_yakin = krippendorff_alpha(yakin, "ordinal")["alpha"]
    a_uzak = krippendorff_alpha(uzak, "ordinal")["alpha"]
    assert a_yakin > a_uzak


def test_alpha_hukumleri_esiklere_uyuyor():
    assert ALPHA_TENTATIVE < ALPHA_ACCEPTABLE
    yuksek = krippendorff_alpha(CANONICAL, "interval")
    assert "KABUL EDİLEBİLİR" in yuksek["verdict"]
    dusuk = krippendorff_alpha(CANONICAL, "nominal")
    assert "GEÇİCİ" in dusuk["verdict"]


def test_alpha_gecersiz_girdiler():
    with pytest.raises(ValueError):
        krippendorff_alpha(CANONICAL, level="yok-boyle")
    with pytest.raises(ValueError):
        krippendorff_alpha({0: [1], 1: [2]})  # eşleştirilebilir birim yok
    with pytest.raises(ValueError):
        krippendorff_alpha({})


# ── boyut analizi ──────────────────────────────────────────────────────────

def test_analyze_ratings_her_boyutu_kendi_olcegiyle_hesapliyor():
    veri = {
        "dogruluk": {i: [4, 5, 4] for i in range(12)} | {0: [1, 2, 1]},
        "halusinasyon_var": {i: [0, 0, 1] for i in range(12)} | {0: [1, 1, 1]},
    }
    sonuc = analyze_ratings(veri)
    assert sonuc["dogruluk"]["level"] == "ordinal"
    assert sonuc["halusinasyon_var"]["level"] == "nominal"
    assert sonuc["_summary"]["dimensions"] == 2


def test_analyze_ratings_tanimsiz_boyutu_reddediyor():
    with pytest.raises(ValueError):
        analyze_ratings({"uydurma_boyut": {0: [1, 2]}})


# ── protokol ve körleme ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rapor():
    return build_human_evaluation_protocol()


def test_protokol_kimligi(rapor):
    assert rapor.protocol == PROTOCOL
    assert rapor.schema_version == 1


def test_tasarim_sartnameye_uyuyor(rapor):
    d = rapor.design
    assert MIN_PROMPTS <= d["prompts"] <= MAX_PROMPTS
    assert MIN_RATERS <= d["raters"] <= MAX_RATERS
    assert d["arms"] == list(ARMS)
    assert d["total_judgements"] > 0
    assert rapor.checks["prompt_count_within_spec"] is True
    assert rapor.checks["rater_count_within_spec"] is True


def test_korleme_gercekten_uygulaniyor(rapor):
    """Kol adı değerlendirici paketinde GEÇMEMELİ."""
    assert rapor.blinding["arm_labels_hidden"] is True
    assert rapor.blinding["leaked_labels"] == []
    metin = json.dumps(rapor.sheets, ensure_ascii=False)
    for kol in ARMS:
        assert f'"{kol}"' not in metin
    assert rapor.checks["arm_labels_hidden_from_raters"] is True


def test_kor_acma_anahtari_ayri_tutuluyor(rapor):
    """Rapora yalnız özet girmeli, eşlemenin kendisi değil."""
    assert len(rapor.unblinding_key_digest) == 16
    metin = json.dumps(rapor.to_dict(), ensure_ascii=False)
    assert "unblinding_key_digest" in metin
    assert rapor.blinding["unblinding_key_held_separately"] is True


def test_paketler_ve_anahtar_tutarli():
    promptlar = [f"P{i}" for i in range(MIN_PROMPTS)]
    raters = [f"R{i}" for i in range(MIN_RATERS)]
    paketler, anahtar = build_evaluation_sheets(promptlar, raters)
    assert len(paketler) == MIN_RATERS
    # Her değerlendirici tüm öğeleri görür (within-subject).
    assert len(paketler[0].items) == MIN_PROMPTS * len(ARMS)
    assert len(anahtar) == MIN_PROMPTS * len(ARMS)
    for oge in paketler[0].items:
        assert oge["item_id"] in anahtar
        assert "arm" not in oge
    # Anahtar her kolu eşit sayıda kapsamalı.
    kollar = [v["arm"] for v in anahtar.values()]
    assert {kollar.count(a) for a in ARMS} == {MIN_PROMPTS}


def test_sunum_sirasi_degerlendiriciler_arasi_farkli():
    promptlar = [f"P{i}" for i in range(MIN_PROMPTS)]
    raters = [f"R{i}" for i in range(MIN_RATERS)]
    paketler, _ = build_evaluation_sheets(promptlar, raters)
    siralar = [tuple(o["item_id"] for o in p.items) for p in paketler]
    assert len(set(siralar)) == len(siralar), "sunum sırası dengelenmemiş"


def test_dikkat_kontrolleri_var(rapor):
    assert rapor.checks["attention_checks_present"] is True
    assert rapor.blinding["attention_checks_per_rater"] > 0


def test_determinizm():
    a = build_human_evaluation_protocol().to_dict()
    b = build_human_evaluation_protocol().to_dict()
    assert a["sheets"] == b["sheets"]
    assert a["unblinding_key_digest"] == b["unblinding_key_digest"]


def test_sartname_disi_girdiler_reddediliyor():
    raters = [f"R{i}" for i in range(MIN_RATERS)]
    with pytest.raises(ValueError):
        build_evaluation_sheets(["tek prompt"], raters)
    with pytest.raises(ValueError):
        build_evaluation_sheets([f"P{i}" for i in range(MAX_PROMPTS + 1)],
                                raters)
    with pytest.raises(ValueError):
        build_evaluation_sheets([f"P{i}" for i in range(MIN_PROMPTS)],
                                ["tek-degerlendirici"])
    with pytest.raises(ValueError):
        build_evaluation_sheets(
            [f"P{i}" for i in range(MIN_PROMPTS)],
            [f"R{i}" for i in range(MAX_RATERS + 1)])


def test_boyutlar_tanimli(rapor):
    assert len(DIMENSIONS) >= 3
    assert rapor.checks["multiple_dimensions_defined"] is True
    for ad, tanim in rapor.dimensions.items():
        assert tanim["question"]
        assert tanim["level"] in ("nominal", "ordinal", "interval")
        assert len(tanim["scale"]) >= 2


# ── en kritik dürüstlük kapıları ───────────────────────────────────────────

def test_gercek_puan_yokken_sonuc_uretilmiyor(rapor):
    assert rapor.results is None
    assert rapor.checks["human_ratings_collected"] is False
    assert rapor.checks["reliability_meets_threshold"] is False
    assert any("GERÇEK İNSAN PUANI YOK" in b for b in rapor.findings)


def test_sinirlar_degerlendirici_olmadigini_soyluyor(rapor):
    assert any("GERÇEK DEĞERLENDİRİCİ YOK" in s for s in rapor.limitations)
    assert any("DOĞRULUĞU değil" in s for s in rapor.limitations)


def test_karne_bolumu_na_kaliyor(rapor):
    """Araç hazır diye puan verilmemeli."""
    assert "human_evaluation" in SCORECARD_SECTIONS
    karne = build_scorecard(human_evaluation=rapor.to_dict())
    bolum = karne.sections["human_evaluation"]
    assert bolum["score"] is None, "gerçek puan yokken skor üretildi"
    assert bolum["inputs"]["protocol_ready"] is True
    assert bolum["inputs"]["ratings_collected"] is False
    assert "human_evaluation" in karne.provenance["reports_supplied"]


def test_gercek_puan_verilince_skor_uretiliyor():
    """Veri gelince bölüm skorlanabilmeli — kapı kalıcı olarak kapalı değil."""
    puanlar = {
        boyut: ({i: [4, 4, 5] for i in range(1, 15)} | {0: [1, 1, 2]}
                if DIMENSIONS[boyut]["level"] == "ordinal"
                else {i: [0, 0, 0] for i in range(1, 15)} | {0: [1, 1, 1]})
        for boyut in DIMENSIONS
    }
    rapor = build_human_evaluation_protocol(collected_ratings=puanlar)
    assert rapor.results is not None
    assert rapor.checks["human_ratings_collected"] is True
    karne = build_scorecard(human_evaluation=rapor.to_dict())
    assert karne.sections["human_evaluation"]["score"] is not None


def test_markdown_uretimi(rapor):
    md = human_evaluation_markdown(rapor)
    assert "# İnsan Değerlendirme Protokolü (P1)" in md
    assert "insan değerlendirme SONUCU içermez" in md
    assert "Krippendorff" in md
    assert "Körleme" in md
    assert "**Yok.**" in md  # sonuç bölümü boş olmalı
    for kol in ARMS:
        assert f'"{kol}"' not in md


# ── CSV import/agregasyon pipeline ─────────────────────────────────────────
def _filled_package(tmp_path):
    prompts = [f"Prompt {i}" for i in range(MIN_PROMPTS)]
    responses = {arm: [f"{arm} yanıt {i}" for i in range(MIN_PROMPTS)]
                 for arm in ARMS}
    root = tmp_path / "human_pkg"
    fill_and_export_packages(root, responses, prompts=prompts)
    (root / "rater_attestation.json").write_text(json.dumps({
        "real_human_ratings": True,
        "raters": [f"R{i:02d}" for i in range(1, MIN_RATERS + 1)],
        "collected_by": "unit-test",
        "collected_utc_date": "2026-09-16",
        "statement": "Synthetic unit-test attestation; production requires real raters.",
    }), encoding="utf-8")
    for csv_path in sorted((root / "paketler").glob("*_puanlama.csv")):
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fieldnames = list(rows[0])
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, row in enumerate(rows):
                row.update({
                    "dogruluk": "5" if index % 2 else "1",
                    "tutarlilik": "5" if index % 2 else "1",
                    "dil_kalitesi": "4" if index % 2 else "2",
                    "belirsizlik_durustlugu": "5" if index % 2 else "1",
                    "halusinasyon_var": "0" if index % 2 else "1",
                })
                writer.writerow(row)
    return root


def test_csv_import_alpha_ve_kol_ozeti_uretiyor(tmp_path):
    root = _filled_package(tmp_path)
    ratings, summary = collect_ratings_from_csv(root, strict_complete=True)
    assert summary["raters_found"] == MIN_RATERS
    assert summary["filled_ratio"] == 1.0
    assert set(ratings) == set(DIMENSIONS)

    import_report = human_rating_import_report(root)
    assert import_report["status"] == "COMPLETE_READY_FOR_REPORT"
    assert import_report["reliability"] is not None
    assert import_report["arm_results"] is not None
    assert [k for k, v in import_report["checks"].items() if not v] == []

    report = build_human_evaluation_from_csv(root)
    assert report.results is not None
    assert report.arm_results is not None
    assert report.checks["human_ratings_collected"] is True
    assert "rating_import_summary" in report.design


def test_csv_import_markdown_ve_olcek_dogrulama(tmp_path):
    root = _filled_package(tmp_path)
    md = human_rating_import_markdown(human_rating_import_report(root))
    assert "CSV Import" in md
    assert "Krippendorff" in md
    bad_csv = next((root / "paketler").glob("*_puanlama.csv"))
    text = bad_csv.read_text(encoding="utf-8")
    bad_csv.write_text(text.replace(",5,5,4,5,0", ",6,5,4,5,0", 1),
                       encoding="utf-8")
    with pytest.raises(ValueError, match="ölçek dışında"):
        collect_ratings_from_csv(root)


def test_csv_import_paket_yokken_na_raporu(tmp_path):
    report = human_rating_import_report(tmp_path / "yok")
    assert report["status"] == "NO_CSV_NA"
    assert report["checks"]["csv_files_found"] is False
    md = human_rating_import_markdown(report)
    assert "NO_CSV_NA" in md


def test_dolu_csv_attestation_yoksa_ana_sonuc_olmaz(tmp_path):
    root = _filled_package(tmp_path)
    (root / "rater_attestation.json").unlink()
    import_report = human_rating_import_report(root)
    assert import_report["status"] == "CSV_COMPLETE_UNATTESTED_NA"
    assert import_report["checks"]["rater_attestation_present"] is False
    report = build_human_evaluation_from_csv(root)
    assert report.results is None
    assert report.checks["human_ratings_collected"] is False
