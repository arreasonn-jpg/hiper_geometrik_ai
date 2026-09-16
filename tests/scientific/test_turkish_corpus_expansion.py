# -*- coding: utf-8 -*-
"""P1 Türkçe korpus genişletme pipeline testleri."""
from __future__ import annotations

import json

import pytest

from hga.evaluation.turkish_corpus_expansion import (
    ALLOWED_LICENSES,
    PROTOCOL,
    run_turkish_corpus_expansion_pipeline,
    turkish_corpus_expansion_markdown,
)


def test_smoke_pipeline_release_degil_ama_hat_calısıyor():
    rapor = run_turkish_corpus_expansion_pipeline(target_words=2_000_000)
    assert rapor.protocol == PROTOCOL
    assert rapor.mode == "embedded_smoke_candidates"
    assert rapor.candidate_summary["accepted_documents"] > 0
    assert rapor.release_gates["external_candidates_supplied"] is False
    assert rapor.release_gates["expanded_corpus_reaches_target_words"] is False
    assert rapor.checks["base_provenance_loaded"] is True
    assert rapor.checks["accepted_licenses_allowed"] is True
    assert all(doc["license"] in ALLOWED_LICENSES
               for doc in rapor.candidate_summary["accepted_preview"])


def test_dis_candidate_jsonl_kabul_ve_split(tmp_path):
    aday = tmp_path / "candidates"
    aday.mkdir()
    rows = [
        {
            "doc_id": "external_cc0_001",
            "source": "unit_test_source",
            "license": "CC0 1.0",
            "provenance_url": "https://example.invalid/unit",
            "text": "Ankara'da bugün hava serindi. Öğrenciler okulda deney yaptı.",
        },
        {
            "doc_id": "external_bad_license_001",
            "source": "unit_test_source",
            "license": "CC BY-NC-SA 4.0",
            "provenance_url": "https://example.invalid/unit",
            "text": "Bu metin Türkçe olsa bile lisans nedeniyle reddedilir.",
        },
    ]
    (aday / "part-000.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    rapor = run_turkish_corpus_expansion_pipeline(
        aday, use_smoke_candidates=False, target_words=1)
    assert rapor.mode == "external_candidate_dir"
    assert rapor.candidate_summary["accepted_documents"] == 1
    assert rapor.release_gates["external_candidates_supplied"] is True
    assert rapor.release_gates["provenance_urls_present_for_accepted"] is True
    assert rapor.rejection_summary["by_reason"]["license_not_allowed"] == 1
    accepted = rapor.candidate_summary["accepted_preview"][0]
    assert accepted["split"] in {"train", "dev", "test"}


def test_gecersiz_aday_hata(tmp_path):
    aday = tmp_path / "bad"
    aday.mkdir()
    (aday / "bad.jsonl").write_text('{"doc_id":"x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="eksik alanlar"):
        run_turkish_corpus_expansion_pipeline(aday, use_smoke_candidates=False)
    with pytest.raises(ValueError):
        run_turkish_corpus_expansion_pipeline(target_words=0)


def test_markdown_release_gate_ve_sinirlar():
    rapor = run_turkish_corpus_expansion_pipeline(target_words=2_000_000)
    md = turkish_corpus_expansion_markdown(rapor)
    assert "Korpus Genişletme" in md
    assert "Release gate" in md
    assert "yeni büyük korpus release'i değildir" in md
