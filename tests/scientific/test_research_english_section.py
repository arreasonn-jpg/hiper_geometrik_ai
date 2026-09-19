# -*- coding: utf-8 -*-
"""Research Suite must expose the pinned English baseline as a first-class section."""
from hga.evaluation.research import run_research_benchmark, validate_sections


def test_english_section_manifestli_seed_kosularina_girer(tmp_path):
    report = run_research_benchmark(
        root=tmp_path / "experiments", seeds=(1, 2), profile="smoke",
        sections=("english-nlp",),
    )

    assert validate_sections(("english-nlp",)) == ["english_nlp"]
    section = report.sections["english_nlp"]
    assert report.outcome == "COMPLETED"
    assert section["status"] == "COMPLETED"
    assert section["checks_all_seeds"]["official_english_ud_sources_hash_verified"]
    assert section["checks_all_seeds"]["contributes_one_seed_to_suite_aggregate"]
    reference = section["metrics_reference_seed"]["english_ewt"]
    assert reference["dataset_hash"] == "2f852e91da70301e5a572f717b3e9d6998d14ca6bf38daa8bb77f4baa5b3ddd3"
    assert set(reference["aggregate"]) == {"dense", "transformer", "bert_style", "gpt_style"}
