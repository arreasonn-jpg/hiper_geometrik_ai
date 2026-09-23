# -*- coding: utf-8 -*-
"""Coklu UD dilleri icin generic reader.

EnglishEWT sinifini miras alir; parser ve aday uretimi
dil-bagimsiz (CoNLL-U standardi).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from .english_ewt import (
    EnglishEWT,
    EWTTaskData,
    SPLITS,
    _candidates,
    _hash_records,
)
from .experiment import canonical_hash, file_sha256

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"

# (lang_dir, sinif_adi, dil_adi)
LANGUAGES: Dict[str, Tuple[str, str]] = {
    "fr_gsd_v1": ("FrenchGSD", "Fransizca"),
    "es_gsd_v1": ("SpanishGSD", "Ispanyolca"),
    "it_gsd_v1": ("ItalianISDT", "Italyanca"),
    "nl_gsd_v1": ("DutchAlpino", "Hollandaca"),
    "zh_gsd_v1": ("ChineseGSD", "Cince"),
    "ja_gsd_v1": ("JapaneseGSD", "Japonca"),
    "ru_gsd_v1": ("RussianGSD", "Rusca"),
    "ar_padt_v1": ("ArabicPADT", "Arapca"),
}


class _UDReaderBase(EnglishEWT):
    """Generic UD reader."""

    _data_dir_name: str = ""

    def __init__(self, data_dir: Path | None = None):
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = DATASETS_DIR / self._data_dir_name
        self.provenance = json.loads(
            (self.data_dir / "PROVENANCE.json").read_text(encoding="utf-8")
        )
        if self.provenance.get("schema_version") != 1:
            raise ValueError(f"Unsupported schema for {self._data_dir_name}")
        self._verify()

    def _verify(self) -> None:
        for source in self.provenance["upstream"]["source_files"]:
            actual = file_sha256(self.data_dir / source["vendored_path"])
            if actual != source["sha256"]:
                raise ValueError(
                    f"SHA-256 mismatch {self._data_dir_name}: {source['vendored_path']}"
                )


def _prepare_task(dataset) -> EWTTaskData:
    sentences = {split: dataset.split_sentences(split) for split in SPLITS}
    candidates = {split: _candidates(split, sentences[split]) for split in SPLITS}
    source_hashes = dataset.source_hashes()
    config = dataset.provenance["benchmark_config"]
    candidate_hashes = {
        split: _hash_records(item.hash_record() for item in candidates[split])
        for split in SPLITS
    }
    dataset_hash = canonical_hash({"source_hashes": source_hashes, "config": config})
    return EWTTaskData(
        dataset_hash=dataset_hash,
        config_hash=canonical_hash(config),
        source_hashes=source_hashes,
        candidate_hashes=candidate_hashes,
        sentence_counts={split: len(sentences[split]) for split in SPLITS},
        train=candidates["train"], dev=candidates["dev"], test=candidates["test"],
    )


def prepare_language_task(lang_dir: str):
    """Verilen dil dizini icin EWTTaskData uret."""
    if lang_dir not in LANGUAGES:
        raise ValueError(f"Bilinmeyen dil: {lang_dir}")
    _, dil_adi = LANGUAGES[lang_dir]

    class _Reader(_UDReaderBase):
        _data_dir_name = lang_dir

    reader = _Reader()
    return _prepare_task(reader)


__all__ = ["LANGUAGES", "prepare_language_task"]
