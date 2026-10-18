# -*- coding: utf-8 -*-
"""German GSD — English EWT'nin Almanca versiyonu.

Parser ve aday uretimi dil-bagimsiz (CoNLL-U standardi).
Sadece DATA_DIR ve PROVENANCE farkli.
"""
from __future__ import annotations

import json
from pathlib import Path

from .english_ewt import (
    SPLITS,
    EnglishEWT,
    EWTTaskData,
    _candidates,
    _hash_records,
)
from .experiment import canonical_hash, file_sha256

DATA_DIR_DE = Path(__file__).resolve().parent / "datasets" / "de_gsd_v1"


class GermanGSD(EnglishEWT):
    """Almanca GSD okuyucu (hash dogrulamali)."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR_DE
        self.provenance = json.loads(
            (self.data_dir / "PROVENANCE.json").read_text(encoding="utf-8")
        )
        if self.provenance.get("schema_version") != 1:
            raise ValueError("Unsupported German provenance schema")
        self._verify()

    def _verify(self) -> None:
        for source in self.provenance["upstream"]["source_files"]:
            actual = file_sha256(self.data_dir / source["vendored_path"])
            if actual != source["sha256"]:
                raise ValueError(
                    f"German source SHA-256 mismatch: {source['vendored_path']} "
                    f"expected={source['sha256']} actual={actual}"
                )


def prepare_german_gsd_task(data_dir: Path | None = None) -> EWTTaskData:
    """Almanca GSD adaylarini hazirla (EWT ile ayni islem)."""
    dataset = GermanGSD(data_dir)
    sentences = {split: dataset.split_sentences(split) for split in SPLITS}
    candidates = {split: _candidates(split, sentences[split]) for split in SPLITS}
    source_hashes = dataset.source_hashes()
    config = dataset.provenance["benchmark_config"]
    candidate_hashes = {
        split: _hash_records(item.hash_record() for item in candidates[split])
        for split in SPLITS
    }
    dataset_hash = canonical_hash(
        {"source_hashes": source_hashes, "config": config}
    )
    return EWTTaskData(
        dataset_hash=dataset_hash,
        config_hash=canonical_hash(config),
        source_hashes=source_hashes,
        candidate_hashes=candidate_hashes,
        sentence_counts={split: len(sentences[split]) for split in SPLITS},
        train=candidates["train"],
        dev=candidates["dev"],
        test=candidates["test"],
    )


__all__ = ["GermanGSD", "prepare_german_gsd_task"]
