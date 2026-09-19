# -*- coding: utf-8 -*-
"""CKPT-000 baseline'ın makinece doğrulanabilir sınırları."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "docs" / "checkpoints" / "CKPT-000.json"
TWT = ROOT / "hga" / "evaluation" / "datasets" / "twt_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ckpt_000_manifest_kapasite_ve_iddia_sinirlarini_sabitler():
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))

    assert checkpoint["checkpoint"] == "CKPT-000"
    assert checkpoint["release"] == "v0.1.0-prototype"
    assert checkpoint["metrics"] == {
        "dense_params": 6_970_433,
        "sparse_physical_params": 33_554_432,
        "total_trainable_params": 40_524_865,
        "statistical_reporting": "mean ± population standard deviation",
        "research_suite_default_seeds": [1, 2, 3, 4, 5],
        "license": "Apache-2.0",
    }
    assert checkpoint["capacity_contract"] == {
        "physical_trainable_parameters": "P",
        "interaction_upper_bound": "C_I^UB",
        "memory_address_upper_bound": "C_M^UB",
        "prohibition": "P, C_I^UB ve C_M^UB birbirinin yerine kullanılamaz; üst sınırlar parametre sayısı değildir.",
    }
    assert checkpoint["known_limitations"]


def test_ckpt_000_twt_revision_ve_kaynak_hashleri_vendored_veriyle_uyusur():
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    provenance = json.loads((TWT / "PROVENANCE.json").read_text(encoding="utf-8"))
    twt = checkpoint["twt_v1"]

    assert twt["upstream_revision"] == provenance["upstream"]["revision"]
    hashes = {entry["vendored_path"]: entry["sha256"]
              for entry in provenance["upstream"]["source_files"]}
    assert twt["web_conllu_sha256"] == hashes["web.conllu"] == _sha256(TWT / "web.conllu")
    assert twt["wiki_conllu_sha256"] == hashes["wiki.conllu"] == _sha256(TWT / "wiki.conllu")


def test_repo_apache_lisansi_ve_container_lock_sozlesmesini_tasir():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)

    assert pyproject["project"]["license"] == "Apache-2.0"
    assert pyproject["project"]["license-files"] == ["LICENSE"]
    assert "Apache License" in (ROOT / "LICENSE").read_text(encoding="utf-8")

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "requirements-lock.txt" in dockerfile
    assert "--no-build-isolation --no-deps ." in dockerfile
    assert 'ENTRYPOINT ["/opt/hga/docker-entrypoint.sh"]' in dockerfile
    assert "docker-entrypoint.sh" in dockerfile


def test_ckpt_000_ewt_revision_lisans_ve_kaynak_hashlerini_kaydeder():
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    ewt_dir = ROOT / "hga" / "evaluation" / "datasets" / "ewt_v1"
    provenance = json.loads((ewt_dir / "PROVENANCE.json").read_text(encoding="utf-8"))
    ewt = checkpoint["ewt_v1"]

    assert ewt["upstream_revision"] == provenance["upstream"]["revision"]
    assert ewt["license"] == provenance["license"]["spdx_id"] == "CC-BY-SA-4.0"
    hashes = {entry["vendored_path"]: entry["sha256"]
              for entry in provenance["upstream"]["source_files"]}
    assert all(_sha256(ewt_dir / path) == digest for path, digest in hashes.items())
    assert _sha256(ewt_dir / provenance["license"]["vendored_license_path"]) == (
        provenance["license"]["license_sha256"]
    )
