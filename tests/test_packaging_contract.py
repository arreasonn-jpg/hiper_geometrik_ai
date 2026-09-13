# -*- coding: utf-8 -*-
"""Paketleme sözleşmesi: desteklenen Python sürümü TEK ve tutarlı olmalı.

Bu dosya bir "iddia denetimi"dir. Proje bir Python sürümünü destekliyor
sayıyorsa o sürüm CI matrisinde test edilmeli; test edilmeyen bir sürüm
`requires-python` veya trove classifier üzerinden desteklenir gibi
gösterilmemelidir.

Denetlenen üç kaynak:

* ``pyproject.toml`` → ``requires-python`` tabanı
* ``pyproject.toml`` → ``Programming Language :: Python :: X.Y`` classifier'ları
* ``.github/workflows/ci.yml`` → ``matrix.python-version``

Ayrıca ruff ``target-version`` ve mypy ``python_version`` ayarlarının taban
sürümün gerisinde kalmadığı kontrol edilir.
"""
from __future__ import annotations

import os
import re
import sys
from typing import List, Tuple

import tomllib

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYPROJECT = os.path.join(KOK, "pyproject.toml")
CI_YML = os.path.join(KOK, ".github", "workflows", "ci.yml")


def _pyproject() -> dict:
    with open(PYPROJECT, "rb") as handle:
        return tomllib.load(handle)


def _surum(metin: str) -> Tuple[int, int]:
    major, minor = metin.strip().split(".")[:2]
    return int(major), int(minor)


def _requires_python_taban(deger: str) -> Tuple[int, int]:
    eslesme = re.search(r">=\s*(\d+\.\d+)", deger)
    assert eslesme, f"requires-python '>=X.Y' içermeli: {deger!r}"
    return _surum(eslesme.group(1))


def _classifier_surumleri(classifiers: List[str]) -> List[Tuple[int, int]]:
    bulunan = []
    for satir in classifiers:
        eslesme = re.fullmatch(r"Programming Language :: Python :: (\d+\.\d+)", satir.strip())
        if eslesme:
            bulunan.append(_surum(eslesme.group(1)))
    return sorted(bulunan)


def _ci_matris_surumleri() -> List[Tuple[int, int]]:
    with open(CI_YML, "r", encoding="utf-8") as handle:
        icerik = handle.read()
    eslesme = re.search(r"python-version:\s*\[([^\]]+)\]", icerik)
    assert eslesme, "CI workflow'unda matrix.python-version listesi bulunamadı"
    ham = [parca.strip().strip('"').strip("'") for parca in eslesme.group(1).split(",")]
    return sorted(_surum(parca) for parca in ham if parca)


def test_ci_matrisi_ile_classifierlar_birebir_ayni():
    """Desteklenir ilan edilen her sürüm CI'da test edilmeli (ve tersi)."""
    veri = _pyproject()
    classifiers = _classifier_surumleri(veri["project"]["classifiers"])
    ci = _ci_matris_surumleri()
    assert classifiers, "En az bir Python classifier'ı olmalı"
    assert classifiers == ci, (
        "Classifier'lar ile CI matrisi ayrışmış. "
        f"classifier={classifiers} ci={ci}. "
        "Bir sürüm ya test edilir ya da desteklenir ilan edilmez."
    )


def test_requires_python_tabani_ci_matrisinin_en_dusugu():
    veri = _pyproject()
    taban = _requires_python_taban(veri["project"]["requires-python"])
    ci = _ci_matris_surumleri()
    assert taban == ci[0], (
        f"requires-python tabanı {taban} ama CI'nın en düşük sürümü {ci[0]}. "
        "Taban sürüm CI tarafından doğrulanmıyor."
    )


def test_ruff_ve_mypy_hedefi_taban_surumden_geride_degil():
    veri = _pyproject()
    taban = _requires_python_taban(veri["project"]["requires-python"])

    ruff_hedef = veri["tool"]["ruff"]["target-version"]
    eslesme = re.fullmatch(r"py(\d)(\d+)", ruff_hedef)
    assert eslesme, f"Beklenmeyen ruff target-version: {ruff_hedef!r}"
    ruff_surum = (int(eslesme.group(1)), int(eslesme.group(2)))
    assert ruff_surum == taban, (
        f"ruff target-version={ruff_surum} ile requires-python tabanı {taban} "
        "aynı olmalı; aksi hâlde lint gerçek taban sürümü denetlemez."
    )

    mypy_surum = _surum(str(veri["tool"]["mypy"]["python_version"]))
    assert mypy_surum >= taban, (
        f"mypy python_version={mypy_surum}, taban {taban} sürümünün gerisinde."
    )


def test_calisan_yorumlayici_desteklenen_araliktadir():
    """Testi çalıştıran yorumlayıcı requires-python sözleşmesine uymalı."""
    veri = _pyproject()
    taban = _requires_python_taban(veri["project"]["requires-python"])
    assert sys.version_info[:2] >= taban, (
        f"Çalışan Python {sys.version_info[:2]}, desteklenen taban {taban}'den düşük."
    )
