# -*- coding: utf-8 -*-
"""Google Forms CSV → HGA değerlendirme CSV köprüsünün sözleşme testleri."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "insan_degerlendirme_paketleri"
    / "google_forms"
    / "forms_yaniti_donustur.py"
)
SPEC = importlib.util.spec_from_file_location("hga_google_forms_converter", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = converter
SPEC.loader.exec_module(converter)

ITEMS = ("0123456789abcdef", "fedcba9876543210")


def _make_package(root: Path) -> Path:
    packages = root / "paketler"
    packages.mkdir(parents=True)
    payload = {
        "rater_id": "R01",
        "items": [
            {"item_id": item_id, "dimensions": list(converter.DIMENSIONS)}
            for item_id in ITEMS
        ],
    }
    (packages / "R01.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def _headers() -> list[str]:
    return [
        "Timestamp",
        converter.RATER_HEADER,
        *[
            f"HGA|{item_id}|{dimension}"
            for item_id in ITEMS
            for dimension in converter.DIMENSIONS
        ],
    ]


def _write_response(path: Path, *, rater: str = "R01", dogruluk: str = "5 — tamamen doğru") -> None:
    headers = _headers()
    row = {header: "" for header in headers}
    row["Timestamp"] = "2026-09-15T12:00:00+00:00"
    row[converter.RATER_HEADER] = rater
    for item_id in ITEMS:
        for dimension in converter.DIMENSIONS:
            row[f"HGA|{item_id}|{dimension}"] = (
                "0 — uydurma bilgi yok"
                if dimension == "halusinasyon_var"
                else dogruluk
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row)


def test_google_forms_csv_is_converted_to_existing_hga_schema(tmp_path: Path) -> None:
    package_root = _make_package(tmp_path / "packages")
    response = tmp_path / "answers.csv"
    _write_response(response)

    summary = converter.convert(
        responses=response,
        packages_dir=package_root,
        output_dir=tmp_path / "out",
    )

    assert summary.raters["R01"]["complete"] is True
    with (tmp_path / "out" / "R01_puanlama.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["item_id"] for row in rows] == list(ITEMS)
    assert rows[0]["dogruluk"] == "5"
    assert rows[0]["halusinasyon_var"] == "0"


def test_grid_csv_uses_filename_rater_and_is_converted(tmp_path: Path) -> None:
    package_root = _make_package(tmp_path / "packages")
    response = tmp_path / "HGA Değerlendirme — R01.csv"
    headers = [
        "Zaman damgası",
        *[
            (
                f"{item_id} | halusinasyon_var"
                if dimension == "halusinasyon_var"
                else f"{item_id} | puanlar [{dimension}]"
            )
            for item_id in ITEMS
            for dimension in converter.DIMENSIONS
        ],
    ]
    row = {header: "5" for header in headers}
    row["Zaman damgası"] = "2026/09/16 12:00:00 ÖS GMT+3"
    for item_id in ITEMS:
        row[f"{item_id} | halusinasyon_var"] = "1 — uydurma var"
    with response.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row)

    summary = converter.convert(response, package_root, tmp_path / "out")

    assert summary.raters["R01"]["complete"] is True
    with (tmp_path / "out" / "R01_puanlama.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["dogruluk"] == "5"
    assert rows[0]["halusinasyon_var"] == "1"


def test_grid_csv_without_rater_in_filename_is_rejected(tmp_path: Path) -> None:
    package_root = _make_package(tmp_path / "packages")
    response = tmp_path / "answers.csv"
    response.write_text(
        f'"{ITEMS[0]} | puanlar [dogruluk]"\n"5"\n', encoding="utf-8"
    )
    with pytest.raises(converter.ConversionError, match="dosya adından Rxx"):
        converter.convert(response, package_root, tmp_path / "out")


def test_invalid_scale_is_rejected_before_a_rating_file_is_written(tmp_path: Path) -> None:
    package_root = _make_package(tmp_path / "packages")
    response = tmp_path / "answers.csv"
    _write_response(response, dogruluk="0 — geçersiz ordinal puan")

    with pytest.raises(converter.ConversionError, match="1–5"):
        converter.convert(response, package_root, tmp_path / "out")


def test_duplicate_submission_requires_explicit_policy(tmp_path: Path) -> None:
    package_root = _make_package(tmp_path / "packages")
    response = tmp_path / "answers.csv"
    _write_response(response)
    with response.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_headers())
        row = {header: "" for header in _headers()}
        row["Timestamp"] = "2026-09-15T13:00:00+00:00"
        row[converter.RATER_HEADER] = "R01"
        for item_id in ITEMS:
            for dimension in converter.DIMENSIONS:
                row[f"HGA|{item_id}|{dimension}"] = (
                    "1 — uydurma bilgi var"
                    if dimension == "halusinasyon_var"
                    else "1 — tamamen yanlış"
                )
        writer.writerow(row)

    with pytest.raises(converter.ConversionError, match="Yinelenen puan"):
        converter.convert(response, package_root, tmp_path / "out")

    converter.convert(
        response, package_root, tmp_path / "out", duplicate_policy="latest"
    )
    with (tmp_path / "out" / "R01_puanlama.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["dogruluk"] == "1"
    assert rows[0]["halusinasyon_var"] == "1"
