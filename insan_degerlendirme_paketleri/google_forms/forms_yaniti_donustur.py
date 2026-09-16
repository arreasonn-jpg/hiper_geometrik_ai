#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Google Forms CSV yanıtlarını HGA'nın kör değerlendirme CSV şemasına dönüştür.

Google Forms'a aktarılan formlar iki desteklenen başlık şemasından birini yazar::

    HGA|<16-hex-item-id>|<boyut>
    <16-hex-item-id> | puanlar [<boyut>]
    <16-hex-item-id> | halusinasyon_var

İkinci şema, dört ölçek satırını tek ``MultipleChoiceGrid`` sorusunda tutan
ızgara sürümünün Google Sheets dışa aktarımıdır. Izgara CSV'sinde ayrı rater
alanı bulunmadığında araç Rxx kodunu dosya adından güvenli biçimde çıkarır.

Bu araç, form bölümlerinden indirilen CSV dosyalarını item_id üzerinden
birleştirir, değer/rater/paket bütünlüğünü doğrular ve hga'nın mevcut
``collect_ratings_from_csv`` aracının okuyacağı ``Rxx_puanlama.csv`` dosyalarını
üretir. Kör açma anahtarına ihtiyaç duymaz ve onu okumaz.

Örnek::

    python insan_degerlendirme_paketleri/google_forms/forms_yaniti_donustur.py \
      --responses ~/Downloads/hga_forms_csv \
      --packages insan_degerlendirme_paketleri \
      --output /tmp/hga_puanlar

Varsayılan olarak eksik veya çakışan puanlar hata verir. Ara teslimleri yalnız
``--allow-partial`` ile üretin; bu çıktı insan değerlendirmesi sonucu değildir.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

DIMENSIONS: Tuple[str, ...] = (
    "dogruluk",
    "tutarlilik",
    "dil_kalitesi",
    "belirsizlik_durustlugu",
    "halusinasyon_var",
)
RATER_HEADER = "HGA|meta|rater_id"
QUESTION_RE = re.compile(
    r"^\s*HGA\|(?P<item>[0-9a-f]{16})\|(?P<dimension>"
    + "|".join(DIMENSIONS)
    + r")\s*$",
    re.IGNORECASE,
)
GRID_QUESTION_RE = re.compile(
    r"^\s*(?P<item>[0-9a-f]{16})\s*\|\s*(?:"
    r"puanlar\s*\[(?P<grid_dimension>dogruluk|tutarlilik|dil_kalitesi|"
    r"belirsizlik_durustlugu)\]|(?P<hallucination>halusinasyon_var))\s*$",
    re.IGNORECASE,
)
RATER_ID_RE = re.compile(r"^R\d{2}$")
RATER_FILENAME_RE = re.compile(r"(?:^|[^A-Z0-9])(R\d{2})(?:[^A-Z0-9]|$)", re.IGNORECASE)
LEADING_SCORE_RE = re.compile(r"^\s*([01-5])(?:\s|—|-|$)")


class ConversionError(ValueError):
    """İndirilen Forms yanıtı HGA şemasıyla uyumsuz olduğunda yükselir."""


@dataclass(frozen=True)
class Package:
    """Kör paketten alınan, kol etiketi içermeyen beklenen değerlendirme kümesi."""

    rater_id: str
    items: Tuple[str, ...]
    path: str


@dataclass(frozen=True)
class Rating:
    rater_id: str
    item_id: str
    dimension: str
    value: int
    source: str
    row_number: int
    timestamp: Optional[str]


@dataclass
class ImportSummary:
    schema_version: int = 1
    protocol: str = "hga_google_forms_import_v1"
    response_files: List[str] = field(default_factory=list)
    rows_seen: int = 0
    response_rows_used: int = 0
    raters: Dict[str, Dict[str, object]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _read_package(path: Path) -> Package:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConversionError(f"Paket okunamadı ({path}): {error}") from error
    rater_id = str(data.get("rater_id", "")).strip()
    if not RATER_ID_RE.fullmatch(rater_id):
        raise ConversionError(f"{path}: rater_id R01 biçiminde olmalı")
    raw_items = data.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ConversionError(f"{path}: boş veya geçersiz items dizisi")
    item_ids: List[str] = []
    seen: Set[str] = set()
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, Mapping):
            raise ConversionError(f"{path}: #{index} öğesi nesne değil")
        item_id = str(item.get("item_id", "")).lower()
        if not re.fullmatch(r"[0-9a-f]{16}", item_id):
            raise ConversionError(f"{path}: #{index} geçersiz item_id")
        if item_id in seen:
            raise ConversionError(f"{path}: yinelenen item_id: {item_id}")
        dimensions = item.get("dimensions")
        if list(dimensions) != list(DIMENSIONS):
            raise ConversionError(
                f"{path}: {item_id} boyutları beklenen şemayla eşleşmiyor"
            )
        seen.add(item_id)
        item_ids.append(item_id)
    return Package(rater_id=rater_id, items=tuple(item_ids), path=str(path))


def load_packages(package_dir: Path) -> Dict[str, Package]:
    """``paketler/Rxx.json`` dosyalarını doğrulanmış rater→paket sözlüğü yapar."""
    package_paths = sorted((Path(package_dir) / "paketler").glob("R[0-9][0-9].json"))
    if not package_paths:
        raise ConversionError(f"{package_dir}/paketler altında Rxx.json bulunamadı")
    packages: Dict[str, Package] = {}
    for path in package_paths:
        package = _read_package(path)
        if package.rater_id in packages:
            raise ConversionError(f"Yinelenen değerlendirici paketi: {package.rater_id}")
        packages[package.rater_id] = package
    return packages


def find_csv_files(path: Path) -> List[Path]:
    """Tek CSV veya alt dizinleri dahil bir CSV dizini döndürür."""
    source = Path(path)
    if source.is_file():
        if source.suffix.lower() != ".csv":
            raise ConversionError(f"Yanıt dosyası CSV olmalı: {source}")
        return [source]
    if source.is_dir():
        files = sorted(item for item in source.rglob("*.csv") if item.is_file())
        if files:
            return files
    raise ConversionError(f"CSV yanıtı bulunamadı: {source}")


def _question_columns(fieldnames: Sequence[str]) -> Dict[Tuple[str, str], str]:
    """Eski makine başlığını ve Forms ızgara dışa aktarımını tanır."""
    columns: Dict[Tuple[str, str], str] = {}
    for header in fieldnames:
        cleaned = (header or "").strip()
        match = QUESTION_RE.fullmatch(cleaned)
        if match:
            item_id = match.group("item").lower()
            dimension = match.group("dimension").lower()
        else:
            grid_match = GRID_QUESTION_RE.fullmatch(cleaned)
            if not grid_match:
                continue
            item_id = grid_match.group("item").lower()
            dimension = (
                grid_match.group("grid_dimension")
                or grid_match.group("hallucination")
            ).lower()
        key = (item_id, dimension)
        if key in columns:
            raise ConversionError(f"CSV'de yinelenen HGA soru sütunu: {header}")
        columns[key] = header
    return columns


def _rater_from_filename(path: Path) -> Optional[str]:
    match = RATER_FILENAME_RE.search(path.stem.upper())
    return match.group(1).upper() if match else None


def _parse_score(value: str, dimension: str, context: str) -> int:
    match = LEADING_SCORE_RE.match(value or "")
    if not match:
        raise ConversionError(f"{context}: boş/geçersiz puan: {value!r}")
    score = int(match.group(1))
    allowed = {0, 1} if dimension == "halusinasyon_var" else {1, 2, 3, 4, 5}
    if score not in allowed:
        allowed_text = "0 veya 1" if dimension == "halusinasyon_var" else "1–5"
        raise ConversionError(f"{context}: {dimension} için puan {allowed_text} olmalı")
    return score


def _timestamp_key(value: Optional[str]) -> Tuple[int, str]:
    """Google'ın yerel biçimleri değişse bile deterministik duplicate davranışı.

    ISO-8601'e benzer bir zaman damgası tanınırsa gerçek kronolojik sıralama,
    aksi halde ham metnin sözlüksel sırası kullanılır. Varsayılan politika
    ``error`` olduğundan bu yalnız ``--duplicate-policy latest`` içindir.
    """
    if not value:
        return (0, "")
    cleaned = value.strip()
    try:
        return (1, datetime.fromisoformat(cleaned.replace("Z", "+00:00")).isoformat())
    except ValueError:
        return (0, cleaned)


def read_form_responses(paths: Iterable[Path], packages: Mapping[str, Package]) -> Tuple[List[Rating], ImportSummary]:
    """CSV satırlarını normalize edilmiş derecelendirmelere çevirir.

    Kimliği boş olan satırlar hata verir: hangi Rxx paketine ait oldukları
    güvenilir biçimde çıkarılamaz. Makine başlığı olmayan CSV de reddedilir;
    yanlış bir Form dışa aktarımı sessizce kabul edilmez.
    """
    ratings: List[Rating] = []
    summary = ImportSummary()
    for path in paths:
        summary.response_files.append(str(path))
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            has_rater_column = RATER_HEADER in fieldnames
            filename_rater = _rater_from_filename(path)
            if not has_rater_column and filename_rater is None:
                raise ConversionError(
                    f"{path}: {RATER_HEADER!r} sütunu yok ve dosya adından Rxx çıkarılamadı"
                )
            question_columns = _question_columns(fieldnames)
            if not question_columns:
                raise ConversionError(
                    f"{path}: desteklenen HGA veya item_id | puanlar [boyut] sütunu yok"
                )
            timestamp_header = next(
                (
                    header
                    for header in fieldnames
                    if header.strip().lower() in {"timestamp", "zaman damgası"}
                ),
                None,
            )
            for row_number, row in enumerate(reader, start=2):
                summary.rows_seen += 1
                rater_id = (
                    (row.get(RATER_HEADER) or "").strip().upper()
                    if has_rater_column
                    else filename_rater or ""
                )
                if not RATER_ID_RE.fullmatch(rater_id):
                    raise ConversionError(f"{path}:{row_number}: geçersiz rater_id {rater_id!r}")
                if rater_id not in packages:
                    raise ConversionError(f"{path}:{row_number}: tanınmayan rater_id {rater_id}")
                expected_items = set(packages[rater_id].items)
                timestamp = row.get(timestamp_header) if timestamp_header else None
                used_in_row = 0
                for (item_id, dimension), header in question_columns.items():
                    if item_id not in expected_items:
                        raise ConversionError(
                            f"{path}:{row_number}: {item_id} {rater_id} paketinde yok"
                        )
                    value = _parse_score(
                        row.get(header, ""), dimension,
                        f"{path}:{row_number} ({item_id}/{dimension})",
                    )
                    ratings.append(Rating(
                        rater_id=rater_id, item_id=item_id, dimension=dimension,
                        value=value, source=str(path), row_number=row_number, timestamp=timestamp,
                    ))
                    used_in_row += 1
                if used_in_row:
                    summary.response_rows_used += 1
    return ratings, summary


def resolve_ratings(
    ratings: Iterable[Rating], duplicate_policy: str = "error"
) -> Dict[Tuple[str, str, str], Rating]:
    """Bir rater/item/boyut için tek puan seçer veya çakışmayı görünür kılar."""
    if duplicate_policy not in {"error", "first", "latest"}:
        raise ConversionError("duplicate_policy: error, first veya latest olmalı")
    resolved: Dict[Tuple[str, str, str], Rating] = {}
    for rating in ratings:
        key = (rating.rater_id, rating.item_id, rating.dimension)
        previous = resolved.get(key)
        if previous is None:
            resolved[key] = rating
            continue
        if duplicate_policy == "error":
            raise ConversionError(
                "Yinelenen puan: "
                f"{rating.rater_id}/{rating.item_id}/{rating.dimension}; "
                f"{previous.source}:{previous.row_number} ve {rating.source}:{rating.row_number}. "
                "Düzeltme formu tekrar gönderildiyse --duplicate-policy latest kullanın."
            )
        if duplicate_policy == "latest" and _timestamp_key(rating.timestamp) >= _timestamp_key(previous.timestamp):
            resolved[key] = rating
    return resolved


def write_rating_csvs(
    resolved: Mapping[Tuple[str, str, str], Rating],
    packages: Mapping[str, Package],
    output_dir: Path,
    allow_partial: bool = False,
) -> ImportSummary:
    """Paket sırasını koruyarak hga uyumlu Rxx_puanlama.csv dosyaları üretir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = ImportSummary()
    for rater_id, package in sorted(packages.items()):
        expected = [(item_id, dimension) for item_id in package.items for dimension in DIMENSIONS]
        missing = [(item_id, dimension) for item_id, dimension in expected
                   if (rater_id, item_id, dimension) not in resolved]
        present = len(expected) - len(missing)
        summary.raters[rater_id] = {
            "package": package.path,
            "expected_cells": len(expected),
            "present_cells": present,
            "missing_cells": len(missing),
            "complete": not missing,
            "output": str(output_dir / f"{rater_id}_puanlama.csv"),
            "package_sha256": hashlib.sha256(
                Path(package.path).read_bytes()).hexdigest(),
        }
        if missing and not allow_partial:
            first = ", ".join(f"{item}/{dimension}" for item, dimension in missing[:3])
            raise ConversionError(
                f"{rater_id}: {len(missing)} eksik puan var ({first}). "
                "Ara çıktı için yalnız --allow-partial kullanın."
            )
        with (output_dir / f"{rater_id}_puanlama.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["item_id", *DIMENSIONS])
            for item_id in package.items:
                values = []
                for dimension in DIMENSIONS:
                    rating = resolved.get((rater_id, item_id, dimension))
                    values.append("" if rating is None else rating.value)
                writer.writerow([item_id, *values])
    return summary


def convert(
    responses: Path,
    packages_dir: Path,
    output_dir: Path,
    allow_partial: bool = False,
    duplicate_policy: str = "error",
) -> ImportSummary:
    packages = load_packages(packages_dir)
    response_paths = find_csv_files(responses)
    ratings, input_summary = read_form_responses(response_paths, packages)
    resolved = resolve_ratings(ratings, duplicate_policy=duplicate_policy)
    output_summary = write_rating_csvs(
        resolved, packages, output_dir, allow_partial=allow_partial
    )
    output_summary.response_files = input_summary.response_files
    output_summary.rows_seen = input_summary.rows_seen
    output_summary.response_rows_used = input_summary.response_rows_used
    output_summary.warnings = list(input_summary.warnings)
    output_summary.warnings.append(
        f"duplicate_policy={duplicate_policy}; resolved_rating_cells={len(resolved)}"
    )
    if allow_partial:
        output_summary.warnings.append(
            "--allow-partial kullanıldı: bu aktarım tamamlanmış insan değerlendirmesi değildir."
        )
    return output_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", required=True, type=Path,
                        help="Google Forms'tan indirilen CSV veya bunları içeren dizin")
    parser.add_argument("--packages", required=True, type=Path,
                        help="insan_degerlendirme_paketleri kök dizini")
    parser.add_argument("--output", required=True, type=Path,
                        help="Rxx_puanlama.csv çıktı dizini (boş/ayrı olmalı)")
    parser.add_argument("--summary", type=Path,
                        help="İçe aktarım denetim özeti için JSON dosyası")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Eksik hücrelerle ara CSV üret; sonuç olarak yorumlama")
    parser.add_argument("--duplicate-policy", choices=("error", "first", "latest"),
                        default="error",
                        help="Aynı rater/item/boyut birden çok kez gelirse davranış (varsayılan: error)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = convert(
            responses=args.responses, packages_dir=args.packages, output_dir=args.output,
            allow_partial=args.allow_partial, duplicate_policy=args.duplicate_policy,
        )
    except ConversionError as error:
        print(f"HATA: {error}", file=sys.stderr)
        return 2
    payload = json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
