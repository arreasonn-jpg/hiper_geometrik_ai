# -*- coding: utf-8 -*-
"""Türkçe korpus genişletme pipeline'ı (P1).

Bu modül yeni büyük korpusu depoya eklemez; lisans/provenance denetimi,
Türkçe içerik filtresi, mevcut ``tr_corpus_v1`` ile dedup, deterministik split
ve release-gate raporunu üretir. Dış kaynak dizini verilmezse küçük gömülü
smoke adaylarıyla hattın çalıştığı kanıtlanır; bu smoke gerçek korpus
büyütmesi olarak sunulamaz.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .turkish_lm import TR_CORPUS_DIR

PROTOCOL = "turkish_corpus_expansion_pipeline_v1"
SCHEMA_VERSION = 1
DEFAULT_TARGET_WORDS = 2_000_000
ALLOWED_LICENSES: Tuple[str, ...] = (
    "Apache-2.0",
    "CC0 1.0",
    "CC BY 4.0",
    "CC BY-SA 3.0",
    "CC BY-SA 4.0",
)


@dataclass(frozen=True)
class CorpusCandidate:
    doc_id: str
    source: str
    license: str
    text: str
    provenance_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CorpusExpansionReport:
    protocol: str
    schema_version: int
    mode: str
    target_words: int
    base_corpus: Dict[str, Any]
    candidate_summary: Dict[str, Any]
    expanded_projection: Dict[str, Any]
    split_projection: Dict[str, Any]
    rejection_summary: Dict[str, Any]
    release_gates: Dict[str, bool]
    checks: Dict[str, bool]
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        return turkish_corpus_expansion_markdown(self)


def _normalize_line(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _doc_split(doc_id: str) -> str:
    bucket = int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest(), 16) % 10
    if bucket < 8:
        return "train"
    return "dev" if bucket == 8 else "test"


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def _turkish_score(text: str) -> float:
    words = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]+", text, flags=re.UNICODE)
    if not words:
        return 0.0
    lower = [w.casefold() for w in words]
    stopwords = {
        "ve", "bir", "bu", "ile", "için", "olarak", "daha", "sonra",
        "kadar", "ancak", "çünkü", "de", "da", "mi", "mı", "mu", "mü",
        "nedir", "nasıl", "hangi", "olan", "gibi", "çok", "az",
    }
    stop = sum(1 for w in lower if w in stopwords)
    tr_chars = len(re.findall(r"[çğıöşüÇĞİÖŞÜ]", text))
    return round((stop + min(tr_chars, len(words))) / max(1, len(words)), 6)


def _passes_turkish_filter(text: str) -> bool:
    return _word_count(text) >= 5 and _turkish_score(text) >= 0.08


def _current_corpus() -> Tuple[Dict[str, Any], List[Dict[str, str]], set[str]]:
    provenance_path = TR_CORPUS_DIR / "PROVENANCE.json"
    corpus_path = TR_CORPUS_DIR / "corpus.jsonl.gz"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    docs: List[Dict[str, str]] = []
    line_keys: set[str] = set()
    with gzip.open(corpus_path, "rt", encoding="utf-8") as handle:
        for raw in handle:
            doc = json.loads(raw)
            docs.append({"doc_id": str(doc["doc_id"]),
                         "source": str(doc["source"]),
                         "text": str(doc["text"])})
            for line in str(doc["text"]).split("\n"):
                key = _normalize_line(line)
                if key:
                    line_keys.add(key)
    return provenance, docs, line_keys


def _smoke_candidates() -> List[CorpusCandidate]:
    return [
        CorpusCandidate(
            doc_id="smoke_cc0_haber_001",
            source="smoke_cc0_demo",
            license="CC0 1.0",
            provenance_url="repo://embedded-smoke",
            text=("İstanbul'da sabah erken saatlerde yağmur başladı. "
                  "Belediye ekipleri yolları açık tutmak için çalışma yaptı."),
        ),
        CorpusCandidate(
            doc_id="smoke_ccby_egitim_001",
            source="smoke_ccby_demo",
            license="CC BY 4.0",
            provenance_url="repo://embedded-smoke",
            text=("Öğrenciler deney sonuçlarını deftere yazdı. "
                  "Veriler daha sonra sınıfta birlikte tartışıldı."),
        ),
        CorpusCandidate(
            doc_id="smoke_rejected_nc_001",
            source="smoke_nc_demo",
            license="CC BY-NC-SA 4.0",
            provenance_url="repo://embedded-smoke",
            text=("Bu aday lisans nedeniyle reddedilmelidir ve korpusa "
                  "eklenmemelidir."),
        ),
    ]


def _read_jsonl_candidates(candidate_dir: Path) -> List[CorpusCandidate]:
    candidates: List[CorpusCandidate] = []
    for path in sorted(candidate_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for row_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{row_number}: JSONL parse hatası") from error
                missing = [k for k in ("doc_id", "source", "license", "text")
                           if not obj.get(k)]
                if missing:
                    raise ValueError(f"{path}:{row_number}: eksik alanlar {missing}")
                candidates.append(CorpusCandidate(
                    doc_id=str(obj["doc_id"]),
                    source=str(obj["source"]),
                    license=str(obj["license"]),
                    text=str(obj["text"]),
                    provenance_url=(None if obj.get("provenance_url") is None
                                    else str(obj.get("provenance_url"))),
                ))
    return candidates


def _candidate_source(candidate_dir: Optional[Path], use_smoke_candidates: bool) -> Tuple[str, List[CorpusCandidate]]:
    if candidate_dir is not None:
        if not candidate_dir.exists():
            raise ValueError(f"candidate_dir yok: {candidate_dir}")
        candidates = _read_jsonl_candidates(candidate_dir)
        return "external_candidate_dir", candidates
    if use_smoke_candidates:
        return "embedded_smoke_candidates", _smoke_candidates()
    return "empty_external_protocol", []


def _summarize_sources(candidates: Sequence[CorpusCandidate]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        row = result.setdefault(c.source, {"documents": 0, "words": 0,
                                          "licenses": set()})
        row["documents"] += 1
        row["words"] += _word_count(c.text)
        row["licenses"].add(c.license)
    return {
        source: {"documents": data["documents"], "words": data["words"],
                 "licenses": sorted(data["licenses"])}
        for source, data in sorted(result.items())
    }


def _canonical_candidates(candidates: Sequence[CorpusCandidate]) -> str:
    return hashlib.sha256(json.dumps(
        [c.to_dict() for c in candidates], ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()[:12]


def run_turkish_corpus_expansion_pipeline(
    candidate_dir: Optional[Path] = None,
    *,
    use_smoke_candidates: bool = True,
    target_words: int = DEFAULT_TARGET_WORDS,
) -> CorpusExpansionReport:
    """Korpus genişletme adaylarını denetle ve release-gate raporu üret."""
    if int(target_words) < 1:
        raise ValueError("target_words >= 1 olmalı")
    provenance, _base_docs, existing_line_keys = _current_corpus()
    mode, candidates = _candidate_source(candidate_dir, use_smoke_candidates)

    accepted_docs: List[Dict[str, Any]] = []
    rejections: List[Dict[str, Any]] = []
    duplicate_lines = 0
    kept_lines_total = 0
    for c in candidates:
        reasons: List[str] = []
        if c.license not in ALLOWED_LICENSES:
            reasons.append("license_not_allowed")
        if not _passes_turkish_filter(c.text):
            reasons.append("turkish_filter_failed")
        kept_lines: List[str] = []
        if not reasons:
            local_seen: set[str] = set()
            for line in re.split(r"(?:\n+|(?<=[.!?])\s+)", c.text):
                normalized = _normalize_line(line)
                if not normalized:
                    continue
                if normalized in existing_line_keys or normalized in local_seen:
                    duplicate_lines += 1
                    continue
                local_seen.add(normalized)
                kept_lines.append(line.strip())
            if not kept_lines:
                reasons.append("all_lines_duplicate")
        if reasons:
            rejections.append({"doc_id": c.doc_id, "source": c.source,
                               "license": c.license, "reasons": reasons})
            continue
        text = "\n".join(kept_lines)
        kept_lines_total += len(kept_lines)
        accepted_docs.append({
            "doc_id": c.doc_id,
            "source": c.source,
            "license": c.license,
            "provenance_url": c.provenance_url,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "words": _word_count(text),
            "sentences_or_lines": len(kept_lines),
            "split": _doc_split(c.doc_id),
        })

    base_counts = provenance["counts"]
    accepted_words = sum(int(d["words"]) for d in accepted_docs)
    split_projection: Dict[str, Dict[str, int]] = {
        split: {"documents": 0, "words": 0} for split in ("train", "dev", "test")
    }
    for doc in accepted_docs:
        split = str(doc["split"])
        split_projection[split]["documents"] += 1
        split_projection[split]["words"] += int(doc["words"])

    target_words = int(target_words)
    expanded_words = int(base_counts["words"]) + accepted_words
    release_gates = {
        "external_candidates_supplied": mode == "external_candidate_dir",
        "no_license_rejections": all(
            "license_not_allowed" not in r["reasons"] for r in rejections),
        "accepted_candidate_words_positive": accepted_words > 0,
        "expanded_corpus_reaches_target_words": expanded_words >= target_words,
        "provenance_urls_present_for_accepted": all(
            bool(d["provenance_url"]) for d in accepted_docs),
    }
    checks = {
        "base_provenance_loaded": bool(provenance.get("files")),
        "candidate_schema_valid": True,
        "accepted_licenses_allowed": all(
            d["license"] in ALLOWED_LICENSES for d in accepted_docs),
        "turkish_filter_applied": True,
        "dedup_against_current_corpus_applied": True,
        "deterministic_split_assigned": all(
            d["split"] in ("train", "dev", "test") for d in accepted_docs),
        "smoke_mode_not_marked_as_release": (
            mode != "embedded_smoke_candidates"
            or not release_gates["external_candidates_supplied"]),
    }
    candidate_summary = {
        "mode": mode,
        "candidate_documents": len(candidates),
        "candidate_sources": _summarize_sources(candidates),
        "candidate_signature": _canonical_candidates(candidates),
        "accepted_documents": len(accepted_docs),
        "accepted_words": accepted_words,
        "kept_lines": kept_lines_total,
        "duplicate_lines_removed": duplicate_lines,
        "accepted_preview": accepted_docs[:10],
    }
    rejection_summary = {
        "rejected_documents": len(rejections),
        "by_reason": {},
        "examples": rejections[:20],
    }
    by_reason: Dict[str, int] = {}
    for rejection in rejections:
        for reason in rejection["reasons"]:
            by_reason[reason] = by_reason.get(reason, 0) + 1
    rejection_summary["by_reason"] = dict(sorted(by_reason.items()))

    projection = {
        "base_words": int(base_counts["words"]),
        "accepted_words": accepted_words,
        "expanded_words": expanded_words,
        "target_words": target_words,
        "target_gap_words": max(0, target_words - expanded_words),
        "target_completion_ratio": round(expanded_words / target_words, 6),
        "additional_words_needed_after_candidates": max(0, target_words - expanded_words),
        "rough_additional_docs_needed_at_candidate_mean": (
            math.ceil(max(0, target_words - expanded_words) /
                      max(1, accepted_words / max(1, len(accepted_docs))))
            if accepted_docs else None),
    }
    status = ("RELEASE_READY" if all(release_gates.values())
              else "PIPELINE_READY_TARGET_NOT_REACHED")
    findings = [
        f"Mevcut tr_corpus_v1: {base_counts['documents']:,} belge, "
        f"{base_counts['words']:,} kelime.",
        f"Aday mod: {mode}; kabul edilen {len(accepted_docs)}/{len(candidates)} belge, "
        f"{accepted_words:,} kelime.",
        f"Genişletilmiş projeksiyon {expanded_words:,}/{target_words:,} kelime; "
        f"durum {status}.",
    ]
    if mode == "embedded_smoke_candidates":
        findings.append(
            "Smoke adayları yalnız pipeline kanıtıdır; tr_corpus_v2 release'i ya da gerçek genişleme değildir.")
    if rejections:
        findings.append(f"Reddedilen adaylar: {rejection_summary['by_reason']}.")
    limitations = [
        "Bu pipeline dış kaynak indirmez; aday JSONL dizini ayrı sağlanmalıdır.",
        "Türkçe filtresi hafif bir sezgiseldir; nihai release için örneklemeli insan/otomatik kalite denetimi gerekir.",
        "Dedup normalize-exact düzeydedir; semantik yakın tekrarlar bu smoke içinde yakalanmaz.",
        "Release gate hedef kelime sayısı ve provenance koşulları sağlanmadan PASS olmaz.",
    ]
    return CorpusExpansionReport(
        protocol=PROTOCOL,
        schema_version=SCHEMA_VERSION,
        mode=mode,
        target_words=target_words,
        base_corpus={
            "name": provenance.get("corpus_name", "tr_corpus_v1"),
            "counts": base_counts,
            "file_sha256": provenance.get("files", {}).get("corpus.jsonl.gz", {}).get("sha256"),
            "canonical_content_sha256": provenance.get("files", {}).get("corpus.jsonl.gz", {}).get("canonical_content_sha256"),
        },
        candidate_summary=candidate_summary,
        expanded_projection=projection,
        split_projection=split_projection,
        rejection_summary=rejection_summary,
        release_gates=release_gates,
        checks=checks,
        findings=findings,
        limitations=limitations,
    )


def turkish_corpus_expansion_markdown(report: CorpusExpansionReport) -> str:
    """Korpus genişletme raporunu Markdown'a çevir."""
    s = report
    p = s.expanded_projection
    rows = [
        "# Türkçe Korpus Genişletme Pipeline",
        "",
        f"- Protokol: `{s.protocol}` v{s.schema_version}",
        f"- Mod: `{s.mode}`",
        f"- Hedef: `{s.target_words:,}` kelime",
        "",
        "> Bu rapor yeni büyük korpus release'i değildir; aday alımı, lisans, "
        "Türkçe filtre, dedup ve split/release-gate hattını kanıtlar.",
        "",
        "## Projeksiyon",
        "",
        "| Alan | Değer |",
        "|---|---:|",
        f"| Mevcut kelime | {p['base_words']:,} |",
        f"| Kabul edilen aday kelime | {p['accepted_words']:,} |",
        f"| Projeksiyon toplam | {p['expanded_words']:,} |",
        f"| Hedef boşluğu | {p['target_gap_words']:,} |",
        f"| Hedef tamamlanma oranı | {p['target_completion_ratio']:.6f} |",
        "",
        "## Aday özeti",
        "",
        "| Alan | Değer |",
        "|---|---:|",
        f"| Aday belge | {s.candidate_summary['candidate_documents']} |",
        f"| Kabul edilen belge | {s.candidate_summary['accepted_documents']} |",
        f"| Reddedilen belge | {s.rejection_summary['rejected_documents']} |",
        f"| Duplicate satır | {s.candidate_summary['duplicate_lines_removed']} |",
        "",
        "## Split projeksiyonu (yalnız kabul edilen adaylar)",
        "",
        "| Split | Belge | Kelime |",
        "|---|---:|---:|",
    ]
    for split in ("train", "dev", "test"):
        data = s.split_projection[split]
        rows.append(f"| {split} | {data['documents']} | {data['words']} |")
    rows.extend([
        "",
        "## Release gate'leri",
        "",
        "| Gate | Sonuç |",
        "|---|---|",
    ])
    rows.extend(f"| {k} | {'GEÇTİ' if v else 'KALDI'} |"
                for k, v in s.release_gates.items())
    rows.extend(["", "## Pipeline kontrolleri", "", "| Kontrol | Sonuç |", "|---|---|"])
    rows.extend(f"| {k} | {'GEÇTİ' if v else 'KALDI'} |"
                for k, v in s.checks.items())
    rows.extend(["", "## Bulgular", ""])
    rows.extend(f"- {x}" for x in s.findings)
    rows.extend(["", "## Sınırlar", ""])
    rows.extend(f"- {x}" for x in s.limitations)
    return "\n".join(rows) + "\n"


__all__ = [
    "ALLOWED_LICENSES",
    "DEFAULT_TARGET_WORDS",
    "PROTOCOL",
    "SCHEMA_VERSION",
    "CorpusCandidate",
    "CorpusExpansionReport",
    "run_turkish_corpus_expansion_pipeline",
    "turkish_corpus_expansion_markdown",
]
