# -*- coding: utf-8 -*-
"""tr_corpus_v1 derleme aracı — milyon-kelime GERÇEK Türkçe korpus.

Bu script bir benchmark DEĞİLDİR; vendored korpusu üreten geliştirme
aracıdır. Ağ erişimi gerektirir ve normalde YALNIZ korpus sürümü
yükseltilirken koşulur. Üretilen çıktı (`corpus.jsonl.gz` + PROVENANCE)
depoya sabitlenir; benchmark tarafı ağ olmadan, hash doğrulamasıyla okur.

Kaynak seçim ilkeleri:

* **Gerçek, insan üretimi Türkçe metin.** Sentetik/otomatik çeviri yok.
* **Yeniden dağıtılabilir lisans.** CC BY-SA / CC0 / Apache-2.0.
  UD_Turkish-IMST bilinçli DIŞARIDA bırakıldı: BY-NC-SA (NonCommercial)
  bu deponun lisans profiliyle uyumsuz.
* **Pinli revizyon.** Her kaynak tag/commit SHA'sı ile indirilir; "en son
  sürüm" diye bir şey yoktur.

Kaynaklar (UD r2.14 + bible-corpus + TWT):

* UD_Turkish-{Kenet, BOUN, Penn, Tourism, Atis, FrameNet, PUD, GB} —
  ``# text =`` satırları, dosya sırasıyla. CC BY-SA.
* christos-c/bible-corpus ``Turkish.xml`` — ``<seg>`` ayetleri, kitap
  bazında gruplanır. CC0 1.0.
* TWT v1 — repoda zaten vendored (Apache-2.0); kendi doğal belge
  kimlikleriyle (``tr-forum:00000222`` gibi) dahil edilir. NOT: TWT arc
  doğrulama benchmarkının da verisidir; bu ÇAPRAZ-GÖREV ilişkisi
  PROVENANCE'a yazılır, gizlenmez.

Belge birimi neden önemli: LM split'i belge-ayrıktır. Bible kitap başına
bir belgedir (66 belge); UD treebank'lerinde doğal belge sınırı olmadığı
için ardışık ``CHUNK`` cümle deterministik bir sözde-belge sayılır. Tek
dev belge tek split'e düşerdi ve split oranlarını çökertirdi.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import unicodedata
import urllib.request
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SCHEMA_VERSION = 1
CORPUS_NAME = "tr_corpus_v1"
UD_TAG = "r2.14"
CHUNK = 40  # UD sözde-belge başına ardışık cümle sayısı

#: (kaynak adı, GitHub org/repo, lisans, atıf)
UD_SOURCES: Tuple[Tuple[str, str, str], ...] = (
    ("ud_kenet", "UniversalDependencies/UD_Turkish-Kenet", "CC BY-SA 4.0"),
    ("ud_boun", "UniversalDependencies/UD_Turkish-BOUN", "CC BY-SA 4.0"),
    ("ud_penn", "UniversalDependencies/UD_Turkish-Penn", "CC BY-SA 4.0"),
    ("ud_tourism", "UniversalDependencies/UD_Turkish-Tourism", "CC BY-SA 4.0"),
    ("ud_atis", "UniversalDependencies/UD_Turkish-Atis", "CC BY-SA 4.0"),
    ("ud_framenet", "UniversalDependencies/UD_Turkish-FrameNet", "CC BY-SA 4.0"),
    ("ud_pud", "UniversalDependencies/UD_Turkish-PUD", "CC BY-SA 3.0"),
    ("ud_gb", "UniversalDependencies/UD_Turkish-GB", "CC BY-SA 4.0"),
)
BIBLE_REPO = "christos-c/bible-corpus"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _http_get(url: str, dest: Path) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "HGA-corpus/1.0"})
    with urllib.request.urlopen(request, timeout=300) as response:
        dest.write_bytes(response.read())
    return sha256_file(dest)


def _resolve_tag_sha(repo: str, tag: str) -> str:
    out = subprocess.check_output(
        ["git", "ls-remote", f"https://github.com/{repo}.git", f"refs/tags/{tag}"],
        text=True, timeout=120)
    sha = out.split()[0] if out.strip() else ""
    if not sha:
        raise RuntimeError(f"{repo} deposunda {tag} tag'i bulunamadı")
    return sha


def _resolve_head_sha(repo: str) -> str:
    out = subprocess.check_output(
        ["git", "ls-remote", f"https://github.com/{repo}.git", "HEAD"],
        text=True, timeout=120)
    return out.split()[0]


def normalize_line(text: str) -> str:
    """Dedup anahtarı: NFKC + casefold + boşluk sadeleştirme."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def extract_ud_texts(extracted_dir: Path) -> List[str]:
    """Bir UD treebank dizinindeki tüm ``# text =`` satırlarını sırayla al."""
    sentences: List[str] = []
    for conllu in sorted(extracted_dir.glob("*.conllu")):
        for line in conllu.read_text(encoding="utf-8").splitlines():
            if line.startswith("# text = "):
                sentence = line[len("# text = "):].strip()
                if sentence:
                    sentences.append(sentence)
    return sentences


def extract_bible_books(xml_path: Path) -> Dict[str, List[str]]:
    """Ayetleri kitap koduna göre grupla (``<seg id="b.GEN.1.1">`` → GEN)."""
    content = xml_path.read_text(encoding="utf-8")
    books: Dict[str, List[str]] = {}
    for match in re.finditer(
            r'<seg[^>]*id="b\.([A-Z0-9]+)\.[^"]*"[^>]*>(.*?)</seg>',
            content, re.S):
        book, verse = match.group(1), " ".join(match.group(2).split())
        if verse:
            books.setdefault(book, []).append(verse)
    if not books:
        raise RuntimeError("Bible XML'den hiç ayet çıkarılamadı")
    return books


def chunk_documents(source: str, sentences: List[str]) -> List[Dict[str, str]]:
    docs = []
    for start in range(0, len(sentences), CHUNK):
        blok = sentences[start:start + CHUNK]
        docs.append({
            "doc_id": f"{source}:{start // CHUNK:05d}",
            "source": source,
            "text": "\n".join(blok),
        })
    return docs


def load_twt_documents() -> List[Dict[str, str]]:
    """Repodaki vendored TWT'yi kendi doğal belge kimlikleriyle oku."""
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from hga.evaluation.real_turkish import TurkishWebTreebank

    twt = TurkishWebTreebank()
    gruplar: Dict[str, List[Tuple[str, str]]] = {}
    for sentence in twt.sentences:
        parts = sentence.sentence_id.split(":")
        doc_id = ":".join(parts[:2]) if len(parts) >= 2 else sentence.sentence_id
        gruplar.setdefault(doc_id, []).append((sentence.sentence_id, sentence.text))
    docs = []
    for doc_id in sorted(gruplar):
        cumleler = [t for _, t in sorted(gruplar[doc_id])]
        docs.append({"doc_id": f"twt:{doc_id}", "source": "twt",
                     "text": "\n".join(cumleler)})
    return docs


def dedup_documents(documents: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], int]:
    """Korpus genelinde normalize-exact cümle dedup'u; boşalan belge düşer."""
    seen: set = set()
    dropped = 0
    result: List[Dict[str, str]] = []
    for doc in documents:
        kept: List[str] = []
        for line in doc["text"].split("\n"):
            key = normalize_line(line)
            if not key or key in seen:
                dropped += 1
                continue
            seen.add(key)
            kept.append(line)
        if kept:
            result.append({**doc, "text": "\n".join(kept)})
    return result, dropped


def canonical_corpus_bytes(documents: List[Dict[str, str]]) -> bytes:
    lines = [json.dumps(d, ensure_ascii=False, sort_keys=True,
                        separators=(",", ":")) for d in documents]
    return ("\n".join(lines) + "\n").encode("utf-8")


def build(work_dir: Path, out_dir: Path,
          skip_download: bool = False) -> Dict[str, object]:
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    upstream: List[Dict[str, object]] = []
    documents: List[Dict[str, str]] = []

    # ── UD treebank'leri ──
    for source, repo, license_name in UD_SOURCES:
        tag_sha = _resolve_tag_sha(repo, UD_TAG)
        tarball = work_dir / f"{source}.tar.gz"
        if not (skip_download and tarball.exists()):
            _http_get(
                f"https://codeload.github.com/{repo}/tar.gz/refs/tags/{UD_TAG}",
                tarball)
        tar_sha = sha256_file(tarball)
        extract_root = work_dir / source
        if extract_root.exists():
            import shutil
            shutil.rmtree(extract_root)
        with tarfile.open(tarball) as tf:
            tf.extractall(extract_root)
        inner = next(extract_root.iterdir())
        sentences = extract_ud_texts(inner)
        docs = chunk_documents(source, sentences)
        documents.extend(docs)
        upstream.append({
            "source": source,
            "repository": f"https://github.com/{repo}",
            "revision": tag_sha, "tag": UD_TAG,
            "license": license_name,
            "artifact_sha256": tar_sha,
            "sentences": len(sentences),
            "documents": len(docs),
            "words": sum(len(s.split()) for s in sentences),
        })

    # ── Bible (CC0) ──
    # Depo 100+ dilin İncil'ini içerir (~yüzlerce MB); tamamını indirmek
    # yerine pinli SHA'da sığ klon yapılır ve YALNIZ Turkish.xml hash'lenir.
    bible_root = work_dir / "bible-corpus"
    if bible_root.exists():
        bible_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(bible_root),
            text=True, timeout=60).strip()
    else:
        bible_sha = _resolve_head_sha(BIBLE_REPO)
        subprocess.check_call(
            ["git", "clone", "-q", "--depth", "1",
             f"https://github.com/{BIBLE_REPO}.git", str(bible_root)],
            timeout=600)
    xml_path = bible_root / "bibles" / "Turkish.xml"
    bible_xml_sha = sha256_file(xml_path)
    books = extract_bible_books(xml_path)
    bible_docs = [{"doc_id": f"bible:{book}", "source": "bible",
                   "text": "\n".join(verses)}
                  for book, verses in sorted(books.items())]
    documents.extend(bible_docs)
    upstream.append({
        "source": "bible",
        "repository": f"https://github.com/{BIBLE_REPO}",
        "revision": bible_sha, "tag": None,
        "license": "CC0 1.0",
        "artifact_sha256": bible_xml_sha,
        "artifact_note": "yalnız bibles/Turkish.xml hash'i (depo 100+ dil içerir)",
        "sentences": sum(len(v) for v in books.values()),
        "documents": len(bible_docs),
        "words": sum(len(s.split()) for v in books.values() for s in v),
    })

    # ── TWT (repoda vendored) ──
    twt_docs = load_twt_documents()
    documents.extend(twt_docs)
    upstream.append({
        "source": "twt",
        "repository": "vendored: hga/evaluation/datasets/twt_v1",
        "revision": "40838e5cbe3f2882d4e768a3d782e6219e50b52a", "tag": None,
        "license": "Apache-2.0",
        "artifact_sha256": None,
        "sentences": sum(d["text"].count("\n") + 1 for d in twt_docs),
        "documents": len(twt_docs),
        "words": sum(len(d["text"].split()) for d in twt_docs),
        "cross_benchmark_note": (
            "TWT aynı zamanda twt_real_results_v1 arc-doğrulama "
            "benchmarkının verisidir. LM görevi farklıdır (next-token); "
            "bu çapraz-görev ilişkisi bilinçli ve belgelidir."),
    })

    # ── Dedup + yaz ──
    documents, dropped = dedup_documents(documents)
    documents.sort(key=lambda d: d["doc_id"])
    canonical = canonical_corpus_bytes(documents)
    corpus_path = out_dir / "corpus.jsonl.gz"
    with open(corpus_path, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            gz.write(canonical)

    total_words = sum(len(d["text"].split()) for d in documents)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "corpus_name": CORPUS_NAME,
        "built_utc_date": date.today().isoformat(),
        "upstream": upstream,
        "excluded_sources": [{
            "source": "ud_imst",
            "repository": "https://github.com/UniversalDependencies/UD_Turkish-IMST",
            "reason": "CC BY-NC-SA (NonCommercial) — depo lisans profiliyle uyumsuz",
        }],
        "processing": {
            "ud_extraction": "# text = satırları, dosya adı sırasıyla",
            "ud_pseudo_document_chunk": CHUNK,
            "bible_document_unit": "kitap (66 belge)",
            "twt_document_unit": "doğal sent_id belge öneki",
            "dedup": "korpus geneli normalize-exact cümle dedup",
            "dedup_dropped_sentences": dropped,
        },
        "counts": {
            "documents": len(documents),
            "words": total_words,
            "sentences": sum(d["text"].count("\n") + 1 for d in documents),
        },
        "files": {
            "corpus.jsonl.gz": {
                "sha256": sha256_file(corpus_path),
                "canonical_content_sha256": sha256_bytes(canonical),
            },
        },
    }
    (out_dir / "PROVENANCE.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n", encoding="utf-8")
    return provenance


def main(argv: Optional[Iterable[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "hga" / "evaluation" / "datasets" / "tr_corpus_v1"
    work_dir = Path("/tmp/tr_corpus_build")
    provenance = build(work_dir, out_dir)
    counts: Dict[str, int] = provenance["counts"]  # type: ignore[assignment]
    print(f"{CORPUS_NAME}: {counts['documents']:,} belge, "
          f"{counts['words']:,} kelime → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
