# -*- coding: utf-8 -*-
"""Hafif veri versiyonlama manifesti.

DVC yerine geçmez; ama veri dosyalarının hash/byte/satır sayısını kaydederek
tekrarlanabilir deneylerin minimum izini tutar.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List


@dataclass
class DosyaManifesti:
    path: str
    bytes: int
    sha256: str
    lines: int

    def to_dict(self) -> Dict:
        return asdict(self)


def dosya_hashle(path: str, chunk_size: int = 1024 * 1024) -> DosyaManifesti:
    h = hashlib.sha256()
    lines = 0
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            size += len(chunk)
            lines += chunk.count(b"\n")
            h.update(chunk)
    return DosyaManifesti(path=path, bytes=size, sha256=h.hexdigest(), lines=lines)


def manifest_olustur(paths: Iterable[str]) -> Dict:
    dosyalar: List[Dict] = []
    for p in paths:
        dosyalar.append(dosya_hashle(p).to_dict())
    return {"dosyalar": dosyalar}


def manifest_kaydet(paths: Iterable[str], cikis: str) -> Dict:
    m = manifest_olustur(paths)
    os.makedirs(os.path.dirname(os.path.abspath(cikis)) or ".", exist_ok=True)
    tmp = cikis + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    os.replace(tmp, cikis)
    return m


def manifest_yukle(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        veri: Dict = json.load(f)
    return veri


__all__ = ["DosyaManifesti", "dosya_hashle", "manifest_olustur", "manifest_kaydet", "manifest_yukle"]
