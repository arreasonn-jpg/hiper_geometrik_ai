"""CLI komutlarının paylaştığı yardımcı fonksiyonlar."""
from __future__ import annotations

import json
import os


def _yaz_rapor(cikti, out=None, markdown=None, md_metin=None):
    """Ortak JSON/Markdown yazıcı — rapor komutları bunu paylaşır."""
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(cikti, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"  report: {out}")
    if markdown and md_metin is not None:
        os.makedirs(os.path.dirname(os.path.abspath(markdown)) or ".", exist_ok=True)
        with open(markdown, "w", encoding="utf-8") as handle:
            handle.write(md_metin)
        print(f"  markdown: {markdown}")
