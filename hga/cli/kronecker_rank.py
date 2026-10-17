"""Kronecker effective rank CLI komutu (Faz 19/20).

`python -m hga kronecker-rank` için ana giriş noktası.
"""
from __future__ import annotations

import json
import os

from hga.evaluation.kronecker_rank import run_nk_rank_sweep


def _kronecker_rank(n_values: str, k_values: str, seed: int = 1,
                    out: str | None = None, markdown: str | None = None,
                    force: bool = False) -> None:
    """Faz 19/20: effective rank + n×K taraması ve zincir çöküş testi."""
    n_list = [int(v.strip()) for v in n_values.split(",") if v.strip()]
    k_list = [int(v.strip()) for v in k_values.split(",") if v.strip()]
    rapor = run_nk_rank_sweep(
        n_values=n_list, k_values=k_list, seed=int(seed), force=bool(force)
    )
    print("Kronecker effective rank ve n×K taraması (Faz 19/20):\n")
    print(rapor.markdown())
    print()
    for bulgu in rapor.findings:
        print(f"  - {bulgu}")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(rapor.to_dict(), handle, ensure_ascii=False, indent=2,
                      sort_keys=True)
        print(f"  report: {out}")
    if markdown:
        os.makedirs(os.path.dirname(os.path.abspath(markdown)) or ".", exist_ok=True)
        with open(markdown, "w", encoding="utf-8") as handle:
            handle.write("# Faz 19/20 — Kronecker Effective Rank ve n×K Taraması\n\n")
            handle.write(rapor.markdown() + "\n\n")
            for bulgu in rapor.findings:
                handle.write(f"- {bulgu}\n")
        print(f"  markdown: {markdown}")
