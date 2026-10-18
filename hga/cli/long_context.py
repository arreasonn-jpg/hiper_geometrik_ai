"""HGA CLI komutu (Faz 2 refactor)."""
from __future__ import annotations

from hga.cli.common import _yaz_rapor


def _uzun_baglam(profile="smoke", seeds=None, out=None, markdown=None):
    """P1: Uzun bağlam LM taraması (opsiyonel 512/1024 smoke)."""
    from hga.evaluation.long_context import (
        long_context_markdown,
        run_long_context_benchmark,
    )
    tohumlar = ([int(v) for v in seeds.split(",") if v.strip()]
                if seeds else None)
    rapor = run_long_context_benchmark(profile=profile, seeds=tohumlar)
    md = long_context_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)
