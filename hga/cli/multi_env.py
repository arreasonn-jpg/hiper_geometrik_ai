"""HGA CLI komutu (Faz 2 refactor)."""
from __future__ import annotations

from hga.cli.common import _yaz_rapor


def _cok_ortam(seeds=None, out=None, markdown=None):
    """P1: Çoklu ortam + cross-domain verifier izolasyonu."""
    from hga.evaluation.multi_environment import (
        multi_environment_markdown,
        run_multi_environment_benchmark,
    )
    tohumlar = [int(v) for v in (seeds or "1,2,3,4,5").split(",") if v.strip()]
    rapor = run_multi_environment_benchmark(seeds=tohumlar)
    md = multi_environment_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)
