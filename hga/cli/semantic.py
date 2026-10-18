"""HGA CLI komutu (Faz 2 refactor)."""
from __future__ import annotations

from hga.cli.common import _yaz_rapor


def _semantik(out=None, markdown=None):
    """P0-5: Türkçe semantik çıkarım v2 gold benchmarkı."""
    from hga.experiments.semantic_extraction import (
        run_semantic_extraction_v2_benchmark,
        semantic_extraction_markdown,
    )
    rapor = run_semantic_extraction_v2_benchmark()
    print(semantic_extraction_markdown(rapor))
    _yaz_rapor(rapor.to_dict(), out, markdown,
               semantic_extraction_markdown(rapor))
