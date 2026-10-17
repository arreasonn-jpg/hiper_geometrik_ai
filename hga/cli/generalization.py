"""HGA CLI komutu (Faz 2 refactor)."""
from __future__ import annotations

from hga.cli.common import _yaz_rapor


def _genelleme_v2(out=None, markdown=None):
    """P0-6: ham metinden keşif + kompozisyon (C_G v2)."""
    from hga.evaluation.compositional_v2 import (
        compositional_v2_markdown,
        run_compositional_v2_benchmark,
    )
    rapor = run_compositional_v2_benchmark()
    print(compositional_v2_markdown(rapor))
    _yaz_rapor(rapor.to_dict(), out, markdown, compositional_v2_markdown(rapor))
