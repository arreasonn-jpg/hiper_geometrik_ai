"""HGA CLI komutu (Faz 2 refactor)."""
from __future__ import annotations

from hga.cli.common import _yaz_rapor


def _cikarim_derinligi(profile="standard", seeds=None, threshold=1.0,
                       out=None, markdown=None):
    """P0-7: C_R ve C_RD ölçümü."""
    from hga.experiments.reasoning_depth import (
        measure_reasoning_depth,
        reasoning_depth_markdown,
    )
    tohumlar = ([int(v) for v in seeds.split(",") if v.strip()]
                if seeds else None)
    rapor = measure_reasoning_depth(profile=profile, seeds=tohumlar,
                                    reliability_threshold=float(threshold))
    print(reasoning_depth_markdown(rapor))
    _yaz_rapor(rapor.to_dict(), out, markdown, reasoning_depth_markdown(rapor))
