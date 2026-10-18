"""HGA CLI komutu (Faz 2 refactor)."""
from __future__ import annotations

from hga.cli.common import _yaz_rapor


def _foundation_analysis(n_values="4,8,16", k_values="1,2,4", seed=1,
                         out=None, markdown=None):
    """CKPT-001 teorik sözleşme, rank/gradient grid'i ve hash modeli."""
    from hga.experiments.foundation_analysis import (
        foundation_analysis_markdown,
        run_foundation_analysis,
    )

    nler = [int(value) for value in n_values.split(",") if value.strip()]
    kler = [int(value) for value in k_values.split(",") if value.strip()]
    rapor = run_foundation_analysis(n_values=nler, k_values=kler, seed=int(seed))
    md = foundation_analysis_markdown(rapor.to_dict())
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)
