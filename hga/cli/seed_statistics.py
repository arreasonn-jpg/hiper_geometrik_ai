"""HGA CLI komutu (Faz 2 refactor)."""
from __future__ import annotations

from hga.cli.common import _yaz_rapor


def _tohum_istatistik(profile="smoke", seeds=None, out=None, markdown=None):
    """P1: Çekirdek protokollerde 20 tohum + istatistiksel çıkarım."""
    from hga.evaluation.seed_statistics import (
        run_core_seed_statistics,
        seed_statistics_markdown,
    )
    tohumlar = ([int(v) for v in seeds.split(",") if v.strip()]
                if seeds else None)
    rapor = run_core_seed_statistics(profile=profile, seeds=tohumlar)
    md = seed_statistics_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)
