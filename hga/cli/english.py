"""HGA CLI komutu (Faz 2 refactor)."""
from __future__ import annotations

from hga.cli.common import _yaz_rapor


def _english_scaling(seeds=None, profile="smoke", out=None, markdown=None):
    """HGA'nın gerçek English EWT üzerinde küçük parameter-scaling probe'u."""
    from hga.experiments.english_ewt import (
        english_hga_scaling_markdown,
        run_english_hga_scaling_probe,
    )

    tohumlar = [int(v) for v in (seeds or "1,2,3,4,5").split(",") if v.strip()]
    rapor = run_english_hga_scaling_probe(seeds=tohumlar, profile=profile)
    md = english_hga_scaling_markdown(rapor)
    print(md)
    _yaz_rapor(rapor, out, markdown, md)

def _english_ewt(seeds=None, profile="smoke", out=None, markdown=None):
    """Pinned UD English EWT üzerinde Transformer/BERT/GPT-style kontroller."""
    from hga.experiments.english_ewt import english_ewt_markdown, run_english_ewt_baselines

    tohumlar = [int(v) for v in (seeds or "1,2,3,4,5").split(",") if v.strip()]
    rapor = run_english_ewt_baselines(seeds=tohumlar, profile=profile)
    md = english_ewt_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)
