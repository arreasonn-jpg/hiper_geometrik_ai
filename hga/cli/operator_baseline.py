"""HGA CLI komutu (Faz 2 refactor)."""
from __future__ import annotations

from hga.cli.common import _yaz_rapor


def _operator_baseline(n=8, steps=300, seeds=None, device="cpu",
                       out=None, markdown=None):
    """P0-2: Kronecker vs gerçek Dense baseline ailesi."""
    from hga.experiments.operator_baselines import (
        operator_baseline_markdown,
        run_operator_baseline_benchmark,
    )
    tohumlar = [int(v) for v in (seeds or "1,2,3").split(",") if v.strip()]
    rapor = run_operator_baseline_benchmark(
        n=int(n), steps=int(steps), seeds=tohumlar, device=device)
    print(operator_baseline_markdown(rapor))
    _yaz_rapor(rapor.to_dict(), out, markdown, operator_baseline_markdown(rapor))
