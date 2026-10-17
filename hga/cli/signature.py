"""HGA CLI komutu (Faz 2 refactor)."""
from __future__ import annotations

import json
from pathlib import Path

from hga.cli.common import _yaz_rapor


def _signature_gate(yol=None, profile="standard", seeds=None,
                    out=None, markdown=None):
    """Signature Benchmark release gate."""
    from hga.evaluation.signature import (
        run_signature_benchmark,
        run_signature_release_gate,
        signature_release_gate_markdown,
    )
    if yol:
        rapor_dict = json.loads(Path(yol).read_text(encoding="utf-8"))
    else:
        tohumlar = [int(v) for v in (seeds or "1").split(",") if v.strip()]
        rapor_dict = run_signature_benchmark(
            profile=profile, seeds=tohumlar).to_dict()
    gate = run_signature_release_gate(rapor_dict)
    md = signature_release_gate_markdown(gate)
    print(md)
    _yaz_rapor(gate.to_dict(), out, markdown, md)

def _signature(profile="smoke", seeds=None, out=None, markdown=None):
    """P0-4: HGA Signature Benchmark."""
    from hga.evaluation.signature import (
        run_signature_benchmark,
        signature_markdown,
    )
    tohumlar = [int(v) for v in (seeds or "1,2,3").split(",") if v.strip()]
    rapor = run_signature_benchmark(profile=profile, seeds=tohumlar)
    print(signature_markdown(rapor))
    _yaz_rapor(rapor.to_dict(), out, markdown, signature_markdown(rapor))
