"""HGA CLI komutu (Faz 2 refactor)."""
from __future__ import annotations

from hga.cli.common import _yaz_rapor


def _verifier_adversarial(out=None, markdown=None):
    """Genişletilmiş verifier adversarial suite v2."""
    from hga.evaluation.verifier_adversarial import (
        run_verifier_adversarial_v2_benchmark,
    )
    rapor = run_verifier_adversarial_v2_benchmark()
    md = rapor.markdown()
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)

def _verifier_ensemble(out=None, markdown=None):
    """Çoklu verifier ensemble benchmarkı."""
    from hga.evaluation.verifier_ensemble import (
        run_verifier_ensemble_benchmark,
        verifier_ensemble_markdown,
    )
    rapor = run_verifier_ensemble_benchmark()
    md = verifier_ensemble_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)
