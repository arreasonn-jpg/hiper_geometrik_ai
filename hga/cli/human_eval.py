"""HGA CLI komutu (Faz 2 refactor)."""
from __future__ import annotations

from pathlib import Path

from hga.cli.common import _yaz_rapor


def _insan_degerlendirme(package_dir=None, out=None, markdown=None):
    """P1: İnsan değerlendirme protokolü + Krippendorff alfa aracı."""
    from hga.experiments.human_evaluation import human_evaluation_markdown

    rapor = _insan_degerlendirme_raporu(package_dir)
    md = human_evaluation_markdown(rapor)
    print(md)
    _yaz_rapor(rapor.to_dict(), out, markdown, md)

def _insan_degerlendirme_raporu(package_dir=None):
    """Varsa tamamlanmış puanları, yoksa yalnız protokolü yükle."""
    from hga.experiments.human_eval_fill import build_human_evaluation_from_csv
    from hga.experiments.human_evaluation import build_human_evaluation_protocol

    paket_dizini = Path(package_dir or "insan_degerlendirme_paketleri")
    try:
        return build_human_evaluation_from_csv(paket_dizini)
    except ValueError:
        return build_human_evaluation_protocol()
