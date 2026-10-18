"""Geriye uyumluluk shim'i (DEPRECATED).

`hga.evaluation` → `hga.experiments` olarak yeniden adlandırıldı.
Bu modül yalnızca eski import yollarını kırmamak için var.

Yeni kod: ``from hga.experiments import ...``
"""
import sys as _sys
import warnings

import hga.experiments as _yeni
from hga.experiments import *  # noqa: F401,F403,E402

warnings.warn(
    "`hga.evaluation` yeniden adlandırıldı → `hga.experiments`. "
    "Lütfen `from hga.experiments import ...` kullanın.",
    DeprecationWarning,
    stacklevel=2,
)

# Kullanıcıların `hga.evaluation.foo` erişimi için
_sys.modules[__name__].__dict__.update(
    {k: v for k, v in vars(_yeni).items() if not k.startswith("_")}
)
