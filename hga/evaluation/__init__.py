"""Geriye uyumluluk shim'i (DEPRECATED).

`hga.evaluation` → `hga.experiments` olarak yeniden adlandırıldı.
Bu modül yalnızca eski import yollarını kırmamak için var.

Yeni kod: ``from hga.experiments import ...``
"""
import importlib
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

# 1) Üst seviye fonksiyon/sınıfları kopyala
_sys.modules[__name__].__dict__.update(
    {k: v for k, v in vars(_yeni).items() if not k.startswith("_")}
)

# 2) Alt modülleri de alias'la (örn: hga.evaluation.capacity)
import pkgutil  # noqa: E402

for _mod_info in pkgutil.walk_packages(_yeni.__path__, prefix="hga.experiments."):
    _eski_ad = _mod_info.name.replace("hga.experiments", "hga.evaluation", 1)
    try:
        _eski_mod = importlib.import_module(_mod_info.name)
        _sys.modules[_eski_ad] = _eski_mod
    except ImportError:
        pass
