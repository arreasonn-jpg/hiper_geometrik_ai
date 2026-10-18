# -*- coding: utf-8 -*-
# ruff: noqa: E402
"""HGA Model Architecture — İngilizce public API.

`kuresel_model.py` göreli-olmayan importlar (`from decoder import ...`) kullanır;
bu yüzden `mimari/` dizinini `sys.path`'e ekleyip öyle import ederiz.
Bu kalıp, `hga/__main__.py`'deki bootstrap ile aynıdır.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_MIMARI = _os.path.dirname(_os.path.abspath(__file__))
if _MIMARI not in _sys.path:
    _sys.path.insert(0, _MIMARI)

from kuresel_model import (  # noqa: E402
    HiperGeometrikAI as HyperGeometricAI,
)
from kuresel_model import (
    agirlik_kaydet as save_weights,
)
from kuresel_model import (
    agirlik_yukle as load_weights,
)
from kuresel_model import (
    model_olustur as build_model,
)
from kuresel_model import (
    model_olustur_config as build_model_from_config,
)

__all__ = [
    "HyperGeometricAI",
    "build_model",
    "build_model_from_config",
    "load_weights",
    "save_weights",
]
