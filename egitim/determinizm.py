# -*- coding: utf-8 -*-
"""
Deterministik çalışma yardımcıları
==================================

Aynı seed ile iki denemenin aynı sonucu vermesini kolaylaştırır. Torch yoksa
Python standard library kısmı yine çalışır; torch varsa CUDA dahil deterministik
ayarlar uygulanır.
"""
from __future__ import annotations

import os
import random
from typing import Dict


def tohumla(seed: int = 42, deterministic: bool = True) -> Dict[str, object]:
    """Python/Torch seed'lerini ayarla ve yapılanları raporla.

    ``PYTHONHASHSEED`` süreç başladıktan sonra hash sırasını geçmişe dönük
    değiştiremez; yine de alt süreçler için ortam değişkeni yazılır. Tam
    tekrarlanabilir deney için komutu ``PYTHONHASHSEED=<seed>`` ile başlatmak
    en güvenlisidir.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    rapor: Dict[str, object] = {
        "seed": seed,
        "python_random": True,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "torch": False,
        "cuda": False,
        "deterministic": bool(deterministic),
    }
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        rapor["torch"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            rapor["cuda"] = True
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:  # eski torch sürümleri
                torch.use_deterministic_algorithms(True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
    except Exception as e:  # torch yok veya sınırlı kurulum
        rapor["torch_error"] = str(e)
    return rapor


__all__ = ["tohumla"]
