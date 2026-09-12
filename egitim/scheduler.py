# -*- coding: utf-8 -*-
"""Learning-rate scheduler yardımcıları.

Warmup + cosine decay, temel ve talimat eğitim motorlarında aynı matematikle
kullanılır. Saf ``warmup_cosine_factor`` fonksiyonu torch olmadan test edilebilir.
"""
from __future__ import annotations

import math
from typing import Optional


def warmup_cosine_factor(step: int, total_steps: int, warmup_steps: int = 0,
                         min_factor: float = 0.01) -> float:
    """0..1 aralığında warmup + cosine decay katsayısı.

    * ``step`` 0 tabanlıdır.
    * warmup döneminde doğrusal 0→1 artar.
    * sonra 1→min_factor cosine decay uygulanır.
    """
    step = max(0, int(step))
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))
    min_factor = float(min_factor)
    if warmup_steps > 0 and step < warmup_steps:
        return max(min_factor, (step + 1) / warmup_steps)
    kalan = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / kalan))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_factor + (1.0 - min_factor) * cosine


def warmup_cosine_scheduler(optimizer, total_steps: int, warmup_steps: int = 0,
                            min_factor: float = 0.01):
    """Torch LambdaLR scheduler oluştur."""
    try:
        import torch  # noqa: F401
        from torch.optim.lr_scheduler import LambdaLR
    except Exception as e:  # pragma: no cover
        raise ImportError("Scheduler için torch gerekli") from e
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_cosine_factor(step, total_steps, warmup_steps, min_factor),
    )


__all__ = ["warmup_cosine_factor", "warmup_cosine_scheduler"]
