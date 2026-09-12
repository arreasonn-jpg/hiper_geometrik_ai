# -*- coding: utf-8 -*-
"""
Dil modeli değerlendirme yardımcıları
=====================================

Standart perplexity ölçümü için bağımsız fonksiyonlar. Torch kurulu değilse
modül import edilebilir; gerçek değerlendirme çağrısında açıklayıcı hata verir.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore


def _torch_gerekli():
    if torch is None:
        raise ImportError("Perplexity değerlendirmesi için torch gerekli.")


class _NgramDataset:
    def __init__(self, token_ids: List[int], baglam: int):
        _torch_gerekli()
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.baglam = int(baglam)

    def __len__(self):
        return max(0, len(self.token_ids) - self.baglam)

    def __getitem__(self, idx):
        return (self.token_ids[idx: idx + self.baglam],
                self.token_ids[idx + self.baglam])


def perplexity(model, token_ids: Iterable[int], baglam: Optional[int] = None,
               batch_size: int = 64, ignore_index: int = 0,
               aygit: Optional[str] = None) -> Tuple[float, float]:
    """N-gram language-model loss ve perplexity döndür.

    Dönüş: ``(ortalama_cross_entropy_loss, exp(loss))``. Loss 50 ile
    sınırlandırılarak ``math.exp`` taşması engellenir.
    """
    _torch_gerekli()
    ids = [int(i) for i in token_ids]
    baglam = int(baglam or getattr(model, "baglam_penceresi", 16))
    dataset = _NgramDataset(ids, baglam)
    if len(dataset) == 0:
        raise ValueError(f"Değerlendirme verisi çok kısa: {len(ids)} token, baglam={baglam}")
    device = torch.device(aygit) if aygit else next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    was_training = model.training
    model.eval()
    toplam, adim = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Model logitleri NaN/Inf içeriyor")
            loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Perplexity loss NaN/Inf")
            toplam += float(loss.item())
            adim += 1
    if was_training:
        model.train()
    loss = toplam / max(1, adim)
    return loss, math.exp(min(50.0, loss))


__all__ = ["perplexity"]
