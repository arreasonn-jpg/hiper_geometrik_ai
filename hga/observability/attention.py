# -*- coding: utf-8 -*-
"""Attention heatmap yardımcıları."""
from __future__ import annotations

from typing import List, Optional


def attention_heatmap(attention_module, x, attention_mask=None, tokens: Optional[List[str]] = None,
                      kafa_ortalama: bool = True) -> dict:
    """HiperGeometrikAttention için ısı haritası üret.

    ``attention_module`` içinde ``attention_haritasi`` metodu beklenir. Dönüşte
    hem sayısal matris hem opsiyonel token etiketleri yer alır.
    """
    if not hasattr(attention_module, "attention_haritasi"):
        raise TypeError("attention_module.attention_haritasi(...) metodu gerekli")
    agirlik = attention_module.attention_haritasi(x, attention_mask=attention_mask,
                                                  kafa_ortalama=kafa_ortalama)
    matris = agirlik.detach().cpu().tolist() if hasattr(agirlik, "detach") else agirlik
    return {
        "shape": tuple(agirlik.shape) if hasattr(agirlik, "shape") else None,
        "tokens": list(tokens) if tokens is not None else None,
        "weights": matris,
        "head_diversity": head_diversity(matris) if not kafa_ortalama else None,
    }


def _flat(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().tolist()
    if isinstance(x, (int, float)):
        return [float(x)]
    out = []
    for v in x:
        if isinstance(v, (list, tuple)):
            out.extend(_flat(v))
        else:
            out.append(float(v))
    return out


def _cos(a, b):
    import math
    a, b = _flat(a), _flat(b)
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)


def head_diversity(weights) -> dict:
    """Attention head çeşitliliği metriği.

    Beklenen şekil ``(B, H, S, S)``. Dönüşte head'ler arası ortalama cosine
    benzerliği ve çeşitlilik ``1 - mean_cosine`` verilir.
    """
    if hasattr(weights, "detach"):
        weights = weights.detach().cpu().tolist()
    if not weights:
        return {"head_sayisi": 0, "mean_pairwise_cosine": 0.0, "diversity": 0.0}
    # Tek batch veya batch listesi: (H,S,S) ise batch'e sar.
    if weights and weights[0] and isinstance(weights[0][0][0], (int, float)):
        weights = [weights]
    coslar = []
    head_sayisi = 0
    for batch in weights:
        head_sayisi = max(head_sayisi, len(batch))
        for i in range(len(batch)):
            for j in range(i + 1, len(batch)):
                coslar.append(_cos(batch[i], batch[j]))
    mean = sum(coslar) / len(coslar) if coslar else 1.0
    return {
        "head_sayisi": head_sayisi,
        "pair_count": len(coslar),
        "mean_pairwise_cosine": mean,
        "diversity": 1.0 - mean,
    }


__all__ = ["attention_heatmap", "head_diversity"]
