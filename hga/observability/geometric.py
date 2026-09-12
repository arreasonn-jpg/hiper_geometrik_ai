# -*- coding: utf-8 -*-
"""Geometrik/Kronecker katman benzerliği analizi."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple


def _flatten(v) -> List[float]:
    if hasattr(v, "detach"):
        v = v.detach().cpu().reshape(-1).tolist()
    out: List[float] = []
    for x in v:
        if isinstance(x, (list, tuple)):
            out.extend(_flatten(x))
        else:
            out.append(float(x))
    return out


def _cos(a, b) -> float:
    a, b = _flatten(a), _flatten(b)
    if len(a) != len(b):
        raise ValueError("vektör boyutları eşit olmalı")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)


def liste_katman_benzerligi(mercek_ciftleri: Sequence[Tuple[Iterable, Iterable]]) -> Dict:
    """A/B mercek listelerinden Kronecker operatör cosine benzerlik matrisi.

    Frobenius iç çarpımı özelliği: ``cos(Bᵀ⊗A, Dᵀ⊗C) = cos(A,C) * cos(B,D)``.
    Bu yüzden tam n⁴ operatörü oluşturmadan katman benzerliği ölçülür.
    """
    k = len(mercek_ciftleri)
    a_mat = [[0.0 for _ in range(k)] for _ in range(k)]
    b_mat = [[0.0 for _ in range(k)] for _ in range(k)]
    op_mat = [[0.0 for _ in range(k)] for _ in range(k)]
    for i in range(k):
        for j in range(k):
            ca = _cos(mercek_ciftleri[i][0], mercek_ciftleri[j][0])
            cb = _cos(mercek_ciftleri[i][1], mercek_ciftleri[j][1])
            a_mat[i][j] = ca
            b_mat[i][j] = cb
            op_mat[i][j] = ca * cb
    off = [op_mat[i][j] for i in range(k) for j in range(k) if i != j]
    return {
        "katman_sayisi": k,
        "A_cosine": a_mat,
        "B_cosine": b_mat,
        "kronecker_operator_cosine": op_mat,
        "ortalama_offdiag": sum(off) / len(off) if off else 0.0,
        "max_offdiag": max(off) if off else 0.0,
    }


def kronecker_katman_benzerligi(zincir) -> Dict:
    """``KureselZincir`` benzeri nesnenin katman merceklerini analiz et."""
    if not hasattr(zincir, "katmanlar"):
        raise TypeError("KureselZincir benzeri .katmanlar alanı bekleniyor")
    ciftler = []
    for katman in zincir.katmanlar:
        ciftler.append((katman.mercek_A, katman.mercek_B))
    return liste_katman_benzerligi(ciftler)


__all__ = ["liste_katman_benzerligi", "kronecker_katman_benzerligi"]
