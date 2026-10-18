# -*- coding: utf-8 -*-
"""n/K/context mimari taraması için hızlı tahminleyici."""
from __future__ import annotations

from typing import Dict, Iterable, List


def parametre_tahmini(n: int, k: int, baglam: int, vocab: int = 8000,
                      emb: int = 128, heads: int = 4,
                      seyrek_satir: int = 1_048_576,
                      seyrek_boyut: int = 32) -> Dict:
    """Modeli kurmadan yaklaşık parametre/VRAM/işlem tahmini yap.

    Bu hesap, hızlı karar vermek içindir; kesin sayı için model kurup
    ``kapasite_raporu()`` kullanın. Formüller mimariyle uyumludur.
    """
    n, k, baglam, vocab, emb = map(int, (n, k, baglam, vocab, emb))
    if emb % int(heads) != 0:
        raise ValueError("emb, heads'e tam bölünmelidir")
    embedding = vocab * emb + baglam * emb
    attention = 4 * emb * emb + 2 * emb          # qkv/out + LayerNorm yaklaşık
    encoder = 2 * ((baglam * emb) * n + n)       # proj_u/proj_v
    zincir = k * (2 * n * n + 2 * n)             # A/B + LayerNorm
    decoder = 2 * n * n + (2 * n * vocab + vocab)
    norm_cikis = 2 * n
    seyrek = max(0, int(seyrek_satir)) * int(seyrek_boyut)
    gen_kopru = (int(seyrek_boyut) * baglam * emb + baglam * emb) if seyrek else 0
    toplam = embedding + attention + encoder + zincir + decoder + norm_cikis + seyrek + gen_kopru
    return {
        "n": n,
        "K": k,
        "baglam": baglam,
        "vocab": vocab,
        "emb": emb,
        "tahmini_parametre": toplam,
        "tahmini_yogun_parametre": toplam - seyrek,
        "seyrek_parametre": seyrek,
        "tahmini_ram_mb_fp32": toplam * 4 / 1024 ** 2,
        "katman_basi_sanal_operator": n ** 4,
        "etkilesim_uzayi_ust_siniri": n ** (2 * k),
        "zincir_ops_yaklasik": 2 * k * n ** 3,
        "attention_ops_yaklasik": baglam * baglam * emb,
    }


def nk_taramasi(n_degerleri: Iterable[int] = (128, 256),
                k_degerleri: Iterable[int] = (2, 4, 8),
                baglam_degerleri: Iterable[int] = (16, 64, 128),
                vocab: int = 8000, emb: int = 128, heads: int = 4,
                seyrek_satir: int = 1_048_576,
                seyrek_boyut: int = 32) -> List[Dict]:
    """n × K × context taramasını tablo verisi olarak üret."""
    satirlar: List[Dict] = []
    for n in n_degerleri:
        for k in k_degerleri:
            for baglam in baglam_degerleri:
                satirlar.append(parametre_tahmini(n, k, baglam, vocab=vocab,
                                                  emb=emb, heads=heads,
                                                  seyrek_satir=seyrek_satir,
                                                  seyrek_boyut=seyrek_boyut))
    return satirlar


__all__ = ["parametre_tahmini", "nk_taramasi"]
