# -*- coding: utf-8 -*-
"""HiperGeometrikAIHiyerarsik — ana modelin hiyerarşik zincirli sürümü.

Mevcut HiperGeometrikAI'yı miras alır, sadece `kuresel_bag` katmanını
HiyerarsikZincir ile değiştirir. Diğer her şey (attention, encoder,
decoder, seyrek bellek) AYNEN kalır.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Mevcut model
import sys
from pathlib import Path
_MIMARI_DIR = Path(__file__).resolve().parent
if str(_MIMARI_DIR) not in sys.path:
    sys.path.insert(0, str(_MIMARI_DIR))

from kuresel_model import HiperGeometrikAI
from hiyerarsik_bag import HiyerarsikZincir


class HiperGeometrikAIHiyerarsik(HiperGeometrikAI):
    """Hiyerarşik piramit zincirli HGA.

    Ek parametre:
        expansion (int): her up katmanında boyut çarpanı (örn. 2).

    Mevcut `katman_sayisi` çift olmalı: K/2 up, K/2 down.
    Örnek: katman_sayisi=4, expansion=2 → 256 → 512 → 1024 → 512 → 256.
    """

    def __init__(self, *args, expansion: int = 2, **kwargs):
        # Önce parent'ı çağır (tüm katmanlar kurulur)
        super().__init__(*args, **kwargs)

        # Şimdi kuresel_bag'i hiyerarşik versiyonla değiştir
        if self.katman_sayisi < 2 or self.katman_sayisi % 2 != 0:
            raise ValueError(
                f"Hiyerarşik zincir için katman_sayisi çift olmalı, "
                f"gelen: {self.katman_sayisi}"
            )
        self.expansion = expansion
        self.kuresel_bag = HiyerarsikZincir(
            n_base=self.n,
            katman_sayisi=self.katman_sayisi,
            expansion=expansion,
            dropout=0.0,  # dropout zaten parent'ta yok, burada da sıfır
            aktivasyon="silu",
        )

    def kapasite_raporu(self):
        """Parent'ın raporunu genişlet: hiyerarşik zincir bilgisi ekle."""
        rapor = super().kapasite_raporu()
        if hasattr(self.kuresel_bag, "kapasite"):
            k = self.kuresel_bag.kapasite()
            rapor["hiyerarsik"] = {
                "expansion": self.expansion,
                "zincir_gercek_parametre": k["gercek_parametre"],
                "zincir_sanal_operator": k["sanal_operator_girdisi"],
            }
        return rapor


def model_olustur_hiyerarsik(
    n: int = 256, katman_sayisi: int = 4, expansion: int = 2,
    baglam_penceresi: int = 16, emb_dim: int = 128,
    num_heads: int = 4, sozluk_boyutu: int = 8000,
    dropout: float = 0.1,
    seyrek_tablo_boyutu: int = 1_048_576,
    seyrek_boyut: int = 32,
    bilgilendir: bool = True,
) -> HiperGeometrikAIHiyerarsik:
    """Tek çağrıda hiyerarşik HGA modeli kur."""
    return HiperGeometrikAIHiyerarsik(
        n=n, katman_sayisi=katman_sayisi, expansion=expansion,
        baglam_penceresi=baglam_penceresi, emb_dim=emb_dim,
        num_heads=num_heads, sozluk_boyutu=sozluk_boyutu,
        dropout=dropout, bilgilendir=bilgilendir,
        seyrek_tablo_boyutu=seyrek_tablo_boyutu,
        seyrek_boyut=seyrek_boyut,
    )


__all__ = ["HiperGeometrikAIHiyerarsik", "model_olustur_hiyerarsik"]
