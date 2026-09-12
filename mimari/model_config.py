# -*- coding: utf-8 -*-
"""
ModelConfig — mimari/eğitim varsayılanları için tek kaynak
==========================================================

Kod içinde dağınık ``n=256``, ``K=4``, ``context=16`` gibi sihirli sayıların
tek yerde tutulması için hafif dataclass tabanlı config. PyYAML kuruluysa YAML
okur; değilse düz ``bölüm: anahtar: değer`` ayrıştırıcısına düşer.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    n: int = 256
    katman_sayisi: int = 4
    baglam_penceresi: int = 16
    emb_dim: int = 128
    num_heads: int = 4
    sozluk_boyutu: int = 8000
    dropout: float = 0.1
    encoder_aktivasyon: str = "tanh"
    zincir_aktivasyon: str = "silu"
    checkpoint_kullan: bool = False
    seyrek_tablo_boyutu: int = 1_048_576
    seyrek_boyut: int = 32
    seyrek_tablo_sayisi: int = 1
    seyrek_erisim_izleme: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]] = None) -> "ModelConfig":
        d = dict(d or {})
        alanlar = {f.name for f in fields(cls)}
        int_alanlar = {"n", "katman_sayisi", "baglam_penceresi", "emb_dim",
                       "num_heads", "sozluk_boyutu", "seyrek_tablo_boyutu",
                       "seyrek_boyut", "seyrek_tablo_sayisi"}
        temiz: Dict[str, Any] = {}
        for ad, deger in d.items():
            if ad not in alanlar:
                continue
            try:
                if ad in ("checkpoint_kullan", "seyrek_erisim_izleme"):
                    if isinstance(deger, str):
                        deger = deger.strip().lower() in ("1", "true", "evet", "yes")
                    else:
                        deger = bool(deger)
                elif ad in int_alanlar:
                    deger = int(deger)
                elif ad == "dropout":
                    deger = float(deger)
            except Exception:
                pass
            temiz[ad] = deger
        return cls(**temiz)


@dataclass
class TrainingConfig:
    ogrenme_hizi: float = 1e-3
    toplam_cag: int = 5
    batch_size: int = 64
    grad_clip: float = 1.0
    amp: str = "auto"
    validation_split: float = 0.0
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 0.0
    warmup_cag: int = 0
    lr_min_factor: float = 0.01
    log_dizini: Optional[str] = None
    checkpoint_dizini: Optional[str] = None
    son_checkpoint_sayisi: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]] = None) -> "TrainingConfig":
        d = dict(d or {})
        alanlar = {f.name: f for f in fields(cls)}
        temiz: Dict[str, Any] = {}
        for ad, deger in d.items():
            if ad not in alanlar:
                continue
            if ad in ("ogrenme_hizi", "grad_clip", "validation_split", "early_stopping_min_delta", "lr_min_factor"):
                deger = float(deger)
            elif ad in ("toplam_cag", "batch_size", "son_checkpoint_sayisi", "warmup_cag"):
                deger = int(deger)
            elif ad == "early_stopping_patience" and deger is not None:
                deger = int(deger)
            temiz[ad] = deger
        return cls(**temiz)


VARSAYILAN_MODEL_CONFIG = ModelConfig()
VARSAYILAN_TRAINING_CONFIG = TrainingConfig()


def _coerce(v: str):
    v = str(v).strip().strip('"').strip("'")
    if v.lower() in ("true", "evet", "yes"):
        return True
    if v.lower() in ("false", "hayir", "hayır", "no"):
        return False
    if v.lower() in ("none", "null", ""):
        return None
    try:
        f = float(v)
        return int(f) if f == int(f) else f
    except ValueError:
        return v


def _basit_yaml(metin: str) -> Dict[str, Dict[str, Any]]:
    sonuc: Dict[str, Dict[str, Any]] = {}
    bolum: Optional[str] = None
    for ham in metin.splitlines():
        satir = ham.split("#", 1)[0].rstrip()
        if not satir.strip():
            continue
        if not satir.startswith((" ", "\t")) and satir.endswith(":"):
            bolum = satir[:-1].strip()
            sonuc.setdefault(bolum, {})
            continue
        if bolum and ":" in satir:
            k, v = satir.strip().split(":", 1)
            sonuc[bolum][k.strip()] = _coerce(v)
    return sonuc


def yukle(yol: Optional[str] = None) -> Dict[str, Any]:
    """YAML/dict config yükle.

    Dönüş: ``{"model": ModelConfig, "training": TrainingConfig}``.
    """
    if yol is None:
        yol = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "hga", "config", "model_config.yaml")
    veri: Dict[str, Any] = {}
    if yol and os.path.exists(yol):
        with open(yol, "r", encoding="utf-8") as f:
            metin = f.read()
        try:
            import yaml  # type: ignore
            veri = yaml.safe_load(metin) or {}
        except Exception:
            veri = _basit_yaml(metin)
    return {
        "model": ModelConfig.from_dict(veri.get("model", {})),
        "training": TrainingConfig.from_dict(veri.get("training", {})),
    }


__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "VARSAYILAN_MODEL_CONFIG",
    "VARSAYILAN_TRAINING_CONFIG",
    "yukle",
]
