# -*- coding: utf-8 -*-
"""
Config — experience_config.yaml yükleyici
==========================================
(v0.1 — rapor §13)

Bağımlılıksız çalışır: PyYAML kuruluysa onu kullanır; değilse bu dosyadaki
basit, güvenli "düz anahtar: değer" ayrıştırıcı devreye girer. Dosya yoksa
veya okunamıyorsa gömülü varsayılanlara düşer — sistem asla yapılandırma
eksikliğinden çökmez.
"""
import os
from typing import Dict, Optional

from ..experience.evaluator import VARSAYILAN_ESIKLER
from ..experience.scoring import VARSAYILAN_AGIRLIKLAR

VARSAYILAN_GENERATOR = {"source_confidence": 0.5, "tip_filtresi": True}
VARSAYILAN_BELLEK = {
    "policy": "DYNAMIC_KV",
    "eviction_policy": "lru",
    "max_idle_ticks": None,
}

_YAML_YOLU = os.path.join(os.path.dirname(__file__), "experience_config.yaml")


def _ayrikla_bolum(metin: str, baslik: str) -> Dict:
    """'baslik:' altındaki 'anahtar: deger' satırlarını ayrıştır (düz YAML)."""
    satirlar = metin.splitlines()
    sonuc: Dict = {}
    icinde = False
    girinti = None
    for satir in satirlar:
        temiz = satir.strip()
        if not temiz or temiz.startswith("#"):
            continue
        if ":" in temiz and not temiz.startswith(" ") and not temiz.startswith("\t"):
            anahtar = temiz.split(":", 1)[0].strip()
            icinde = (anahtar == baslik)
            girinti = len(satir) - len(satir.lstrip())
            continue
        if icinde and ":" in temiz:
            k, v = temiz.split(":", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            # yalnızca bu bölümün alt satırları (girinti anahtar satırından büyük)
            mevcut_girinti = len(satir) - len(satir.lstrip())
            if girinti is not None and mevcut_girinti > girinti:
                if v.lower() == "true":
                    v = True
                elif v.lower() == "false":
                    v = False
                elif v.lower() in ("null", "none", "~"):
                    v = None
                else:
                    try:
                        v = float(v)
                        v = int(v) if v == int(v) else v
                    except ValueError:
                        pass
                sonuc[k] = v
    return sonuc


def yukle(yol: Optional[str] = None) -> Dict:
    """experience_config.yaml → {'agirliklar': ..., 'esikler': ..., 'generator': ...}."""
    yol = yol or _YAML_YOLU
    agirliklar = dict(VARSAYILAN_AGIRLIKLAR)
    esikler = dict(VARSAYILAN_ESIKLER)
    generator = dict(VARSAYILAN_GENERATOR)
    bellek = dict(VARSAYILAN_BELLEK)

    metin = None
    if os.path.exists(yol):
        try:
            with open(yol, "r", encoding="utf-8") as f:
                metin = f.read()
        except OSError:
            metin = None

    if metin:
        try:
            import yaml  # PyYAML kuruluysa tercih et
            veri = yaml.safe_load(metin) or {}
            if isinstance(veri.get("agirliklar"), dict):
                agirliklar.update(veri["agirliklar"])
            if isinstance(veri.get("esikler"), dict):
                esikler.update(veri["esikler"])
            if isinstance(veri.get("generator"), dict):
                generator.update(veri["generator"])
            if isinstance(veri.get("bellek"), dict):
                bellek.update(veri["bellek"])
        except Exception:
            # PyYAML yok veya bozuksa düz ayrıştırıcı
            agirliklar.update(_ayrikla_bolum(metin, "agirliklar"))
            esikler.update(_ayrikla_bolum(metin, "esikler"))
            generator.update(_ayrikla_bolum(metin, "generator"))
            bellek.update(_ayrikla_bolum(metin, "bellek"))

    return {
        "agirliklar": agirliklar,
        "esikler": esikler,
        "generator": generator,
        "bellek": bellek,
    }


def varsayilanlar() -> Dict:
    return {
        "agirliklar": dict(VARSAYILAN_AGIRLIKLAR),
        "esikler": dict(VARSAYILAN_ESIKLER),
        "generator": dict(VARSAYILAN_GENERATOR),
        "bellek": dict(VARSAYILAN_BELLEK),
    }
