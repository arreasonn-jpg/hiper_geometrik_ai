# -*- coding: utf-8 -*-
"""Canlı Türkçe veri hattı smoke yardımcıları.

Büyük corpus indirme değil; birkaç Wikipedia başlığıyla ağ/kalite/manifest
zincirinin çalıştığını ölçen küçük ve güvenli prova.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

from hga.experience.corpus import cumlelere_bol
from .quality import temizle_cumleler
from .versioning import dosya_hashle


@dataclass
class CanliVeriSmokeRaporu:
    durum: str
    konular: List[str]
    cekilen_konu: int = 0
    ham_karakter: int = 0
    ham_cumle: int = 0
    temiz_cumle: int = 0
    kabul_orani: float = 0.0
    cikis: Optional[str] = None
    manifest: Optional[Dict] = None
    hata: Optional[str] = None
    notlar: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def _varsayilan_wiki_fetcher(proje_kok: str) -> Callable[[str], str]:
    try:
        from egitim.veri_toplayici import OtomatikVeriToplayici, _requests_gerekli
        _requests_gerekli()
    except Exception as e:  # ImportError veya ağ modülü yok
        raise ImportError(str(e)) from e
    toplayici = OtomatikVeriToplayici(proje_kok)
    return lambda konu: toplayici.wikipedia_cek(konu) or ""


def canli_wikipedia_smoke(konular: Optional[Iterable[str]] = None,
                          proje_kok: Optional[str] = None,
                          cikis: Optional[str] = None,
                          fetcher: Optional[Callable[[str], str]] = None,
                          min_temiz_cumle: int = 1) -> CanliVeriSmokeRaporu:
    """Wikipedia → kalite filtresi → opsiyonel manifest zincirini küçük ölçekte koş.

    ``fetcher`` testlerde/ağsız ortamda enjekte edilebilir. Gerçek ağ için
    ``requests`` opsiyoneldir; yoksa ``durum='skipped_dependency'`` döner.
    """
    ham_konular = ["Türkçe", "İstanbul", "Yapay zekâ"] if konular is None else konular
    konular = [str(k).strip() for k in ham_konular if str(k).strip()]
    proje_kok = os.path.abspath(proje_kok or os.getcwd())
    if not konular:
        return CanliVeriSmokeRaporu(durum="error", konular=[], hata="konu listesi boş")

    if fetcher is None:
        try:
            fetcher = _varsayilan_wiki_fetcher(proje_kok)
        except ImportError as e:
            return CanliVeriSmokeRaporu(
                durum="skipped_dependency", konular=konular, hata=str(e),
                notlar=["requests kurulursa canlı smoke çalışır"])

    parcalar: List[str] = []
    hatalar: List[str] = []
    for konu in konular:
        try:
            metin = fetcher(konu) or ""
        except Exception as e:  # ağ/API hata toleransı
            hatalar.append(f"{konu}: {e}")
            continue
        if metin.strip():
            parcalar.append(metin.strip())

    ham = "\n\n".join(parcalar)
    cumleler = cumlelere_bol(ham)
    temiz, kalite = temizle_cumleler(cumleler)
    durum = "ok" if len(temiz) >= int(min_temiz_cumle) else "insufficient_data"
    notlar = []
    if not parcalar:
        notlar.append("canlı kaynak boş döndü veya erişilemedi; kontrollü smoke için --kontrollu kullan")
    rapor = CanliVeriSmokeRaporu(
        durum=durum,
        konular=konular,
        cekilen_konu=len(parcalar),
        ham_karakter=len(ham),
        ham_cumle=len(cumleler),
        temiz_cumle=len(temiz),
        kabul_orani=kalite.to_dict().get("kabul_orani", 0.0),
        hata="; ".join(hatalar) if hatalar else None,
        notlar=notlar,
    )

    if cikis and temiz:
        os.makedirs(os.path.dirname(os.path.abspath(cikis)) or ".", exist_ok=True)
        with open(cikis, "w", encoding="utf-8") as f:
            f.write("\n".join(temiz) + "\n")
        rapor.cikis = cikis
        rapor.manifest = dosya_hashle(cikis).to_dict()
    elif temiz:
        # Hash/manifest yolunu da test edebilmek için kalıcı dosya istenmediyse
        # geçici dosyada hesapla, sonra sil.
        fd, tmp = tempfile.mkstemp(prefix="hga_live_smoke_", suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write("\n".join(temiz) + "\n")
            rapor.manifest = dosya_hashle(tmp).to_dict()
            rapor.manifest["path"] = "<temporary>"
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    return rapor


__all__ = ["CanliVeriSmokeRaporu", "canli_wikipedia_smoke"]
