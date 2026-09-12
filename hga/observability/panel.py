# -*- coding: utf-8 -*-
"""Tek dosyalık gözlemlenebilirlik paneli üretimi."""
from __future__ import annotations

import html
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from .experience_flow import deneyim_akisi
from .geometric import liste_katman_benzerligi
from .memory import ascii_bellek_haritasi, bellek_doluluk_haritasi


def gozlem_paneli_olustur(adaylar: Optional[Iterable] = None,
                          slotlar: Any = None,
                          mercek_ciftleri: Optional[Iterable] = None,
                          baslik: str = "HGA Gözlemlenebilirlik Paneli") -> Dict[str, Any]:
    """Bellek/deneyim/geometri metriklerini JSON-serileştirilebilir panelde birleştir."""
    panel: Dict[str, Any] = {
        "rapor_tipi": "hga-observability-panel-v1",
        "baslik": baslik,
        "olusturma_zamani_utc": datetime.now(timezone.utc).isoformat(),
        "deneyim_akisi": deneyim_akisi(adaylar or []),
    }
    if slotlar is not None:
        panel["bellek"] = bellek_doluluk_haritasi(slotlar)
        panel["bellek_ascii"] = ascii_bellek_haritasi(slotlar)
    else:
        panel["bellek"] = None
        panel["bellek_ascii"] = ""
    if mercek_ciftleri is not None:
        panel["geometri"] = liste_katman_benzerligi(list(mercek_ciftleri))
    else:
        panel["geometri"] = None
    return panel


def gozlem_paneli_markdown(panel: Dict[str, Any]) -> str:
    ak = panel.get("deneyim_akisi", {})
    bellek = panel.get("bellek") or {}
    geo = panel.get("geometri") or {}
    lines = [
        f"# {panel.get('baslik', 'HGA Gözlemlenebilirlik Paneli')}",
        "",
        f"- Rapor tipi: `{panel.get('rapor_tipi')}`",
        f"- Zaman (UTC): `{panel.get('olusturma_zamani_utc')}`",
        "",
        "## Deneyim Akışı",
        f"- Toplam: `{ak.get('toplam', 0)}`",
        f"- Kabul oranı: `{ak.get('acceptance_rate', 0.0)}`",
        f"- Red oranı: `{ak.get('rejection_rate', 0.0)}`",
        f"- Çatışma oranı: `{ak.get('conflict_rate', 0.0)}`",
        f"- Durumlar: `{ak.get('durum', {})}`",
        "",
        "## Bellek",
        f"- Doluluk: `{bellek.get('dolu', 0)}/{bellek.get('toplam', 0)}`",
        f"- Doluluk oranı: `{bellek.get('doluluk_orani', 0.0)}`",
        "",
        "```text",
        str(panel.get("bellek_ascii", "")),
        "```",
        "",
        "## Geometri",
        f"- Katman sayısı: `{geo.get('katman_sayisi', 0)}`",
        f"- Ortalama offdiag: `{geo.get('ortalama_offdiag', 0.0)}`",
        f"- Maks offdiag: `{geo.get('max_offdiag', 0.0)}`",
    ]
    return "\n".join(lines) + "\n"


def gozlem_paneli_html(panel: Dict[str, Any]) -> str:
    md = gozlem_paneli_markdown(panel)
    # Dış bağımlılıksız, statik ve Arena preview'de güvenli tek HTML.
    body = html.escape(md)
    raw = html.escape(json.dumps(panel, ensure_ascii=False, indent=2, sort_keys=True))
    return f"""<!doctype html>
<html lang=\"tr\">
<head>
<meta charset=\"utf-8\">
<title>{html.escape(panel.get('baslik', 'HGA Gözlem Paneli'))}</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 2rem; line-height: 1.45; }}
pre {{ background: #111827; color: #f9fafb; padding: 1rem; border-radius: 0.75rem; overflow: auto; }}
.card {{ border: 1px solid #e5e7eb; border-radius: 1rem; padding: 1rem; margin: 1rem 0; }}
</style>
</head>
<body>
<div class=\"card\"><pre>{body}</pre></div>
<h2>Ham JSON</h2>
<pre>{raw}</pre>
</body>
</html>
"""


def gozlem_paneli_kaydet(panel: Dict[str, Any], json_yol: Optional[str] = None,
                         markdown_yol: Optional[str] = None,
                         html_yol: Optional[str] = None) -> Dict[str, str]:
    yollar: Dict[str, str] = {}
    if json_yol:
        os.makedirs(os.path.dirname(os.path.abspath(json_yol)) or ".", exist_ok=True)
        with open(json_yol, "w", encoding="utf-8") as f:
            json.dump(panel, f, ensure_ascii=False, indent=2, sort_keys=True)
        yollar["json"] = json_yol
    if markdown_yol:
        os.makedirs(os.path.dirname(os.path.abspath(markdown_yol)) or ".", exist_ok=True)
        with open(markdown_yol, "w", encoding="utf-8") as f:
            f.write(gozlem_paneli_markdown(panel))
        yollar["markdown"] = markdown_yol
    if html_yol:
        os.makedirs(os.path.dirname(os.path.abspath(html_yol)) or ".", exist_ok=True)
        with open(html_yol, "w", encoding="utf-8") as f:
            f.write(gozlem_paneli_html(panel))
        yollar["html"] = html_yol
    return yollar


__all__ = [
    "gozlem_paneli_olustur",
    "gozlem_paneli_markdown",
    "gozlem_paneli_html",
    "gozlem_paneli_kaydet",
]
