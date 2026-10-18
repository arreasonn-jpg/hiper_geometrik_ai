"""CLI komutlarının paylaştığı yardımcı fonksiyonlar."""
from __future__ import annotations

import json
import os
import sys


def _yaz_rapor(cikti, out=None, markdown=None, md_metin=None):
    """Ortak JSON/Markdown yazıcı — rapor komutları bunu paylaşır."""
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(cikti, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"  report: {out}")
    if markdown and md_metin is not None:
        os.makedirs(os.path.dirname(os.path.abspath(markdown)) or ".", exist_ok=True)
        with open(markdown, "w", encoding="utf-8") as handle:
            handle.write(md_metin)
        print(f"  markdown: {markdown}")


def _toplu_sinirlar(sinirlar, tohum_sayisi):
    """Tek koşu için yazılan sınır cümlesini çok tohumlu rapora uyarla.

    run_yield_experiment tek koşu ürettiği için "güven aralığı yok" der. Bu
    cümle çok tohumlu toplu raporda YANLIŞ olur: tablo zaten bootstrap GA
    içerir. Tohum sayısına göre cümleyi değiştiriyoruz.
    """
    if tohum_sayisi <= 1:
        return list(sinirlar)
    uyarlanmis = []
    for sinir in sinirlar:
        if sinir.startswith(_TEK_TOHUM_SINIRI):
            uyarlanmis.append(
                f"Güven aralıkları {tohum_sayisi} tohumun bootstrap dağılımından "
                "hesaplandı; tohum sayısı düşük olduğu için aralıklar geniştir."
            )
        else:
            uyarlanmis.append(sinir)
    return uyarlanmis


# ── Paylaşılan modül seviyesi sabitler ──

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MIMARI = os.path.join(KOK, "mimari")
_TEK_TOHUM_SINIRI = "Her rapor tek tohumun tek koşusudur"
for _p in [KOK, MIMARI]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
