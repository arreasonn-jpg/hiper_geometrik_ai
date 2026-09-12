# -*- coding: utf-8 -*-
"""
Kalıcılık — KnowledgeStore'u diske yaz / diskten oku
=====================================================
(rapor §3, §12, EK-C: "VERIFIED → kalıcı bilgiye yükselt")

Bilgi tabanının oturumlar arasında yaşaması için atomik JSON kalıcılığı:

  * `kaydet(store, yol)` — store.to_dict()'i UTF-8 JSON'a yazar. Yazma
    ATOMİKTİR: önce geçici dosyaya yazılır, sonra `os.replace` ile taşınır;
    böylece yarıda kesilen bir yazma mevcut dosyayı bozmaz.
  * `yukle(yol)` — JSON'dan KnowledgeStore kurar (from_dict ile).

Türkçe karakterler `ensure_ascii=False` ile korunur (ör. "Gökyüzü" değil
`"G\u00f6ky\u00fcz\u00fc"` yazılır). Dosya yolu yoksa `yukle` doğal olarak
`FileNotFoundError` fırlatır — çağıran taraf karar verir.
"""
import json
import os
import tempfile

from .knowledge_store import KnowledgeStore


def kaydet(store: KnowledgeStore, yol: str) -> str:
    """KnowledgeStore'u atomik biçimde JSON dosyasına yaz; yolu döner."""
    yol = os.path.abspath(yol)
    dizin = os.path.dirname(yol)
    os.makedirs(dizin, exist_ok=True)

    veri = json.dumps(store.to_dict(), ensure_ascii=False, indent=2)
    fd, gecici = tempfile.mkstemp(dir=dizin, prefix=".hga_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(veri)
        os.replace(gecici, yol)
    finally:
        if os.path.exists(gecici):
            os.remove(gecici)
    return yol


def yukle(yol: str) -> KnowledgeStore:
    """JSON dosyasından KnowledgeStore kur."""
    with open(yol, "r", encoding="utf-8") as f:
        veri = json.load(f)
    return KnowledgeStore.from_dict(veri)
