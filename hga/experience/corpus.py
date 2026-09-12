# -*- coding: utf-8 -*-
"""
Korpus Yükleyici — metin dosyasından cümle → üçlü → REAL_DATA
===============================================================
(rapor §2, §12; `egitim/veri_toplayici.py` çıktısıyla bağlantı)

Döngünün "Gerçek veri" aşamasını dosyadan besler:

    `dosyadan_cumleler(yol)`   — metin dosyasını cümlelere böler
    `dosyadan_bilgi_aktar(...)`— cümleleri CumleAyiklayici ile üçlülere
                                  ayırıp REAL_DATA olarak KnowledgeStore'a yazar

Cümle bölme basittir: `.`, `!`, `?` sonrası boşluk/ satır sonu. Kısaltma
bilgisi (örn. "vs.", "Dr.") gibi tam bir cümle ayırıcı KAPSAM DIŞIDIR; bu
modül korpus ölçeğine geçişte değiştirilebilir tek noktadır.
"""
import re
from typing import Dict, List, Optional

from .cumle_ayiklayici import CumleAyiklayici, cumlelerden_bilgi_aktar


def cumlelere_bol(metin: str) -> List[str]:
    """Metni cümlelere böler (noktalama + boşluk/satır sonu)."""
    parcalar = re.split(r"(?<=[.!?])\s+", (metin or "").strip())
    return [p.strip() for p in parcalar if p.strip()]


def dosyadan_cumleler(yol: str) -> List[str]:
    """Bir metin dosyasını okuyup cümlelere böler."""
    with open(yol, "r", encoding="utf-8") as f:
        return cumlelere_bol(f.read())


def dosyadan_bilgi_aktar(store, yol: str,
                         ayiklayici: Optional[CumleAyiklayici] = None,
                         iliski_kisitlari: Optional[Dict[str, Dict]] = None):
    """Dosyadaki cümleleri ayıklayıp REAL_DATA olarak KnowledgeStore'a yazar.

    Dönen liste, aktarılan üçlülerdir (sözlük dışı cümleler atlanır).
    """
    cumleler = dosyadan_cumleler(yol)
    return cumlelerden_bilgi_aktar(store, cumleler, ayiklayici=ayiklayici,
                                   iliski_kisitlari=iliski_kisitlari)
