# -*- coding: utf-8 -*-
"""
LEGACY katmanı — dondurulmuş eski prototipler
==============================================

Bu paketteki modüller `docs/KOD_TABANI_VE_MIMARI_DUZENI.md` içinde `LEGACY`
olarak sınıflandırılmıştır. Aktif mimari onları KULLANMAZ:

* `legacy/bilgi_katmani.py` → yerini `hga.knowledge.KnowledgeStore` ve
  `hga.evaluation.hallucination` aldı.
* `legacy/calistir.py` → yerini `python -m hga` CLI'si aldı.
* `legacy/arayuz.py` → yerini `python -m hga` ve `hga.observability`
  panelleri aldı.

Kurallar:

* Yeni kod bu paketten **import etmez**. Tek istisna `hga.ui_runtime`, o da
  yalnız eski sohbet arayüzlerini ayakta tutmak içindir.
* Buradaki modüller yeni özellik almaz; yalnız geriye dönük test uyumluluğu
  için korunur ve v1.1'de kaldırılmaları planlanır.
* Legacy sızıntısını `tests/test_legacy_isolation.py` otomatik denetler.
"""

__all__: list = []
