# -*- coding: utf-8 -*-
"""hiper_geometrik_ai eğitim paketi.

Modüller (çalıştırma sırasına göre):
- ``veri_toplayici``   : Türkçe korpus toplama (Hugging Face / Wikipedia)
- ``egitici``          : Ön-eğitim (n-gram/next-token) — hiper_model_<n>.pt üretir
- ``talimat_toplayici``: Soru–cevap talimat verisi (talimat_verisi.json)
- ``talimat_egitici``  : Instruction fine-tuning — hiper_model_<n>_talimat.pt üretir
"""
