# -*- coding: utf-8 -*-
"""hiper_geometrik_ai mimari paketi.

Modüller:
- ``kuresel_model``  : HiperGeometrikAI (pencere tabanlı küçük dil modeli)
- ``hiper_attention``: Fused-QKV + Pre-LN + ReZero + SDPA dikkat katmanı
- ``kuresel_bag``    : İki lineer izdüşümü birleştiren katman
- ``kuresel_loss``   : CrossEntropyLoss sarmalayıcısı
- ``decoder``        : n boyutundan sözlük logits'ine lineer katman
- ``tokenizer``      : Kelime düzeyi GeometrikTokenizer
"""
