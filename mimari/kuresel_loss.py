# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class KureselGeometrikLoss(nn.Module):
    """Etiket yumuşatmalı standart çapraz entropi (CrossEntropyLoss) sarmalayıcısı.

    Dürüst not: Bu sınıfta özel bir 'geometrik bükülme' terimi YOKTUR; adı yalnızca
    projenin sözlüğüyle uyumlu tutulmuştur. İçinde gerçekten olan şey:
    ``nn.CrossEntropyLoss(label_smoothing=..., ignore_index=...)``.
    """

    def __init__(self, label_smoothing=0.05, ignore_index=-100):
        super(KureselGeometrikLoss, self).__init__()
        self.temel_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, ignore_index=ignore_index
        )

    def forward(self, model_ciktisi, gercek_hedef):
        return self.temel_loss(model_ciktisi, gercek_hedef)
