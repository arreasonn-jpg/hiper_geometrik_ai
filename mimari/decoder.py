# -*- coding: utf-8 -*-
"""
Fraktal Decoder — Bilinear "Odak Merceği" Okuma Katmanı
=======================================================

Küresel bağ zincirinin (B, n, n) çıktısını sözlük logitlerine çevirir.

1) Odak mercekleri (1.0 belgesindeki "odak merceği" fikri GERİ DÖNDÜ):
       odak = M1 @ X @ M2        (bilinear sandviç — 2n² gerçek parametre,
                                  n⁴ sanal operatör girdisi)
2) Çift yönlü ortalama okuma:
       y1 = odak.mean(dim=1) → (B, n)   (mercek_2'nin tamamı devrede)
       y2 = odak.mean(dim=2) → (B, n)   (mercek_1'in tamamı devrede)
       y  = tanh(concat(y1, y2))
3) anlam_cozucu: Linear(2n, sozluk_boyutu) → ham logit

NOT (softmax YOK): nn.CrossEntropyLoss içeride log_softmax uyguladığı için
burada softmax UYGULANMAZ — 1.0 belgesindeki "çifte softmax" hatası bilinçli
olarak düzeltilmiştir (GitHub sürümündeki doğru karar korunmuştur; rapor 8.6.4).
"""
import torch
import torch.nn as nn


class FraktalDecoder(nn.Module):
    def __init__(self, n: int, sozluk_boyutu: int):
        super().__init__()
        self.n = n
        self.mercek_1 = nn.Parameter(torch.empty(n, n))
        self.mercek_2 = nn.Parameter(torch.empty(n, n))
        nn.init.xavier_uniform_(self.mercek_1)
        nn.init.xavier_uniform_(self.mercek_2)
        self.anlam_cozucu = nn.Linear(2 * n, sozluk_boyutu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n, n) → odak: (B, n, n)
        odak = torch.einsum("ij,bjk,kl->bil", self.mercek_1, x, self.mercek_2)
        y1 = odak.mean(dim=1)   # (B, n)
        y2 = odak.mean(dim=2)   # (B, n)
        y = torch.tanh(torch.cat([y1, y2], dim=-1))
        cikti: torch.Tensor = self.anlam_cozucu(y)  # ham logit — softmax YOK
        return cikti


# Geriye dönük uyumluluk
AnlamCozucu = FraktalDecoder
