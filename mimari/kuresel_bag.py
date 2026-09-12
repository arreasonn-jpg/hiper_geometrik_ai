# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class OptimizeEdilmisKureselBag(nn.Module):
    """İki lineer 'mercek' izdüşümünü birleştiren yardımcı katman.

    Dürüst not: Bu katman, (n x n) boyutlu iki kare matristen oluşan sade bir
    yapıdır (toplam 2·n² parametre). Eski sürümde ``forward`` iki ayrı tensör
    döndürüyor ve model bunlardan yalnızca birincisini kullandığı için
    ``mercek_B`` hiç eğitilmiyordu. Artık her iki izdüşüm de toplanarak
    tek bir çıktıya birleştirilir; böylece iki mercek de gradyan alır.
    """

    def __init__(self, n=1000):
        super(OptimizeEdilmisKureselBag, self).__init__()
        self.n = n
        self.mercek_A = nn.Linear(n, n, bias=False)
        self.mercek_B = nn.Linear(n, n, bias=False)
        self.aktivasyon = nn.SiLU()

    def forward(self, u, v):
        # Her iki izdüşüm de hesaplanır ve BİRLEŞTİRİLİR (u + v).
        # Dönüş değeri tek bir tensördür; (u, v) tuple'ı artık yok.
        u_yeni = self.aktivasyon(self.mercek_A(u))
        v_yeni = self.aktivasyon(self.mercek_B(v))
        return u_yeni + v_yeni
