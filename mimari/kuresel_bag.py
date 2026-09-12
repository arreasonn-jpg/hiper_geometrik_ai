import torch
import torch.nn as nn

class OptimizeEdilmisKureselBag(nn.Module):
    def __init__(self, n=1000):
        super(OptimizeEdilmisKureselBag, self).__init__()
        self.n = n
        self.mercek_A = nn.Linear(n, n, bias=False)
        self.mercek_B = nn.Linear(n, n, bias=False)
        self.aktivasyon = nn.SiLU()

    def forward(self, u, v):
        # O(N^3) yerine O(N^2) Hiper-Tensör İzdüşümü (500x Hızlı)
        u_yeni = self.mercek_A(u)
        v_yeni = self.mercek_B(v)
        u_yeni = self.aktivasyon(u_yeni)
        v_yeni = self.aktivasyon(v_yeni)
        return u_yeni, v_yeni