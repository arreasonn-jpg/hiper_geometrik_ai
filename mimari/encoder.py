# -*- coding: utf-8 -*-
"""
Geometrik Veri Encoder — 0→1 Katmanı (Dış Çarpım Köprüsü)
=========================================================

Önceki sürümde bu modül ÖLÜ KODTU: 512 boyutlu "önceden vektörleştirilmiş ham
veri" bekliyor ama modelde ona bu formatta veri üreten hiçbir yol yoktu
(rapor, Bölüm 4). Bu yeniden tasarımda encoder, modelin GERÇEKTEN kullandığı
köprüye dönüştürüldü (rapor 5.3: "ya köprüle ya kaldır" — köprüledik):

    düzleştirilmiş bağlam vektörü (B, baglam × emb_dim)
        → iki n boyutlu izdüşüm:  u = proj_u(duz),  v = proj_v(duz)
        → dış çarpım + tanh:      X = tanh(u ⊗ v)      → (B, n, n)

Böylece "0→1 katmanı" geometrik büyümesi gerçekleşir: n boyutlu iki vektörden
n² adet "sanal algı köşesi" üreten matris elde edilir (README: 1. Katman) ve
bu matris, KureselZincir'deki bilinear katmanlara girdi olur.

tanh burada hem 1.0 belgesindeki "bükme" adımına sadık kalır hem de önemli bir
matematiksel görev görür: u ⊗ v dış çarpımı rank-1'dir; eleman bazlı tanh bu
kısıtı KIRAR ve zincire tam ranklı bir başlangıç matrisi verir.
"""
import torch
import torch.nn as nn


class GeometrikVeriEncoder(nn.Module):
    """Düz vektör → (u, v) → tanh(u ⊗ v) → (B, n, n) sanal geometri matrisi."""

    def __init__(self, giris_boyutu: int, n: int):
        super().__init__()
        self.n = n
        self.proj_u = nn.Linear(giris_boyutu, n)
        self.proj_v = nn.Linear(giris_boyutu, n)

    def forward(self, duz: torch.Tensor) -> torch.Tensor:
        # duz: (B, giris_boyutu) → (B, n, n)
        u = self.proj_u(duz)
        v = self.proj_v(duz)
        # Dış çarpım: X[b, i, j] = u[b, i] * v[b, j]  → n² sanal köşe
        return torch.tanh(torch.einsum("bi,bj->bij", u, v))

    def kapasite(self) -> dict:
        return {
            "n": self.n,
            "sanal_kose": self.n ** 2,
            "gercek_parametre": sum(p.numel() for p in self.parameters()),
        }
