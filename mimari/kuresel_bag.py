# -*- coding: utf-8 -*-
"""
Küresel Bağ — Gerçek Bilinear (A @ X @ B) Katmanı ve K Katmanlı Zinciri
=======================================================================

Projenin "geometrik" kimliğinin kalbi: bir (n, n) boyutlu ara temsili İKİ
taraftan matris çarpımıyla dönüştüren gerçek "matris sandviçi" (bilinear)
katmanı ve bu katmanların artık bağlantılı zinciri.

Matematik:
    Y = A @ X @ B        (X: (B, n, n) girdi, A ve B: (n, n) "mercek" matrisleri)

Flatten edilmiş uzayda bu işlem  Y_vec = (Bᵀ ⊗ A) X_vec  doğrusal dönüşümüdür;
yani katmanın TEMSİL ETTİĞİ tam operatör iki merceğin Kronecker çarpımıdır ve
(n², n²) boyutludur → n⁴ sanal operatör girdisi. Bu devasa operatörün tüm
girdilerini ayrı parametre olarak depolamak yerine yalnızca 2n² gerçek
parametre tutulur. README'deki "Kronecker İllüzyonu" tam olarak budur.

DİKKAT — dürüst kapasite notu:
    n⁴ "sanal bağ" sayısı GERÇEK EĞİTİLEBİLİR PARAMETRE sayısı değildir.
    Katmanın öğrenebileceği fonksiyon ailesi 2n² gerçek serbestlik derecesiyle
    sınırlıdır (düşük-rank / Kronecker kısıtı). Tam sayımlar için
    `kuresel_model.kapasite_raporu()`'na bakın.

Geçmiş notu: Bu dosyanın eski sürümündeki `OptimizeEdilmisKureselBag` sınıfı
(iki bağımsız nn.Linear + model tarafından çöpe atılan `v_yeni` çıktısı ve hiç
gradyan almayan `mercek_B`) bilinçli olarak KALDIRILDI: matris çarpımı
içermeyen o sürüm, projenin geometrik iddiasını taşımıyordu (rapor 8.2).
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _grad_checkpoint


class KureselBagKatmani(nn.Module):
    """Tek bir bilinear "matris sandviçi" katmanı: X ↦ A @ X @ B.

    - Girdi / çıktı: (batch, n, n)
    - Gerçek parametre: 2n²  (mercek_A + mercek_B — İKİSİ DE eğitilir)
    - Temsil edilen tam operatör: Bᵀ ⊗ A → (n², n²) → n⁴ sanal girdi
    - Başlatma: Xavier → std = 1/√n → varyans ileri geçişte korunur.
    """

    def __init__(self, n: int):
        super().__init__()
        self.n = n
        self.mercek_A = nn.Parameter(torch.empty(n, n))
        self.mercek_B = nn.Parameter(torch.empty(n, n))
        nn.init.xavier_uniform_(self.mercek_A)
        nn.init.xavier_uniform_(self.mercek_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n, n)  →  A @ X @ B
        # einsum, PyTorch'un kontraksiyon optimizasyonundan yararlanır
        # (bak: rapor 8.2'de önerilen implementasyon).
        return torch.einsum("ij,bjk,kl->bil", self.mercek_A, x, self.mercek_B)

    def kapasite(self) -> dict:
        return {
            "n": self.n,
            "gercek_parametre": 2 * self.n ** 2,
            "sanal_operator_girdisi": self.n ** 4,  # (Bᵀ ⊗ A) operatörünün boyutu
        }


class KureselZincir(nn.Module):
    """K adet bilinear küresel bağ katmanının Pre-Norm + artık bağlantılı zinciri.

    Her katman:
        X ← X + Dropout(SiLU( Katman_i( LayerNorm(X) ) ))

    Kapasite matematiği (rapor 8.3):
    - Gerçek parametre: K × 2n²            → katman sayısıyla DOĞRUSAL büyür
    - Katman başına temsil edilen Kronecker operatörü: n⁴ sanal girdi
    - README anlatısındaki "geometrik büyüme" (katman başına ~n² çarpan)
      zincir boyunca n^(2K) etkileşim uzayı üst sınırı verir.
      Örnek: n=256, K=4 → 256^8 ≈ 1,8×10¹⁹ → katrilyonun (10¹⁵) üzerinde.
      BU BİR ÜST SINIR/TEMsil ölçüsüdür; öğrenilebilir serbestlik yine
      K × 2n² gerçek parametredir.

    Kararlılık (rapor 8.3: "aralarına SiLU/LayerNorm koymayı unutma"):
    Pre-Norm + artık bağlantı derin zincirde gradyan patlamasını/kaybolmasını
    önler. `checkpoint_kullan=True` ile gradient checkpointing açılır
    (rapor 8.4.2): ara aktivasyonlar bellekte tutulmak yerine geri yayılımda
    yeniden hesaplanır.
    """

    def __init__(self, n: int, katman_sayisi: int = 4, dropout: float = 0.0,
                 checkpoint_kullan: bool = False):
        super().__init__()
        if katman_sayisi < 1:
            raise ValueError("katman_sayisi en az 1 olmalı")
        self.n = n
        self.katman_sayisi = katman_sayisi
        self.checkpoint_kullan = checkpoint_kullan

        self.katmanlar = nn.ModuleList(
            [KureselBagKatmani(n) for _ in range(katman_sayisi)]
        )
        self.normlar = nn.ModuleList(
            [nn.LayerNorm(n) for _ in range(katman_sayisi)]
        )
        self.aktivasyon = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for katman, norm in zip(self.katmanlar, self.normlar):
            h = norm(x)
            if self.checkpoint_kullan and self.training and torch.is_grad_enabled():
                h = _grad_checkpoint(katman, h, use_reentrant=False)
            else:
                h = katman(h)
            x = x + self.dropout(self.aktivasyon(h))
        return x

    def kapasite(self) -> dict:
        return {
            "n": self.n,
            "katman_sayisi": self.katman_sayisi,
            "gercek_parametre": sum(p.numel() for p in self.parameters()),
            "katman_basi_sanal_operator": self.n ** 4,
            "etkilesim_uzayi_ust_siniri": self.n ** (2 * self.katman_sayisi),
        }
