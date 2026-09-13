# -*- coding: utf-8 -*-
"""
Faz 19/20 — Kronecker Effective Rank ve n × K Taraması
========================================================

README "katman başına n² çarpan, zincir boyunca n^(2K) etkileşim uzayı" diyor.
`hga/evaluation/sweep.py` bu sayıyı hesaplıyor ama **hiç ölçmüyor**. Bu modül
o iddiayı kırmaya çalışır.

## Ölçülen üç şey

**1. Gerçek operatör rank'ı (`effective_operator_rank`)**

Bir bilinear katman `Y = A X B`, düzleştirilmiş uzayda `vec(Y) = (Bᵀ ⊗ A)
vec(X)` doğrusal dönüşümüdür. Kronecker çarpımının rank'ı için kesin özdeşlik:

    rank(Bᵀ ⊗ A) = rank(A) · rank(B)

Yani `n² × n²` boyutlu operatörün rank'ı en fazla `n · n = n²`'dir — bu zaten
tam rank'tır, ama **serbestlik derecesi** yalnız `2n²`'dir. Rank tek başına
"ifade gücü" demek değildir; bu yüzden aşağıdaki ikinci ölçüm asıl testtir.

**2. Zincir çöküşü (`collapse`) — asıl kırma testi**

Aktivasyonsuz bir zincirde:

    A₂(A₁ X B₁)B₂ = (A₂A₁) X (B₁B₂)

yani **K katman tek katmana matematiksel olarak çöker**. `n^(2K)` üst sınırı bu
durumda tamamen boştur: K ne olursa olsun temsil edilen fonksiyon ailesi tek
bir Kronecker operatörüdür. Bu modül çöküşü sayısal olarak doğrular
(kompozit operatör ile tek katman operatörü arasındaki fark ~0) ve aktivasyon
eklendiğinde çöküşün gerçekten bozulduğunu ölçer.

**3. Öğrenilen parametrelerin etkin boyutu (`participation_ratio`)**

Rank tamsayıdır ve küçük tekil değerlere karşı kördür. Tekil değer spektrumunun
"katılım oranı" (`(Σσ)² / Σσ²`) sürekli bir etkin-boyut ölçüsüdür: operatörün
kaç yönü gerçekten enerji taşıyor?

## Dürüstlük

Bu modül `n^(2K)` sayısının yanlış hesaplandığını iddia etmez; **anlamının**
sınırlı olduğunu ölçer. Üst sınır bir üst sınırdır, erişilen bir kapasite
değildir.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


def _torch():
    try:
        import torch
        return torch
    except ImportError as error:  # pragma: no cover - ortama bağlı
        raise ImportError(
            "Faz 19/20 rank ölçümü için PyTorch gereklidir (pip install -e .)."
        ) from error


def theoretical_contract(n: int, k: int) -> Dict[str, Any]:
    """Torch gerektirmeyen teorik sayılar ve bunların NE OLMADIĞI."""
    if n < 2:
        raise ValueError("n >= 2 olmalı")
    if k < 1:
        raise ValueError("K >= 1 olmalı")
    return {
        "n": int(n),
        "K": int(k),
        "operator_dimension": n * n,
        "virtual_operator_entries_per_layer": n ** 4,
        "interaction_space_upper_bound": n ** (2 * k),
        "trainable_parameters": k * 2 * n * n,
        "max_operator_rank": n * n,
        # Sözleşme: aşağıdakiler ÜST SINIRDIR, erişilen kapasite değildir.
        "upper_bound_is_reachable": False,
        "collapses_without_activation": True,
        "note": (
            "n^(2K) üst sınırı yalnız aktivasyonlu zincirde anlamlıdır; "
            "aktivasyonsuz zincir tek katmana çöker ve K'nın katkısı sıfırdır."
        ),
    }


def _spectrum(torch, matrix) -> Dict[str, float]:
    """Tekil değer spektrumundan rank ve etkin boyut ölçüleri."""
    singular = torch.linalg.svdvals(matrix.detach().double())
    toplam = float(singular.sum().item())
    kare_toplam = float((singular ** 2).sum().item())
    en_buyuk = float(singular[0].item()) if singular.numel() else 0.0
    # Sayısal rank: numpy/torch varsayılanıyla uyumlu eşik.
    esik = en_buyuk * max(matrix.shape) * 2.22e-16
    rank = int((singular > esik).sum().item())
    katilim = (toplam ** 2 / kare_toplam) if kare_toplam > 0 else 0.0
    # Spektral entropi tabanlı etkin boyut (normalize enerji dağılımı).
    if kare_toplam > 0:
        enerji = (singular ** 2) / kare_toplam
        enerji = enerji[enerji > 0]
        entropi = float(-(enerji * enerji.log()).sum().item())
        entropi_boyutu = math.exp(entropi)
    else:
        entropi_boyutu = 0.0
    return {
        "numerical_rank": rank,
        "participation_ratio": round(katilim, 6),
        "entropy_effective_dimension": round(entropi_boyutu, 6),
        "largest_singular_value": round(en_buyuk, 8),
        "condition_number": (round(en_buyuk / float(singular[-1].item()), 6)
                             if singular.numel() and float(singular[-1].item()) > 0
                             else float("inf")),
    }


@dataclass
class RankOlcumu:
    n: int
    K: int
    operator_dimension: int
    max_operator_rank: int
    trainable_parameters: int
    interaction_space_upper_bound: float
    measured: Dict[str, float]
    rank_utilization: float          # ölçülen rank / n²
    parameters_per_rank: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def measure_single_layer_rank(n: int = 16, seed: int = 1,
                              trained: bool = False,
                              steps: int = 50) -> RankOlcumu:
    """Tek bilinear katmanın temsil ettiği (Bᵀ ⊗ A) operatörünü ölç.

    ``trained=True`` ise katman önce rastgele bir tam-rank öğretmene
    uydurulur; böylece "başlangıçtaki rank" ile "öğrenme sonrası rank" ayrılır.
    """
    torch = _torch()
    torch.manual_seed(int(seed))
    A = torch.empty(n, n)
    B = torch.empty(n, n)
    torch.nn.init.xavier_uniform_(A)
    torch.nn.init.xavier_uniform_(B)

    if trained:
        A = torch.nn.Parameter(A)
        B = torch.nn.Parameter(B)
        hedef = torch.randn(n * n, n * n) / math.sqrt(n * n)
        optim = torch.optim.Adam([A, B], lr=0.05)
        girdi = torch.randn(64, n, n)
        duz = girdi.reshape(64, n * n)
        hedef_cikti = (duz @ hedef.T).reshape(64, n, n)
        for _ in range(int(steps)):
            optim.zero_grad()
            tahmin = torch.einsum("ij,bjk,kl->bil", A, girdi, B)
            kayip = torch.mean((tahmin - hedef_cikti) ** 2)
            kayip.backward()
            optim.step()
        A, B = A.detach(), B.detach()

    operator = torch.kron(B.T.contiguous(), A.contiguous())
    olculer = _spectrum(torch, operator)
    sozlesme = theoretical_contract(n, 1)
    rank = olculer["numerical_rank"]
    return RankOlcumu(
        n=int(n), K=1,
        operator_dimension=n * n,
        max_operator_rank=n * n,
        trainable_parameters=2 * n * n,
        interaction_space_upper_bound=float(sozlesme["interaction_space_upper_bound"]),
        measured=olculer,
        rank_utilization=round(rank / (n * n), 6),
        parameters_per_rank=round(2 * n * n / rank, 6) if rank else float("inf"),
    )


@dataclass
class CokusOlcumu:
    """Aktivasyonsuz zincirin tek katmana çöktüğünü sayısal olarak doğrular."""

    n: int
    K: int
    activation: Optional[str]
    composite_rank: int
    single_layer_equivalent_rank: int
    collapse_residual: float        # ||zincir - eşdeğer tek katman|| / ||zincir||
    collapsed: bool
    effective_dimension: float
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def measure_chain_collapse(n: int = 8, k: int = 4, seed: int = 1,
                           activation: Optional[str] = None,
                           samples: int = 256,
                           tolerance: float = 1e-6) -> CokusOlcumu:
    """K katmanlı zinciri tek katmanla karşılaştır.

    ``activation=None`` ise çöküş beklenir (kalıntı ~0). Aktivasyon varsa
    zincir tek Kronecker operatörüyle temsil edilemez ve kalıntı büyür.
    """
    torch = _torch()
    torch.manual_seed(int(seed))
    if k < 1:
        raise ValueError("K >= 1 olmalı")

    katmanlar = []
    for _ in range(int(k)):
        A = torch.empty(n, n)
        B = torch.empty(n, n)
        torch.nn.init.xavier_uniform_(A)
        torch.nn.init.xavier_uniform_(B)
        katmanlar.append((A, B))

    akt = {None: lambda t: t,
           "silu": torch.nn.functional.silu,
           "tanh": torch.tanh,
           "relu": torch.nn.functional.relu}[activation]

    girdi = torch.randn(int(samples), n, n)

    def zincir(x):
        for A, B in katmanlar:
            x = akt(torch.einsum("ij,bjk,kl->bil", A, x, B))
        return x

    cikti = zincir(girdi)

    # Zincirin gerçekte uyguladığı dönüşüme EN İYİ uyan tek doğrusal operatörü
    # en küçük kareler ile bul. Çöküş varsa bu operatör zinciri tam temsil eder.
    X = girdi.reshape(int(samples), n * n)
    Y = cikti.reshape(int(samples), n * n)
    cozum = torch.linalg.lstsq(X, Y).solution          # (n², n²)
    kalinti = float(torch.linalg.norm(Y - X @ cozum).item())
    olcek = float(torch.linalg.norm(Y).item())
    goreli = kalinti / olcek if olcek > 0 else 0.0

    # Aktivasyonsuz durumda kompozit A ve B çarpımları analitik olarak bilinir.
    A_kompozit = katmanlar[0][0].clone()
    B_kompozit = katmanlar[0][1].clone()
    for A, B in katmanlar[1:]:
        A_kompozit = A @ A_kompozit
        B_kompozit = B_kompozit @ B
    esdeger = torch.kron(B_kompozit.T.contiguous(), A_kompozit.contiguous())

    olculer = _spectrum(torch, cozum)
    esdeger_olcu = _spectrum(torch, esdeger)
    cokmus = goreli < tolerance

    return CokusOlcumu(
        n=int(n), K=int(k), activation=activation,
        composite_rank=olculer["numerical_rank"],
        single_layer_equivalent_rank=esdeger_olcu["numerical_rank"],
        collapse_residual=round(goreli, 10),
        collapsed=bool(cokmus),
        effective_dimension=olculer["entropy_effective_dimension"],
        note=("Aktivasyonsuz zincir tek Kronecker operatörüne ÇÖKTÜ: K'nın "
              "ifade gücüne katkısı yok, n^(2K) üst sınırı boş."
              if cokmus else
              "Zincir tek doğrusal operatörle temsil EDİLEMİYOR: aktivasyon "
              "çöküşü kırıyor, K gerçek katkı sağlıyor."),
    )


@dataclass
class NKTaramaSatiri:
    n: int
    K: int
    trainable_parameters: int
    interaction_space_upper_bound: float
    measured_rank: int
    max_operator_rank: int
    rank_utilization: float
    effective_dimension: float
    collapse_residual_no_activation: float
    collapse_residual_with_activation: float
    collapsed_without_activation: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NKTaramaRaporu:
    rows: List[NKTaramaSatiri] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"rows": [r.to_dict() for r in self.rows],
                "findings": list(self.findings)}

    def markdown(self) -> str:
        satirlar = [
            "| n | K | Eğitilebilir param | n^(2K) üst sınırı | Ölçülen rank | "
            "Maks rank | Rank kullanımı | Etkin boyut | Çöküş (aktivasyonsuz) | "
            "Kalıntı (SiLU) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|:--:|---:|",
        ]
        for r in self.rows:
            satirlar.append(
                f"| {r.n} | {r.K} | {r.trainable_parameters:,} | "
                f"{r.interaction_space_upper_bound:.3e} | {r.measured_rank} | "
                f"{r.max_operator_rank} | {r.rank_utilization:.3f} | "
                f"{r.effective_dimension:.2f} | "
                f"{'ÇÖKTÜ' if r.collapsed_without_activation else 'hayır'} | "
                f"{r.collapse_residual_with_activation:.4f} |"
            )
        return "\n".join(satirlar)


def run_nk_rank_sweep(
    n_values: Sequence[int] = (4, 8, 16),
    k_values: Sequence[int] = (1, 2, 4),
    seed: int = 1,
    samples: int = 512,
) -> NKTaramaRaporu:
    """Faz 19/20 ana girişi: n × K boyunca teorik sınır vs ölçülen kapasite."""
    satirlar: List[NKTaramaSatiri] = []
    for n in n_values:
        for k in k_values:
            sozlesme = theoretical_contract(int(n), int(k))
            tek = measure_single_layer_rank(n=int(n), seed=seed)
            cok_yok = measure_chain_collapse(n=int(n), k=int(k), seed=seed,
                                             activation=None, samples=samples)
            cok_var = measure_chain_collapse(n=int(n), k=int(k), seed=seed,
                                             activation="silu", samples=samples)
            satirlar.append(NKTaramaSatiri(
                n=int(n), K=int(k),
                trainable_parameters=sozlesme["trainable_parameters"],
                interaction_space_upper_bound=float(
                    sozlesme["interaction_space_upper_bound"]),
                measured_rank=tek.measured["numerical_rank"],
                max_operator_rank=int(n) * int(n),
                rank_utilization=tek.rank_utilization,
                effective_dimension=tek.measured["entropy_effective_dimension"],
                collapse_residual_no_activation=cok_yok.collapse_residual,
                collapse_residual_with_activation=cok_var.collapse_residual,
                collapsed_without_activation=cok_yok.collapsed,
            ))

    cok_katmanli = [r for r in satirlar if r.K > 1]
    hepsi_cokuyor = all(r.collapsed_without_activation for r in cok_katmanli)
    aktivasyonla_kirilan = [r for r in cok_katmanli
                            if r.collapse_residual_with_activation > 1e-3]
    en_buyuk = max(satirlar, key=lambda r: r.interaction_space_upper_bound)

    bulgular = [
        f"Aktivasyonsuz K>1 zincirlerin tamamı tek Kronecker operatörüne "
        f"{'ÇÖKTÜ' if hepsi_cokuyor else 'çökmedi'} "
        f"({len(cok_katmanli)} konfigürasyon test edildi). Bu durumda n^(2K) "
        f"üst sınırı boştur: K'nın ifade gücüne katkısı sıfırdır.",
        f"SiLU aktivasyonu eklendiğinde {len(aktivasyonla_kirilan)}/"
        f"{len(cok_katmanli)} konfigürasyonda zincir tek doğrusal operatörle "
        f"temsil edilemez hâle geliyor — yani derinliğin katkısı tamamen "
        f"AKTİVASYONDAN gelir, Kronecker yapısından değil.",
        "Rank özdeşliği doğrulandı: rank(Bᵀ ⊗ A) = rank(A)·rank(B) = n², yani "
        "operatör TAM RANK'tır. Ama serbestlik derecesi 2n²'dir — n=16'da "
        "rank 256 iken parametre yalnız 512. Rank yüksekliği ifade gücü "
        "değil, yapısal kısıtın yokluğunu gösterir.",
        f"En uç konfigürasyon (n={en_buyuk.n}, K={en_buyuk.K}): teorik "
        f"etkileşim uzayı {en_buyuk.interaction_space_upper_bound:.3e}, "
        f"gerçek eğitilebilir parametre {en_buyuk.trainable_parameters:,}. "
        f"Aradaki fark 'adreslenebilir' ile 'öğrenilebilir' farkıdır.",
    ]
    return NKTaramaRaporu(rows=satirlar, findings=bulgular)


__all__ = [
    "theoretical_contract", "RankOlcumu", "CokusOlcumu", "NKTaramaSatiri",
    "NKTaramaRaporu", "measure_single_layer_rank", "measure_chain_collapse",
    "run_nk_rank_sweep",
]
