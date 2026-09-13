# -*- coding: utf-8 -*-
"""Bağımsız istatistiksel çıkarım katmanı: CI, etki büyüklüğü, eşleşmiş test.

`mean ± std` bir **betimleyicidir**, kanıt değildir. İki kol arasındaki farkın
"gerçek" olup olmadığını söylemek için üç ayrı şey gerekir:

1. **Belirsizlik aralığı** — ortalamanın kendisi ne kadar oynak?
   (`bootstrap_ci`, percentile bootstrap)
2. **Etki büyüklüğü** — fark pratikte ne kadar büyük?
   (`cohens_d_paired` / Hedges düzeltmeli `g`)
3. **Anlamlılık** — bu fark yalnız şans eseri çıkmış olabilir mi?
   (`paired_permutation_test`, `wilcoxon_signed_rank`)

## Dürüstlük notu: 5 seed ile neyi ölçebilirsiniz?

Bu repo varsayılan olarak 5 seed kullanır. Eşleşmiş **işaret-çevirme
(sign-flip) permütasyon testinde** n çift için mümkün permütasyon sayısı
``2^n``'dir; iki yönlü en küçük erişilebilir p-değeri ``2 / 2^n``'dir:

| n (seed) | permütasyon | mümkün en küçük iki yönlü p |
|---:|---:|---:|
| 5  | 32   | 0.0625 |
| 6  | 64   | 0.03125 |
| 8  | 256  | 0.0078 |
| 10 | 1024 | 0.00195 |

Yani **5 seed ile p < 0.05 elde etmek matematiksel olarak imkânsızdır** —
sonuç ne kadar tutarlı olursa olsun. Bu modül bunu gizlemez;
``minimum_achievable_p_value`` alanını her raporda döndürür. "p anlamlı
çıkmadı" ile "bu tasarımla p anlamlı çıkamaz" ayrı şeylerdir.

Tüm fonksiyonlar saf stdlib'dir (numpy/scipy gerekmez) ve bootstrap
deterministiktir: aynı girdi + aynı ``seed`` → aynı çıktı.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Any, Dict, List, Sequence

# Sign-flip permütasyonu bu eşiğe kadar TAM (exact) sayılır; üstünde
# deterministik Monte Carlo örneklemesine düşülür.
EXACT_PERMUTATION_LIMIT = 18          # 2^18 = 262.144
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_MONTE_CARLO_SAMPLES = 20_000


def _validate(values: Sequence[float], minimum: int = 2) -> List[float]:
    data = [float(value) for value in values]
    if len(data) < minimum:
        raise ValueError(f"En az {minimum} gözlem gerekli (verilen: {len(data)})")
    if any(math.isnan(value) or math.isinf(value) for value in data):
        raise ValueError("NaN/Inf içeren gözlem istatistiksel olarak işlenemez")
    return data


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _sample_std(values: Sequence[float]) -> float:
    """Örneklem standart sapması (ddof=1) — çıkarım için doğru olan budur."""
    if len(values) < 2:
        return 0.0
    ortalama = _mean(values)
    return math.sqrt(sum((value - ortalama) ** 2 for value in values) / (len(values) - 1))


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Doğrusal interpolasyonlu yüzdelik (numpy 'linear' ile uyumlu)."""
    if not sorted_values:
        raise ValueError("Boş dizide yüzdelik hesaplanamaz")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    konum = fraction * (len(sorted_values) - 1)
    alt = math.floor(konum)
    ust = math.ceil(konum)
    if alt == ust:
        return float(sorted_values[int(konum)])
    agirlik = konum - alt
    return float(sorted_values[alt] * (1 - agirlik) + sorted_values[ust] * agirlik)


@dataclass
class ConfidenceInterval:
    """Percentile bootstrap güven aralığı."""

    statistic: str
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    resamples: int
    sample_size: int
    seed: int
    method: str = "percentile-bootstrap"
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def bootstrap_ci(
    values: Sequence[float],
    confidence_level: float = 0.95,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 12345,
    statistic: str = "mean",
) -> ConfidenceInterval:
    """Ortalamanın percentile bootstrap güven aralığı (deterministik).

    Küçük örneklemde bootstrap **dağılımı genişletmez**: gözlenmeyen değerler
    yeniden örneklemede de belirmez. n=5 için aralık gerçek belirsizliği
    olduğundan dar gösterebilir; bu sınır ``limitations`` içinde raporlanır.
    """
    data = _validate(values, minimum=2)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level (0,1) aralığında olmalı")
    if resamples < 1000:
        raise ValueError("Kararlı yüzdelik için en az 1000 yeniden örnekleme gerekli")

    rng = random.Random(seed)
    n = len(data)
    dagilim: List[float] = []
    for _ in range(int(resamples)):
        ornek = [data[rng.randrange(n)] for _ in range(n)]
        dagilim.append(_mean(ornek))
    dagilim.sort()

    alfa = 1.0 - confidence_level
    alt = _percentile(dagilim, alfa / 2.0)
    ust = _percentile(dagilim, 1.0 - alfa / 2.0)

    sinirlar = [
        "Percentile bootstrap; küçük n'de gerçek kapsama nominal seviyenin altında kalabilir.",
        "Aralık yalnız SEED VARYANSINI tanımlar; dataset/protokol yanlılığını kapsamaz.",
    ]
    if n < 10:
        sinirlar.append(
            f"n={n} küçük: bootstrap yalnız gözlenen {n} değeri yeniden örnekler, "
            "aralık yanıltıcı biçimde dar olabilir."
        )
    return ConfidenceInterval(
        statistic=str(statistic),
        point_estimate=round(_mean(data), 12),
        lower=round(alt, 12),
        upper=round(ust, 12),
        confidence_level=float(confidence_level),
        resamples=int(resamples),
        sample_size=n,
        seed=int(seed),
        limitations=sinirlar,
    )


@dataclass
class EffectSize:
    """Eşleşmiş (paired) etki büyüklüğü."""

    measure: str
    value: float
    hedges_g: float
    mean_difference: float
    std_difference: float
    sample_size: int
    magnitude: str
    interpretation: str
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _magnitude(value: float) -> str:
    """Cohen'in geleneksel eşikleri — evrensel değil, yalnız kaba etikettir."""
    buyukluk = abs(value)
    if buyukluk < 0.2:
        return "negligible"
    if buyukluk < 0.5:
        return "small"
    if buyukluk < 0.8:
        return "medium"
    return "large"


def cohens_d_paired(
    treatment: Sequence[float],
    baseline: Sequence[float],
) -> EffectSize:
    """Eşleşmiş Cohen's d_z ve küçük örneklem düzeltmeli Hedges' g.

    ``d_z = mean(fark) / sd(fark)``. Aynı seed'lerde çalışan iki kol
    eşleşmiştir; eşleşmeyi yok sayıp bağımsız-örneklem d'si kullanmak
    varyansı şişirir ve etkiyi küçük gösterir.
    """
    if len(treatment) != len(baseline):
        raise ValueError("Eşleşmiş etki büyüklüğü için diziler aynı uzunlukta olmalı")
    islem = _validate(treatment, minimum=2)
    kontrol = _validate(baseline, minimum=2)

    farklar = [a - b for a, b in zip(islem, kontrol)]
    ortalama_fark = _mean(farklar)
    sapma = _sample_std(farklar)
    n = len(farklar)

    if sapma == 0.0:
        # Tüm seed'lerde fark aynı: d tanımsız (0/0 veya ∞).
        d = 0.0 if ortalama_fark == 0.0 else math.inf * (1 if ortalama_fark > 0 else -1)
    else:
        d = ortalama_fark / sapma

    # Hedges düzeltmesi (df = n-1); küçük örneklemde d yukarı yanlıdır.
    df = n - 1
    j = 1.0 - (3.0 / (4.0 * df - 1.0)) if df > 1 else 1.0
    g = d * j if math.isfinite(d) else d

    if math.isfinite(d):
        buyukluk = _magnitude(g)
        yorum = (
            f"Eşleşmiş fark ortalaması {ortalama_fark:.6g}, d_z={d:.4f}, "
            f"g={g:.4f} ({buyukluk})."
        )
    else:
        buyukluk = "undefined"
        yorum = (
            f"Tüm {n} eşleşmede fark sabit ({ortalama_fark:.6g}); "
            "varyans sıfır olduğu için standardize etki büyüklüğü tanımsız."
        )

    return EffectSize(
        measure="cohens_d_z_paired",
        value=(round(d, 12) if math.isfinite(d) else d),
        hedges_g=(round(g, 12) if math.isfinite(g) else g),
        mean_difference=round(ortalama_fark, 12),
        std_difference=round(sapma, 12),
        sample_size=n,
        magnitude=buyukluk,
        interpretation=yorum,
        limitations=[
            "Cohen eşikleri (0.2/0.5/0.8) sözleşmedir; alan-bağımsız gerçek anlam taşımaz.",
            "Etki büyüklüğü YÖN ve BÜYÜKLÜK verir; nedensellik veya genellenebilirlik vermez.",
        ],
    )


@dataclass
class SignificanceTest:
    """Eşleşmiş anlamlılık testi sonucu."""

    test: str
    statistic: float
    p_value: float
    alternative: str
    sample_size: int
    exact: bool
    permutations: int
    minimum_achievable_p_value: float
    significant_at_0_05: bool
    underpowered_by_design: bool
    interpretation: str
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def minimum_two_sided_p(n: int) -> float:
    """n çift için sign-flip permütasyon testinin ulaşabileceği en küçük p."""
    if n < 1:
        raise ValueError("n >= 1 olmalı")
    return min(1.0, 2.0 / (2.0 ** n))


def paired_permutation_test(
    treatment: Sequence[float],
    baseline: Sequence[float],
    alternative: str = "two-sided",
    seed: int = 12345,
    monte_carlo_samples: int = DEFAULT_MONTE_CARLO_SAMPLES,
) -> SignificanceTest:
    """Eşleşmiş işaret-çevirme permütasyon testi.

    Sıfır hipotezi: fark dağılımı 0 etrafında simetriktir (etki yok). Her
    eşleşmenin işareti bağımsız çevrilebilir; ``2^n`` olası atama üzerinden
    gözlenen ortalama farkın ne kadar uç olduğu hesaplanır.

    ``n <= EXACT_PERMUTATION_LIMIT`` iken TAM sayım yapılır (p kesindir);
    üstünde deterministik Monte Carlo kullanılır ve ``exact=False`` döner.
    """
    if len(treatment) != len(baseline):
        raise ValueError("Eşleşmiş test için diziler aynı uzunlukta olmalı")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative: 'two-sided' | 'greater' | 'less'")
    islem = _validate(treatment, minimum=2)
    kontrol = _validate(baseline, minimum=2)

    farklar = [a - b for a, b in zip(islem, kontrol)]
    n = len(farklar)
    gozlenen = _mean(farklar)
    exact = n <= EXACT_PERMUTATION_LIMIT

    def _en_az_kadar_uc(aday: float) -> bool:
        if alternative == "two-sided":
            # Kayan nokta gürültüsüne karşı küçük tolerans.
            return abs(aday) >= abs(gozlenen) - 1e-15
        if alternative == "greater":
            return aday >= gozlenen - 1e-15
        return aday <= gozlenen + 1e-15

    if exact:
        toplam = 0
        uc = 0
        for isaretler in product((1.0, -1.0), repeat=n):
            aday = _mean([isaret * fark for isaret, fark in zip(isaretler, farklar)])
            toplam += 1
            if _en_az_kadar_uc(aday):
                uc += 1
        p_value = uc / toplam
        kullanilan = toplam
    else:
        rng = random.Random(seed)
        kullanilan = int(monte_carlo_samples)
        uc = 0
        for _ in range(kullanilan):
            aday = _mean([fark if rng.random() < 0.5 else -fark for fark in farklar])
            if _en_az_kadar_uc(aday):
                uc += 1
        # (uc + 1) / (m + 1): Monte Carlo'da p=0 raporlamayı önleyen standart düzeltme.
        p_value = (uc + 1) / (kullanilan + 1)

    minimum_p = minimum_two_sided_p(n) if alternative == "two-sided" else 1.0 / (2.0 ** n)
    yetersiz_guc = minimum_p > 0.05

    if yetersiz_guc:
        yorum = (
            f"n={n} ile ulaşılabilecek en küçük p={minimum_p:.5f} > 0.05. "
            "Bu tasarımda anlamlılık İDDİA EDİLEMEZ; p değeri yalnız tanımlayıcıdır. "
            f"p<0.05 için en az {math.ceil(math.log2(2 / 0.05))} eşleşmiş seed gerekir."
        )
    elif p_value < 0.05:
        yorum = (
            f"p={p_value:.5f} < 0.05: gözlenen fark, sıfır hipotezi altında "
            f"{n} eşleşmede bu kadar uç çıkmazdı."
        )
    else:
        yorum = (
            f"p={p_value:.5f} >= 0.05: gözlenen fark şans eseri açıklanabilir; "
            "etki yok demek DEĞİLDİR (kanıt yokluğu, yokluk kanıtı değildir)."
        )

    sinirlar = [
        "Sıfır hipotezi fark dağılımının 0 etrafında simetrik olmasıdır.",
        "p-değeri etki BÜYÜKLÜĞÜ değildir; effect size ile birlikte okunmalıdır.",
        "Seed varyansı dışındaki (dataset, protokol, uygulama) yanlılıkları kapsamaz.",
    ]
    if not exact:
        sinirlar.append(
            f"n={n} > {EXACT_PERMUTATION_LIMIT}: p deterministik Monte Carlo ile "
            f"{kullanilan} örnekten tahmin edildi, tam sayım değildir."
        )
    if yetersiz_guc:
        sinirlar.append(
            "TASARIM GEREĞİ YETERSİZ GÜÇ: seed sayısı artırılmadan p<0.05 imkânsızdır."
        )

    return SignificanceTest(
        test=("exact-paired-sign-flip-permutation" if exact
              else "monte-carlo-paired-sign-flip-permutation"),
        statistic=round(gozlenen, 12),
        p_value=round(p_value, 12),
        alternative=alternative,
        sample_size=n,
        exact=exact,
        permutations=kullanilan,
        minimum_achievable_p_value=round(minimum_p, 12),
        significant_at_0_05=bool(p_value < 0.05),
        underpowered_by_design=bool(yetersiz_guc),
        interpretation=yorum,
        limitations=sinirlar,
    )


def wilcoxon_signed_rank(
    treatment: Sequence[float],
    baseline: Sequence[float],
    alternative: str = "two-sided",
) -> SignificanceTest:
    """Eşleşmiş Wilcoxon işaretli sıra testi (sıfır farklar atılır).

    Ortalama yerine sıralara dayandığı için aykırı değerlere daha dayanıklıdır.
    Sıfır farklar (tam eşitlik) klasik Wilcoxon'da atılır; bu, etkin örneklem
    boyutunu düşürür ve raporlanır.
    """
    if len(treatment) != len(baseline):
        raise ValueError("Eşleşmiş test için diziler aynı uzunlukta olmalı")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative: 'two-sided' | 'greater' | 'less'")
    islem = _validate(treatment, minimum=2)
    kontrol = _validate(baseline, minimum=2)

    ham_farklar = [a - b for a, b in zip(islem, kontrol)]
    farklar = [fark for fark in ham_farklar if fark != 0.0]
    atilan = len(ham_farklar) - len(farklar)
    n = len(farklar)

    if n == 0:
        return SignificanceTest(
            test="wilcoxon-signed-rank",
            statistic=0.0, p_value=1.0, alternative=alternative,
            sample_size=0, exact=True, permutations=0,
            minimum_achievable_p_value=1.0,
            significant_at_0_05=False, underpowered_by_design=True,
            interpretation=("Tüm eşleşmelerde fark tam sıfır; test tanımsız, "
                            "p=1.0 olarak raporlanır."),
            limitations=["Sıfır farklar atıldığı için etkin örneklem 0'a düştü."],
        )

    # Mutlak farklara ortalama-sıra (tie) ataması.
    sirali = sorted(range(n), key=lambda i: abs(farklar[i]))
    siralar = [0.0] * n
    konum = 0
    while konum < n:
        son = konum
        while (son + 1 < n
               and abs(farklar[sirali[son + 1]]) == abs(farklar[sirali[konum]])):
            son += 1
        ortalama_sira = (konum + son + 2) / 2.0   # 1-tabanlı sıraların ortalaması
        for indeks in range(konum, son + 1):
            siralar[sirali[indeks]] = ortalama_sira
        konum = son + 1

    w_arti = sum(sira for fark, sira in zip(farklar, siralar) if fark > 0)
    w_eksi = sum(sira for fark, sira in zip(farklar, siralar) if fark < 0)

    # n küçükken sıra toplamının TAM dağılımı sign-flip ile sayılabilir.
    exact = n <= EXACT_PERMUTATION_LIMIT
    if exact:
        toplam = 0
        uc = 0
        for isaretler in product((1.0, -1.0), repeat=n):
            aday_arti = sum(sira for isaret, sira in zip(isaretler, siralar) if isaret > 0)
            toplam += 1
            if alternative == "two-sided":
                merkez = sum(siralar) / 2.0
                if abs(aday_arti - merkez) >= abs(w_arti - merkez) - 1e-12:
                    uc += 1
            elif alternative == "greater":
                if aday_arti >= w_arti - 1e-12:
                    uc += 1
            else:
                if aday_arti <= w_arti + 1e-12:
                    uc += 1
        p_value = uc / toplam
        kullanilan = toplam
    else:
        # Normal yaklaşım (süreklilik düzeltmeli, tie düzeltmeli).
        beklenen = n * (n + 1) / 4.0
        tie_gruplari: Dict[float, int] = {}
        for sira in siralar:
            tie_gruplari[sira] = tie_gruplari.get(sira, 0) + 1
        tie_duzeltme = sum(adet ** 3 - adet for adet in tie_gruplari.values()) / 48.0
        varyans = n * (n + 1) * (2 * n + 1) / 24.0 - tie_duzeltme
        if varyans <= 0:
            p_value = 1.0
        else:
            z = (w_arti - beklenen) / math.sqrt(varyans)
            kuyruk = 0.5 * math.erfc(abs(z) / math.sqrt(2.0))
            if alternative == "two-sided":
                p_value = min(1.0, 2.0 * kuyruk)
            elif alternative == "greater":
                p_value = kuyruk if z > 0 else 1.0 - kuyruk
            else:
                p_value = kuyruk if z < 0 else 1.0 - kuyruk
        kullanilan = 0

    minimum_p = minimum_two_sided_p(n) if alternative == "two-sided" else 1.0 / (2.0 ** n)
    yetersiz_guc = minimum_p > 0.05

    sinirlar = [
        "Wilcoxon sıralara dayanır; etki büyüklüğünü DEĞİL, sıralama tutarlılığını sınar.",
        "p-değeri tek başına bilimsel kanıt değildir; CI ve effect size ile okunmalıdır.",
    ]
    if atilan:
        sinirlar.append(
            f"{atilan} eşleşmede fark tam sıfırdı ve klasik Wilcoxon gereği atıldı; "
            f"etkin örneklem {n}."
        )
    if yetersiz_guc:
        sinirlar.append(
            "TASARIM GEREĞİ YETERSİZ GÜÇ: bu etkin örneklemle p<0.05 imkânsızdır."
        )
    if not exact:
        sinirlar.append("Normal yaklaşım kullanıldı (tie ve süreklilik düzeltmeli).")

    return SignificanceTest(
        test=("exact-wilcoxon-signed-rank" if exact
              else "normal-approximation-wilcoxon-signed-rank"),
        statistic=round(min(w_arti, w_eksi), 12),
        p_value=round(p_value, 12),
        alternative=alternative,
        sample_size=n,
        exact=exact,
        permutations=kullanilan,
        minimum_achievable_p_value=round(minimum_p, 12),
        significant_at_0_05=bool(p_value < 0.05),
        underpowered_by_design=bool(yetersiz_guc),
        interpretation=(
            f"W={min(w_arti, w_eksi):.4g}, p={p_value:.5f}"
            + (" (tasarım gereği anlamlılık iddia edilemez)" if yetersiz_guc else "")
        ),
        limitations=sinirlar,
    )


@dataclass
class ComparisonReport:
    """İki kolun tam istatistiksel karşılaştırması."""

    metric: str
    treatment_label: str
    baseline_label: str
    treatment_mean: float
    baseline_mean: float
    treatment_ci: Dict[str, Any]
    baseline_ci: Dict[str, Any]
    difference_ci: Dict[str, Any]
    effect_size: Dict[str, Any]
    permutation_test: Dict[str, Any]
    wilcoxon_test: Dict[str, Any]
    verdict: str
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compare_paired(
    metric: str,
    treatment: Sequence[float],
    baseline: Sequence[float],
    treatment_label: str = "treatment",
    baseline_label: str = "baseline",
    confidence_level: float = 0.95,
    seed: int = 12345,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> ComparisonReport:
    """Tek çağrıda CI + etki büyüklüğü + iki eşleşmiş test üretir.

    ``verdict`` asla yalnız p-değerine bakmaz: farkın güven aralığı sıfırı
    içeriyorsa ya da tasarım yetersiz güçlüyse bunu açıkça söyler.
    """
    if len(treatment) != len(baseline):
        raise ValueError("Eşleşmiş karşılaştırma için diziler aynı uzunlukta olmalı")
    islem = _validate(treatment, minimum=2)
    kontrol = _validate(baseline, minimum=2)
    farklar = [a - b for a, b in zip(islem, kontrol)]

    islem_ci = bootstrap_ci(islem, confidence_level, resamples, seed, f"{metric}::treatment")
    kontrol_ci = bootstrap_ci(kontrol, confidence_level, resamples, seed + 1,
                              f"{metric}::baseline")
    fark_ci = bootstrap_ci(farklar, confidence_level, resamples, seed + 2,
                           f"{metric}::difference")
    etki = cohens_d_paired(islem, kontrol)
    permutasyon = paired_permutation_test(islem, kontrol, "two-sided", seed)
    wilcoxon = wilcoxon_signed_rank(islem, kontrol, "two-sided")

    ci_sifiri_iceriyor = fark_ci.lower <= 0.0 <= fark_ci.upper
    yetersiz_guc = permutasyon.underpowered_by_design

    if yetersiz_guc:
        karar = (
            f"YETERSİZ GÜÇ: n={len(farklar)} ile p<0.05 ulaşılamaz "
            f"(en küçük p={permutasyon.minimum_achievable_p_value:.5f}). "
            f"Fark ortalaması {etki.mean_difference:.6g}, "
            f"%{confidence_level * 100:.0f} CI [{fark_ci.lower:.6g}, {fark_ci.upper:.6g}]"
            + (" — sıfırı içeriyor." if ci_sifiri_iceriyor else " — sıfırı içermiyor.")
            + " Anlamlılık İDDİA EDİLMEZ; sonuç betimleyicidir."
        )
    elif permutasyon.significant_at_0_05 and not ci_sifiri_iceriyor:
        karar = (
            f"AYRIŞMA: p={permutasyon.p_value:.5f}, g={etki.hedges_g:.4f} "
            f"({etki.magnitude}), CI [{fark_ci.lower:.6g}, {fark_ci.upper:.6g}] sıfırı içermiyor."
        )
    elif ci_sifiri_iceriyor:
        karar = (
            f"AYRIM YOK: fark CI'si [{fark_ci.lower:.6g}, {fark_ci.upper:.6g}] "
            "sıfırı içeriyor; kollar ayrıştırılamadı."
        )
    else:
        karar = (
            f"KARARSIZ: p={permutasyon.p_value:.5f} eşiği geçmedi ama CI sıfırı içermiyor; "
            "daha fazla seed gerekir."
        )

    return ComparisonReport(
        metric=str(metric),
        treatment_label=str(treatment_label),
        baseline_label=str(baseline_label),
        treatment_mean=round(_mean(islem), 12),
        baseline_mean=round(_mean(kontrol), 12),
        treatment_ci=islem_ci.to_dict(),
        baseline_ci=kontrol_ci.to_dict(),
        difference_ci=fark_ci.to_dict(),
        effect_size=etki.to_dict(),
        permutation_test=permutasyon.to_dict(),
        wilcoxon_test=wilcoxon.to_dict(),
        verdict=karar,
        limitations=[
            "Tüm çıkarım YALNIZ seed varyansı üzerinedir; dataset ve protokol sabittir.",
            "Çoklu metrik karşılaştırılıyorsa çoklu test düzeltmesi uygulanmamıştır.",
            "İstatistiksel ayrışma, gerçek dünya genellemesi anlamına gelmez.",
        ],
    )


def summarize_seed_metric(
    values: Sequence[float],
    confidence_level: float = 0.95,
    seed: int = 12345,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> Dict[str, Any]:
    """Tek kolun seed dağılımı için mean/std/CI özeti (karşılaştırmasız)."""
    data = _validate(values, minimum=2)
    aralik = bootstrap_ci(data, confidence_level, resamples, seed)
    return {
        "n": len(data),
        "mean": round(_mean(data), 12),
        "std_sample": round(_sample_std(data), 12),
        "min": min(data),
        "max": max(data),
        "ci_lower": aralik.lower,
        "ci_upper": aralik.upper,
        "confidence_level": float(confidence_level),
        "method": aralik.method,
        "seed": int(seed),
    }


__all__ = [
    "ComparisonReport",
    "ConfidenceInterval",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "EXACT_PERMUTATION_LIMIT",
    "EffectSize",
    "SignificanceTest",
    "bootstrap_ci",
    "cohens_d_paired",
    "compare_paired",
    "minimum_two_sided_p",
    "paired_permutation_test",
    "summarize_seed_metric",
    "wilcoxon_signed_rank",
]
