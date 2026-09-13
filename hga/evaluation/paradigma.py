# -*- coding: utf-8 -*-
"""
Faz 21 — Neural-only vs Symbolic-only vs Hybrid Ablasyonu
===========================================================

Projenin en büyük açıklarından biri şuydu: "hibrit mimari" iddiası hiçbir zaman
**kontrollü** olarak test edilmemişti. Bu modül üç kolu **aynı veri, aynı
split, aynı metrikler** altında karşılaştırır:

* ``symbolic``  — yalnız kural/bilgi tabanı. Öğrenme yok. Özellik bilinmiyorsa
  ÇEKİMSER kalır (``None``), tahmin uydurmaz.
* ``neural``    — yalnız öğrenilmiş gömme + MLP. Kural bilgisi yok; varlık
  kimliklerinden istatistik öğrenir. Görülmemiş varlıkta çuvallaması beklenir.
* ``hybrid``    — sembolik kesin konuştuğunda sembolik kazanır (veto),
  çekimser kaldığında nöral doldurur.

Görev, sentetik ama **prosedürel** bir doğruluk görevidir: bir ``(özne, ilişki,
nesne)`` üçlüsü, ilişkinin gerektirdiği özellikleri özne ve nesne taşıyorsa
doğrudur. Zorluk üç kaynaktan gelir:

1. **Gizli özellik**: varlıkların bir kısmının özellikleri bilgi tabanında
   YOKTUR. Sembolik kol burada çekimser kalmak zorundadır — mükemmel skorun
   neden sahte olduğunu gösteren kısım budur.
2. **Görülmemiş varlık (cold-start)**: test setinin bir bölümü eğitimde hiç
   geçmemiş varlıklar içerir. Nöral kol burada gömme öğrenememiştir.
3. **Dengeli etiket**: %50 doğru / %50 yanlış → şans seviyesi 0.5.

Dürüstlük kuralları (kullanıcı kısıtları):
* FAR tek başına rapor edilmez; FRR/Precision/Recall/F1 birlikte verilir.
* Her koşu 5 seed ile ``mean ± std`` olarak özetlenir.
* Çekimserlik gizlenmez: ``coverage`` ayrı raporlanır, çekimser cevap doğru
  sayılmaz.
* Nöral kol torch gerektirir; torch yoksa açık hata verilir, sessizce
  atlanmaz.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

KOLLAR = ("symbolic", "neural", "hybrid")

# ── Görev üretimi ────────────────────────────────────────────────────────────

OZELLIK_SAYISI = 6


@dataclass
class Ornek:
    subject_id: str
    relation_id: str
    object_id: str
    label: int                 # 1 = doğru, 0 = yanlış
    unseen_entity: bool        # test anında görülmemiş varlık içeriyor mu?
    symbolic_decidable: bool   # bilgi tabanı bu örneği çözebilir mi?

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Gorev:
    """Prosedürel görev: varlıklar, ilişkiler, bilinen özellikler ve split'ler."""

    entity_features: Dict[str, Tuple[int, ...]]          # gerçek (oracle) özellikler
    known_features: Dict[str, Tuple[Optional[int], ...]]  # bilgi tabanının gördüğü
    relation_requirements: Dict[str, Tuple[int, ...]]     # gerekli özellik indeksleri
    train: List[Ornek]
    test: List[Ornek]

    def ozet(self) -> Dict[str, Any]:
        return {
            "entities": len(self.entity_features),
            "relations": len(self.relation_requirements),
            "train": len(self.train),
            "test": len(self.test),
            "test_unseen": sum(o.unseen_entity for o in self.test),
            "test_symbolic_decidable": sum(o.symbolic_decidable for o in self.test),
        }


def gorev_uret(
    entity_count: int = 120,
    relation_count: int = 8,
    train_size: int = 1200,
    test_size: int = 400,
    hidden_feature_ratio: float = 0.30,
    unseen_entity_ratio: float = 0.20,
    seed: int = 1,
) -> Gorev:
    """Üç kolun da aynı şekilde göreceği dengeli, prosedürel görev üret."""
    if not 0.0 <= hidden_feature_ratio <= 1.0:
        raise ValueError("hidden_feature_ratio [0,1] olmalı")
    if not 0.0 <= unseen_entity_ratio < 1.0:
        raise ValueError("unseen_entity_ratio [0,1) olmalı")
    if entity_count < 10 or relation_count < 1:
        raise ValueError("entity_count >= 10 ve relation_count >= 1 olmalı")

    rng = random.Random(seed)
    varliklar = [f"E{i:04d}" for i in range(entity_count)]
    entity_features = {
        e: tuple(rng.randint(0, 1) for _ in range(OZELLIK_SAYISI)) for e in varliklar
    }
    # Bazı varlıkların özellikleri bilgi tabanında YOK (gizli).
    gizli_sayi = int(round(entity_count * hidden_feature_ratio))
    gizliler = set(rng.sample(varliklar, gizli_sayi))
    known_features: Dict[str, Tuple[Optional[int], ...]] = {
        e: (tuple([None] * OZELLIK_SAYISI) if e in gizliler else entity_features[e])
        for e in varliklar
    }
    relation_requirements = {
        f"R{i:02d}": tuple(sorted(rng.sample(range(OZELLIK_SAYISI), rng.randint(1, 2))))
        for i in range(relation_count)
    }

    # Görülmemiş varlıklar yalnız test'te geçer (cold-start).
    unseen_sayi = int(round(entity_count * unseen_entity_ratio))
    gorulmemisler = set(rng.sample(varliklar, unseen_sayi))
    gorulenler = [e for e in varliklar if e not in gorulmemisler]
    if len(gorulenler) < 2:
        raise ValueError("unseen_entity_ratio çok yüksek: eğitimde varlık kalmadı")

    def dogru_mu(s: str, r: str, o: str) -> int:
        gerekli = relation_requirements[r]
        return int(all(entity_features[s][i] == 1 for i in gerekli)
                   and all(entity_features[o][i] == 1 for i in gerekli))

    def sembolik_cozulur_mu(s: str, r: str, o: str) -> bool:
        gerekli = relation_requirements[r]
        return all(known_features[s][i] is not None for i in gerekli) and \
            all(known_features[o][i] is not None for i in gerekli)

    def ornek_uret(havuz: Sequence[str], hedef_etiket: int,
                   unseen: bool) -> Optional[Ornek]:
        for _ in range(200):
            s, o = rng.choice(havuz), rng.choice(havuz)
            r = rng.choice(list(relation_requirements))
            if dogru_mu(s, r, o) == hedef_etiket:
                return Ornek(s, r, o, hedef_etiket, unseen,
                             sembolik_cozulur_mu(s, r, o))
        return None

    def set_uret(n: int, havuz: Sequence[str], unseen: bool) -> List[Ornek]:
        cikti: List[Ornek] = []
        for index in range(n):
            ornek = ornek_uret(havuz, index % 2, unseen)
            if ornek is not None:
                cikti.append(ornek)
        rng.shuffle(cikti)
        return cikti

    train = set_uret(train_size, gorulenler, unseen=False)
    unseen_test = int(round(test_size * unseen_entity_ratio))
    test = set_uret(test_size - unseen_test, gorulenler, unseen=False)
    if gorulmemisler:
        test += set_uret(unseen_test, sorted(gorulmemisler), unseen=True)
    rng.shuffle(test)

    return Gorev(entity_features, known_features, relation_requirements, train, test)


# ── Metrikler ────────────────────────────────────────────────────────────────

@dataclass
class KolMetrikleri:
    arm: str
    total: int
    answered: int
    abstained: int
    coverage: float
    correct: int
    accuracy: float                # çekimser = yanlış (tüm set üzerinden)
    accuracy_on_answered: float    # yalnız cevaplananlar
    precision: float
    recall: float
    f1: float
    far: float                     # yanlış kabul oranı (FP / gerçek negatif)
    frr: float                     # yanlış ret oranı (FN / gerçek pozitif)
    accuracy_seen: float
    accuracy_unseen: float
    accuracy_symbolic_decidable: float
    accuracy_symbolic_undecidable: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _oran(pay: int, payda: int) -> float:
    return round(pay / payda, 6) if payda else 0.0


def _metrik_hesapla(arm: str, ornekler: Sequence[Ornek],
                    tahminler: Sequence[Optional[int]]) -> KolMetrikleri:
    tp = tn = fp = fn = 0
    cevaplanan = dogru = 0
    for ornek, tahmin in zip(ornekler, tahminler):
        if tahmin is None:
            continue
        cevaplanan += 1
        if tahmin == 1 and ornek.label == 1:
            tp += 1
            dogru += 1
        elif tahmin == 0 and ornek.label == 0:
            tn += 1
            dogru += 1
        elif tahmin == 1 and ornek.label == 0:
            fp += 1
        else:
            fn += 1

    def alt_dogruluk(secici) -> float:
        alt = [(o, t) for o, t in zip(ornekler, tahminler) if secici(o)]
        if not alt:
            return 0.0
        return _oran(sum(1 for o, t in alt if t is not None and t == o.label), len(alt))

    toplam = len(ornekler)
    return KolMetrikleri(
        arm=arm,
        total=toplam,
        answered=cevaplanan,
        abstained=toplam - cevaplanan,
        coverage=_oran(cevaplanan, toplam),
        correct=dogru,
        accuracy=_oran(dogru, toplam),
        accuracy_on_answered=_oran(dogru, cevaplanan),
        precision=_oran(tp, tp + fp),
        recall=_oran(tp, tp + fn),
        f1=round(2 * tp / (2 * tp + fp + fn), 6) if (2 * tp + fp + fn) else 0.0,
        far=_oran(fp, fp + tn),
        frr=_oran(fn, fn + tp),
        accuracy_seen=alt_dogruluk(lambda o: not o.unseen_entity),
        accuracy_unseen=alt_dogruluk(lambda o: o.unseen_entity),
        accuracy_symbolic_decidable=alt_dogruluk(lambda o: o.symbolic_decidable),
        accuracy_symbolic_undecidable=alt_dogruluk(lambda o: not o.symbolic_decidable),
    )


# ── Kollar ───────────────────────────────────────────────────────────────────

def sembolik_tahmin(gorev: Gorev, ornekler: Sequence[Ornek]) -> List[Optional[int]]:
    """Yalnız kural: özellik biliniyorsa kesin karar, bilinmiyorsa çekimser."""
    tahminler: List[Optional[int]] = []
    for ornek in ornekler:
        gerekli = gorev.relation_requirements[ornek.relation_id]
        s = gorev.known_features[ornek.subject_id]
        o = gorev.known_features[ornek.object_id]
        degerler = [s[i] for i in gerekli] + [o[i] for i in gerekli]
        if any(d is None for d in degerler):
            tahminler.append(None)              # kanıt yok → uydurmuyoruz
        else:
            tahminler.append(int(all(d == 1 for d in degerler)))
    return tahminler


def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "Faz 21 nöral kolu için PyTorch gerekli (pip install -e .). "
            "Sembolik kol torch'suz çalışır."
        ) from exc


def noral_tahmin(
    gorev: Gorev,
    ornekler: Sequence[Ornek],
    seed: int = 1,
    embedding_dim: int = 16,
    hidden_dim: int = 32,
    epochs: int = 60,
    lr: float = 0.05,
) -> Tuple[List[Optional[int]], Dict[str, Any]]:
    """Yalnız öğrenme: varlık/ilişki gömmeleri + MLP. Kural bilgisi YOK."""
    torch = _torch()
    torch.manual_seed(seed)

    varliklar = sorted(gorev.entity_features)
    iliskiler = sorted(gorev.relation_requirements)
    e_index = {e: i for i, e in enumerate(varliklar)}
    r_index = {r: i for i, r in enumerate(iliskiler)}

    e_emb = torch.nn.Embedding(len(varliklar), embedding_dim)
    r_emb = torch.nn.Embedding(len(iliskiler), embedding_dim)
    model = torch.nn.Sequential(
        torch.nn.Linear(embedding_dim * 3, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 1),
    )
    parametreler = list(e_emb.parameters()) + list(r_emb.parameters()) + \
        list(model.parameters())
    optim = torch.optim.Adam(parametreler, lr=lr)
    kayip_fn = torch.nn.BCEWithLogitsLoss()

    def toplu(ornek_listesi):
        s = torch.tensor([e_index[o.subject_id] for o in ornek_listesi])
        r = torch.tensor([r_index[o.relation_id] for o in ornek_listesi])
        ob = torch.tensor([e_index[o.object_id] for o in ornek_listesi])
        y = torch.tensor([float(o.label) for o in ornek_listesi])
        return s, r, ob, y

    s_tr, r_tr, o_tr, y_tr = toplu(gorev.train)
    son_kayip = float("nan")
    for _ in range(int(epochs)):
        optim.zero_grad()
        girdi = torch.cat([e_emb(s_tr), r_emb(r_tr), e_emb(o_tr)], dim=1)
        kayip = kayip_fn(model(girdi).squeeze(1), y_tr)
        kayip.backward()
        optim.step()
        son_kayip = float(kayip.item())

    with torch.no_grad():
        s_te, r_te, o_te, _ = toplu(ornekler)
        girdi = torch.cat([e_emb(s_te), r_emb(r_te), e_emb(o_te)], dim=1)
        olasilik = torch.sigmoid(model(girdi).squeeze(1))
        tahminler = [int(p >= 0.5) for p in olasilik.tolist()]
        guvenler = [abs(float(p) - 0.5) * 2 for p in olasilik.tolist()]

    # Eğitim doğruluğu: ezberleme ile genellemeyi ayırmak için raporlanır.
    with torch.no_grad():
        girdi_tr = torch.cat([e_emb(s_tr), r_emb(r_tr), e_emb(o_tr)], dim=1)
        egitim_dogruluk = float(
            ((torch.sigmoid(model(girdi_tr).squeeze(1)) >= 0.5).float() == y_tr)
            .float().mean().item()
        )
    return list(tahminler), {"train_accuracy": round(egitim_dogruluk, 6),
                             "final_loss": round(son_kayip, 6),
                             "confidences": guvenler}


def hibrit_tahmin(
    sembolik: Sequence[Optional[int]],
    noral: Sequence[Optional[int]],
) -> List[Optional[int]]:
    """Sembolik kesin konuşuyorsa o kazanır (veto); yoksa nöral doldurur."""
    return [s if s is not None else n for s, n in zip(sembolik, noral)]


# ── Deney ────────────────────────────────────────────────────────────────────

@dataclass
class ParadigmaRaporu:
    seed: int
    task: Dict[str, Any]
    arms: Dict[str, Dict[str, Any]]
    neural_diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_paradigm_ablation(
    entity_count: int = 120,
    relation_count: int = 8,
    train_size: int = 1200,
    test_size: int = 400,
    hidden_feature_ratio: float = 0.30,
    unseen_entity_ratio: float = 0.20,
    seed: int = 1,
    epochs: int = 60,
) -> ParadigmaRaporu:
    """Üç kolu tek seed'de aynı görev üzerinde koştur."""
    gorev = gorev_uret(
        entity_count=entity_count, relation_count=relation_count,
        train_size=train_size, test_size=test_size,
        hidden_feature_ratio=hidden_feature_ratio,
        unseen_entity_ratio=unseen_entity_ratio, seed=seed,
    )
    sembolik = sembolik_tahmin(gorev, gorev.test)
    noral, teshis = noral_tahmin(gorev, gorev.test, seed=seed, epochs=epochs)
    hibrit = hibrit_tahmin(sembolik, noral)

    kollar = {
        "symbolic": _metrik_hesapla("symbolic", gorev.test, sembolik).to_dict(),
        "neural": _metrik_hesapla("neural", gorev.test, noral).to_dict(),
        "hybrid": _metrik_hesapla("hybrid", gorev.test, hibrit).to_dict(),
    }
    teshis.pop("confidences", None)
    return ParadigmaRaporu(seed=int(seed), task=gorev.ozet(), arms=kollar,
                           neural_diagnostics=teshis)


@dataclass
class ParadigmaSweepRaporu:
    """5 seed mean ± std özeti (kullanıcı kısıtı: her benchmark 5 seed)."""

    seeds: List[int]
    task: Dict[str, Any]
    summary: Dict[str, Dict[str, Dict[str, float]]]
    neural_train_accuracy: Dict[str, float] = field(default_factory=dict)
    reports: List[Dict[str, Any]] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self, metrikler: Sequence[str] = (
        "accuracy", "coverage", "precision", "recall", "f1", "far", "frr",
        "accuracy_seen", "accuracy_unseen",
    )) -> str:
        basliklar = {
            "accuracy": "Accuracy", "coverage": "Coverage",
            "precision": "Precision", "recall": "Recall", "f1": "F1",
            "far": "FAR", "frr": "FRR",
            "accuracy_seen": "Acc (görülen)", "accuracy_unseen": "Acc (görülmemiş)",
        }
        satirlar = ["| Metrik | " + " | ".join(KOLLAR) + " |",
                    "|---|" + "---:|" * len(KOLLAR)]
        for metrik in metrikler:
            hucreler = []
            for kol in KOLLAR:
                istat = self.summary[kol][metrik]
                hucreler.append(f"{istat['mean']:.3f} ± {istat['std']:.3f}")
            satirlar.append(f"| {basliklar.get(metrik, metrik)} | "
                            + " | ".join(hucreler) + " |")
        return "\n".join(satirlar)


def _mean_std(degerler: Sequence[float]) -> Dict[str, float]:
    n = len(degerler)
    ortalama = sum(degerler) / n if n else 0.0
    varyans = sum((d - ortalama) ** 2 for d in degerler) / n if n else 0.0
    return {"mean": round(ortalama, 6), "std": round(math.sqrt(varyans), 6),
            "min": round(min(degerler), 6) if n else 0.0,
            "max": round(max(degerler), 6) if n else 0.0}


def run_paradigm_sweep(
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    **kwargs: Any,
) -> ParadigmaSweepRaporu:
    """Faz 21'in ana girişi: 5 seed, üç kol, mean ± std."""
    kwargs.pop("seed", None)
    raporlar = [run_paradigm_ablation(seed=int(s), **kwargs) for s in seeds]
    sayisal = [k for k, v in raporlar[0].arms["symbolic"].items()
               if isinstance(v, (int, float))]
    ozet = {
        kol: {metrik: _mean_std([float(r.arms[kol][metrik]) for r in raporlar])
              for metrik in sayisal}
        for kol in KOLLAR
    }

    sym, neu, hyb = ozet["symbolic"], ozet["neural"], ozet["hybrid"]
    bulgular = [
        f"Sembolik kol cevapladığı örneklerde çok güçlü "
        f"(accuracy_on_answered={sym['accuracy_on_answered']['mean']:.3f}) ama "
        f"coverage={sym['coverage']['mean']:.3f}: gizli özellikli örneklerde "
        f"çekimser kalır. Tek başına 'yüksek doğruluk' iddiası bu yüzden "
        f"kapsam belirtilmeden anlamsızdır.",
        f"Nöral kol tam kapsam verir (coverage={neu['coverage']['mean']:.3f}) "
        f"ama doğruluğu düşüktür (accuracy={neu['accuracy']['mean']:.3f}) ve "
        f"görülmemiş varlıklarda "
        f"{neu['accuracy_unseen']['mean']:.3f}'e çöker "
        f"(görülen: {neu['accuracy_seen']['mean']:.3f}).",
        f"Hibrit kol her iki zaafı da kapatır: "
        f"accuracy={hyb['accuracy']['mean']:.3f} ± {hyb['accuracy']['std']:.3f}, "
        f"coverage={hyb['coverage']['mean']:.3f}.",
        f"Hibritin sembolik üzerine net kazancı "
        f"{hyb['accuracy']['mean'] - sym['accuracy']['mean']:+.3f}, "
        f"nöral üzerine {hyb['accuracy']['mean'] - neu['accuracy']['mean']:+.3f} "
        f"accuracy puanıdır.",
        f"FAR tek başına raporlanmaz: hibrit FAR={hyb['far']['mean']:.3f}, "
        f"FRR={hyb['frr']['mean']:.3f}, F1={hyb['f1']['mean']:.3f}.",
    ]
    egitim_dog = _mean_std([float(r.neural_diagnostics["train_accuracy"])
                            for r in raporlar])
    bulgular.append(
        f"Nöral kol eğitim doğruluğu {egitim_dog['mean']:.3f}, test doğruluğu "
        f"{neu['accuracy']['mean']:.3f} → genelleme açığı "
        f"{egitim_dog['mean'] - neu['accuracy']['mean']:+.3f}. Görülmemiş "
        f"varlıkta {neu['accuracy_unseen']['mean']:.3f} ≈ şans (0.5): kural "
        f"öğrenmiyor, varlık kimliği ezberliyor."
    )
    return ParadigmaSweepRaporu(
        seeds=[int(s) for s in seeds],
        task=raporlar[0].task,
        summary=ozet,
        neural_train_accuracy=egitim_dog,
        reports=[r.to_dict() for r in raporlar],
        findings=bulgular,
    )


__all__ = [
    "KOLLAR", "Ornek", "Gorev", "KolMetrikleri", "ParadigmaRaporu",
    "ParadigmaSweepRaporu", "gorev_uret", "sembolik_tahmin", "noral_tahmin",
    "hibrit_tahmin", "run_paradigm_ablation", "run_paradigm_sweep",
]
