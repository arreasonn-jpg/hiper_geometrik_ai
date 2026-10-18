# -*- coding: utf-8 -*-
"""
Faz 3/6 — Ölçeklendirilmiş Golden Benchmark (100 / 1.000 / 10.000)
====================================================================

`golden_dataset/` elle sabitlenmiş 8 kayıt içerir. Bu, regresyon koruması
olarak değerlidir ama **istatistiksel bir iddia taşıyamaz**: 5 test örneğinde
FAR=0 görmek, gürültüden ayırt edilemez.

Bu modül aynı epistemik sözleşmeyi (`VALID` / `INVALID` / `UNCERTAIN` /
`CONFLICT`) koruyarak veri setini **prosedürel olarak** 100, 1.000, 10.000
örneğe ölçekler.

## Ground truth nereden geliyor? (tautoloji riski ve kaçınma)

En büyük tehlike, beklenen etiketi evaluator'ın karar ağacını taklit ederek
üretmektir — o zaman test kendini doğrular. Burada bunun yerine **inşa yöntemi
ground truth'tur**: her örnek, önceden seçilmiş bir hedef sınıfın tanımına göre
kurulur ve bilgi tabanı ona göre doldurulur.

| Hedef sınıf | Nasıl inşa edilir |
|---|---|
| `VALID` | İlişkinin gerektirdiği tüm özellikler bilinir, yüksek güvenli ve doğrudur. Tip kısıtları uyar. |
| `INVALID` | Gerekli bir özellik bilinir, yüksek güvenlidir ve **yanlıştır**; ya da nesne tipi izinli değildir. |
| `UNCERTAIN` | Gerekli bir özellik bilgi tabanında **yoktur** ya da güveni eşiğin altındadır. |
| `CONFLICT` | Yapısal kurallar olumlu der ama kayıtlı yüksek güvenli kanıt olumsuzdur (ya da tersi). |

Yani üretici "evaluator ne der?" diye sormaz; "kanıt durumu nedir?" diye sorar.
Evaluator'ın o kanıt durumundan doğru epistemik sınıfı çıkarması **ölçülen
şeydir**, varsayılan değil.

## Zor mod (`hard=True`)

Kolay modda doğruluk 1.000 çıkar. Bu evaluator'ın gücünü değil, görevin
kolaylığını ölçer — ve kullanıcı ilkesi gereği (**sistemi kırmaya çalışan
ölçüm**) tek başına yayımlanamaz. Zor mod, karar ağacının gerçekten ayrım
yapması gereken sınır vakalarını ekler:

* **Eşik sınırı**: güven tam `0.5` (belirsizlik eşiğinin tam üstü) ve özellik
  değeri tam `0.5` — "yanlış mı, belirsiz mi?" ayrımı.
* **Kısıt önceliği**: aynı ilişkide bir özellik BİLİNMİYOR, diğeri bilinen ve
  YANLIŞ. Doğru cevap `INVALID`'dir (kesin ihlal, eksik kanıta baskın gelir),
  ama sıra bağımlı bir uygulama `UNCERTAIN` döndürebilir.
* **Kaynak çatışması**: aynı özelliğe düşük güvenli `MODEL_GENERATED` ve
  yüksek güvenli `REAL_DATA` yazılır; yüksek güvenli gerçek veri kazanmalıdır.
* **Zayıf kanıt**: kayıtlı kanıt olumsuz ama güveni çelişki eşiğinin ALTINDA —
  bu `CONFLICT` değildir, `VALID` kalmalıdır.

## Sızıntı

Her ölçekte `audit_partitions` ile train/test semantik parmak izi denetimi
yapılır ve rapora yazılır. Test üçlüleri bilgi tabanına olgu olarak asla
yazılmaz.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from hga.experience.evaluator import ExperienceEvaluator
from hga.knowledge import ExperienceCandidate, KaynakTuru, KnowledgeStore

from .golden import GoldenMetrics, _metrics
from .leakage import audit_partitions

SINIFLAR = ("VALID", "INVALID", "UNCERTAIN", "CONFLICT")

# Evaluator eşikleriyle uyumlu sabitler (bkz. experience/evaluator.py)
YUKSEK_GUVEN = 0.95
DUSUK_GUVEN = 0.2          # belirsiz_guven_esik = 0.5 → altında kalmalı
CELISKI_GUVENI = 0.9       # celiski_kanit_esik = 0.6 → üstünde olmalı

VARLIK_TIPLERI = ("canli", "nesne", "kavram")


@dataclass
class OlceklenmisKayit:
    experience_id: str
    subject_id: str
    relation_id: str
    object_id: str
    expected_state: str
    construction: str          # hangi mekanizmayla kuruldu (izlenebilirlik)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OlceklenmisVeriSeti:
    size: int
    seed: int
    store: KnowledgeStore
    train_records: List[Dict[str, Any]]
    test_records: List[OlceklenmisKayit]
    class_counts: Dict[str, int]

    def dataset_hash(self) -> str:
        ham = json.dumps(
            {"train": self.train_records,
             "test": [r.to_dict() for r in self.test_records]},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(ham).hexdigest()


ZOR_SINIFLAR = (
    "esik_siniri_belirsiz",
    "kisit_onceligi_invalid",
    "kaynak_catismasi_invalid",
    "zayif_kanit_valid",
)


def veri_seti_uret(size: int = 100, seed: int = 1,
                   train_ratio: float = 1.0,
                   hard: bool = False) -> OlceklenmisVeriSeti:
    """``size`` test örneği içeren, dört sınıfa dengeli dağılmış veri seti kur.

    ``train_ratio`` test başına kaç train olgusu yazılacağını belirler; bu
    olgular test üçlülerinden **farklıdır** (sızıntı denetimiyle doğrulanır).

    ``hard=True`` ise örneklerin yarısı sınır vakalarından seçilir (bkz. modül
    docstring'i).
    """
    size = int(size)
    if size < len(SINIFLAR):
        raise ValueError(f"size >= {len(SINIFLAR)} olmalı")
    rng = random.Random(seed)
    store = KnowledgeStore()

    # İlişkiler: her biri bir özellik gerektirir; biri tip kısıtlı.
    iliskiler = []
    for i in range(4):
        rid = f"SR_{i:02d}"
        ozellik = f"ozellik_{i}"
        store.iliski_tanimla(
            f"iliski{i}", relation_id=rid,
            subject_types=["canli"] if i == 0 else None,
            object_types=["nesne"] if i == 1 else None,
            requires_object_props={ozellik: 1.0},
            source=KaynakTuru.REAL_DATA,
        )
        iliskiler.append((rid, ozellik))

    test_kayitlari: List[OlceklenmisKayit] = []
    train_kayitlari: List[Dict[str, Any]] = []
    sinif_sayilari = {ad: 0 for ad in SINIFLAR}

    for index in range(size):
        # Zor modda 8'lik döngü: 0-3 temel epistemik sınıflar (CONFLICT dâhil
        # korunur), 4-7 sınır vakaları. Tek/çift ayrımı CONFLICT'i tamamen
        # yutuyordu.
        zor_vaka = bool(hard and index % 8 >= 4)
        hedef = SINIFLAR[index % len(SINIFLAR)]
        rid, ozellik = iliskiler[index % len(iliskiler)]
        # Her örnek kendi taze varlıklarını kullanır → örnekler bağımsız.
        s_id = f"SE_S_{index:06d}"
        o_id = f"SE_O_{index:06d}"
        store.varlik_ekle(f"ozne{index}", entity_type="canli", entity_id=s_id)

        if zor_vaka:
            zor = ZOR_SINIFLAR[index % len(ZOR_SINIFLAR)]
            store.varlik_ekle(f"nesne{index}", entity_type="nesne", entity_id=o_id)
            if zor == "esik_siniri_belirsiz":
                # Güven tam eşikte (0.5): belirsiz_guven_esik karşılaştırması
                # "<" olduğu için bu değer YETERLİ sayılır; değer 0.5 ise
                # _ozellik_yanlis_mi(hedef=1.0) False döner → VALID.
                store.ozellik_koy(o_id, ozellik, 0.5, source=KaynakTuru.REAL_DATA,
                                  confidence=0.5)
                hedef, insa = "VALID", (
                    "sınır: güven tam eşikte (0.5) ve değer tam 0.5 — "
                    "ihlal sayılmamalı")
            elif zor == "kisit_onceligi_invalid":
                # İki gerekli özellik: biri BİLİNMİYOR, biri bilinen ve YANLIŞ.
                # Kesin ihlal, eksik kanıta baskın gelmeli → INVALID.
                # ÖNEMLİ: paylaşılan ilişkiyi değiştirmek sonraki örnekleri
                # kirletir; bu vakaya ÖZEL bir ilişki tanımlanır.
                rid = f"SR_HARD_{index:06d}"
                ikinci = f"{ozellik}_ek"
                store.iliski_tanimla(
                    f"zoriliski{index}", relation_id=rid,
                    requires_object_props={ozellik: 1.0, ikinci: 1.0},
                    source=KaynakTuru.REAL_DATA,
                )
                # ozellik bilinmiyor (hiç yazılmadı), ikinci bilinen ve yanlış
                store.ozellik_koy(o_id, ikinci, 0.0, source=KaynakTuru.REAL_DATA,
                                  confidence=YUKSEK_GUVEN)
                hedef, insa = "INVALID", (
                    "biri bilinmeyen biri kesin yanlış iki kısıt — kesin ihlal "
                    "baskın gelmeli")
            elif zor == "kaynak_catismasi_invalid":
                # Aynı özelliğe önce düşük güvenli model çıktısı, sonra yüksek
                # güvenli gerçek veri (yanlış) yazılır; gerçek veri kazanmalı.
                store.ozellik_koy(o_id, ozellik, 1.0,
                                  source=KaynakTuru.MODEL_GENERATED,
                                  confidence=DUSUK_GUVEN)
                store.ozellik_koy(o_id, ozellik, 0.0, source=KaynakTuru.REAL_DATA,
                                  confidence=YUKSEK_GUVEN)
                hedef, insa = "INVALID", (
                    "kaynak çatışması: yüksek güvenli REAL_DATA düşük güvenli "
                    "MODEL_GENERATED'ı yenmeli")
            else:  # zayif_kanit_valid
                store.ozellik_koy(o_id, ozellik, 1.0, source=KaynakTuru.REAL_DATA,
                                  confidence=YUKSEK_GUVEN)
                store.olgu_kaydet(s_id, rid, o_id, score=0.0,
                                  source=KaynakTuru.MODEL_GENERATED,
                                  confidence=0.3)
                hedef, insa = "VALID", (
                    "olumsuz kanıt var ama güveni çelişki eşiğinin ALTINDA — "
                    "CONFLICT olmamalı")
            test_kayitlari.append(OlceklenmisKayit(
                experience_id=f"SGOLD_{index:06d}",
                subject_id=s_id, relation_id=rid, object_id=o_id,
                expected_state=hedef, construction=f"[zor:{zor}] {insa}",
            ))
            sinif_sayilari[hedef] += 1
            continue

        if hedef == "VALID":
            store.varlik_ekle(f"nesne{index}", entity_type="nesne", entity_id=o_id)
            store.ozellik_koy(o_id, ozellik, 1.0, source=KaynakTuru.REAL_DATA,
                              confidence=YUKSEK_GUVEN)
            insa = "gerekli özellik bilinir, yüksek güvenli ve doğru"
        elif hedef == "INVALID":
            store.varlik_ekle(f"nesne{index}", entity_type="nesne", entity_id=o_id)
            store.ozellik_koy(o_id, ozellik, 0.0, source=KaynakTuru.REAL_DATA,
                              confidence=YUKSEK_GUVEN)
            insa = "gerekli özellik bilinir, yüksek güvenli ve YANLIŞ"
        elif hedef == "UNCERTAIN":
            store.varlik_ekle(f"nesne{index}", entity_type="nesne", entity_id=o_id)
            if index % 8 < 4:
                insa = "gerekli özellik bilgi tabanında YOK"
            else:
                store.ozellik_koy(o_id, ozellik, 1.0, source=KaynakTuru.MODEL_GENERATED,
                                  confidence=DUSUK_GUVEN)
                insa = "gerekli özellik var ama güveni belirsizlik eşiğinin altında"
        else:  # CONFLICT
            store.varlik_ekle(f"nesne{index}", entity_type="nesne", entity_id=o_id)
            store.ozellik_koy(o_id, ozellik, 1.0, source=KaynakTuru.REAL_DATA,
                              confidence=YUKSEK_GUVEN)
            # Yapısal kural olumlu der; kayıtlı yüksek güvenli kanıt olumsuz.
            store.olgu_kaydet(s_id, rid, o_id, score=0.0,
                              source=KaynakTuru.REAL_DATA,
                              confidence=CELISKI_GUVENI)
            insa = "yapısal tahmin olumlu, kayıtlı yüksek güvenli kanıt olumsuz"

        test_kayitlari.append(OlceklenmisKayit(
            experience_id=f"SGOLD_{index:06d}",
            subject_id=s_id, relation_id=rid, object_id=o_id,
            expected_state=hedef, construction=insa,
        ))
        sinif_sayilari[hedef] += 1

        # Ayrı train olguları (test üçlüsüyle asla aynı değil).
        if rng.random() < train_ratio:
            ts_id = f"SE_TS_{index:06d}"
            to_id = f"SE_TO_{index:06d}"
            store.varlik_ekle(f"tozne{index}", entity_type="canli", entity_id=ts_id)
            store.varlik_ekle(f"tnesne{index}", entity_type="nesne", entity_id=to_id)
            store.ozellik_koy(to_id, ozellik, 1.0, source=KaynakTuru.REAL_DATA,
                              confidence=YUKSEK_GUVEN)
            store.olgu_kaydet(ts_id, rid, to_id, score=1.0,
                              source=KaynakTuru.REAL_DATA, confidence=1.0)
            train_kayitlari.append({"experience_id": f"STRAIN_{index:06d}",
                                    "subject_id": ts_id, "relation_id": rid,
                                    "object_id": to_id, "truth": True,
                                    "split": "train"})

    return OlceklenmisVeriSeti(size=size, seed=int(seed), store=store,
                               train_records=train_kayitlari,
                               test_records=test_kayitlari,
                               class_counts=sinif_sayilari)


@dataclass
class OlcekliGoldenRaporu:
    size: int
    seed: int
    hard: bool
    elapsed_seconds: float
    dataset_hash: str
    leakage_clean: bool
    class_counts: Dict[str, int]
    metrics: Dict[str, Any]
    per_class_accuracy: Dict[str, float]
    misclassifications: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_scaled_golden(size: int = 100, seed: int = 1,
                      evaluator: Optional[ExperienceEvaluator] = None,
                      max_misclassification_samples: int = 25,
                      hard: bool = False,
                      ) -> OlcekliGoldenRaporu:
    """Tek ölçek/seed için evaluator'ı prosedürel golden üzerinde koştur."""
    import time as _time
    _basla = _time.perf_counter()
    veri = veri_seti_uret(size=size, seed=seed, hard=hard)
    sizinti = audit_partitions(
        veri.train_records,
        [{"subject_id": r.subject_id, "relation_id": r.relation_id,
          "object_id": r.object_id} for r in veri.test_records],
    )
    evaluator = evaluator or ExperienceEvaluator()

    beklenen: Dict[str, str] = {}
    tahmin: Dict[str, str] = {}
    hatalar: List[Dict[str, str]] = []
    sinif_dogru = {ad: 0 for ad in SINIFLAR}
    for kayit in veri.test_records:
        aday = ExperienceCandidate(kayit.experience_id, kayit.subject_id,
                                   kayit.relation_id, kayit.object_id)
        evaluator.degerlendir(aday, veri.store)
        beklenen[kayit.experience_id] = kayit.expected_state
        tahmin[kayit.experience_id] = aday.state.value
        if aday.state.value == kayit.expected_state:
            sinif_dogru[kayit.expected_state] += 1
        elif len(hatalar) < int(max_misclassification_samples):
            hatalar.append({
                "experience_id": kayit.experience_id,
                "expected": kayit.expected_state,
                "predicted": aday.state.value,
                "construction": kayit.construction,
                "rationale": "; ".join(aday.rationale[-2:]),
            })

    olculer: GoldenMetrics = _metrics(beklenen, tahmin)
    sinif_dogruluk = {
        ad: round(sinif_dogru[ad] / veri.class_counts[ad], 6)
        if veri.class_counts[ad] else 0.0
        for ad in SINIFLAR
    }
    return OlcekliGoldenRaporu(
        size=veri.size, seed=veri.seed, hard=bool(hard),
        elapsed_seconds=round(_time.perf_counter() - _basla, 4),
        dataset_hash=veri.dataset_hash(),
        leakage_clean=not sizinti.findings,
        class_counts=dict(veri.class_counts),
        metrics=asdict(olculer), per_class_accuracy=sinif_dogruluk,
        misclassifications=hatalar,
    )


def _mean_std(degerler: Sequence[float]) -> Dict[str, float]:
    n = len(degerler)
    ortalama = sum(degerler) / n if n else 0.0
    varyans = sum((d - ortalama) ** 2 for d in degerler) / n if n else 0.0
    return {"mean": round(ortalama, 6), "std": round(math.sqrt(varyans), 6)}


@dataclass
class OlcekliGoldenSweep:
    sizes: List[int]
    seeds: List[int]
    hard: bool
    summary: Dict[str, Dict[str, Dict[str, float]]]   # size → metrik → istat
    per_class: Dict[str, Dict[str, Dict[str, float]]]
    leakage_clean: bool
    cost: Dict[str, Dict[str, float]] = field(default_factory=dict)
    reports: List[Dict[str, Any]] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        satirlar = [
            "| Ölçek | Accuracy | Precision | Recall | F1 | FAR | FRR |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for size in self.sizes:
            o = self.summary[str(size)]
            satirlar.append(
                f"| {size:,} | "
                + " | ".join(
                    f"{o[m]['mean']:.4f} ± {o[m]['std']:.4f}"
                    for m in ("accuracy", "precision", "recall", "f1", "far", "frr")
                ) + " |"
            )
        satirlar.append("")
        satirlar.append("| Ölçek | Süre (sn) | Örnek başına (ms) |")
        satirlar.append("|---:|---:|---:|")
        for size in self.sizes:
            c = self.cost.get(str(size), {})
            satirlar.append(f"| {size:,} | {c.get('mean', 0):.2f} ± "
                            f"{c.get('std', 0):.2f} | {c.get('per_example_ms', 0):.3f} |")
        satirlar.append("")
        satirlar.append("| Ölçek | " + " | ".join(SINIFLAR) + " |")
        satirlar.append("|---:|" + "---:|" * len(SINIFLAR))
        for size in self.sizes:
            p = self.per_class[str(size)]
            satirlar.append(f"| {size:,} | " + " | ".join(
                f"{p[ad]['mean']:.4f} ± {p[ad]['std']:.4f}" for ad in SINIFLAR) + " |")
        return "\n".join(satirlar)


def run_scaled_golden_sweep(
    sizes: Sequence[int] = (100, 1000, 10000),
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    hard: bool = False,
) -> OlcekliGoldenSweep:
    """Faz 3/6 ana girişi: ölçek × seed matrisi, mean ± std."""
    raporlar: List[OlcekliGoldenRaporu] = []
    ozet: Dict[str, Dict[str, Dict[str, float]]] = {}
    sinif_ozet: Dict[str, Dict[str, Dict[str, float]]] = {}
    temiz = True
    maliyet: Dict[str, Dict[str, float]] = {}
    for size in sizes:
        grup = [run_scaled_golden(size=int(size), seed=int(s), hard=hard)
                for s in seeds]
        raporlar.extend(grup)
        temiz = temiz and all(r.leakage_clean for r in grup)
        ozet[str(size)] = {
            metrik: _mean_std([float(r.metrics[metrik]) for r in grup])
            for metrik in ("accuracy", "precision", "recall", "f1", "far", "frr")
        }
        sinif_ozet[str(size)] = {
            ad: _mean_std([r.per_class_accuracy[ad] for r in grup])
            for ad in SINIFLAR
        }
        sureler = [r.elapsed_seconds for r in grup]
        maliyet[str(size)] = {
            **_mean_std(sureler),
            "per_example_ms": round(1000 * sum(sureler) / len(sureler) / int(size), 4),
        }

    en_kucuk, en_buyuk = str(min(sizes)), str(max(sizes))
    bulgular = [
        f"Ölçek {en_kucuk} → {en_buyuk}: accuracy "
        f"{ozet[en_kucuk]['accuracy']['mean']:.4f} → "
        f"{ozet[en_buyuk]['accuracy']['mean']:.4f}; std "
        f"{ozet[en_kucuk]['accuracy']['std']:.4f} → "
        f"{ozet[en_buyuk]['accuracy']['std']:.4f}.",
        f"En büyük ölçekte FAR={ozet[en_buyuk]['far']['mean']:.4f}, "
        f"FRR={ozet[en_buyuk]['frr']['mean']:.4f}, "
        f"F1={ozet[en_buyuk]['f1']['mean']:.4f} — FAR tek başına raporlanmaz.",
        "Beklenen etiket evaluator karar ağacından değil, kanıt durumunun "
        "inşasından türetilir; ölçüm tautolojik değildir.",
        f"Sızıntı denetimi (train/test semantik parmak izi) tüm ölçeklerde "
        f"{'temiz' if temiz else 'KİRLİ'}.",
    ]
    if len(sizes) > 1 and maliyet[en_kucuk]["per_example_ms"] > 0:
        buyume = maliyet[en_buyuk]["per_example_ms"] / maliyet[en_kucuk]["per_example_ms"]
        olcek_orani = max(sizes) / min(sizes)
        bulgular.append(
            f"MALİYET: örnek başına süre {maliyet[en_kucuk]['per_example_ms']:.3f} ms "
            f"→ {maliyet[en_buyuk]['per_example_ms']:.3f} ms ({buyume:.1f}×) "
            f"ölçek {olcek_orani:.0f}× büyürken. Sabit olmaması değerlendirmenin "
            f"O(N) değil O(N²) olduğunu gösterir: Scoring.information_gain "
            f"adayı bilgi tabanındaki diğer varlıklarla karşılaştırır. Ters "
            f"indeksle sabit çarpan düşürüldü ama asimptotik sınıf metriğin "
            f"tanımı gereği korunuyor — 10⁵+ ölçekte örnekleme/ANN gerekir.")
    if ozet[en_buyuk]["accuracy"]["mean"] >= 0.999:
        bulgular.append(
            "UYARI: doğruluk ~1.0. Bu, evaluator'ın gücünden çok görevin "
            "deterministik oluşundandır; sentetik oracle tam olduğu için üst "
            "sınıra çarpılıyor. Gerçek Türkçe veride düşmesi beklenir "
            "(Faz 27/28).")
    return OlcekliGoldenSweep(
        sizes=[int(s) for s in sizes], seeds=[int(s) for s in seeds],
        hard=bool(hard),
        summary=ozet, per_class=sinif_ozet, leakage_clean=temiz, cost=maliyet,
        reports=[r.to_dict() for r in raporlar], findings=bulgular,
    )


__all__ = [
    "SINIFLAR", "OlceklenmisKayit", "OlceklenmisVeriSeti",
    "OlcekliGoldenRaporu", "OlcekliGoldenSweep",
    "veri_seti_uret", "run_scaled_golden", "run_scaled_golden_sweep",
]
