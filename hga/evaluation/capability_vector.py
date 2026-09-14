"""HGA Capability Vector ve otomatik araştırma karnesi (P3).

Neden tek sayı yetmez
---------------------
"HGA 9.1/10" gibi bir puan projeyi tek bir eksene indirger ve hangi iddianın
ölçülmüş, hangisinin üst sınır olduğunu gizler. Bu modül iki şey yapar:

1. **Resmî terminoloji**: her büyüklük ölçülen mi, üst sınır mı açıkça
   etiketlenir.

   ==========  ==========================================  ==========
   Sembol      Anlam                                        Tür
   ==========  ==========================================  ==========
   ``P``       fiziksel/eğitilebilir parametre              ölçülen
   ``C_I^UB``  etkileşim ÜST SINIRI (parametre değil)       üst sınır
   ``C_M^UB``  bellek adres ÜST SINIRI (tablo değil)        üst sınır
   ``C_E``     üretilebilir deneyim kapasitesi              ölçülen
   ``C_V``     doğrulanabilir kapasite                      ölçülen
   ``C_G``     genelleme kapasitesi                         ölçülen
   ``C_R``     güvenilir çıkarım derinliği                  ölçülen
   ``C_RD``    dolguya dayanıklı çıkarım derinliği          ölçülen
   ``C_MR``    bellek geri çağırma                          ölçülen
   ``C_H``     halüsinasyon direnci                         ölçülen
   ``C_U``     belirsizlik kalibrasyonu                     ölçülen
   ==========  ==========================================  ==========

2. **Otomatik karne**: skorlar benchmark çıktılarından hesaplanır. README'ye
   elle yazılmış bir "10/10" buraya giremez; her skorun kaynağı
   ``evidence`` alanında protokol adı + veri imzasıyla birlikte durur. Kanıtı
   olmayan bir alan ``None`` döner ve "n/a" basılır — asla 0 ya da 10 değil.

Kural: **kanıt yoksa skor yok.**
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

#: Terminoloji sözleşmesi. README ve raporlar bunu tek kaynak olarak kullanır.
TERMINOLOGY: Dict[str, Dict[str, str]] = {
    "P": {"name": "Physical Parameters", "kind": "measured",
          "note": "RAM'de ayrılan, optimizer'ın güncellediği tensör sayısı."},
    "C_I^UB": {"name": "Interaction Upper Bound", "kind": "upper_bound",
               "note": "Kronecker zincirinin temsil edebileceği operatör "
                       "girdi sayısının üst sınırı. PARAMETRE DEĞİLDİR."},
    "C_M^UB": {"name": "Memory Address Upper Bound", "kind": "upper_bound",
               "note": "sözlük^pencere kavramsal anahtar uzayı. Fiziksel "
                       "tablo boyutu DEĞİLDİR."},
    "C_E": {"name": "Measured Experience Capacity", "kind": "measured",
            "note": "Kısıtlar altında gerçekten üretilebilen deneyim sayısı."},
    "C_V": {"name": "Measured Verified Capacity", "kind": "measured",
            "note": "Bağımsız verifier'ın karara bağladığı alt küme."},
    "C_G": {"name": "Measured Generalization Capacity", "kind": "measured",
            "note": "Görülmemiş eksenlerde kompozisyon başarısı."},
    "C_R": {"name": "Measured Reliable Reasoning Depth", "kind": "measured",
            "note": "Dolgu yokken güvenilir kalınan en derin zincir."},
    "C_RD": {"name": "Distractor-Resistant Reasoning Depth", "kind": "measured",
             "note": "Belirtilen dolgu yükü altında güvenilir derinlik."},
    "C_MR": {"name": "Measured Memory Recall", "kind": "measured",
             "note": "Yazılan kaydın geri çağrılabilme oranı."},
    "C_H": {"name": "Hallucination Resistance", "kind": "measured",
            "note": "1 − (çıkarılamayan iddiayı kesin doğrulama oranı)."},
    "C_U": {"name": "Uncertainty Calibration", "kind": "measured",
            "note": "1 − ECE; kalibrasyon raporundan gelir."},
}

#: Karne bölümleri ve her birinin hangi kanıttan beslendiği.
SCORECARD_SECTIONS: Tuple[str, ...] = (
    "architecture", "memory", "verification", "generalization", "reasoning",
    "self_learning", "statistical_rigor", "human_evaluation", "turkish_nlp",
    "language_modeling", "reproducibility",
    "scientific_evidence", "engineering",
)


@dataclass
class CapabilityEntry:
    """Vektörün tek bir bileşeni."""

    symbol: str
    name: str
    kind: str                 # measured | upper_bound
    value: Optional[float]
    unit: str
    evidence: Optional[str]   # protokol adı + veri imzası
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScorecardSection:
    section: str
    score: Optional[float]     # 0..10 veya None (kanıt yok)
    inputs: Dict[str, Any]
    evidence: List[str]
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Scorecard:
    generated_at_utc: str
    capability_vector: List[Dict[str, Any]]
    sections: Dict[str, Dict[str, Any]]
    overall: Optional[float]
    scored_sections: int
    unscored_sections: List[str]
    provenance: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        return scorecard_markdown(self)


def _clamp10(value: float) -> float:
    return round(max(0.0, min(10.0, value)), 2)


def _gate_score(checks: Dict[str, bool]) -> Optional[float]:
    """Kabul kapılarının geçme oranını 0..10'a çevir."""
    if not checks:
        return None
    return _clamp10(10.0 * sum(1 for v in checks.values() if v) / len(checks))


def _get(d: Optional[Dict[str, Any]], *path, default=None):
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_capability_vector(
    *,
    priority_ablation: Optional[Dict[str, Any]] = None,
    operator_baselines: Optional[Dict[str, Any]] = None,
    reasoning_depth: Optional[Dict[str, Any]] = None,
    signature: Optional[Dict[str, Any]] = None,
    semantic_extraction: Optional[Dict[str, Any]] = None,
    compositional_v2: Optional[Dict[str, Any]] = None,
    capacity: Optional[Dict[str, Any]] = None,
    memory: Optional[Dict[str, Any]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    twt_results: Optional[Dict[str, Any]] = None,
) -> List[CapabilityEntry]:
    """Verilen rapor sözlüklerinden Capability Vector'ü kur.

    Eksik rapor bir hata değildir: ilgili bileşen ``value=None`` ve
    ``evidence=None`` ile döner. Kanıtsız sayı üretilmez.
    """
    girdiler: List[CapabilityEntry] = []

    def ekle(sembol: str, deger: Optional[float], birim: str,
             kanit: Optional[str], not_: str = "") -> None:
        meta = TERMINOLOGY[sembol]
        girdiler.append(CapabilityEntry(
            symbol=sembol, name=meta["name"], kind=meta["kind"],
            value=None if deger is None else round(float(deger), 6),
            unit=birim, evidence=kanit, note=not_ or meta["note"]))

    # P / C_I^UB / C_M^UB — kapasite raporundan.
    ekle("P", _get(capacity, "physical_parameters"), "parametre",
         "capacity_framework" if capacity else None)
    ekle("C_I^UB", _get(capacity, "c_i_interaction"), "operatör girdisi",
         "capacity_framework" if capacity else None)
    ekle("C_M^UB", _get(capacity, "c_m_conceptual"), "adres",
         "capacity_framework" if capacity else None)
    ekle("C_E", _get(capacity, "c_e_total"), "deneyim",
         "capacity_framework" if capacity else None)
    ekle("C_V", _get(capacity, "c_v_total"), "deneyim",
         "capacity_framework" if capacity else None)

    # C_G — v2 raporundan (ham metin; şema verilmeden).
    ekle("C_G", _get(compositional_v2, "overall_c_g_v2"), "oran",
         (f"{_get(compositional_v2, 'protocol')}@"
          f"{_get(compositional_v2, 'dataset_hash')}") if compositional_v2 else None)

    # C_R / C_RD — derinlik raporundan.
    c_r = _get(reasoning_depth, "c_r")
    kanit_derinlik = (f"{_get(reasoning_depth, 'protocol')}@"
                      f"{_get(reasoning_depth, 'dataset_hash')}"
                      ) if reasoning_depth else None
    ekle("C_R", c_r, "hop", kanit_derinlik)
    c_rd = _get(reasoning_depth, "c_rd", default={}) or {}
    en_yuksek = None
    if c_rd:
        anahtar = max(c_rd, key=lambda k: int(k))
        en_yuksek = c_rd[anahtar]
        ekle("C_RD", en_yuksek, f"hop @ {anahtar} dolgu", kanit_derinlik,
             not_=f"{anahtar} dolgu olgu altında güvenilir derinlik.")
    else:
        ekle("C_RD", None, "hop", None)

    # C_MR — kanıt gücü sırasına göre TEK bir kaynaktan alınır:
    # (1) doğrudan bellek benchmarkı, (2) hiyerarşik bellek recall'ı,
    # (3) derinlik ızgarasının en düşük hücresi (dolaylı). Birden fazla
    # kaynak varsa en güçlüsü kazanır; vektöre yalnız bir satır girer.
    bellek_kanit = None
    if memory:
        bellek_kanit = _get(memory, "protocol") or "memory_benchmark"
        imza_bellek = _get(memory, "config", "signature")
        if imza_bellek:
            bellek_kanit = f"{bellek_kanit}@{imza_bellek}"

    c_mr = _get(memory, "active_recall")
    c_mr_kanit = bellek_kanit
    c_mr_not = ""
    if c_mr is None:
        c_mr = _get(memory, "recall", "recall")
        if c_mr is not None:
            c_mr_not = ("Hiyerarşik bellek benchmarkında yazılan kayıtların "
                        "geri okunma oranı (tüm katmanlar dahil).")
    if c_mr is None and reasoning_depth:
        izgara = _get(reasoning_depth, "memory_recall_grid", default={}) or {}
        degerler = [v for alt in izgara.values() for v in alt.values()]
        if degerler:
            c_mr = min(degerler)
            c_mr_kanit = kanit_derinlik
            c_mr_not = ("Derinlik ızgarasındaki EN DÜŞÜK bellek geri çağırma "
                        "oranı (dolaylı kanıt).")
    ekle("C_MR", c_mr, "oran", c_mr_kanit if c_mr is not None else None,
         not_=c_mr_not)

    # C_H — signature benchmark'ın halüsinasyon oranından.
    c_h = None
    if signature:
        oranlar = [
            _get(signature, "results", task, "hga", "hallucination_rate_mean")
            for task in _get(signature, "tasks", default=[]) or []
        ]
        oranlar = [o for o in oranlar if o is not None]
        if oranlar:
            c_h = 1.0 - (sum(oranlar) / len(oranlar))
    ekle("C_H", c_h, "oran",
         (f"{_get(signature, 'protocol')}@{_get(signature, 'dataset_hash')}"
          ) if signature else None)

    # C_U — kalibrasyon raporundan (1 − ECE).
    ece = _get(calibration, "ece")
    ekle("C_U", None if ece is None else 1.0 - float(ece), "1−ECE",
         "calibration" if calibration else None)

    return girdiler


def build_scorecard(
    *,
    priority_ablation: Optional[Dict[str, Any]] = None,
    operator_baselines: Optional[Dict[str, Any]] = None,
    reasoning_depth: Optional[Dict[str, Any]] = None,
    signature: Optional[Dict[str, Any]] = None,
    semantic_extraction: Optional[Dict[str, Any]] = None,
    compositional_v2: Optional[Dict[str, Any]] = None,
    capacity: Optional[Dict[str, Any]] = None,
    memory: Optional[Dict[str, Any]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    twt_results: Optional[Dict[str, Any]] = None,
    multi_environment: Optional[Dict[str, Any]] = None,
    self_learning_scaling: Optional[Dict[str, Any]] = None,
    seed_statistics: Optional[Dict[str, Any]] = None,
    depth_diagnosis: Optional[Dict[str, Any]] = None,
    priority_optimization: Optional[Dict[str, Any]] = None,
    human_evaluation: Optional[Dict[str, Any]] = None,
    language_modeling: Optional[Dict[str, Any]] = None,
    reproducibility: Optional[Dict[str, Any]] = None,
    engineering: Optional[Dict[str, Any]] = None,
) -> Scorecard:
    """Benchmark çıktılarından **otomatik** araştırma karnesi üret.

    Her bölüm skoru o bölümün kabul kapılarının geçme oranıdır. Bu kasıtlı
    olarak sert bir kuraldır: bir kapı "KALDI" ise skor düşer ve yükseltmek
    için ya ölçüm iyileşmeli ya kapı gerekçeli değiştirilmelidir. Metin
    yazarak puan yükseltilemez.
    """
    bolumler: Dict[str, ScorecardSection] = {}
    uyarilar: List[str] = []

    def kanit(rapor: Optional[Dict[str, Any]]) -> List[str]:
        if not rapor:
            return []
        etiket = rapor.get("protocol") or rapor.get("report_type") or "report"
        imza = (rapor.get("dataset_hash") or rapor.get("config_hash")
                or ((rapor.get("config") or {}).get("signature")) or "")
        # Uzun SHA-256'lar tabloyu okunamaz yapar; ilk 12 karakter kimlik
        # için yeterlidir ve tam hash zaten JSON raporda durur.
        imza = str(imza)[:12]
        return [f"{etiket}@{imza}" if imza else str(etiket)]

    def bolum(ad: str, checks: Optional[Dict[str, bool]],
              girdiler: Dict[str, Any], kanitlar: List[str],
              gerekce: str) -> None:
        skor = _gate_score(checks or {})
        if skor is None:
            uyarilar.append(f"`{ad}` bölümü için kanıt yok; skor üretilmedi.")
        bolumler[ad] = ScorecardSection(
            section=ad, score=skor, inputs=girdiler,
            evidence=kanitlar, rationale=gerekce)

    # architecture ← operatör baseline ailesi (gerçek dense karşılaştırması)
    bolum("architecture", _get(operator_baselines, "checks"),
          {"rankings": _get(operator_baselines, "rankings"),
           "ceiling_gap": _get(operator_baselines, "ceiling_gap")},
          kanit(operator_baselines),
          "Kronecker/rank-1/low-rank/full-dense kolları eşit parametre ve "
          "eşit FLOP rejimlerinde; skor kabul kapılarının geçme oranıdır.")

    # memory ← derinlik ızgarasının bellek satırı + varsa bellek benchmarkı
    bellek_checks: Dict[str, bool] = {}
    if reasoning_depth:
        izgara = _get(reasoning_depth, "memory_recall_grid", default={}) or {}
        degerler = [v for alt in izgara.values() for v in alt.values()]
        if degerler:
            bellek_checks["memory_recall_perfect_at_all_depths"] = min(degerler) >= 1.0
            bellek_checks["memory_recall_above_0_9"] = min(degerler) >= 0.9
    if memory:
        for ad, deger in (memory.get("checks") or {}).items():
            bellek_checks[ad] = bool(deger)
    bolum("memory", bellek_checks or None,
          {"min_recall": min([v for alt in (_get(reasoning_depth,
                                                 "memory_recall_grid",
                                                 default={}) or {}).values()
                              for v in alt.values()], default=None),
           "hierarchical_recall": _get(memory, "recall", "recall"),
           "crash_recovery_recall": _get(memory, "crash_recovery",
                                         "post_recovery_recall"),
           "tier_distribution": _get(memory, "tier_distribution")},
          kanit(reasoning_depth) + kanit(memory),
          "Hiyerarşik bellek (hot/warm/cold/archive) recall, gecikme, tahliye "
          "ve çökme kurtarma kapıları; kanıt yoksa çıkarım derinliği "
          "ızgarasındaki en düşük geri çağırma oranına düşülür.")

    # verification ← Priority(E) nedensel zinciri + verifier izolasyonu.
    # İkisi birlikte "doğrulama" başlığının iki yarısıdır: birincisi neyin
    # doğrulanmaya DEĞER olduğunu seçer, ikincisi doğrulayıcının yetkisi
    # dışında konuşmadığını gösterir.
    dogrulama_checks = dict(_get(priority_ablation, "checks") or {})
    for ad, deger in (_get(priority_optimization, "checks") or {}).items():
        dogrulama_checks[f"weightopt:{ad}"] = bool(deger)
    for ad, deger in (_get(multi_environment, "checks") or {}).items():
        dogrulama_checks[f"multienv:{ad}"] = bool(deger)
    bolum("verification", dogrulama_checks or None,
          {"baseline_downstream": _get(priority_ablation, "baseline_downstream"),
           "verifier_isolation_rate": _get(multi_environment, "contamination",
                                           "isolation_rate"),
           "weights_validated_on_holdout": _get(
               priority_optimization, "overfitting", "gain_survives_holdout"),
           "holdout_gain": _get(
               priority_optimization, "overfitting", "holdout_gain"),
           "adversarial_abstain_rate": _get(multi_environment, "adversarial",
                                            "abstain_rate")},
          kanit(priority_ablation) + kanit(multi_environment)
          + kanit(priority_optimization),
          "Priority(E) ağırlıklarının skor→sıralama→seçim→downstream "
          "zincirini taşıyıp taşımadığı ve doğrulayıcıların alan dışında "
          "çekimser kalıp kalmadığı (cross-domain kontaminasyon) ölçülür.")

    # generalization ← C_G v2 (ham metin)
    bolum("generalization", _get(compositional_v2, "checks"),
          {"overall_c_g_v2": _get(compositional_v2, "overall_c_g_v2")},
          kanit(compositional_v2),
          "Şema ve ontoloji önceden verilmeden, ham metinden keşif + "
          "kompozisyon başarısı.")

    # reasoning ← C_R / C_RD + çöküşün kök neden teşhisi.
    # Bir sınırı ölçmek yarım iştir; NEDEN olduğunu bilmek tam iş. Teşhis
    # kapıları bu yüzden aynı bölümde birleştirilir.
    _cikarim_checks = dict(_get(reasoning_depth, "checks") or {})
    for ad, deger in (_get(depth_diagnosis, "checks") or {}).items():
        _cikarim_checks[f"diagnosis:{ad}"] = bool(deger)
    bolum("reasoning", _cikarim_checks or None,
          {"c_r": _get(reasoning_depth, "c_r"),
           "c_rd": _get(reasoning_depth, "c_rd"),
           "grid_limited": _get(reasoning_depth, "c_r_grid_limited"),
           "collapse_root_cause": _get(
               depth_diagnosis, "diagnosis", "root_cause"),
           "depth_per_slot_log_log_slope": _get(
               depth_diagnosis, "scaling", "log_log_slope")},
          kanit(reasoning_depth) + kanit(depth_diagnosis),
          "Güvenilir çıkarım derinliği ve dolgu baskısı altındaki dayanıklılık.")

    # self_learning ← ölçeklendirme + sürüklenme (model collapse) denetimi
    bolum("self_learning", _get(self_learning_scaling, "checks"),
          {"max_cycles": max(
              [p.get("cycles", 0) for p in
               (_get(self_learning_scaling, "points") or [])] or [0]) or None,
           "total_incorrect_knowledge": _get(
               self_learning_scaling, "drift", "total_incorrect_knowledge"),
           "saturation_detected": any(
               v.get("saturated") for v in
               (_get(self_learning_scaling, "cycle_axis") or {}).values())
           if self_learning_scaling else None},
          kanit(self_learning_scaling),
          "Uzun kapalı döngüde bilgi ölçeklemesi ve yanlış bilgi "
          "birikmemesi (self-training çöküşüne direnç).")

    # statistical_rigor ← 20 tohum + CI/etki büyüklüğü/permütasyon
    _kiyaslar = _get(seed_statistics, "comparisons") or []
    bolum("statistical_rigor", _get(seed_statistics, "checks"),
          {"seeds": len(_get(seed_statistics, "seeds") or []) or None,
           "paired_comparisons": len(_kiyaslar) or None,
           "minimum_attainable_p": _get(
               seed_statistics, "power", "minimum_attainable_two_sided_p"),
           "comparisons_excluding_zero": sum(
               1 for k in _kiyaslar
               if not ((k.get("difference_ci") or {}).get("lower", -1) <= 0.0
                       <= (k.get("difference_ci") or {}).get("upper", 1)))
           if _kiyaslar else None},
          kanit(seed_statistics),
          "Çekirdek protokollerde 20 tohum, eşleşmiş tasarım, %95 bootstrap "
          "CI, etki büyüklüğü ve iki bağımsız anlamlılık testi. Çıplak "
          "p-değeri kabul edilmez.")

    # human_evaluation ← YALNIZ gerçek insan puanı varsa skorlanır.
    # Protokol ve araç hazır olması bir sonuç DEĞİLDİR; araç kapılarına
    # puan vermek "ölçmediğimi ölçtüm" demek olurdu. Bu yüzden gerçek puan
    # toplanana kadar bölüm bilinçli olarak kanıtsız (n/a) bırakılır.
    _insan_toplandi = bool(
        _get(human_evaluation, "checks", "human_ratings_collected"))
    bolum("human_evaluation",
          _get(human_evaluation, "checks") if _insan_toplandi else None,
          {"protocol_ready": bool(human_evaluation),
           "ratings_collected": _insan_toplandi,
           "prompts": _get(human_evaluation, "design", "prompts"),
           "raters": _get(human_evaluation, "design", "raters"),
           "alpha_tool_validated": _get(
               human_evaluation, "checks",
               "alpha_validated_on_reference_data")},
          kanit(human_evaluation),
          "50–100 Türkçe prompt, 10–20 kör değerlendirici ve Krippendorff "
          "α ile kodlayıcılar arası güvenilirlik. Protokol ve araç hazır "
          "olsa bile gerçek insan puanı yoksa skor üretilmez.")

    # turkish_nlp ← semantik çıkarım (sentetik altın set)
    #             + TWT sonuç tablosu (GERÇEK Türkçe treebank)
    turkce_checks = dict(_get(semantic_extraction, "checks") or {})
    for ad, deger in (_get(twt_results, "checks") or {}).items():
        turkce_checks[f"twt:{ad}"] = bool(deger)
    bolum("turkish_nlp", turkce_checks or None,
          {"relation_f1": _get(semantic_extraction, "layers", "relation", "f1"),
           "entity_f1": _get(semantic_extraction, "layers", "entity", "f1"),
           "twt_best_f1_all": _get(twt_results, "results", "hga", "all",
                                   "f1", "mean"),
           "twt_seeds": len(_get(twt_results, "seeds", default=[]) or [])},
          kanit(semantic_extraction) + kanit(twt_results),
          "Elle etiketli altın sette varlık/ilişki/özellik/zaman/olumsuzluk "
          "çıkarımı ve gerçek Türkçe treebank (TWT) üzerinde arc doğrulama.")

    # language_modeling ← henüz gerçek korpus yok
    bolum("language_modeling", _get(language_modeling, "checks"),
          {"perplexity": _get(language_modeling, "perplexity")},
          kanit(language_modeling),
          "Gerçek Türkçe korpusta perplexity ve üretim kalitesi. Kanıt yoksa "
          "skor üretilmez — 'tiny smoke' bir dil modeli iddiası değildir.")

    # reproducibility / engineering
    bolum("reproducibility", _get(reproducibility, "checks"),
          {"protocols_meeting_20_seeds": _get(
              reproducibility, "seed_discipline",
              "protocols_meeting_core_requirement"),
           "protocols_examined": _get(reproducibility, "seed_discipline",
                                      "protocols_examined"),
           "git_commit": _get(reproducibility, "environment", "git_commit"),
           "determinism_measured": bool(_get(reproducibility, "determinism"))},
          kanit(reproducibility),
          "Manifest üretimi, veri/konfig hash'i, ÖLÇÜLEN determinizm "
          "(aynı tohum → bayt-eş çıktı) ve 20 tohum kuralına uyum.")
    bolum("engineering", _get(engineering, "checks"),
          {"python_matrix": _get(engineering, "ci", "python_matrix"),
           "cli_commands_smoke_tested": _get(engineering, "ci",
                                             "cli_command_count"),
           "scientific_test_files": _get(engineering, "test_suite",
                                         "scientific_test_files")},
          kanit(engineering),
          "CI matrisi, lint/type kapıları, paketleme sözleşmesi.")

    # scientific_evidence: kaç bölümün gerçek kanıtı var + kapıların durumu
    kanitli = [ad for ad, b in bolumler.items() if b.score is not None]
    tum_kapilar: Dict[str, bool] = {}
    for rapor in (priority_ablation, operator_baselines, reasoning_depth,
                  signature, semantic_extraction, compositional_v2,
                  self_learning_scaling):
        for ad, deger in (_get(rapor, "checks") or {}).items():
            tum_kapilar[f"{_get(rapor, 'protocol')}:{ad}"] = bool(deger)
    # Hiçbir protokol koşulmadıysa bu bölüm 0.0 DEĞİL, kanıtsız olmalıdır:
    # "ölçtük ve sıfır çıktı" ile "hiç ölçmedik" aynı şey değildir.
    bilim_checks = dict(tum_kapilar)
    if tum_kapilar:
        bilim_checks["all_sections_have_evidence"] = (
            len(kanitli) == len(SCORECARD_SECTIONS))
    bolum("scientific_evidence", bilim_checks,
          {"sections_with_evidence": len(kanitli),
           "total_gates": len(tum_kapilar)},
          [k for rapor in (priority_ablation, operator_baselines,
                           reasoning_depth, signature, semantic_extraction,
                           compositional_v2, self_learning_scaling,
                           seed_statistics, depth_diagnosis,
                           priority_optimization)
           for k in kanit(rapor)],
          "Tüm protokollerin kabul kapılarının birleşik geçme oranı. Bu skor "
          "yalnızca ölçüm iyileşerek yükselir.")

    puanlilar = [b.score for b in bolumler.values() if b.score is not None]
    genel = round(sum(puanlilar) / len(puanlilar), 2) if puanlilar else None
    puansizlar = [ad for ad, b in bolumler.items() if b.score is None]
    if puansizlar:
        uyarilar.append(
            f"Genel skor yalnız {len(puanlilar)}/{len(SCORECARD_SECTIONS)} "
            f"bölüm üzerinden hesaplandı; {puansizlar} kanıtsız. Bu ortalama "
            f"eksik kanıtı gizlemez, onu işaretler.")

    return Scorecard(
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        capability_vector=[e.to_dict() for e in build_capability_vector(
            priority_ablation=priority_ablation,
            operator_baselines=operator_baselines,
            reasoning_depth=reasoning_depth, signature=signature,
            semantic_extraction=semantic_extraction,
            compositional_v2=compositional_v2, capacity=capacity,
            memory=memory, calibration=calibration,
            twt_results=twt_results)],
        sections={ad: b.to_dict() for ad, b in bolumler.items()},
        overall=genel, scored_sections=len(puanlilar),
        unscored_sections=puansizlar,
        provenance={
            "rule": "Skorlar benchmark kabul kapılarından otomatik hesaplanır; "
                    "elle yazılamaz.",
            "reports_supplied": sorted([
                ad for ad, rapor in (
                    ("priority_ablation", priority_ablation),
                    ("operator_baselines", operator_baselines),
                    ("reasoning_depth", reasoning_depth),
                    ("signature", signature),
                    ("semantic_extraction", semantic_extraction),
                    ("compositional_v2", compositional_v2),
                    ("capacity", capacity), ("memory", memory),
                    ("twt_results", twt_results),
                    ("multi_environment", multi_environment),
                    ("self_learning_scaling", self_learning_scaling),
                    ("seed_statistics", seed_statistics),
                    ("human_evaluation", human_evaluation),
                    ("depth_diagnosis", depth_diagnosis),
                    ("priority_optimization", priority_optimization),
                    ("calibration", calibration),
                    ("language_modeling", language_modeling),
                    ("reproducibility", reproducibility),
                    ("engineering", engineering),
                ) if rapor]),
        },
        warnings=uyarilar,
    )


def scorecard_markdown(card: Scorecard) -> str:
    satirlar = [
        "# HGA RESEARCH SCORECARD",
        "",
        f"Üretim zamanı (UTC): `{card.generated_at_utc}`",
        "",
        "> Bu skorlar benchmark kabul kapılarından **otomatik** hesaplanır. "
        "Elle yazılmış bir puan bu tabloya giremez; kanıtı olmayan bölüm "
        "`n/a` döner.",
        "",
        "| Bölüm | Skor | Kanıt |",
        "|---|---:|---|",
    ]
    for ad in SCORECARD_SECTIONS:
        b = card.sections.get(ad)
        if b is None:
            continue
        skor = "n/a" if b["score"] is None else f"{b['score']:.1f}"
        satirlar.append(
            f"| {ad} | {skor} | {', '.join(b['evidence']) or '—'} |")
    genel = "n/a" if card.overall is None else f"{card.overall:.1f}"
    satirlar.append(f"| **Overall Research Readiness** | **{genel}** | "
                    f"{card.scored_sections}/{len(SCORECARD_SECTIONS)} bölüm |")

    satirlar += ["", "## HGA Capability Vector", "",
                 "| Sembol | Ad | Tür | Değer | Birim | Kanıt |",
                 "|---|---|---|---:|---|---|"]
    for e in card.capability_vector:
        deger = "n/a" if e["value"] is None else f"{e['value']:g}"
        tur = "ÜST SINIR" if e["kind"] == "upper_bound" else "ölçülen"
        satirlar.append(
            f"| `{e['symbol']}` | {e['name']} | {tur} | {deger} | "
            f"{e['unit']} | {e['evidence'] or '—'} |")

    if card.warnings:
        satirlar += ["", "## Uyarılar", ""]
        satirlar += [f"- {u}" for u in card.warnings]
    satirlar += ["", "## Bölüm gerekçeleri", ""]
    for ad in SCORECARD_SECTIONS:
        b = card.sections.get(ad)
        if b is None:
            continue
        satirlar.append(f"- **{ad}**: {b['rationale']}")
    return "\n".join(satirlar) + "\n"


def save_scorecard(card: Scorecard, json_path: Optional[str] = None,
                   markdown_path: Optional[str] = None) -> Dict[str, str]:
    """Karneyi diske yaz; yazılan yolları döndür."""
    import os
    yollar: Dict[str, str] = {}
    if json_path:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)) or ".",
                    exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(card.to_dict(), handle, ensure_ascii=False, indent=2,
                      sort_keys=True)
        yollar["json"] = json_path
    if markdown_path:
        os.makedirs(os.path.dirname(os.path.abspath(markdown_path)) or ".",
                    exist_ok=True)
        with open(markdown_path, "w", encoding="utf-8") as handle:
            handle.write(scorecard_markdown(card))
        yollar["markdown"] = markdown_path
    return yollar


__all__ = ["TERMINOLOGY", "SCORECARD_SECTIONS", "CapabilityEntry",
           "ScorecardSection", "Scorecard", "build_capability_vector",
           "build_scorecard", "scorecard_markdown", "save_scorecard"]
