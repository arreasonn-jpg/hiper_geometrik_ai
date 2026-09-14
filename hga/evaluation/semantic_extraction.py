"""Türkçe semantik çıkarım benchmarkı: elle etiketli altın set (P0-5).

``hga/experience/semantik_ayiklayici.py`` bir hat sunar; bu modül o hattın
**ne kadar doğru** olduğunu ölçer. Ölçüm elle yazılmış bir altın set üzerinde
yapılır: her cümle için beklenen varlıklar, ilişkiler, özellikler, zaman ve
olumsuzluk açıkça yazılıdır.

Neden elle etiketli set?
------------------------
Çıkarıcının kendi çıktısını referans almak (self-consistency) hiçbir şey
kanıtlamaz. Altın set çıkarıcıdan **bağımsız** yazılmıştır ve kasıtlı olarak
hattın kapsamı dışında kalan cümleler de içerir (yan cümle, ettirgen çatı,
bilinmeyen fiil). Bu negatif örnekler olmadan precision ölçülemez.

Metrikler
---------
Her katman (entity / relation / property / temporal / negation) için ayrı
precision, recall ve F1. Ayrıca:

* ``relation_exact_match``: özne+yüklem+nesne+rol+kutup+zaman hepsi doğru.
* ``over_extraction_rate``: altın sette olmayan bir şey uydurma oranı —
  halüsinasyonun çıkarım tarafındaki karşılığı.
* ``abstention_rate``: hattın "çıkaramadım" deyip atladığı cümle oranı;
  yanlış çıkarmaktansa atlamak tercih edilir ama gizlenmez.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence, Set, Tuple

from ..experience.semantik_ayiklayici import SemantikCikarim, cumle_coz

PROTOCOL = "turkish_semantic_extraction_v1"
LAYERS = ("entity", "relation", "property", "temporal", "negation")


@dataclass(frozen=True)
class GoldSentence:
    """Elle etiketli tek cümle.

    ``in_scope=False`` cümleler hattın kasıtlı kapsam dışıdır: oradan hiçbir
    ilişki çıkmaması BEKLENEN davranıştır ve precision'ı ölçmeye yarar.
    """

    sentence: str
    entities: Tuple[Tuple[str, str], ...]                 # (lemma, tip)
    relations: Tuple[Tuple[str, str, str, str, str], ...]  # (özne,yüklem,nesne,rol,kutup)
    properties: Tuple[Tuple[str, str, str], ...]          # (varlık, ad, değer)
    temporal: Tuple[str, ...]
    negations: Tuple[str, ...]                            # olumsuzlanan yüklemler
    in_scope: bool = True


#: Altın set. Elle yazılmıştır; çıkarıcının çıktısından TÜRETİLMEMİŞTİR.
#: Ontolojiden türeyen özellikler (canlı/binilebilir) burada YER ALMAZ —
#: yalnız metinde açıkça yazan bilgi altın kabul edilir.
GOLD_SET: Tuple[GoldSentence, ...] = (
    GoldSentence(
        "Ali dün Ankara'ya arabayla gitti.",
        entities=(("ali", "insan"), ("ankara", "mekan"), ("araba", "tasit")),
        relations=(("ali", "gitmek", "ankara", "hedef", "POSITIVE"),),
        properties=(("ali", "travel_mode", "araba"),),
        temporal=("dün",),
        negations=(),
    ),
    GoldSentence(
        "Ayşe kırmızı arabaya bindi.",
        entities=(("ayse", "insan"), ("araba", "tasit")),
        relations=(("ayse", "binmek", "araba", "hedef", "POSITIVE"),),
        properties=(("araba", "renk", "kırmızı"),),
        temporal=(),
        negations=(),
    ),
    GoldSentence(
        "Mehmet yarın okula gitmeyecek.",
        entities=(("mehmet", "insan"), ("okul", "mekan")),
        relations=(("mehmet", "gitmek", "okul", "hedef", "NEGATIVE"),),
        properties=(),
        temporal=("yarın",),
        negations=("gitmek",),
    ),
    GoldSentence(
        "Zeynep kitabı okudu.",
        entities=(("zeynep", "insan"), ("kitap", "nesne")),
        relations=(("zeynep", "okumak", "kitap", "nesne", "POSITIVE"),),
        properties=(),
        temporal=(),
        negations=(),
    ),
    GoldSentence(
        "Ali ata bindi.",
        entities=(("ali", "insan"), ("at", "hayvan")),
        relations=(("ali", "binmek", "at", "hedef", "POSITIVE"),),
        properties=(),
        temporal=(),
        negations=(),
    ),
    GoldSentence(
        "Deniz bugün eve gelmedi.",
        entities=(("deniz", "insan"), ("ev", "mekan")),
        relations=(("deniz", "gelmek", "ev", "hedef", "NEGATIVE"),),
        properties=(),
        temporal=("bugün",),
        negations=("gelmek",),
    ),
    GoldSentence(
        "Fatma mavi bisikletle parka gitti.",
        entities=(("fatma", "insan"), ("bisiklet", "tasit"), ("park", "mekan")),
        relations=(("fatma", "gitmek", "park", "hedef", "POSITIVE"),),
        properties=(("bisiklet", "renk", "mavi"), ("fatma", "travel_mode", "bisiklet")),
        temporal=(),
        negations=(),
    ),
    GoldSentence(
        "Veli okulda kitap okuyor.",
        entities=(("veli", "insan"), ("okul", "mekan"), ("kitap", "nesne")),
        relations=(("veli", "okumak", "okul", "konum", "POSITIVE"),
                   ("veli", "okumak", "kitap", "nesne", "POSITIVE")),
        properties=(),
        temporal=(),
        negations=(),
    ),
    GoldSentence(
        "Can İstanbul'dan trenle geldi.",
        entities=(("can", "insan"), ("istanbul", "mekan"), ("tren", "tasit")),
        relations=(("can", "gelmek", "istanbul", "kaynak", "POSITIVE"),),
        properties=(("can", "travel_mode", "tren"),),
        temporal=(),
        negations=(),
    ),
    GoldSentence(
        "Kedi masada uyuyor.",
        entities=(("kedi", "hayvan"), ("masa", "nesne")),
        relations=(("kedi", "uyumak", "masa", "konum", "POSITIVE"),),
        properties=(),
        temporal=(),
        negations=(),
    ),
    # ── Kapsam dışı: hattın çıkarım YAPMAMASI beklenen yapılar ────────────
    GoldSentence(
        "Ali'nin dün aldığı kitabı Ayşe'ye verdiği söylendi.",
        entities=(), relations=(), properties=(), temporal=(), negations=(),
        in_scope=False,
    ),
    GoldSentence(
        "Öğretmen öğrencilere kitabı okuttu.",
        entities=(), relations=(), properties=(), temporal=(), negations=(),
        in_scope=False,
    ),
)


def _prf(dogru: int, tahmin: int, altin: int) -> Dict[str, float]:
    precision = dogru / tahmin if tahmin else 0.0
    recall = dogru / altin if altin else (1.0 if tahmin == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "true_positive": dogru,
        "predicted": tahmin,
        "gold": altin,
    }


def _tahmin_kumeler(cikarim: SemantikCikarim) -> Dict[str, Set[Tuple[Any, ...]]]:
    return {
        "entity": {(e.lemma, e.entity_type) for e in cikarim.entities},
        "relation": {(r.subject, r.predicate, r.object, r.role, r.polarity)
                     for r in cikarim.relations},
        # Ontolojiden türeyen özellikler altın sette yok; ölçüme de girmemeli,
        # yoksa precision yapay olarak düşer. Kaynak yüzeyi <ontology:...>
        # olanlar bu yüzden ayrılır ve ayrıca sayılır.
        "property": {(p.entity, p.name, str(p.value))
                     for p in cikarim.properties
                     if not p.source_surface.startswith("<ontology:")},
        "temporal": {t["value"] for t in cikarim.temporal
                     if t["kind"] == "adverb"},
        "negation": {n["scope"] for n in cikarim.negations},
    }


def _altin_kumeler(gold: GoldSentence) -> Dict[str, Set[Tuple[Any, ...]]]:
    return {
        "entity": set(gold.entities),
        "relation": set(gold.relations),
        "property": {(e, a, str(d)) for e, a, d in gold.properties},
        "temporal": set(gold.temporal),
        "negation": set(gold.negations),
    }


@dataclass
class SemanticExtractionReport:
    protocol: str
    dataset_hash: str
    sentence_count: int
    in_scope_count: int
    out_of_scope_count: int
    layers: Dict[str, Dict[str, float]]
    relation_exact_match: float
    over_extraction_rate: float
    out_of_scope_false_positive_rate: float
    abstention_rate: float
    mean_relation_confidence: float
    per_sentence: List[Dict[str, Any]]
    checks: Dict[str, bool]
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        return semantic_extraction_markdown(self)


def run_semantic_extraction_benchmark(
    gold_set: Sequence[GoldSentence] = GOLD_SET,
) -> SemanticExtractionReport:
    """Altın set üzerinde katman katman precision/recall/F1 ölç."""
    if not gold_set:
        raise ValueError("gold_set boş olamaz")

    sayaclar = {katman: {"tp": 0, "pred": 0, "gold": 0} for katman in LAYERS}
    per_sentence: List[Dict[str, Any]] = []
    tam_eslesme = 0
    kapsam_ici = 0
    kapsam_disi_fp = 0
    kapsam_disi = 0
    cekimser = 0
    guvenler: List[float] = []

    for gold in gold_set:
        cikarim = cumle_coz(gold.sentence)
        tahmin = _tahmin_kumeler(cikarim)
        altin = _altin_kumeler(gold)
        guvenler.extend(r.confidence for r in cikarim.relations)

        if not cikarim.relations:
            cekimser += 1

        for katman in LAYERS:
            # Kapsam dışı cümlelerde yalnız ilişki katmanı puanlanır: varlık
            # veya zaman çıkarmak orada bir hata değildir (cümle Türkçedir ve
            # varlıklar gerçekten oradadır); asıl iddia yönlü ilişkidir.
            if not gold.in_scope and katman != "relation":
                continue
            ortak = tahmin[katman] & altin[katman]
            sayaclar[katman]["tp"] += len(ortak)
            sayaclar[katman]["pred"] += len(tahmin[katman])
            sayaclar[katman]["gold"] += len(altin[katman])

        if gold.in_scope:
            kapsam_ici += 1
            if tahmin["relation"] == altin["relation"] and altin["relation"]:
                tam_eslesme += 1
        else:
            kapsam_disi += 1
            if tahmin["relation"]:
                kapsam_disi_fp += 1

        per_sentence.append({
            "sentence": gold.sentence,
            "in_scope": gold.in_scope,
            "predicted_relations": sorted(map(list, tahmin["relation"])),
            "gold_relations": sorted(map(list, altin["relation"])),
            "missed": sorted(map(list, altin["relation"] - tahmin["relation"])),
            "spurious": sorted(map(list, tahmin["relation"] - altin["relation"])),
            "skipped_reasons": list(cikarim.skipped_reasons),
        })

    katmanlar = {
        katman: _prf(s["tp"], s["pred"], s["gold"])
        for katman, s in sayaclar.items()
    }
    toplam_tahmin = sum(s["pred"] for s in sayaclar.values())
    toplam_tp = sum(s["tp"] for s in sayaclar.values())
    asiri = round((toplam_tahmin - toplam_tp) / toplam_tahmin, 6) if toplam_tahmin else 0.0

    imza = hashlib.sha256(json.dumps(
        [g.sentence for g in gold_set], ensure_ascii=False,
        sort_keys=True).encode("utf-8")).hexdigest()[:12]

    kontroller = {
        "relation_f1_at_least_0_80": katmanlar["relation"]["f1"] >= 0.80,
        "entity_f1_at_least_0_90": katmanlar["entity"]["f1"] >= 0.90,
        "negation_recall_perfect": katmanlar["negation"]["recall"] >= 1.0,
        "temporal_recall_perfect": katmanlar["temporal"]["recall"] >= 1.0,
        # Kapsam dışı cümlelerden ilişki uydurulmamalı.
        "no_out_of_scope_hallucination": kapsam_disi_fp == 0,
        "property_precision_at_least_0_80": (
            katmanlar["property"]["precision"] >= 0.80),
    }

    bulgular = [
        f"{len(gold_set)} elle etiketli cümle ({kapsam_ici} kapsam içi, "
        f"{kapsam_disi} kapsam dışı).",
        "Katman F1: " + ", ".join(
            f"{k}={katmanlar[k]['f1']:.4f}" for k in LAYERS) + ".",
        f"İlişki tam eşleşme (özne+yüklem+nesne+rol+kutup): "
        f"{round(tam_eslesme / kapsam_ici, 4) if kapsam_ici else 0.0}.",
        f"Aşırı çıkarım oranı {asiri:.4f}; kapsam dışı cümlelerde yanlış "
        f"ilişki {kapsam_disi_fp}/{kapsam_disi}.",
    ]

    sinirlar = [
        "Altın set küçüktür ve elle yazılmıştır; istatistiksel bir Türkçe NER/"
        "RE değerlendirmesi DEĞİLDİR. Sonuçlar hattın kapsamını gösterir, "
        "genel Türkçe performansını değil.",
        "Ontolojiden türetilen özellikler (canlı, binilebilir) altın sette yer "
        "almaz ve precision hesabına katılmaz; onlar metinden değil tip "
        "varsayımından gelir ve düşük güvenle yazılır.",
        "Hat kural tabanlıdır: sözlük büyüdükçe recall artar, bu bir öğrenme "
        "sonucu değildir.",
        "Kapsam dışı örnek sayısı azdır; precision tahmini geniş güven "
        "aralığına sahiptir.",
    ]

    return SemanticExtractionReport(
        protocol=PROTOCOL, dataset_hash=imza, sentence_count=len(gold_set),
        in_scope_count=kapsam_ici, out_of_scope_count=kapsam_disi,
        layers=katmanlar,
        relation_exact_match=round(tam_eslesme / kapsam_ici, 6) if kapsam_ici else 0.0,
        over_extraction_rate=asiri,
        out_of_scope_false_positive_rate=(
            round(kapsam_disi_fp / kapsam_disi, 6) if kapsam_disi else 0.0),
        abstention_rate=round(cekimser / len(gold_set), 6),
        mean_relation_confidence=(
            round(sum(guvenler) / len(guvenler), 6) if guvenler else 0.0),
        per_sentence=per_sentence, checks=kontroller,
        findings=bulgular, limitations=sinirlar,
    )


def semantic_extraction_markdown(report: SemanticExtractionReport) -> str:
    satirlar = [
        "# Türkçe Semantik Çıkarım Benchmarkı (P0-5)",
        "",
        f"Protokol: `{report.protocol}` · veri imzası: `{report.dataset_hash}` · "
        f"{report.sentence_count} cümle "
        f"({report.in_scope_count} kapsam içi / {report.out_of_scope_count} dışı)",
        "",
        "| Katman | Precision | Recall | F1 | TP | Tahmin | Altın |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for katman in LAYERS:
        m = report.layers[katman]
        satirlar.append(
            f"| {katman} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1']:.4f} | {m['true_positive']} | {m['predicted']} | "
            f"{m['gold']} |")
    satirlar += [
        "",
        f"- İlişki tam eşleşme: **{report.relation_exact_match:.4f}**",
        f"- Aşırı çıkarım oranı: {report.over_extraction_rate:.4f}",
        f"- Kapsam dışı yanlış pozitif: {report.out_of_scope_false_positive_rate:.4f}",
        f"- Çekimserlik (ilişki çıkaramama): {report.abstention_rate:.4f}",
        f"- Ortalama ilişki güveni: {report.mean_relation_confidence:.4f}",
        "", "## Kabul kapıları", "", "| kapı | sonuç |", "|---|---|",
    ]
    for ad, sonuc in report.checks.items():
        satirlar.append(f"| {ad} | {'GEÇTİ' if sonuc else 'KALDI'} |")
    satirlar += ["", "## Hatalı cümleler", ""]
    hatali = [s for s in report.per_sentence if s["missed"] or s["spurious"]]
    if hatali:
        for s in hatali:
            satirlar.append(
                f"- `{s['sentence']}` → eksik: {s['missed']}, "
                f"fazla: {s['spurious']}")
    else:
        satirlar.append("- Yok: her cümlede ilişki kümesi altınla birebir.")
    satirlar += ["", "## Bulgular", ""]
    satirlar += [f"- {b}" for b in report.findings]
    satirlar += ["", "## Sınırlar", ""]
    satirlar += [f"- {s}" for s in report.limitations]
    return "\n".join(satirlar) + "\n"


__all__ = ["PROTOCOL", "LAYERS", "GOLD_SET", "GoldSentence",
           "SemanticExtractionReport", "run_semantic_extraction_benchmark",
           "semantic_extraction_markdown"]
