"""C_G v2: ham metinden entity discovery + relation induction (P0-6).

v1'in kapalı açığı
------------------
Mevcut kompozisyonellik benchmarkı görülmemiş varlığın ontoloji kaydını ve
görülmemiş ilişkinin şemasını sisteme **önceden veriyordu**. Yani ölçülen şey
"bilinen sembollerden yeni kombinasyon kurabiliyor mu" idi; bu gerçek bir
genelleme testidir ama *entity discovery* veya *relation induction* değildir.

v2'nin sözleşmesi
-----------------
Sisteme YALNIZ ham Türkçe cümleler verilir. Hiçbir varlık kimliği, tip kaydı
veya ilişki şeması önceden verilmez. Sistem şunları kendi çıkarmak zorundadır::

    ham cümle → entity keşfi → entity linking → relation induction
              → property → temporal → kompozisyon kararı

Eğitim örneği::

    Ali ata bindi.  ·  Ayşe arabaya bindi.  ·  Mehmet otobüse bindi.

Test örneği::

    Ali arabaya bindi.        → görülmüş varlık × görülmüş ilişki, YENİ bileşim
    Zeynep bisiklete bindi.   → görülmemiş varlık
    Ali arabayı sürdü.        → görülmemiş ilişki
    Zeynep bisikleti sürdü.   → ikisi de görülmemiş
    Ali kırmızı araca dün bindi.  → görülmemiş sözcükleme + özellik + zaman

Beş eksen ayrı raporlanır: ``seen`` / ``unseen_entity`` / ``unseen_relation``
/ ``unseen_both`` / ``unseen_wording``.

Ölçülen şey
-----------
``C_G^v2`` = bileşim doğruluğu × keşif doğruluğu. İki bileşen ayrı da
raporlanır, çünkü bir sistem bileşimi doğru kurup varlığı yanlış keşfedebilir
(ya da tersi) ve tek bir sayı bunu gizler.

Dürüstlük
---------
Çıkarım hattı kural tabanlıdır; keşif "öğrenilmiş" değildir. Ölçülen şey,
**önceden şema verilmeden** ham metinden yapı çıkarılıp çıkarılamadığıdır.
Sözlükte olmayan bir kök keşfedilirse tipi ``UNKNOWN`` olur ve bu bir başarı
sayılmaz; ``type_accuracy`` ayrı ölçülür.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..experience.semantik_ayiklayici import cumle_coz

PROTOCOL = "compositional_generalization_v2_raw_text"
AXES = ("seen", "unseen_entity", "unseen_relation", "unseen_both",
        "unseen_wording")


@dataclass(frozen=True)
class V2Case:
    """Test cümlesi + beklenen yapı. Şema ÖNCEDEN VERİLMEZ."""

    sentence: str
    axis: str
    expected_subject: str
    expected_relation: str
    expected_object: str
    expected_object_type: str
    expected_properties: Tuple[Tuple[str, str, str], ...] = ()
    expected_time: Optional[str] = None
    #: True ise DOĞRU davranış ilişki üretmemektir (ör. ettirgen çatı:
    #: üye yapısı yüzey durumlardan çıkarılamaz). Bu vakalar kompozisyon
    #: doğruluğuna GİRMEZ; ayrı "abstention" kapısında sayılır.
    expect_abstention: bool = False


#: Eğitim korpusu — sistemin "görmüş" sayıldığı tek şey budur.
TRAIN_CORPUS: Tuple[str, ...] = (
    "Ali ata bindi.",
    "Ayşe arabaya bindi.",
    "Mehmet otobüse bindi.",
    "Veli trene bindi.",
    "Fatma okula gitti.",
    "Can eve gitti.",
)

TEST_CASES: Tuple[V2Case, ...] = (
    # ── seen: eğitimde geçen varlık + ilişki, YENİ bileşim ────────────────
    V2Case("Ali arabaya bindi.", "seen", "ali", "binmek", "araba", "tasit"),
    V2Case("Mehmet eve gitti.", "seen", "mehmet", "gitmek", "ev", "mekan"),
    V2Case("Ayşe trene bindi.", "seen", "ayse", "binmek", "tren", "tasit"),
    # ── unseen_entity: eğitimde hiç geçmemiş varlık ───────────────────────
    V2Case("Zeynep bisiklete bindi.", "unseen_entity", "zeynep", "binmek",
           "bisiklet", "tasit"),
    V2Case("Deniz parka gitti.", "unseen_entity", "deniz", "gitmek",
           "park", "mekan"),
    # ── unseen_relation: eğitimde hiç geçmemiş ilişki ─────────────────────
    V2Case("Ali kitabı okudu.", "unseen_relation", "ali", "okumak",
           "kitap", "nesne"),
    V2Case("Ayşe masayı gördü.", "unseen_relation", "ayse", "görmek",
           "masa", "nesne"),
    # ── unseen_both ───────────────────────────────────────────────────────
    V2Case("Zeynep kalemi aldı.", "unseen_both", "zeynep", "almak",
           "kalem", "nesne"),
    # ── unseen_wording: aynı olay, farklı sözcükleme + özellik + zaman ────
    V2Case("Ali kırmızı araca dün bindi.", "unseen_wording", "ali", "binmek",
           "arac", "tasit",
           expected_properties=(("arac", "renk", "kırmızı"),),
           expected_time="dün"),
    V2Case("Ayşe mavi taşıta yarın binecek.", "unseen_wording", "ayse",
           "binmek", "tasit", "tasit",
           expected_properties=(("tasit", "renk", "mavi"),),
           expected_time="yarın"),
)

#: ZOR set: hattın SÖZLÜĞÜNDE de bulunmayan varlık ve fiiller. Buradaki
#: başarısızlık beklenen ve raporlanan bir sınırdır — kural tabanlı keşfin
#: nerede bittiğini gösterir. Varsayılan koşuya dahildir; "kolay set 1.0"
#: sonucunun tek başına yanıltıcı olmaması için.
HARD_CASES: Tuple[V2Case, ...] = (
    # Sözlükte olmayan varlık: tip çıkarılamamalı (UNKNOWN) ama kompozisyon
    # yine de kurulabilmeli — bu ayrımı görmek testin asıl amacıdır.
    V2Case("Ali helikoptere bindi.", "unseen_entity", "ali", "binmek",
           "helikopter", "tasit"),
    V2Case("Zeynep laboratuvara gitti.", "unseen_entity", "zeynep", "gitmek",
           "laboratuvar", "mekan"),
    # Sözlükte olmayan fiil: kanonik SOV yapısı varsa mastar İNDÜKLENMELİ
    # (kanıt-tabanlı indüksiyon, düşük güvenle). Bunlar eskiden çekimserlik
    # vakasıydı; indüksiyon eklenince ölçüm hedefi güncellendi.
    V2Case("Ali kitabı inceledi.", "unseen_relation", "ali", "incelemek",
           "kitap", "nesne"),
    V2Case("Ayşe arabayı tamir etti.", "unseen_relation", "ayse",
           "tamir etmek", "araba", "tasit"),
    # İNDÜKSİYONUN SINIRI: burada ilişki üretmek YANLIŞTIR; hat çekimser
    # kalmalı. Ettirgen çatı üye yapısını değiştirir (okuyan öğretmen
    # değildir); kısa/az kanıtlı kökler de indüklenmez.
    V2Case("Öğretmen öğrencilere kitabı okuttu.", "unseen_relation",
           "ogretmen", "okutmak", "kitap", "nesne", expect_abstention=True),
    V2Case("Ali kitabı sattırdı.", "unseen_relation", "ali", "sattırmak",
           "kitap", "nesne", expect_abstention=True),
)


def _discovered_symbols(sentences: Sequence[str]) -> Tuple[Set[str], Set[str]]:
    """Ham metinden keşfedilen varlık kökleri ve ilişkiler."""
    varliklar: Set[str] = set()
    iliskiler: Set[str] = set()
    for cumle in sentences:
        cikarim = cumle_coz(cumle)
        varliklar.update(e.lemma for e in cikarim.entities)
        iliskiler.update(r.predicate for r in cikarim.relations)
    return varliklar, iliskiler


@dataclass
class AxisMetrics:
    axis: str
    total: int
    composition_correct: int
    composition_accuracy: float
    entity_discovery_accuracy: float
    relation_induction_accuracy: float
    type_accuracy: float
    property_accuracy: float
    temporal_accuracy: float
    c_g_v2: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompositionalV2Report:
    protocol: str
    dataset_hash: str
    train_sentences: List[str]
    discovered_train_entities: List[str]
    discovered_train_relations: List[str]
    axes: Dict[str, Dict[str, Any]]
    overall_c_g_v2: float
    schema_leakage: Dict[str, Any]
    per_case: List[Dict[str, Any]]
    checks: Dict[str, bool]
    abstention_cases: List[Dict[str, Any]] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        return compositional_v2_markdown(self)


def _oran(pay: int, payda: int) -> float:
    return round(pay / payda, 6) if payda else 0.0


def run_compositional_v2_benchmark(
    train_corpus: Sequence[str] = TRAIN_CORPUS,
    test_cases: Sequence[V2Case] = TEST_CASES,
    include_hard: bool = True,
) -> CompositionalV2Report:
    """Ham metinden keşif + indüksiyon + kompozisyon genellemesini ölç."""
    if not test_cases:
        raise ValueError("test_cases boş olamaz")
    zor_vakalar = list(HARD_CASES) if include_hard else []
    kolay_sayisi = len(test_cases)
    test_cases = list(test_cases) + zor_vakalar

    train_entities, train_relations = _discovered_symbols(train_corpus)

    eksen_vakalari: Dict[str, List[Dict[str, Any]]] = {a: [] for a in AXES}
    per_case: List[Dict[str, Any]] = []

    cekimserlik_vakalari: List[Dict[str, Any]] = []
    for case in test_cases:
        if case.axis not in AXES:
            raise ValueError(f"bilinmeyen eksen: {case.axis}")
        cikarim = cumle_coz(case.sentence)
        if case.expect_abstention:
            # Doğru davranış ilişki ÜRETMEMEK: ettirgen çatı gibi üye
            # yapısı belirsiz cümlelerde indüksiyon yanlış bilgi üretirdi.
            # Bu vakalar kompozisyon doğruluğuna girmez; ayrı kapıda sayılır.
            cekimserlik_vakalari.append({
                "sentence": case.sentence,
                "axis": case.axis,
                "abstained": not cikarim.relations,
                "predicted_relations": sorted(
                    [r.subject, r.predicate, r.object]
                    for r in cikarim.relations),
                "skipped_reasons": list(cikarim.skipped_reasons),
            })
            continue
        varliklar = {e.lemma: e for e in cikarim.entities}
        iliskiler = {(r.subject, r.predicate, r.object) for r in cikarim.relations}
        ozellikler = {(p.entity, p.name, str(p.value)) for p in cikarim.properties
                      if not p.source_surface.startswith("<ontology:")}
        zamanlar = {t["value"] for t in cikarim.temporal if t["kind"] == "adverb"}

        beklenen_uclu = (case.expected_subject, case.expected_relation,
                         case.expected_object)
        kompozisyon = beklenen_uclu in iliskiler
        varlik_kesfi = (case.expected_subject in varliklar
                        and case.expected_object in varliklar)
        iliski_induksiyonu = any(r.predicate == case.expected_relation
                                 for r in cikarim.relations)
        nesne = varliklar.get(case.expected_object)
        tip_dogru = bool(nesne and nesne.entity_type == case.expected_object_type)
        ozellik_dogru = all(
            (e, a, str(d)) in ozellikler for e, a, d in case.expected_properties)
        zaman_dogru = (case.expected_time is None
                       or case.expected_time in zamanlar)

        kayit = {
            "sentence": case.sentence,
            "axis": case.axis,
            "composition_correct": kompozisyon,
            "entity_discovered": varlik_kesfi,
            "relation_induced": iliski_induksiyonu,
            "type_correct": tip_dogru,
            "property_correct": ozellik_dogru,
            "temporal_correct": zaman_dogru,
            "predicted_relations": sorted(map(list, iliskiler)),
            "expected_relation": list(beklenen_uclu),
            "skipped_reasons": list(cikarim.skipped_reasons),
        }
        eksen_vakalari[case.axis].append(kayit)
        per_case.append(kayit)

    eksenler: Dict[str, Dict[str, Any]] = {}
    for eksen in AXES:
        vakalar = eksen_vakalari[eksen]
        if not vakalar:
            continue
        n = len(vakalar)
        komp = _oran(sum(1 for v in vakalar if v["composition_correct"]), n)
        kesif = _oran(sum(1 for v in vakalar if v["entity_discovered"]), n)
        eksenler[eksen] = AxisMetrics(
            axis=eksen, total=n,
            composition_correct=sum(1 for v in vakalar if v["composition_correct"]),
            composition_accuracy=komp,
            entity_discovery_accuracy=kesif,
            relation_induction_accuracy=_oran(
                sum(1 for v in vakalar if v["relation_induced"]), n),
            type_accuracy=_oran(sum(1 for v in vakalar if v["type_correct"]), n),
            property_accuracy=_oran(
                sum(1 for v in vakalar if v["property_correct"]), n),
            temporal_accuracy=_oran(
                sum(1 for v in vakalar if v["temporal_correct"]), n),
            c_g_v2=round(komp * kesif, 6),
        ).to_dict()

    toplam = len(per_case)
    genel = round(
        _oran(sum(1 for v in per_case if v["composition_correct"]), toplam)
        * _oran(sum(1 for v in per_case if v["entity_discovered"]), toplam), 6)

    # Şema sızıntısı denetimi: test hedeflerinin eğitimde keşfedilmemiş olması
    # gereken eksenlerde gerçekten keşfedilmemiş olduğunu doğrula.
    sizinti_varlik = []
    sizinti_iliski = []
    for case in test_cases:
        if case.axis in ("unseen_entity", "unseen_both"):
            if case.expected_object in train_entities:
                sizinti_varlik.append(case.expected_object)
        if case.axis in ("unseen_relation", "unseen_both"):
            if case.expected_relation in train_relations:
                sizinti_iliski.append(case.expected_relation)
    sizinti = {
        "leaked_entities": sorted(set(sizinti_varlik)),
        "leaked_relations": sorted(set(sizinti_iliski)),
        "clean": not sizinti_varlik and not sizinti_iliski,
        "schema_provided_in_advance": False,
        "ontology_provided_in_advance": False,
    }

    imza = hashlib.sha256(json.dumps({
        "protocol": PROTOCOL,
        "train": list(train_corpus),
        "test": [c.sentence for c in test_cases],
    }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    kontroller = {
        "no_schema_leakage": sizinti["clean"],
        # Görülmemiş varlık ekseninde kompozisyon çalışmalı: keşif gerçekse
        # yeni varlık yeni şema gerektirmemeli.
        "unseen_entity_composition_works": (
            eksenler.get("unseen_entity", {}).get("composition_accuracy", 0.0) >= 1.0),
        "unseen_relation_induction_works": (
            eksenler.get("unseen_relation", {}).get(
                "relation_induction_accuracy", 0.0) >= 1.0),
        "unseen_wording_property_and_time": (
            eksenler.get("unseen_wording", {}).get("property_accuracy", 0.0) >= 1.0
            and eksenler.get("unseen_wording", {}).get("temporal_accuracy", 0.0) >= 1.0),
        # En zor eksen: ikisi de görülmemiş.
        "unseen_both_composition_works": (
            eksenler.get("unseen_both", {}).get("composition_accuracy", 0.0) >= 1.0),
        # Zor set dahilken her eksende tip doğruluğunun 1.0 çıkması ölçümün
        # kolay olduğuna işarettir; kapı bu yüzden zor sette ayrıca bakar.
        "hard_subset_included": bool(zor_vakalar),
        # İndüksiyonun SINIRI: üye yapısı belirsiz cümlelerde (ettirgen çatı)
        # ilişki üretmek yanlış bilgidir; hat çekimser kalmalı. Bu kapı
        # olmadan indüksiyon "her fiile mastar tak" dejenerasyonuna kayabilir
        # — çekimserlik vakaları o dejenerasyonu anında yakalar.
        "induction_abstains_on_ambiguous_voice": (
            bool(cekimserlik_vakalari)
            and all(v["abstained"] for v in cekimserlik_vakalari)),
    }

    bulgular = [
        f"Eğitim korpusundan keşfedilen {len(train_entities)} varlık, "
        f"{len(train_relations)} ilişki: hiçbiri elle verilmedi.",
        "Eksen bazında C_G v2: " + ", ".join(
            f"{a}={eksenler[a]['c_g_v2']:.4f}" for a in AXES if a in eksenler) + ".",
        f"Genel C_G v2 = {genel:.4f} (kompozisyon × keşif).",
    ]
    if zor_vakalar:
        zor_kayitlar = per_case[kolay_sayisi:]
        zor_komp = _oran(sum(1 for v in zor_kayitlar
                             if v["composition_correct"]), len(zor_kayitlar))
        zor_tip = _oran(sum(1 for v in zor_kayitlar if v["type_correct"]),
                        len(zor_kayitlar))
        bulgular.append(
            f"Sözlük-dışı ZOR alt kümede ({len(zor_kayitlar)} cümle) "
            f"kompozisyon {zor_komp:.4f}, tip doğruluğu {zor_tip:.4f}. "
            f"Kural tabanlı keşfin sınırı buradadır ve gizlenmemiştir.")
    basarisiz = [v["sentence"] for v in per_case if not v["composition_correct"]]
    if basarisiz:
        bulgular.append(f"Kompozisyonu kurulamayan cümleler: {basarisiz}.")
    if cekimserlik_vakalari:
        cekimser_ok = sum(1 for v in cekimserlik_vakalari if v["abstained"])
        bulgular.append(
            f"Çekimserlik vakaları (ettirgen çatı vb.): "
            f"{cekimser_ok}/{len(cekimserlik_vakalari)} doğru çekimser. "
            "İlişki üretmek bu cümlelerde YANLIŞ olurdu; indüksiyonun "
            "'her fiile mastar tak' dejenerasyonuna kaymadığının kanıtıdır.")

    sinirlar = [
        "Çıkarım hattı kural tabanlıdır; 'keşif' istatistiksel öğrenme değil, "
        "morfolojik analiz + sözlük araması sonucudur.",
        "Sözlükte olmayan kök UNKNOWN tiple keşfedilir; type_accuracy bunu "
        "ayrı ölçer ve kompozisyon doğruluğuyla karıştırılmamalıdır.",
        "Test seti küçüktür (elle küratörlü); güven aralıkları geniştir.",
        "'Görülmemiş ilişki' hattın fiil sözlüğünde OLABİLİR; görülmemişlik "
        "EĞİTİM KORPUSUNA göredir, hattın kapsamına göre değil. Bu ayrım "
        "schema_leakage bölümünde açıkça raporlanır.",
        "Mastar indüksiyonu kanıt-tabanlıdır (kök ≥4 harf + yalın özne + "
        "durum ekli nesne) ve DÜŞÜK güvenle (0.55) işaretlenir; ünlü uyumu "
        "ascii üzerinde yaklaşıktır (ı→i eşlemesi kimi art ünlülü köklerde "
        "-mek seçtirir). Çatı ekli fiillerde (ettirgen/edilgen) hat KASITLI "
        "çekimserdir: üye yapısı yüzey durumlardan çıkarılamaz.",
    ]

    return CompositionalV2Report(
        protocol=PROTOCOL, dataset_hash=imza,
        train_sentences=list(train_corpus),
        discovered_train_entities=sorted(train_entities),
        discovered_train_relations=sorted(train_relations),
        axes=eksenler, overall_c_g_v2=genel, schema_leakage=sizinti,
        per_case=per_case, checks=kontroller,
        abstention_cases=cekimserlik_vakalari,
        findings=bulgular, limitations=sinirlar,
    )


def compositional_v2_markdown(report: CompositionalV2Report) -> str:
    satirlar = [
        "# C_G v2 — Ham Metinden Keşif ve Kompozisyon (P0-6)",
        "",
        f"Protokol: `{report.protocol}` · veri imzası: `{report.dataset_hash}`",
        "",
        f"Eğitimde keşfedilen varlıklar: `{report.discovered_train_entities}`",
        f"Eğitimde keşfedilen ilişkiler: `{report.discovered_train_relations}`",
        "",
        "**Sisteme hiçbir entity id, ontoloji kaydı veya relation şeması "
        "önceden verilmedi.**",
        "",
        "| Eksen | n | Kompozisyon | Entity keşfi | Relation indüksiyonu | "
        "Tip | Özellik | Zaman | C_G v2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for eksen in AXES:
        m = report.axes.get(eksen)
        if m is None:
            continue
        satirlar.append(
            f"| {eksen} | {m['total']} | {m['composition_accuracy']:.4f} | "
            f"{m['entity_discovery_accuracy']:.4f} | "
            f"{m['relation_induction_accuracy']:.4f} | "
            f"{m['type_accuracy']:.4f} | {m['property_accuracy']:.4f} | "
            f"{m['temporal_accuracy']:.4f} | {m['c_g_v2']:.4f} |")
    satirlar += ["", f"**Genel C_G v2 = {report.overall_c_g_v2:.4f}**", "",
                 "## Şema sızıntısı denetimi", "",
                 f"- Sızan varlık: {report.schema_leakage['leaked_entities']}",
                 f"- Sızan ilişki: {report.schema_leakage['leaked_relations']}",
                 f"- Temiz mi: {'EVET' if report.schema_leakage['clean'] else 'HAYIR'}",
                 "", "## Kabul kapıları", "", "| kapı | sonuç |", "|---|---|"]
    for ad, sonuc in report.checks.items():
        satirlar.append(f"| {ad} | {'GEÇTİ' if sonuc else 'KALDI'} |")
    if report.abstention_cases:
        satirlar += ["", "## Çekimserlik vakaları (ilişki üretmek YANLIŞ)",
                     "", "| cümle | çekimser mi | neden |", "|---|---|---|"]
        for v in report.abstention_cases:
            satirlar.append(
                f"| {v['sentence']} | "
                f"{'EVET' if v['abstained'] else 'HAYIR (İHLAL)'} | "
                f"{', '.join(v['skipped_reasons']) or '-'} |")
    satirlar += ["", "## Bulgular", ""]
    satirlar += [f"- {b}" for b in report.findings]
    satirlar += ["", "## Sınırlar", ""]
    satirlar += [f"- {s}" for s in report.limitations]
    return "\n".join(satirlar) + "\n"


__all__ = ["PROTOCOL", "AXES", "TRAIN_CORPUS", "TEST_CASES", "V2Case",
           "AxisMetrics", "CompositionalV2Report",
           "run_compositional_v2_benchmark", "compositional_v2_markdown"]
