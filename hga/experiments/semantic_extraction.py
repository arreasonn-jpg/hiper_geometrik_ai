"""Türkçe semantik çıkarım benchmarkı: bağımsız altın set protokolleri (P0-5).

``hga/experience/semantik_ayiklayici.py`` bir hat sunar; bu modül o hattın
**ne kadar doğru** olduğunu ölçer. Ölçüm, çıkarıcının kendi çıktısından değil,
beklenen varlık/ilişki/özellik/zaman/olumsuzluk etiketleri açıkça yazılmış
altın setlerden yapılır.

v1 ve v2 ayrımı
---------------
``turkish_semantic_extraction_v1`` tarihsel smoke/regresyon setidir: 12 elle
sabitlenmiş cümleyle hattın temel katmanlarını korur.

``turkish_semantic_extraction_v2_curated_gold`` daha geniş bir *protokol* setidir:
136 deterministik, kaynak-kodda gözden geçirilebilir altın cümle içerir. Cümleler
şablon ailesiyle üretilir ama her şablonun beklenen semantik etiketi kaynakta
bağımsız olarak yazılıdır; çıkarıcı çıktısı referans alınmaz. Bu hâlâ genel bir
Türkçe NER/RE skoru değildir: amaç sınırlı kapsamlı hattın relation extraction,
özellik, zaman, olumsuzluk ve kapsam-dışı çekimserlik davranışını büyütülmüş bir
regresyon sözleşmesine bağlamaktır.

Metrikler
---------
Her katman (entity / relation / property / temporal / negation) için ayrı
precision, recall ve F1. Ayrıca:

* ``relation_exact_match``: özne+yüklem+nesne+rol+kutup hepsi doğru olan
  kapsam-içi cümle oranı.
* ``extraction_yield``: kapsam-içi cümlelerde relation kümesinin altınla birebir
  eşleşme oranı; v2 kabul kapısı bunu en az 0.85 ister.
* ``over_extraction_rate``: altın sette olmayan bir şey uydurma oranı —
  halüsinasyonun çıkarım tarafındaki karşılığı.
* ``out_of_scope_false_positive_rate``: kapsam-dışı cümlelerde ilişki uydurma
  oranı.
* ``abstention_rate``: hattın ilişki çıkarmayı reddettiği cümle oranı; yanlış
  çıkarmaktansa atlamak tercih edilir ama gizlenmez.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence, Set, Tuple

from ..experience.semantik_ayiklayici import SemantikCikarim, cumle_coz

PROTOCOL_V1 = "turkish_semantic_extraction_v1"
PROTOCOL_V2 = "turkish_semantic_extraction_v2_curated_gold"
# Geriye dönük uyumluluk: eski ithaller PROTOCOL değişkenini v1 olarak görür.
PROTOCOL = PROTOCOL_V1
LAYERS = ("entity", "relation", "property", "temporal", "negation")
V2_MIN_SENTENCES = 100
V2_MIN_EXTRACTION_YIELD = 0.85


@dataclass(frozen=True)
class GoldSentence:
    """Etiketli tek cümle.

    ``in_scope=False`` cümleler hattın kasıtlı kapsam dışıdır: oradan hiçbir
    ilişki çıkmaması BEKLENEN davranıştır ve precision'ı ölçmeye yarar.

    ``uid``/``split``/``phenomena`` alanları v2'nin denetlenebilir kapsama
    raporu içindir; v1 ve dış çağıranlar için varsayılanları vardır.
    """

    sentence: str
    entities: Tuple[Tuple[str, str], ...]                 # (lemma, tip)
    relations: Tuple[Tuple[str, str, str, str, str], ...]  # (özne,yüklem,nesne,rol,kutup)
    properties: Tuple[Tuple[str, str, str], ...]          # (varlık, ad, değer)
    temporal: Tuple[str, ...]
    negations: Tuple[str, ...]                            # olumsuzlanan yüklemler
    in_scope: bool = True
    uid: str = ""
    split: str = "test"
    phenomena: Tuple[str, ...] = ()
    source: str = "hand_labeled_v1"


#: v1 altın set. Elle yazılmıştır; çıkarıcının çıktısından TÜRETİLMEMİŞTİR.
#: Ontolojiden türeyen özellikler (canlı/binilebilir) burada YER ALMAZ —
#: yalnız metinde açıkça yazan bilgi altın kabul edilir.
GOLD_SET_V1: Tuple[GoldSentence, ...] = (
    GoldSentence(
        "Ali dün Ankara'ya arabayla gitti.",
        entities=(("ali", "insan"), ("ankara", "mekan"), ("araba", "tasit")),
        relations=(("ali", "gitmek", "ankara", "hedef", "POSITIVE"),),
        properties=(("ali", "travel_mode", "araba"),),
        temporal=("dün",),
        negations=(),
        uid="sem-v1-001",
        phenomena=("movement", "instrument", "temporal", "case:yonelme"),
    ),
    GoldSentence(
        "Ayşe kırmızı arabaya bindi.",
        entities=(("ayse", "insan"), ("araba", "tasit")),
        relations=(("ayse", "binmek", "araba", "hedef", "POSITIVE"),),
        properties=(("araba", "renk", "kırmızı"),),
        temporal=(),
        negations=(),
        uid="sem-v1-002",
        phenomena=("property", "case:yonelme"),
    ),
    GoldSentence(
        "Mehmet yarın okula gitmeyecek.",
        entities=(("mehmet", "insan"), ("okul", "mekan")),
        relations=(("mehmet", "gitmek", "okul", "hedef", "NEGATIVE"),),
        properties=(),
        temporal=("yarın",),
        negations=("gitmek",),
        uid="sem-v1-003",
        phenomena=("movement", "negation", "temporal", "case:yonelme"),
    ),
    GoldSentence(
        "Zeynep kitabı okudu.",
        entities=(("zeynep", "insan"), ("kitap", "nesne")),
        relations=(("zeynep", "okumak", "kitap", "nesne", "POSITIVE"),),
        properties=(),
        temporal=(),
        negations=(),
        uid="sem-v1-004",
        phenomena=("object_relation", "case:belirtme"),
    ),
    GoldSentence(
        "Ali ata bindi.",
        entities=(("ali", "insan"), ("at", "hayvan")),
        relations=(("ali", "binmek", "at", "hedef", "POSITIVE"),),
        properties=(),
        temporal=(),
        negations=(),
        uid="sem-v1-005",
        phenomena=("movement", "case:yonelme"),
    ),
    GoldSentence(
        "Deniz bugün eve gelmedi.",
        entities=(("deniz", "insan"), ("ev", "mekan")),
        relations=(("deniz", "gelmek", "ev", "hedef", "NEGATIVE"),),
        properties=(),
        temporal=("bugün",),
        negations=("gelmek",),
        uid="sem-v1-006",
        phenomena=("movement", "negation", "temporal", "case:yonelme"),
    ),
    GoldSentence(
        "Fatma mavi bisikletle parka gitti.",
        entities=(("fatma", "insan"), ("bisiklet", "tasit"), ("park", "mekan")),
        relations=(("fatma", "gitmek", "park", "hedef", "POSITIVE"),),
        properties=(("bisiklet", "renk", "mavi"), ("fatma", "travel_mode", "bisiklet")),
        temporal=(),
        negations=(),
        uid="sem-v1-007",
        phenomena=("movement", "instrument", "property", "case:yonelme"),
    ),
    GoldSentence(
        "Veli okulda kitap okuyor.",
        entities=(("veli", "insan"), ("okul", "mekan"), ("kitap", "nesne")),
        relations=(("veli", "okumak", "okul", "konum", "POSITIVE"),
                   ("veli", "okumak", "kitap", "nesne", "POSITIVE")),
        properties=(),
        temporal=(),
        negations=(),
        uid="sem-v1-008",
        phenomena=("object_relation", "location", "case:bulunma"),
    ),
    GoldSentence(
        "Can İstanbul'dan trenle geldi.",
        entities=(("can", "insan"), ("istanbul", "mekan"), ("tren", "tasit")),
        relations=(("can", "gelmek", "istanbul", "kaynak", "POSITIVE"),),
        properties=(("can", "travel_mode", "tren"),),
        temporal=(),
        negations=(),
        uid="sem-v1-009",
        phenomena=("movement", "instrument", "case:ayrilma"),
    ),
    GoldSentence(
        "Kedi masada uyuyor.",
        entities=(("kedi", "hayvan"), ("masa", "nesne")),
        relations=(("kedi", "uyumak", "masa", "konum", "POSITIVE"),),
        properties=(),
        temporal=(),
        negations=(),
        uid="sem-v1-010",
        phenomena=("location", "case:bulunma"),
    ),
    # ── Kapsam dışı: hattın çıkarım YAPMAMASI beklenen yapılar ────────────
    GoldSentence(
        "Ali'nin dün aldığı kitabı Ayşe'ye verdiği söylendi.",
        entities=(), relations=(), properties=(), temporal=(), negations=(),
        in_scope=False,
        uid="sem-v1-011",
        phenomena=("out_of_scope", "subordinate_clause"),
    ),
    GoldSentence(
        "Öğretmen öğrencilere kitabı okuttu.",
        entities=(), relations=(), properties=(), temporal=(), negations=(),
        in_scope=False,
        uid="sem-v1-012",
        phenomena=("out_of_scope", "causative_voice"),
    ),
)
# Eski API adı v1'i gösterir; v2 bilinçli olarak ayrı fonksiyonla çağrılır.
GOLD_SET = GOLD_SET_V1


NAMES_V2: Tuple[Tuple[str, str], ...] = (
    ("ali", "Ali"), ("ayse", "Ayşe"), ("mehmet", "Mehmet"), ("veli", "Veli"),
    ("zeynep", "Zeynep"), ("fatma", "Fatma"), ("can", "Can"), ("deniz", "Deniz"),
)
DAT_PLACES_V2: Tuple[Tuple[str, str], ...] = (
    ("ankara", "Ankara'ya"), ("izmir", "İzmir'e"), ("bursa", "Bursa'ya"),
    ("okul", "okula"), ("ev", "eve"), ("park", "parka"),
    ("kutuphane", "kütüphaneye"),
)
ABL_PLACES_V2: Tuple[Tuple[str, str], ...] = (
    ("istanbul", "İstanbul'dan"), ("ankara", "Ankara'dan"),
    ("izmir", "İzmir'den"), ("bursa", "Bursa'dan"),
    ("okul", "okuldan"), ("ev", "evden"), ("park", "parktan"),
)
LOC_PLACES_V2: Tuple[Tuple[str, str], ...] = (
    ("ankara", "Ankara'da"), ("bursa", "Bursa'da"), ("okul", "okulda"),
    ("ev", "evde"), ("park", "parkta"), ("kutuphane", "kütüphanede"),
)
VEHICLES_INS_V2: Tuple[Tuple[str, str], ...] = (
    ("araba", "arabayla"), ("otobus", "otobüsle"), ("tren", "trenle"),
    ("bisiklet", "bisikletle"), ("ucak", "uçakla"), ("gemi", "gemiyle"),
)
VEHICLES_DAT_V2: Tuple[Tuple[str, str], ...] = (
    ("araba", "arabaya"), ("otobus", "otobüse"), ("tren", "trene"),
    ("bisiklet", "bisiklete"), ("ucak", "uçağa"), ("gemi", "gemiye"),
)
OBJECTS_ACC_V2: Tuple[Tuple[str, str], ...] = (
    ("kitap", "kitabı"), ("kalem", "kalemi"),
    ("masa", "masayı"), ("telefon", "telefonu"),
)
OBJECTS_BARE_V2: Tuple[Tuple[str, str], ...] = (
    ("kitap", "kitap"), ("kalem", "kalem"),
)
OBJECT_VERBS_V2: Tuple[Tuple[str, str], ...] = (
    ("okudu", "okumak"), ("gördü", "görmek"),
    ("aldı", "almak"), ("sevdi", "sevmek"),
)
ADVERBS_V2: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("dün ", ("dün",)), ("bugün ", ("bugün",)),
    ("yarın ", ("yarın",)), ("", ()),
)
ADJECTIVES_V2: Tuple[Tuple[str, str, str], ...] = (
    ("kırmızı", "renk", "kırmızı"), ("mavi", "renk", "mavi"),
    ("yeşil", "renk", "yeşil"), ("büyük", "boyut", "büyük"),
    ("küçük", "boyut", "küçük"), ("yeni", "durum", "yeni"),
    ("eski", "durum", "eski"), ("hızlı", "hiz", "hızlı"),
)
SPLITS_V2 = ("calibration", "heldout", "stress")


def _split_v2(index: int) -> str:
    return SPLITS_V2[index % len(SPLITS_V2)]


def _gold_v2(
    index: int,
    sentence: str,
    entities: Iterable[Tuple[str, str]],
    relations: Iterable[Tuple[str, str, str, str, str]],
    properties: Iterable[Tuple[str, str, str]] = (),
    temporal: Iterable[str] = (),
    negations: Iterable[str] = (),
    in_scope: bool = True,
    phenomena: Iterable[str] = (),
) -> GoldSentence:
    return GoldSentence(
        sentence=sentence,
        entities=tuple(entities),
        relations=tuple(relations),
        properties=tuple(properties),
        temporal=tuple(temporal),
        negations=tuple(negations),
        in_scope=in_scope,
        uid=f"sem-v2-{index:04d}",
        split=_split_v2(index),
        phenomena=tuple(phenomena),
        source="curated_template_v2",
    )


def build_semantic_gold_v2() -> Tuple[GoldSentence, ...]:
    """v2 altın setini deterministik ve gözden geçirilebilir biçimde kur.

    Şablonlar, kapsamı büyütmek içindir; beklenen semantik kümeler bu fonksiyonda
    açıkça yazılır. Bu yüzden benchmark self-consistency değil, kaynak-kodda
    denetlenebilir bir sözleşme/regresyon testidir.
    """
    gold: List[GoldSentence] = []
    i = 1

    # 1) Hedefe hareket + vasıta + opsiyonel zaman (42 cümle)
    for n in range(42):
        subj, subj_s = NAMES_V2[n % len(NAMES_V2)]
        place, place_s = DAT_PLACES_V2[(n * 2) % len(DAT_PLACES_V2)]
        vehicle, vehicle_s = VEHICLES_INS_V2[(n * 5) % len(VEHICLES_INS_V2)]
        adv_s, temporal = ADVERBS_V2[n % len(ADVERBS_V2)]
        sentence = f"{subj_s} {adv_s}{place_s} {vehicle_s} gitti."
        gold.append(_gold_v2(
            i, sentence,
            entities=((subj, "insan"), (place, "mekan"), (vehicle, "tasit")),
            relations=((subj, "gitmek", place, "hedef", "POSITIVE"),),
            properties=((subj, "travel_mode", vehicle),),
            temporal=temporal,
            phenomena=("movement", "instrument", "case:yonelme", "polarity:positive"),
        ))
        i += 1

    # 2) Kaynaktan gelme + vasıta (18 cümle)
    for n in range(18):
        subj, subj_s = NAMES_V2[(n + 3) % len(NAMES_V2)]
        place, place_s = ABL_PLACES_V2[(n * 3) % len(ABL_PLACES_V2)]
        vehicle, vehicle_s = VEHICLES_INS_V2[(n * 2 + 1) % len(VEHICLES_INS_V2)]
        adv_s, temporal = ADVERBS_V2[(n + 1) % len(ADVERBS_V2)]
        sentence = f"{subj_s} {adv_s}{place_s} {vehicle_s} geldi."
        gold.append(_gold_v2(
            i, sentence,
            entities=((subj, "insan"), (place, "mekan"), (vehicle, "tasit")),
            relations=((subj, "gelmek", place, "kaynak", "POSITIVE"),),
            properties=((subj, "travel_mode", vehicle),),
            temporal=temporal,
            phenomena=("movement", "instrument", "case:ayrilma", "polarity:positive"),
        ))
        i += 1

    # 3) Olumsuz gelecek hareket (18 cümle)
    for n in range(18):
        subj, subj_s = NAMES_V2[(n + 5) % len(NAMES_V2)]
        place, place_s = DAT_PLACES_V2[(n * 4 + 1) % len(DAT_PLACES_V2)]
        vehicle, vehicle_s = VEHICLES_INS_V2[(n * 3 + 2) % len(VEHICLES_INS_V2)]
        adv_s, temporal = ("yarın ", ("yarın",)) if n % 2 == 0 else ("bugün ", ("bugün",))
        sentence = f"{subj_s} {adv_s}{place_s} {vehicle_s} gitmeyecek."
        gold.append(_gold_v2(
            i, sentence,
            entities=((subj, "insan"), (place, "mekan"), (vehicle, "tasit")),
            relations=((subj, "gitmek", place, "hedef", "NEGATIVE"),),
            properties=((subj, "travel_mode", vehicle),),
            temporal=temporal,
            negations=("gitmek",),
            phenomena=("movement", "instrument", "case:yonelme", "negation", "temporal"),
        ))
        i += 1

    # 4) Olumsuz geçmiş varış (10 cümle)
    for n in range(10):
        subj, subj_s = NAMES_V2[(n + 1) % len(NAMES_V2)]
        place, place_s = DAT_PLACES_V2[(n * 5 + 2) % len(DAT_PLACES_V2)]
        vehicle, vehicle_s = VEHICLES_INS_V2[(n + 4) % len(VEHICLES_INS_V2)]
        adv_s, temporal = ADVERBS_V2[n % 2]
        sentence = f"{subj_s} {adv_s}{place_s} {vehicle_s} gelmedi."
        gold.append(_gold_v2(
            i, sentence,
            entities=((subj, "insan"), (place, "mekan"), (vehicle, "tasit")),
            relations=((subj, "gelmek", place, "hedef", "NEGATIVE"),),
            properties=((subj, "travel_mode", vehicle),),
            temporal=temporal,
            negations=("gelmek",),
            phenomena=("movement", "instrument", "case:yonelme", "negation", "temporal"),
        ))
        i += 1

    # 5) Belirtme durumlu nesne ilişkileri (8 cümle)
    for n in range(8):
        subj, subj_s = NAMES_V2[(n + 2) % len(NAMES_V2)]
        obj, obj_s = OBJECTS_ACC_V2[(n * 3) % len(OBJECTS_ACC_V2)]
        verb_s, predicate = OBJECT_VERBS_V2[(n * 5) % len(OBJECT_VERBS_V2)]
        sentence = f"{subj_s} {obj_s} {verb_s}."
        gold.append(_gold_v2(
            i, sentence,
            entities=((subj, "insan"), (obj, "nesne")),
            relations=((subj, predicate, obj, "nesne", "POSITIVE"),),
            phenomena=("object_relation", "case:belirtme", "polarity:positive"),
        ))
        i += 1

    # 6) Bulunma + belirtisiz nesne; iki relation beklenir (12 cümle)
    for n in range(12):
        subj, subj_s = NAMES_V2[(n + 4) % len(NAMES_V2)]
        place, place_s = LOC_PLACES_V2[(n * 5) % len(LOC_PLACES_V2)]
        obj, obj_s = OBJECTS_BARE_V2[n % len(OBJECTS_BARE_V2)]
        sentence = f"{subj_s} {place_s} {obj_s} okuyor."
        gold.append(_gold_v2(
            i, sentence,
            entities=((subj, "insan"), (place, "mekan"), (obj, "nesne")),
            relations=((subj, "okumak", place, "konum", "POSITIVE"),
                       (subj, "okumak", obj, "nesne", "POSITIVE")),
            phenomena=("object_relation", "location", "case:bulunma"),
        ))
        i += 1

    # 7) Açık sıfat/özellik + relation (20 cümle)
    for n in range(20):
        subj, subj_s = NAMES_V2[(n + 6) % len(NAMES_V2)]
        adj_s, prop_name, prop_value = ADJECTIVES_V2[(n * 3) % len(ADJECTIVES_V2)]
        adv_s, temporal = ADVERBS_V2[(n // 8) % len(ADVERBS_V2)]
        temporal_phenomena = ("temporal",) if temporal else ()
        if n % 2 == 0:
            vehicle, vehicle_s = VEHICLES_DAT_V2[(n * 5) % len(VEHICLES_DAT_V2)]
            sentence = f"{subj_s} {adv_s}{adj_s} {vehicle_s} bindi."
            entities = ((subj, "insan"), (vehicle, "tasit"))
            relations = ((subj, "binmek", vehicle, "hedef", "POSITIVE"),)
            properties = ((vehicle, prop_name, prop_value),)
            phenomena = ("property", "movement", "case:yonelme", "polarity:positive") + temporal_phenomena
        else:
            obj, obj_s = OBJECTS_ACC_V2[(n * 7) % len(OBJECTS_ACC_V2)]
            verb_s, predicate = OBJECT_VERBS_V2[(n // 2) % len(OBJECT_VERBS_V2)]
            sentence = f"{subj_s} {adv_s}{adj_s} {obj_s} {verb_s}."
            entities = ((subj, "insan"), (obj, "nesne"))
            relations = ((subj, predicate, obj, "nesne", "POSITIVE"),)
            properties = ((obj, prop_name, prop_value),)
            phenomena = ("property", "object_relation", "case:belirtme", "polarity:positive") + temporal_phenomena
        gold.append(_gold_v2(
            i, sentence,
            entities=entities,
            relations=relations,
            properties=properties,
            temporal=temporal,
            phenomena=phenomena,
        ))
        i += 1

    # 8) Kapsam dışı / çekimserlik negatifleri (8 cümle)
    out_of_scope = (
        ("Ali'nin dün aldığı kitabı Ayşe'ye verdiği söylendi.",
         ("out_of_scope", "subordinate_clause")),
        ("Öğretmen öğrencilere kitabı okuttu.",
         ("out_of_scope", "causative_voice")),
        ("Kedi masaya çıkartıldı.",
         ("out_of_scope", "passive_or_causative_voice")),
        ("Deniz, Ali'nin evden geldiğini düşündü.",
         ("out_of_scope", "embedded_clause")),
        ("Mehmet kitabın okunduğunu duydu.",
         ("out_of_scope", "passive_subordinate_clause")),
        ("Veli'nin parka gitmesi istendi.",
         ("out_of_scope", "nominalized_clause")),
        ("Ayşe kalemi okutmadı.",
         ("out_of_scope", "causative_voice", "negation")),
        ("Fatma kapıyı açtırdı.",
         ("out_of_scope", "causative_voice")),
    )
    for sentence, phenomena in out_of_scope:
        gold.append(_gold_v2(
            i, sentence,
            entities=(), relations=(), properties=(), temporal=(), negations=(),
            in_scope=False,
            phenomena=phenomena,
        ))
        i += 1

    if len({g.sentence for g in gold}) != len(gold):
        raise RuntimeError("semantic v2 gold set duplicate sentence üretti")
    return tuple(gold)


GOLD_SET_V2: Tuple[GoldSentence, ...] = build_semantic_gold_v2()


def _prf(dogru: int, tahmin: int, altin: int) -> Dict[str, float]:
    if tahmin == 0 and altin == 0:
        precision = recall = f1 = 1.0
    else:
        precision = dogru / tahmin if tahmin else 0.0
        recall = dogru / altin if altin else 0.0
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


def _altin_kumeler(gold: GoldSentence) -> Dict[str, Set[Any]]:
    return {
        "entity": set(gold.entities),
        "relation": set(gold.relations),
        "property": {(e, a, str(d)) for e, a, d in gold.properties},
        "temporal": set(gold.temporal),
        "negation": set(gold.negations),
    }


def _hash_payload(gold_set: Sequence[GoldSentence], include_annotations: bool) -> str:
    if include_annotations:
        payload: Any = [asdict(g) for g in gold_set]
    else:
        payload = [g.sentence for g in gold_set]
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _new_bucket() -> Dict[str, Any]:
    return {
        "layers": {katman: {"tp": 0, "pred": 0, "gold": 0} for katman in LAYERS},
        "sentence_count": 0,
        "in_scope_count": 0,
        "out_of_scope_count": 0,
        "relation_exact": 0,
        "relation_non_empty": 0,
        "out_of_scope_fp": 0,
    }


def _update_bucket(
    bucket: Dict[str, Any],
    gold: GoldSentence,
    tahmin: Dict[str, Set[Any]],
    altin: Dict[str, Set[Any]],
) -> None:
    bucket["sentence_count"] += 1
    for katman in LAYERS:
        # Kapsam dışı cümlelerde yalnız ilişki katmanı puanlanır: varlık
        # veya zaman çıkarmak orada bir hata değildir (cümle Türkçedir ve
        # varlıklar gerçekten oradadır); asıl iddia yönlü ilişkidir.
        if not gold.in_scope and katman != "relation":
            continue
        ortak = tahmin[katman] & altin[katman]
        bucket["layers"][katman]["tp"] += len(ortak)
        bucket["layers"][katman]["pred"] += len(tahmin[katman])
        bucket["layers"][katman]["gold"] += len(altin[katman])
    if gold.in_scope:
        bucket["in_scope_count"] += 1
        if tahmin["relation"]:
            bucket["relation_non_empty"] += 1
        if tahmin["relation"] == altin["relation"] and altin["relation"]:
            bucket["relation_exact"] += 1
    else:
        bucket["out_of_scope_count"] += 1
        if tahmin["relation"]:
            bucket["out_of_scope_fp"] += 1


def _finalize_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
    layers = {
        katman: _prf(s["tp"], s["pred"], s["gold"])
        for katman, s in bucket["layers"].items()
    }
    in_scope = int(bucket["in_scope_count"])
    out_scope = int(bucket["out_of_scope_count"])
    return {
        "sentence_count": int(bucket["sentence_count"]),
        "in_scope_count": in_scope,
        "out_of_scope_count": out_scope,
        "layers": layers,
        "relation_exact_match": round(bucket["relation_exact"] / in_scope, 6) if in_scope else 0.0,
        "relation_coverage": round(bucket["relation_non_empty"] / in_scope, 6) if in_scope else 0.0,
        "out_of_scope_false_positive_rate": (
            round(bucket["out_of_scope_fp"] / out_scope, 6) if out_scope else 0.0),
    }


def _dataset_summary(gold_set: Sequence[GoldSentence]) -> Dict[str, Any]:
    split_counts = Counter(g.split for g in gold_set)
    phenomenon_counts: Counter[str] = Counter()
    predicates: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    polarities: Counter[str] = Counter()
    for gold in gold_set:
        phenomenon_counts.update(gold.phenomena)
        for _, predicate, _, role, polarity in gold.relations:
            predicates[predicate] += 1
            roles[role] += 1
            polarities[polarity] += 1
    return {
        "splits": dict(sorted(split_counts.items())),
        "phenomena": dict(sorted(phenomenon_counts.items())),
        "predicate_coverage": dict(sorted(predicates.items())),
        "role_coverage": dict(sorted(roles.items())),
        "polarity_coverage": dict(sorted(polarities.items())),
        "unique_sentence_count": len({g.sentence for g in gold_set}),
        "construct": "curated_gold_protocol_not_general_turkish_ner_re_score",
    }


@dataclass
class SemanticExtractionReport:
    protocol: str
    dataset_hash: str
    annotation_hash: str
    sentence_count: int
    in_scope_count: int
    out_of_scope_count: int
    layers: Dict[str, Dict[str, float]]
    relation_exact_match: float
    relation_coverage: float
    extraction_yield: float
    over_extraction_rate: float
    out_of_scope_false_positive_rate: float
    abstention_rate: float
    mean_relation_confidence: float
    per_sentence: List[Dict[str, Any]]
    checks: Dict[str, bool]
    dataset_summary: Dict[str, Any] = field(default_factory=dict)
    split_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    phenomenon_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        return semantic_extraction_markdown(self)


def run_semantic_extraction_benchmark(
    gold_set: Sequence[GoldSentence] = GOLD_SET,
    *,
    protocol: str = PROTOCOL,
    min_relation_f1: float = 0.80,
    min_entity_f1: float = 0.90,
    min_property_precision: float = 0.80,
    min_extraction_yield: float = 0.80,
) -> SemanticExtractionReport:
    """Altın set üzerinde katman katman precision/recall/F1 ölç."""
    if not gold_set:
        raise ValueError("gold_set boş olamaz")

    overall = _new_bucket()
    split_buckets: DefaultDict[str, Dict[str, Any]] = defaultdict(_new_bucket)
    phenomenon_buckets: DefaultDict[str, Dict[str, Any]] = defaultdict(_new_bucket)
    per_sentence: List[Dict[str, Any]] = []
    cekimser = 0
    guvenler: List[float] = []

    for gold in gold_set:
        cikarim = cumle_coz(gold.sentence)
        tahmin = _tahmin_kumeler(cikarim)
        altin = _altin_kumeler(gold)
        guvenler.extend(r.confidence for r in cikarim.relations)

        if not cikarim.relations:
            cekimser += 1

        _update_bucket(overall, gold, tahmin, altin)
        _update_bucket(split_buckets[gold.split], gold, tahmin, altin)
        for phenomenon in gold.phenomena or ("unspecified",):
            _update_bucket(phenomenon_buckets[phenomenon], gold, tahmin, altin)

        per_sentence.append({
            "uid": gold.uid,
            "split": gold.split,
            "phenomena": list(gold.phenomena),
            "source": gold.source,
            "sentence": gold.sentence,
            "in_scope": gold.in_scope,
            "predicted_relations": sorted(map(list, tahmin["relation"])),
            "gold_relations": sorted(map(list, altin["relation"])),
            "missed": sorted(map(list, altin["relation"] - tahmin["relation"])),
            "spurious": sorted(map(list, tahmin["relation"] - altin["relation"])),
            "skipped_reasons": list(cikarim.skipped_reasons),
        })

    finalized = _finalize_bucket(overall)
    katmanlar = finalized["layers"]
    toplam_tahmin = sum(s["pred"] for s in overall["layers"].values())
    toplam_tp = sum(s["tp"] for s in overall["layers"].values())
    asiri = round((toplam_tahmin - toplam_tp) / toplam_tahmin, 6) if toplam_tahmin else 0.0
    extraction_yield = finalized["relation_exact_match"]
    summary = _dataset_summary(gold_set)
    splits_present = set(summary["splits"])

    kontroller = {
        "relation_f1_at_least_0_80": katmanlar["relation"]["f1"] >= min_relation_f1,
        "entity_f1_at_least_0_90": katmanlar["entity"]["f1"] >= min_entity_f1,
        "negation_recall_perfect": katmanlar["negation"]["recall"] >= 1.0,
        "temporal_recall_perfect": katmanlar["temporal"]["recall"] >= 1.0,
        # Kapsam dışı cümlelerden ilişki uydurulmamalı.
        "no_out_of_scope_hallucination": overall["out_of_scope_fp"] == 0,
        "property_precision_at_least_0_80": (
            katmanlar["property"]["precision"] >= min_property_precision),
        "extraction_yield_at_least_threshold": extraction_yield >= min_extraction_yield,
    }
    if protocol == PROTOCOL_V2:
        kontroller.update({
            "v2_gold_size_at_least_100": len(gold_set) >= V2_MIN_SENTENCES,
            "v2_unique_sentences": summary["unique_sentence_count"] == len(gold_set),
            "v2_split_metadata_present": set(SPLITS_V2) <= splits_present,
            "v2_out_of_scope_negatives_present": summary["phenomena"].get("out_of_scope", 0) >= 8,
            "v2_extraction_yield_at_least_0_85": extraction_yield >= V2_MIN_EXTRACTION_YIELD,
        })

    bulgular = [
        f"{len(gold_set)} etiketli cümle ({overall['in_scope_count']} kapsam içi, "
        f"{overall['out_of_scope_count']} kapsam dışı).",
        "Katman F1: " + ", ".join(
            f"{k}={katmanlar[k]['f1']:.4f}" for k in LAYERS) + ".",
        f"İlişki tam eşleşme / extraction_yield: {extraction_yield:.4f}.",
        f"Aşırı çıkarım oranı {asiri:.4f}; kapsam dışı cümlelerde yanlış "
        f"ilişki {overall['out_of_scope_fp']}/{overall['out_of_scope_count']}.",
        "Kapsam özeti: " + ", ".join(
            f"{k}={v}" for k, v in summary["predicate_coverage"].items()) + ".",
    ]

    sinirlar = [
        "Bu protokol istatistiksel bir Türkçe NER/RE değerlendirmesi DEĞİLDİR. "
        "Sonuçlar sınırlı, kural-tabanlı hattın kapsamını gösterir; genel "
        "Türkçe performansı veya dış geçerlilik iddiası değildir.",
        "v2 genişletmesi gerçek dış korpus veya bağımsız insan anotasyonu değil, "
        "kaynakta denetlenebilir şablon/altın etiket sözleşmesidir.",
        "Ontolojiden türetilen özellikler (canlı, binilebilir) altın sette yer "
        "almaz ve precision hesabına katılmaz; onlar metinden değil tip "
        "varsayımından gelir ve düşük güvenle yazılır.",
        "Hat kural tabanlıdır: sözlük büyüdükçe recall artar, bu bir öğrenme "
        "sonucu değildir.",
        "Kapsam dışı örnekler precision güvenliği için eklidir ama fuzzing veya "
        "geniş haber/sosyal medya dağılımı yerine geçmez.",
    ]

    return SemanticExtractionReport(
        protocol=protocol,
        dataset_hash=_hash_payload(gold_set, include_annotations=False),
        annotation_hash=_hash_payload(gold_set, include_annotations=True),
        sentence_count=len(gold_set),
        in_scope_count=overall["in_scope_count"],
        out_of_scope_count=overall["out_of_scope_count"],
        layers=katmanlar,
        relation_exact_match=finalized["relation_exact_match"],
        relation_coverage=finalized["relation_coverage"],
        extraction_yield=extraction_yield,
        over_extraction_rate=asiri,
        out_of_scope_false_positive_rate=finalized["out_of_scope_false_positive_rate"],
        abstention_rate=round(cekimser / len(gold_set), 6),
        mean_relation_confidence=(
            round(sum(guvenler) / len(guvenler), 6) if guvenler else 0.0),
        per_sentence=per_sentence,
        checks=kontroller,
        dataset_summary=summary,
        split_metrics={k: _finalize_bucket(v) for k, v in sorted(split_buckets.items())},
        phenomenon_metrics={k: _finalize_bucket(v) for k, v in sorted(phenomenon_buckets.items())},
        findings=bulgular,
        limitations=sinirlar,
    )


def run_semantic_extraction_v2_benchmark(
    gold_set: Sequence[GoldSentence] = GOLD_SET_V2,
) -> SemanticExtractionReport:
    """Semantic Extraction v2 gold protokolünü çalıştır."""
    return run_semantic_extraction_benchmark(
        gold_set,
        protocol=PROTOCOL_V2,
        min_relation_f1=0.95,
        min_entity_f1=0.95,
        min_property_precision=0.95,
        min_extraction_yield=V2_MIN_EXTRACTION_YIELD,
    )


def semantic_extraction_markdown(report: SemanticExtractionReport) -> str:
    satirlar = [
        "# Türkçe Semantik Çıkarım Benchmarkı (P0-5)",
        "",
        f"Protokol: `{report.protocol}` · veri imzası: `{report.dataset_hash}` · "
        f"anotasyon imzası: `{report.annotation_hash}` · {report.sentence_count} cümle "
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
        f"- Extraction yield: **{report.extraction_yield:.4f}**",
        f"- Relation coverage: {report.relation_coverage:.4f}",
        f"- Aşırı çıkarım oranı: {report.over_extraction_rate:.4f}",
        f"- Kapsam dışı yanlış pozitif: {report.out_of_scope_false_positive_rate:.4f}",
        f"- Çekimserlik (ilişki çıkaramama): {report.abstention_rate:.4f}",
        f"- Ortalama ilişki güveni: {report.mean_relation_confidence:.4f}",
        "",
        "## Veri kapsamı",
        "",
        "- Splitler: " + ", ".join(
            f"{k}={v}" for k, v in report.dataset_summary.get("splits", {}).items()),
        "- Yüklem kapsamı: " + ", ".join(
            f"{k}={v}" for k, v in report.dataset_summary.get("predicate_coverage", {}).items()),
        "- Rol kapsamı: " + ", ".join(
            f"{k}={v}" for k, v in report.dataset_summary.get("role_coverage", {}).items()),
        "- Fenomenler: " + ", ".join(
            f"{k}={v}" for k, v in report.dataset_summary.get("phenomena", {}).items()),
        "",
        "## Split metrikleri",
        "",
        "| split | cümle | relation F1 | exact/yield | out-scope FP |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, metrics in report.split_metrics.items():
        satirlar.append(
            f"| {split} | {metrics['sentence_count']} | "
            f"{metrics['layers']['relation']['f1']:.4f} | "
            f"{metrics['relation_exact_match']:.4f} | "
            f"{metrics['out_of_scope_false_positive_rate']:.4f} |")
    satirlar += [
        "", "## Kabul kapıları", "", "| kapı | sonuç |", "|---|---|",
    ]
    for ad, sonuc in report.checks.items():
        satirlar.append(f"| {ad} | {'GEÇTİ' if sonuc else 'KALDI'} |")
    satirlar += ["", "## Hatalı cümleler", ""]
    hatali = [s for s in report.per_sentence if s["missed"] or s["spurious"]]
    if hatali:
        for s in hatali[:50]:
            satirlar.append(
                f"- `{s['uid'] or '-'} · {s['sentence']}` → eksik: {s['missed']}, "
                f"fazla: {s['spurious']}")
        if len(hatali) > 50:
            satirlar.append(f"- … {len(hatali) - 50} ek hata JSON raporda.")
    else:
        satirlar.append("- Yok: her cümlede ilişki kümesi altınla birebir.")
    satirlar += ["", "## Bulgular", ""]
    satirlar += [f"- {b}" for b in report.findings]
    satirlar += ["", "## Sınırlar ve dış geçerlilik", ""]
    satirlar += [f"- {s}" for s in report.limitations]
    return "\n".join(satirlar) + "\n"


__all__ = [
    "PROTOCOL", "PROTOCOL_V1", "PROTOCOL_V2", "LAYERS",
    "GOLD_SET", "GOLD_SET_V1", "GOLD_SET_V2", "GoldSentence",
    "SemanticExtractionReport", "build_semantic_gold_v2",
    "run_semantic_extraction_benchmark", "run_semantic_extraction_v2_benchmark",
    "semantic_extraction_markdown",
]
