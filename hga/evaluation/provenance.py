# -*- coding: utf-8 -*-
"""
Faz 27/28 — Köken (Provenance) Denetimi ve Gerçek Veri Sözleşmesi
===================================================================

Bir bilgi tabanının bilimsel olarak savunulabilir olması için "biliyorum"
demesi yetmez; **"nereden biliyorum"** sorusunu cevaplayabilmesi gerekir.

Bu modül iki şey yapar:

1. **Denetim** (`audit_provenance`): bilgi tabanındaki her olgunun kökenini
   kontrol eder. `REAL_DATA` kaynaklı bir olgunun `source_url` ve
   `document_hash` taşıması **zorunludur**; taşımıyorsa "yetim olgu" sayılır.
   Türetilmiş/sentetik kaynaklar (`VERIFIED_RULE`, `DERIVED`,
   `MODEL_GENERATED`) muaftır çünkü onların kökeni dış belge değildir.

2. **Yeniden üretilebilirlik** (`verify_document_hashes`): kaydedilen
   `document_hash` gerçekten belgenin içeriğinin SHA-256'sı mı? Belge sonradan
   değiştiyse bu yakalanmalıdır — aksi hâlde "kaynak gösterdim" iddiası
   denetlenemez.

## Neden bu bir "kırma" testi?

Milestone koşusunda `C_V/C_E = 0.9965` ölçülmüştü. Bu yüksek oran **yalnız
aritmetik oracle tam olduğu için** mümkündü. Gerçek metinde doğrulama oracle'ı
yoktur; elimizdeki tek şey kaynağın izlenebilirliğidir. Provenance kapsamı
düşükse, verification istatistikleri havada kalır.

Bu modül yeni bir veri toplayıcı EKLEMEZ (feature freeze). Mevcut
`hga/experience/corpus.py` yolunu köken taşıyacak şekilde ölçer ve eksikliği
raporlar.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hga.knowledge.schemas import KaynakTuru

# Dış belgeye dayanması BEKLENEN kaynaklar: köken zorunlu.
DIS_KAYNAKLAR = frozenset({
    KaynakTuru.REAL_DATA,
    KaynakTuru.EXTERNAL_VERIFIED,
    KaynakTuru.HUMAN_CONFIRMED,
})

# Kökeni dış belge OLMAYAN kaynaklar: muaf.
TURETILMIS_KAYNAKLAR = frozenset({
    KaynakTuru.VERIFIED_RULE,
    KaynakTuru.DERIVED,
    KaynakTuru.MODEL_GENERATED,
    KaynakTuru.FREE_GENERATION,
})

ZORUNLU_ALANLAR = ("source_url", "document_hash")

# Kullanıcıya gösterilecek tam köken grafının çekirdek sırası. Graph dallanır
# (iki Entity düğümü vardır) ama bu sıra "Bunu neden biliyorsun?" sorusuna
# okunabilir bir omurga verir.
PROVENANCE_CHAIN_KINDS: Tuple[str, ...] = (
    "Source", "Document", "Sentence", "Extraction", "Entity", "Relation",
    "RelationFact", "KnowledgeVersion", "Experience", "Verification", "Answer",
)


def document_hash(content: str) -> str:
    """Belge içeriğinin kanonik SHA-256'sı (satır sonu normalize edilir)."""
    normalize = (content or "").replace("\r\n", "\n").strip()
    return hashlib.sha256(normalize.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProvenanceNode:
    """Köken grafındaki tek düğüm.

    ``kind`` değerleri ``PROVENANCE_CHAIN_KINDS`` sözleşmesinden gelir. Metadata
    insan açıklaması için zengindir ama düğüm kimliği deterministik ve kısadır.
    """

    node_id: str
    kind: str
    label: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProvenanceEdge:
    """Köken grafında yönlü bağlantı."""

    source: str
    target: str
    relation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProvenanceChain:
    """Source → ... → Answer zincirinin makine-okunur görünümü."""

    nodes: List[ProvenanceNode]
    edges: List[ProvenanceEdge]
    missing: List[str] = field(default_factory=list)
    external_trace_required: bool = False

    @property
    def complete_external_trace(self) -> bool:
        """Dış kaynaklı olguda zorunlu köken alanları tam mı?"""
        return not self.external_trace_required or not self.missing

    def kinds(self) -> List[str]:
        """Düğüm türlerini resmi omurga sırasıyla döndür.

        Graph dallandığı için eklenme sırası her zaman okunabilir değildir
        (ör. RelationFact düğümü pratikte önce kurulabilir). Kullanıcıya
        gösterilen sıra Source→...→Answer sözleşmesini izler.
        """
        siralama = {kind: index for index, kind in enumerate(PROVENANCE_CHAIN_KINDS)}
        return [node.kind for node in sorted(
            self.nodes, key=lambda node: (siralama.get(node.kind, 10_000), node.node_id))]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "missing": list(self.missing),
            "external_trace_required": self.external_trace_required,
            "complete_external_trace": self.complete_external_trace,
            "chain_kinds": self.kinds(),
        }


@dataclass
class YetimOlgu:
    """Kökeni izlenemeyen olgu kaydı."""

    subject_id: str
    relation_id: str
    object_id: str
    source: str
    missing_fields: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProvenanceRaporu:
    total_facts: int
    external_facts: int          # köken zorunlu olanlar
    derived_facts: int           # muaf olanlar
    traceable_facts: int         # kökeni tam olanlar
    orphan_facts: int            # köken zorunlu ama eksik
    coverage: float              # traceable / external
    unique_documents: int
    unique_sources: int
    sentence_coverage: float     # ham cümlesi kayıtlı olan dış olgu oranı
    by_source: Dict[str, Dict[str, int]] = field(default_factory=dict)
    orphans: List[Dict[str, Any]] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        """Dış kaynaklı her olgunun kökeni tam mı?"""
        return self.orphan_facts == 0

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "clean": self.clean}

    def assert_clean(self) -> None:
        """Köken eksikse açık hata ver — sessizce geçme."""
        if not self.clean:
            ornek = self.orphans[:3]
            raise AssertionError(
                f"{self.orphan_facts} olgunun kökeni izlenemiyor "
                f"(kapsam={self.coverage:.4f}). Örnek: {ornek}"
            )

    def markdown(self) -> str:
        satirlar = [
            "| Kaynak | Olgu | Kökeni tam | Yetim |",
            "|---|---:|---:|---:|",
        ]
        for ad in sorted(self.by_source):
            v = self.by_source[ad]
            satirlar.append(f"| `{ad}` | {v['total']:,} | {v['traceable']:,} | "
                            f"{v['orphan']:,} |")
        satirlar.append("")
        satirlar.append(f"- Toplam olgu: **{self.total_facts:,}** "
                        f"(dış kaynaklı {self.external_facts:,}, "
                        f"türetilmiş {self.derived_facts:,})")
        satirlar.append(f"- Köken kapsamı: **{self.coverage:.4f}** "
                        f"(yetim: {self.orphan_facts:,})")
        satirlar.append(f"- Benzersiz belge: {self.unique_documents:,} · "
                        f"benzersiz kaynak: {self.unique_sources:,}")
        satirlar.append(f"- Ham cümle kapsamı: {self.sentence_coverage:.4f}")
        return "\n".join(satirlar)


def _node_id(kind: str, *parts: Any) -> str:
    payload = "|".join(str(part) for part in parts if part is not None)
    digest = hashlib.sha256(f"{kind}|{payload}".encode("utf-8")).hexdigest()[:12]
    return f"{kind.lower()}:{digest}"


def build_provenance_chain(
    store,
    fact,
    *,
    experience=None,
    answer: Optional[str] = None,
) -> ProvenanceChain:
    """Tek olgu için Source→Document→Sentence→Extraction→... zinciri kur.

    Amaç yalnız ``source_url`` ve ``document_hash`` var mı diye bakmak değildir;
    kullanıcının "Bunu neden biliyorsun?" sorusuna cümle, çıkarıcı, varlıklar,
    ilişki, bilgi sürümü, isteğe bağlı deneyim/doğrulama ve cevap düğümlerini
    içeren tam bir graph döndürmektir.

    ``experience`` ve ``answer`` opsiyoneldir çünkü her olgu bir cevap üretim
    yolundan gelmeyebilir. Verildiğinde zincirin sonuna eklenir; verilmediğinde
    ``missing`` sayılmaz.
    """
    nodes: List[ProvenanceNode] = []
    edges: List[ProvenanceEdge] = []
    missing: List[str] = []
    by_key: Dict[Tuple[str, str], ProvenanceNode] = {}

    def add(kind: str, label: str, metadata: Optional[Dict[str, Any]] = None,
            node_id: Optional[str] = None) -> str:
        nid = node_id or _node_id(kind, label, metadata)
        key = (kind, nid)
        if key not in by_key:
            node = ProvenanceNode(nid, kind, label, dict(metadata or {}))
            by_key[key] = node
            nodes.append(node)
        return nid

    def edge(src: Optional[str], dst: Optional[str], relation: str) -> None:
        if src and dst:
            edges.append(ProvenanceEdge(src, dst, relation))

    external_required = getattr(fact, "source", None) in DIS_KAYNAKLAR

    source_id = document_id = sentence_id = extraction_id = None
    if getattr(fact, "source_url", None):
        source_id = add("Source", str(fact.source_url), {
            "source_url": fact.source_url,
            "source_type": getattr(getattr(fact, "source", None), "value", str(getattr(fact, "source", ""))),
            "retrieved_at": getattr(fact, "retrieved_at", None),
        })
    elif external_required:
        missing.append("Source.source_url")

    if getattr(fact, "document_hash", None):
        document_id = add("Document", str(fact.document_hash)[:16], {
            "document_hash": fact.document_hash,
        })
        edge(source_id, document_id, "source_contains_document")
    elif external_required:
        missing.append("Document.document_hash")

    if getattr(fact, "sentence", None):
        sentence_id = add("Sentence", str(fact.sentence), {"sentence": fact.sentence})
        edge(document_id, sentence_id, "document_contains_sentence")
    elif external_required:
        missing.append("Sentence.text")

    if getattr(fact, "extractor", None):
        extraction_id = add("Extraction", str(fact.extractor), {
            "extractor": fact.extractor,
            "confidence": getattr(fact, "confidence", None),
        })
        edge(sentence_id, extraction_id, "sentence_processed_by_extractor")
    elif external_required:
        missing.append("Extraction.extractor")

    # Fact düğümü omurga için her zaman oluşturulur; varlık/relation kayıtları
    # eksikse missing'e düşer ama graph bozulmaz.
    fact_id = add("RelationFact", "/".join(fact.uclusu), {
        "subject_id": fact.subject_id,
        "relation_id": fact.relation_id,
        "object_id": fact.object_id,
        "score": fact.score,
        "confidence": fact.confidence,
        "source": getattr(getattr(fact, "source", None), "value", str(getattr(fact, "source", ""))),
        "fact_version": getattr(fact, "version", None),
    })
    edge(extraction_id, fact_id, "extraction_asserted_fact")

    for role, entity_id in (("subject", fact.subject_id), ("object", fact.object_id)):
        try:
            entity = store.entities.getir(entity_id)
        except Exception:
            missing.append(f"Entity.{role}:{entity_id}")
            continue
        entity_node_id = add("Entity", entity.token, {
            "entity_id": entity.entity_id,
            "entity_type": entity.entity_type,
            "role": role,
            "confidence": entity.confidence,
            "version": entity.version,
        }, node_id=f"entity:{entity.entity_id}")
        edge(extraction_id, entity_node_id, f"extraction_identified_{role}")
        edge(entity_node_id, fact_id, f"{role}_participates_in_fact")

    try:
        relation = store.relations.iliski_al(fact.relation_id)
    except Exception:
        missing.append(f"Relation:{fact.relation_id}")
    else:
        relation_node_id = add("Relation", relation.token, {
            "relation_id": relation.relation_id,
            "subject_types": list(relation.subject_types),
            "object_types": list(relation.object_types),
            "confidence": relation.confidence,
            "version": relation.version,
        }, node_id=f"relation:{relation.relation_id}")
        edge(extraction_id, relation_node_id, "extraction_identified_relation")
        edge(relation_node_id, fact_id, "relation_labels_fact")

    version_id = add("KnowledgeVersion", f"K{getattr(store, 'versiyon', None)}", {
        "knowledge_version": getattr(store, "versiyon", None),
    }, node_id=f"knowledge-version:{getattr(store, 'versiyon', None)}")
    edge(fact_id, version_id, "fact_stored_in_version")

    last_id = version_id
    if experience is not None:
        exp_id = add("Experience", getattr(experience, "experience_id", "experience"), {
            "experience_id": getattr(experience, "experience_id", None),
            "state": getattr(getattr(experience, "state", None), "value", str(getattr(experience, "state", ""))),
            "source": getattr(getattr(experience, "source", None), "value", str(getattr(experience, "source", ""))),
            "source_confidence": getattr(experience, "source_confidence", None),
        }, node_id=f"experience:{getattr(experience, 'experience_id', 'unknown')}")
        edge(last_id, exp_id, "knowledge_version_supports_experience")
        last_id = exp_id
        verified_by = getattr(experience, "verified_by", None)
        state = getattr(getattr(experience, "state", None), "value", str(getattr(experience, "state", "")))
        if verified_by or state in {"VERIFIED", "INVALID", "UNCERTAIN"}:
            ver_id = add("Verification", verified_by or state, {
                "verified_by": verified_by,
                "state": state,
                "belirsizlik_sebebi": getattr(getattr(experience, "belirsizlik_sebebi", None), "value", None),
            })
            edge(last_id, ver_id, "experience_checked_by_verifier")
            last_id = ver_id

    if answer is not None:
        answer_id = add("Answer", answer[:80], {"answer": answer})
        edge(last_id, answer_id, "trace_supports_answer")

    return ProvenanceChain(
        nodes=nodes, edges=edges, missing=missing,
        external_trace_required=external_required,
    )


def audit_provenance(store, max_orphan_samples: int = 25) -> ProvenanceRaporu:
    """Bilgi tabanındaki tüm olguların kökenini denetle."""
    olgular = list(store.relations.olgular())
    kaynak_istat: Dict[str, Dict[str, int]] = {}
    yetimler: List[Dict[str, Any]] = []
    belgeler = set()
    adresler = set()
    izlenebilir = dis_toplam = turetilmis = cumleli = 0

    for olgu in olgular:
        kaynak = olgu.source.value if hasattr(olgu.source, "value") else str(olgu.source)
        istat = kaynak_istat.setdefault(
            kaynak, {"total": 0, "traceable": 0, "orphan": 0})
        istat["total"] += 1

        if olgu.document_hash:
            belgeler.add(olgu.document_hash)
        if olgu.source_url:
            adresler.add(olgu.source_url)

        koken_zorunlu = olgu.source in DIS_KAYNAKLAR
        if not koken_zorunlu:
            turetilmis += 1
            istat["traceable"] += 1     # muaf: eksiklik sayılmaz
            continue

        dis_toplam += 1
        eksik = [ad for ad in ZORUNLU_ALANLAR if not getattr(olgu, ad, None)]
        if eksik:
            istat["orphan"] += 1
            if len(yetimler) < int(max_orphan_samples):
                yetimler.append(YetimOlgu(
                    subject_id=olgu.subject_id, relation_id=olgu.relation_id,
                    object_id=olgu.object_id, source=kaynak,
                    missing_fields=eksik,
                ).to_dict())
        else:
            izlenebilir += 1
            istat["traceable"] += 1
            if olgu.sentence:
                cumleli += 1

    yetim_sayisi = dis_toplam - izlenebilir
    kapsam = round(izlenebilir / dis_toplam, 6) if dis_toplam else 1.0
    cumle_kapsam = round(cumleli / dis_toplam, 6) if dis_toplam else 0.0

    bulgular = [
        f"Dış kaynaklı {dis_toplam:,} olgunun {izlenebilir:,} tanesinin kökeni "
        f"tam (kapsam={kapsam:.4f}); {yetim_sayisi:,} yetim olgu var.",
        f"Türetilmiş/sentetik {turetilmis:,} olgu köken zorunluluğundan muaftır "
        f"— kökenleri dış belge değil, kural veya model çıktısıdır.",
    ]
    if yetim_sayisi:
        bulgular.append(
            "UYARI: yetim olgular 'nereden biliyorum' sorusunu cevaplayamaz. "
            "Bu olgulara dayanan verification istatistikleri bağımsız olarak "
            "denetlenemez.")
    else:
        bulgular.append(
            "Her dış kaynaklı olgu bir belgeye ve o belgenin hash'ine "
            "bağlanabiliyor: verification iddiaları denetlenebilir.")
    return ProvenanceRaporu(
        total_facts=len(olgular), external_facts=dis_toplam,
        derived_facts=turetilmis, traceable_facts=izlenebilir,
        orphan_facts=yetim_sayisi, coverage=kapsam,
        unique_documents=len(belgeler), unique_sources=len(adresler),
        sentence_coverage=cumle_kapsam, by_source=kaynak_istat,
        orphans=yetimler, findings=bulgular,
    )


@dataclass
class HashDogrulamaRaporu:
    """Kaydedilen document_hash gerçekten belgenin içeriğine mi ait?"""

    checked: int
    matched: int
    mismatched: int
    unknown_documents: int
    mismatches: List[Dict[str, str]] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.mismatched == 0 and self.unknown_documents == 0

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "clean": self.clean}


def verify_document_hashes(
    store,
    documents: Mapping[str, str],
    max_samples: int = 25,
) -> HashDogrulamaRaporu:
    """Olguların ``document_hash``'lerini gerçek belge içerikleriyle karşılaştır.

    ``documents``: ``{source_url: içerik}``. Belge sonradan değiştiyse hash
    tutmaz ve bu **yakalanır** — "kaynak gösterdim" iddiasının denetlenebilir
    olması bunu gerektirir.
    """
    beklenen = {url: document_hash(icerik) for url, icerik in documents.items()}
    eslesen = eslesmeyen = bilinmeyen = kontrol = 0
    ornekler: List[Dict[str, str]] = []

    for olgu in store.relations.olgular():
        if not olgu.document_hash:
            continue
        kontrol += 1
        if olgu.source_url not in beklenen:
            bilinmeyen += 1
            if len(ornekler) < int(max_samples):
                ornekler.append({
                    "subject_id": olgu.subject_id,
                    "source_url": olgu.source_url or "",
                    "problem": "belge sağlanmadı, hash doğrulanamıyor",
                })
            continue
        if beklenen[olgu.source_url] == olgu.document_hash:
            eslesen += 1
        else:
            eslesmeyen += 1
            if len(ornekler) < int(max_samples):
                ornekler.append({
                    "subject_id": olgu.subject_id,
                    "source_url": olgu.source_url or "",
                    "problem": "belge içeriği değişmiş (hash tutmuyor)",
                    "recorded": olgu.document_hash,
                    "actual": beklenen[olgu.source_url],
                })

    return HashDogrulamaRaporu(
        checked=kontrol, matched=eslesen, mismatched=eslesmeyen,
        unknown_documents=bilinmeyen, mismatches=ornekler,
    )


def ingest_with_provenance(
    store,
    sentences: Sequence[str],
    source_url: str,
    content: Optional[str] = None,
    extractor: str = "CumleAyiklayici",
    retrieved_at: Optional[float] = None,
    iliski_kisitlari: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Any]:
    """Cümleleri köken bilgisiyle birlikte bilgi tabanına aktar.

    Mevcut ``cumlelerden_bilgi_aktar`` yolunu kullanır (yeni ayıklayıcı
    EKLENMEZ), ardından yazılan olgulara köken damgası vurur. ``content``
    verilmezse belge içeriği cümlelerin birleşimi kabul edilir.
    """
    from hga.experience.cumle_ayiklayici import cumlelerden_bilgi_aktar

    belge = content if content is not None else "\n".join(sentences)
    hash_ = document_hash(belge)
    cumle_listesi = list(sentences)

    # Cümleleri TEK TEK aktarırız: böylece hangi olgunun hangi ham cümleden
    # geldiği kesin olarak bilinir. Toplu aktarımda bu eşleme kaybolur, çünkü
    # ayıklayıcı entity_id değil token döndürür.
    yeni_olgular = []
    aktarilan_toplam = 0
    atlanan_cumleler = []
    for cumle in cumle_listesi:
        onceki = len(store.relations.olgular())
        aktarilan = cumlelerden_bilgi_aktar(store, [cumle],
                                            iliski_kisitlari=iliski_kisitlari)
        aktarilan_toplam += len(aktarilan)
        eklenen = store.relations.olgular()[onceki:]
        if not eklenen:
            atlanan_cumleler.append(cumle)
        for olgu in eklenen:
            olgu.source_url = source_url
            olgu.document_hash = hash_
            olgu.extractor = extractor
            olgu.retrieved_at = retrieved_at
            olgu.sentence = cumle
            yeni_olgular.append(olgu)

    return {
        "source_url": source_url,
        "document_hash": hash_,
        "sentences": len(cumle_listesi),
        "extracted_triples": aktarilan_toplam,
        "facts_written": len(yeni_olgular),
        "skipped_sentences": len(atlanan_cumleler),
        # Ayıklama verimi: sözlük tabanlı ayıklayıcı gerçek metnin ne kadarını
        # yakalayabiliyor? Faz 27/28'in asıl darboğazı budur.
        "extraction_yield": (round(len(yeni_olgular) / len(cumle_listesi), 6)
                             if cumle_listesi else 0.0),
        "extractor": extractor,
    }


__all__ = [
    "DIS_KAYNAKLAR", "TURETILMIS_KAYNAKLAR", "ZORUNLU_ALANLAR",
    "PROVENANCE_CHAIN_KINDS", "ProvenanceNode", "ProvenanceEdge",
    "ProvenanceChain", "YetimOlgu", "ProvenanceRaporu", "HashDogrulamaRaporu",
    "document_hash", "build_provenance_chain", "audit_provenance",
    "verify_document_hashes", "ingest_with_provenance",
]
