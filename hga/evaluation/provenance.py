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
from typing import Any, Dict, List, Mapping, Optional, Sequence

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


def document_hash(content: str) -> str:
    """Belge içeriğinin kanonik SHA-256'sı (satır sonu normalize edilir)."""
    normalize = (content or "").replace("\r\n", "\n").strip()
    return hashlib.sha256(normalize.encode("utf-8")).hexdigest()


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
    "YetimOlgu", "ProvenanceRaporu", "HashDogrulamaRaporu",
    "document_hash", "audit_provenance", "verify_document_hashes",
    "ingest_with_provenance",
]
