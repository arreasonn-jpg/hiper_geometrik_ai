# -*- coding: utf-8 -*-
"""Faz 27/28: köken (provenance) denetimi ve belge hash doğrulaması."""
import pytest

from hga.evaluation.provenance import (
    audit_provenance,
    document_hash,
    ingest_with_provenance,
    verify_document_hashes,
)
from hga.knowledge import KaynakTuru, KnowledgeStore

BELGE = "Ali ata bindi. Ayşe kitabı okudu. Mehmet topu attı."
CUMLELER = [c.strip() + "." for c in BELGE.split(".") if c.strip()]


def _store():
    return KnowledgeStore()


def test_document_hash_satir_sonu_normalize_eder():
    assert document_hash("a\r\nb") == document_hash("a\nb")
    assert document_hash("  x  ") == document_hash("x")
    assert document_hash("a") != document_hash("b")


def test_olgu_provenance_alanlari_geriye_donuk_uyumlu():
    """Köken vermeden yazılan olgu eskisi gibi çalışmalı."""
    s = _store()
    s.varlik_ekle("a", entity_id="A")
    s.varlik_ekle("b", entity_id="B")
    s.iliski_tanimla("r", relation_id="R")
    olgu = s.olgu_kaydet("A", "R", "B", 1.0)
    assert olgu.source_url is None
    assert olgu.provenance_var_mi is False
    assert "source_url" in olgu.to_dict()


def test_gecersiz_provenance_alani_acik_hata():
    s = _store()
    s.varlik_ekle("a", entity_id="A")
    s.varlik_ekle("b", entity_id="B")
    s.iliski_tanimla("r", relation_id="R")
    with pytest.raises(ValueError):
        s.olgu_kaydet("A", "R", "B", 1.0, provenance={"bilinmeyen": 1})


def test_ingest_her_olguya_koken_damgasi_vurur():
    s = _store()
    bilgi = ingest_with_provenance(s, CUMLELER, source_url="test://1",
                                   content=BELGE)
    assert bilgi["facts_written"] >= 1
    assert bilgi["document_hash"] == document_hash(BELGE)
    for olgu in s.relations.olgular():
        assert olgu.source_url == "test://1"
        assert olgu.document_hash == bilgi["document_hash"]
        assert olgu.provenance_var_mi


def test_her_olgu_kendi_ham_cumlesine_baglanir():
    """Cümle eşlemesi doğru olmalı: olgu hangi cümleden geldi?"""
    s = _store()
    ingest_with_provenance(s, CUMLELER, source_url="test://1", content=BELGE)
    for olgu in s.relations.olgular():
        assert olgu.sentence in CUMLELER


def test_ayiklama_verimi_olculur_ve_sozlukten_dusuk_olabilir():
    """Sözlük tabanlı ayıklayıcı her cümleyi yakalayamaz — bu raporlanmalı."""
    s = _store()
    cumleler = CUMLELER + ["Bu cümle sözlükte olmayan kelimeler icerir."]
    bilgi = ingest_with_provenance(s, cumleler, source_url="test://2")
    assert bilgi["skipped_sentences"] >= 1
    assert 0.0 < bilgi["extraction_yield"] < 1.0


def test_denetim_kokeni_tam_olan_veriyi_temiz_bulur():
    s = _store()
    ingest_with_provenance(s, CUMLELER, source_url="test://1", content=BELGE)
    rapor = audit_provenance(s)
    assert rapor.clean
    assert rapor.coverage == 1.0
    assert rapor.orphan_facts == 0
    assert rapor.unique_documents == 1
    rapor.assert_clean()


def test_denetim_yetim_olguyu_yakalar():
    s = _store()
    ingest_with_provenance(s, CUMLELER, source_url="test://1", content=BELGE)
    s.varlik_ekle("y1", entity_id="Y1")
    s.varlik_ekle("y2", entity_id="Y2")
    s.iliski_tanimla("yr", relation_id="YR")
    s.olgu_kaydet("Y1", "YR", "Y2", 1.0, source=KaynakTuru.REAL_DATA)
    rapor = audit_provenance(s)
    assert not rapor.clean
    assert rapor.orphan_facts == 1
    assert rapor.coverage < 1.0
    assert rapor.orphans[0]["missing_fields"] == ["source_url", "document_hash"]
    with pytest.raises(AssertionError):
        rapor.assert_clean()


def test_turetilmis_kaynaklar_koken_zorunlulugundan_muaf():
    s = _store()
    s.varlik_ekle("a", entity_id="A")
    s.varlik_ekle("b", entity_id="B")
    s.iliski_tanimla("r", relation_id="R")
    s.olgu_kaydet("A", "R", "B", 1.0, source=KaynakTuru.MODEL_GENERATED)
    s.olgu_kaydet("A", "R", "B", 1.0, source=KaynakTuru.DERIVED)
    rapor = audit_provenance(s)
    assert rapor.clean
    assert rapor.derived_facts == 2
    assert rapor.external_facts == 0


def test_belge_degisirse_hash_dogrulamasi_yakalar():
    """'Kaynak gösterdim' iddiası denetlenebilir olmalı."""
    s = _store()
    ingest_with_provenance(s, CUMLELER, source_url="test://1", content=BELGE)
    saglam = verify_document_hashes(s, {"test://1": BELGE})
    assert saglam.clean and saglam.mismatched == 0

    bozuk = verify_document_hashes(s, {"test://1": BELGE + " Sonradan eklendi."})
    assert not bozuk.clean
    assert bozuk.mismatched == saglam.checked
    assert "değişmiş" in bozuk.mismatches[0]["problem"]


def test_belge_saglanmazsa_dogrulanamaz_olarak_raporlanir():
    s = _store()
    ingest_with_provenance(s, CUMLELER, source_url="test://1", content=BELGE)
    rapor = verify_document_hashes(s, {})
    assert not rapor.clean
    assert rapor.unknown_documents == rapor.checked
