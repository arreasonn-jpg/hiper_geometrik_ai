# -*- coding: utf-8 -*-
"""Faz 23: bilgi sürümleme, rollback ve diff testleri."""
import json

import pytest

from hga.knowledge import KnowledgeVersionStore


def _kurulum() -> KnowledgeVersionStore:
    kv = KnowledgeVersionStore()
    kv.store.varlik_ekle("Ali", entity_type="insan", entity_id="E_001")
    kv.store.varlik_ekle("Ata", entity_type="hayvan", entity_id="E_002",
                         properties={"binilebilir": 1})
    kv.store.varlik_ekle("Gökyüzü", entity_type="mekan", entity_id="E_003",
                         properties={"binilebilir": 0})
    kv.store.iliski_tanimla("Binmek", relation_id="R_001")
    return kv


def test_k0_baslangic_surumu_olusur():
    kv = KnowledgeVersionStore()
    assert kv.current_id == "K0"
    assert kv.current.parent_id is None
    assert kv.zincir_dogrula()


def test_commit_zinciri_ilerler_ve_hash_degisir():
    kv = _kurulum()
    k1 = kv.commit("K1")
    kv.store.olgu_kaydet("E_001", "R_001", "E_002", score=1.0)
    k2 = kv.commit("K2")
    assert [k1.version_id, k2.version_id] == ["K1", "K2"]
    assert k2.parent_id == "K1"
    assert k1.content_hash != k2.content_hash
    assert kv.zincir_dogrula()


def test_rollback_icerigi_geri_getirir_ve_gecmisi_silmez():
    kv = _kurulum()
    kv.store.olgu_kaydet("E_001", "R_001", "E_002", score=1.0)
    kv.commit("K1 doğru")
    kv.store.olgu_kaydet("E_001", "R_001", "E_003", score=1.0)   # hatalı
    kv.commit("K2 hatalı")
    assert len(kv.store.relations.olgular()) == 2

    geri = kv.rollback("K1")
    assert geri.kind == "ROLLBACK"
    assert geri.rolled_back_to == "K1"
    # içerik geri geldi
    assert len(kv.store.relations.olgular()) == 1
    assert geri.content_hash == kv._getir("K1").content_hash
    # geçmiş silinmedi: hatalı K2 hâlâ zincirde
    assert any(v.version_id == "K2" for v in kv.versions)
    assert kv.zincir_dogrula()


def test_rollback_sonrasi_versiyon_sayaci_geriye_gitmez():
    kv = _kurulum()
    kv.commit("K1")
    onceki = kv.store.versiyon
    kv.store.olgu_kaydet("E_001", "R_001", "E_003", score=1.0)
    kv.commit("K2")
    kv.rollback("K1")
    assert kv.store.versiyon > onceki


def test_diff_eklenen_ve_silinen_olgulari_bulur():
    kv = _kurulum()
    kv.commit("K1")
    kv.store.olgu_kaydet("E_001", "R_001", "E_002", score=1.0)
    kv.commit("K2")
    fark = kv.diff("K1", "K2")
    assert fark.eklenen_olgular == [["E_001", "R_001", "E_002"]]
    assert fark.silinen_olgular == []
    assert not fark.ayni_mi

    ters = kv.diff("K2", "K1")
    assert ters.silinen_olgular == [["E_001", "R_001", "E_002"]]


def test_checkout_yan_etkisizdir():
    kv = _kurulum()
    kv.commit("K1")
    kv.store.olgu_kaydet("E_001", "R_001", "E_002", score=1.0)
    kv.commit("K2")
    eski = kv.checkout("K1")
    eski.olgu_kaydet("E_001", "R_001", "E_003", score=1.0)
    # checkout kopyası değişti ama aktif store ve snapshot bozulmadı
    assert len(kv.store.relations.olgular()) == 1
    assert kv.zincir_dogrula()


def test_snapshot_sonradan_store_degisiminden_etkilenmez():
    kv = _kurulum()
    kayit = kv.commit("K1")
    hash_once = kayit.state_hash
    kv.store.varlik_ekle("Araba", entity_type="tasit", entity_id="E_004")
    kv.store.celiski_gunlugu.append({"test": 1})
    assert kayit.state_hash == hash_once
    assert kv.zincir_dogrula()
    assert kv.commit_gerekli_mi()


def test_bilinmeyen_surum_hata_verir():
    kv = KnowledgeVersionStore()
    with pytest.raises(KeyError):
        kv.checkout("K99")
    with pytest.raises(KeyError):
        kv.rollback("K99")


def test_aktif_surume_rollback_reddedilir():
    kv = KnowledgeVersionStore()
    with pytest.raises(ValueError):
        kv.rollback("K0")


def test_kaydet_yukle_zinciri_korur(tmp_path):
    kv = _kurulum()
    kv.commit("K1")
    kv.store.olgu_kaydet("E_001", "R_001", "E_003", score=1.0)
    kv.commit("K2")
    kv.rollback("K1")

    yol = tmp_path / "surumler.json"
    kv.kaydet(str(yol))
    veri = json.loads(yol.read_text(encoding="utf-8"))
    assert veri["current"] == kv.current_id

    geri = KnowledgeVersionStore.yukle(str(yol))
    assert geri.zincir_dogrula()
    assert geri.current_id == kv.current_id
    assert geri.store.ozet() == kv.store.ozet()


def test_bozulan_snapshot_zincir_dogrulamasinda_yakalanir():
    kv = _kurulum()
    kv.commit("K1")
    kv.versions[-1].state["entities"] = {}
    assert not kv.zincir_dogrula()


def test_gecmis_state_govdesini_disari_sizdirmaz():
    kv = _kurulum()
    kv.commit("K1")
    for kayit in kv.gecmis():
        assert "state" not in kayit
        assert "content_hash" in kayit
