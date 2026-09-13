# -*- coding: utf-8 -*-
"""Faz 13–14: P / C_I / C_M / C_E / C_V kapasite çerçevesi testleri."""
import pytest

from hga.evaluation import (
    capacity_contract,
    measure_experience_capacity,
    run_capacity_benchmark,
)
from hga.experience.mini_env import AritmetikOrtam
from hga.knowledge import KaynakTuru, KnowledgeStore


def test_capacity_contract_ust_sinirlari_parametre_saymaz():
    sozlesme = capacity_contract(physical_parameters=1000, n=256, k=4,
                                 vocab=8000, window=16)
    assert sozlesme["c_i_interaction"] == 256.0 ** 8
    assert sozlesme["c_m_conceptual"] == 8000.0 ** 16
    assert sozlesme["c_i_is_parameter_count"] is False
    assert sozlesme["c_m_is_physical_table_size"] is False


def test_capacity_contract_gecersiz_girdiyi_reddeder():
    with pytest.raises(ValueError):
        capacity_contract(n=1, k=4)
    with pytest.raises(ValueError):
        capacity_contract(vocab=1, window=4)


def test_c_v_c_e_c_m_siralamasi_korunur():
    rapor = run_capacity_benchmark(operands_max=5, sample_size=None)
    assert rapor.c_v_total <= rapor.c_e_total
    assert rapor.c_e_total <= rapor.c_m_conceptual
    assert rapor.ordering_holds


def test_karara_baglanamayan_bolge_c_v_disinda_kalir():
    """Verifier None dönen adaylar C_E'de sayılır ama C_V'ye girmez."""
    store = KnowledgeStore()
    store.varlik_ekle("1+1", entity_type="ifade", entity_id="E_A")
    store.varlik_ekle("bilinmeyen", entity_type="ifade", entity_id="E_B")
    store.varlik_ekle("2", entity_type="sayi", entity_id="E_R")
    store.iliski_tanimla("eşittir", relation_id="R_EQ",
                         subject_types=["ifade"], object_types=["sayi"])
    rapor = measure_experience_capacity(
        store, verifier=AritmetikOrtam().aday_dogrula, relation_ids=["R_EQ"])
    assert rapor.c_e_total == 2          # iki ifade × bir sonuç
    assert rapor.c_v_total == 1          # yalnız "1+1=2" karara bağlanabilir
    assert rapor.c_e_undecidable == 1
    assert rapor.decidability == 0.5
    assert rapor.ordering_holds


def test_ornekleme_raporda_acikca_bildirilir():
    rapor = run_capacity_benchmark(operands_max=7, sample_size=50)
    assert rapor.sampled
    assert rapor.sample_size == 50
    assert any("Örnekleme" in n for n in rapor.notes)


def test_markdown_tablosu_bes_kapasiteyi_icerir():
    rapor = run_capacity_benchmark(operands_max=4, sample_size=None)
    md = rapor.markdown()
    for sembol in ("P", "C_I", "C_M", "C_E", "C_V"):
        assert f"| {sembol} |" in md


def test_dogru_ve_yanlis_adaylar_ayri_sayilir():
    rapor = run_capacity_benchmark(operands_max=4, sample_size=None)
    assert rapor.c_v_true > 0
    assert rapor.c_v_false > 0
    assert rapor.c_v_true + rapor.c_v_false == rapor.c_v_total


def test_verified_rule_kaynakli_tohum_bilgi_kapasiteyi_bozmaz():
    rapor = run_capacity_benchmark(operands_max=4, initial_facts=3,
                                   sample_size=None)
    assert rapor.c_e_total > 0
    assert rapor.physical_parameters is not None
    assert KaynakTuru.VERIFIED_RULE.value == "VERIFIED_RULE"
