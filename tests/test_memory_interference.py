# -*- coding: utf-8 -*-
"""Faz 15-17: kasıtlı çakışma ve sabit/dinamik KV ödünleşimi testleri."""
import pytest

from hga.memory import DeneyimSlotlari
from hga.memory.interference import (
    DYNAMIC_KV,
    FIRST_WINS,
    LAST_WINS,
    POLITIKALAR,
    carpisan_anahtar_bul,
    run_fixed_vs_dynamic_scaling,
    run_interference_test,
    run_policy_comparison,
)


def test_carpisan_anahtar_gercekten_ayni_slota_dusuyor():
    """Adversarial arama gerçekten çakışan anahtar buluyor mu?"""
    memory = DeneyimSlotlari(slot_sayisi=512)
    hedef = ("VICTIM:0", "R", "O")
    carpisan = carpisan_anahtar_bul(memory, hedef)
    assert carpisan is not None, "512 slotta çakışan anahtar bulunmalıydı"
    assert carpisan != hedef
    assert memory.adresler(carpisan) == memory.adresler(hedef)


def test_first_wins_kurbani_korur_saldirgani_kaybeder():
    r = run_interference_test(policy=FIRST_WINS, forced_collisions=10, slot_count=512)
    assert r.collisions_constructed == 10
    assert r.victim_survival_rate == 1.0
    assert r.attacker_survival_rate == 0.0
    assert r.corruption_rate == 0.0
    assert r.both_survival_rate == 0.0


def test_last_wins_kurbani_bozar():
    """LAST_WINS'te bilgi kaybı gerçek ve ölçülebilir olmalı."""
    r = run_interference_test(policy=LAST_WINS, forced_collisions=10, slot_count=512)
    assert r.victim_survival_rate == 0.0
    assert r.attacker_survival_rate == 1.0
    assert r.corruption_rate == 1.0


def test_dynamic_kv_her_iki_bilgiyi_korur():
    r = run_interference_test(policy=DYNAMIC_KV, forced_collisions=10, slot_count=512)
    assert r.both_survival_rate == 1.0
    assert r.corruption_rate == 0.0
    assert r.collision_events == 0


def test_sabit_tabloda_iki_bilgi_ayni_anda_korunamaz():
    """Asıl yapısal iddia: sabit adres uzayında both_survival daima 0."""
    for politika in (FIRST_WINS, LAST_WINS):
        r = run_interference_test(policy=politika, forced_collisions=15, slot_count=256)
        assert r.both_survival_rate == 0.0, politika


def test_politika_kiyasi_dynamic_kv_yi_kazanan_secer():
    rapor = run_policy_comparison(forced_collisions=10, slot_count=512)
    assert set(rapor.reports) == set(POLITIKALAR)
    assert rapor.best_both_survival == DYNAMIC_KV
    assert rapor.storage_ratio_dynamic_over_fixed > 1.0
    assert "|" in rapor.markdown()


def test_olcekleme_yuk_faktoru_arttikca_recall_dusuyor():
    rapor = run_fixed_vs_dynamic_scaling(context_counts=(100, 1000, 5000),
                                         slot_count=1024)
    recalls = [p.fixed_retrieval_accuracy for p in rapor.points]
    assert recalls == sorted(recalls, reverse=True), "recall monoton düşmeli"
    assert recalls[-1] < recalls[0]
    assert all(p.dynamic_retrieval_accuracy == 1.0 for p in rapor.points)


def test_onceden_tahsisli_tablo_context_sayisindan_bagimsiz():
    """nn.Embedding gerçeği: sabit maliyet, veriyle büyümez."""
    rapor = run_fixed_vs_dynamic_scaling(context_counts=(100, 1000, 10000),
                                         slot_count=2048, vector_dim=16)
    baytlar = {p.preallocated_table_bytes for p in rapor.points}
    assert len(baytlar) == 1, "önceden tahsisli tablo boyutu sabit olmalı"
    assert baytlar.pop() == 2048 * 1 * 16 * 4


def test_dinamik_kv_bellek_maliyeti_girdiyle_buyur():
    rapor = run_fixed_vs_dynamic_scaling(context_counts=(100, 1000, 10000),
                                         slot_count=2048)
    baytlar = [p.dynamic_storage_bytes for p in rapor.points]
    assert baytlar == sorted(baytlar)
    assert baytlar[-1] > 5 * baytlar[0]
    oranlar = [p.ratio_dynamic_over_preallocated for p in rapor.points]
    assert oranlar[0] < oranlar[-1], "ödünleşim ölçekle tersine dönmeli"


def test_gecersiz_politika_ve_parametre_acik_hata():
    with pytest.raises(ValueError):
        run_interference_test(policy="YOK")
    with pytest.raises(ValueError):
        run_interference_test(forced_collisions=0)
