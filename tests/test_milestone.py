# -*- coding: utf-8 -*-
"""Faz 10–12: milestone tablosu (K₀→Kₙ) deneyi testleri."""
import json
import subprocess
import sys

import pytest

from hga.experience import milestone_markdown, run_milestone_experiment


@pytest.fixture(scope="module")
def rapor():
    return run_milestone_experiment(
        cycles=10, batch_size=16, initial_facts=20, operands_max=12,
        negatives_per_fact=3, seed=1, memory_slots=256,
        checkpoints=(0, 5, 10),
    )


def test_kontrol_noktalari_istenildigi_gibi_uretilir(rapor):
    assert [c.cycle for c in rapor.checkpoints] == [0, 5, 10]
    assert rapor.cycles_completed == 10
    assert not rapor.candidate_pool_exhausted


def test_bilgi_yalniz_dogrulanmis_olgularla_buyur(rapor):
    ilk, son = rapor.checkpoints[0], rapor.checkpoints[-1]
    assert son.verified_knowledge > ilk.verified_knowledge
    assert son.new_knowledge == son.verified_knowledge - ilk.verified_knowledge
    # kapalı döngü hiç yanlış olgu yazmamalı
    assert all(c.incorrect_knowledge == 0 for c in rapor.checkpoints)


def test_far_ve_frr_raporlanir(rapor):
    for nokta in rapor.checkpoints:
        assert 0.0 <= nokta.far <= 1.0
        assert 0.0 <= nokta.frr <= 1.0


def test_uncertain_ve_conflict_gercekten_olculur(rapor):
    son = rapor.checkpoints[-1]
    # epistemik prob: "bilmiyorum" ile "yanlış" ayrı sütunlarda
    assert son.uncertain > 0
    assert son.conflict > 0
    assert son.invalid > 0


def test_experience_yield_tanimi_dogru(rapor):
    son = rapor.checkpoints[-1]
    assert 0.0 < son.experience_yield < 1.0


def test_verim_ayristirmasi_milestone_tablosunda_var(rapor):
    """P1-005: EY tek başına raporlanmamalı; NY ve UEY de tabloda olmalı."""
    son = rapor.checkpoints[-1]
    assert 0.0 <= son.novelty_yield <= 1.0
    assert 0.0 <= son.useful_experience_yield <= 1.0
    # Kullanışlı bilgi, doğrulanmış bilginin alt kümesidir.
    assert son.useful_experience_yield <= son.experience_yield
    metin = milestone_markdown([rapor])
    assert "Novelty Yield (NY)" in metin
    assert "Useful Exp. Yield (UEY)" in metin


def test_uey_bellek_kaybini_yakalar_ey_gizler():
    """Dar bellekte doğrulanan bilgi kaybolur; EY bunu göremez, UEY görür.

    Ölçülen: 16 slotta EY≈0.19 iken UEY≈0.04 — doğrulanmış bilginin büyük
    kısmı geri çağrılamıyor. EY'nin neden tek başına yanıltıcı olduğunun
    milestone protokolündeki somut kanıtıdır.
    """
    dar = run_milestone_experiment(
        cycles=20, batch_size=16, initial_facts=10, operands_max=12,
        negatives_per_fact=3, seed=1, memory_slots=16, checkpoints=(20,),
    ).checkpoints[-1]
    genis = run_milestone_experiment(
        cycles=20, batch_size=16, initial_facts=10, operands_max=12,
        negatives_per_fact=3, seed=1, memory_slots=4096, checkpoints=(20,),
    ).checkpoints[-1]

    # EY bellek darlığından etkilenmez: aynı doğrulama, aynı oran.
    assert dar.experience_yield == genis.experience_yield
    # UEY ise çöker.
    assert dar.useful_experience_yield < genis.useful_experience_yield
    assert dar.useful_experience_yield < dar.experience_yield / 2
    assert dar.memory_collision > genis.memory_collision


def test_zincirler_gecerli_ve_defter_dolu(rapor):
    assert rapor.knowledge_chain_valid
    assert rapor.ledger_chain_valid
    assert rapor.ledger_total_entries > 0
    assert rapor.final_knowledge_version.startswith("K")


def test_rollback_tatbikati_hatayi_geri_alir(rapor):
    drill = rapor.rollback_drill
    assert drill["incorrect_facts_when_corrupted"] == 1
    assert drill["incorrect_facts_after_rollback"] == 0
    assert drill["content_restored"]
    assert drill["history_preserved"]


def test_kapasite_olcumu_rapora_gomulur(rapor):
    assert rapor.capacity["c_v_total"] <= rapor.capacity["c_e_total"]
    assert rapor.capacity["ordering_holds"]


def test_test_izolasyonu_temiz(rapor):
    assert rapor.isolation_clean
    assert rapor.generation_test_overlap == 0
    assert rapor.memory_test_overlap == 0
    assert rapor.test_holdout_size > 0


def test_aday_havuzu_tukenirse_acikca_bildirilir():
    rapor = run_milestone_experiment(
        cycles=100, batch_size=32, initial_facts=5, operands_max=6,
        negatives_per_fact=2, seed=1, checkpoints=(0,),
    )
    assert rapor.candidate_pool_exhausted
    assert rapor.cycles_completed < 100
    assert any("tükendi" in x for x in rapor.limitations)


def test_sinirlar_belirtilir(rapor):
    assert any("sentetik" in x for x in rapor.limitations)
    assert any("failure injection" in x for x in rapor.limitations)


def test_markdown_tablosu_mean_std_uretir():
    raporlar = [
        run_milestone_experiment(cycles=4, batch_size=8, initial_facts=10,
                                 operands_max=8, negatives_per_fact=3,
                                 seed=seed, checkpoints=(0, 4))
        for seed in (1, 2)
    ]
    md = milestone_markdown(raporlar)
    assert "Experience Yield" in md
    assert "±" in md
    assert "Cycle 4" in md


def test_markdown_bos_liste_reddeder():
    with pytest.raises(ValueError):
        milestone_markdown([])


def test_cli_milestone_manifestli(tmp_path):
    out = tmp_path / "rapor.json"
    md = tmp_path / "tablo.md"
    sonuc = subprocess.run(
        [sys.executable, "-m", "hga", "milestone", "--cycles", "4",
         "--batch", "8", "--initial-facts", "10", "--operands-max", "8",
         "--negatives-per-fact", "3", "--seeds", "1",
         "--checkpoints", "0,4",
         "--experiment-root", str(tmp_path / "exp"),
         "--out", str(out), "--markdown", str(md)],
        capture_output=True, text=True, timeout=300,
    )
    assert sonuc.returncode == 0, sonuc.stderr
    veri = json.loads(out.read_text(encoding="utf-8"))
    assert veri["experiment_ids"]
    assert veri["manifests"][0]["dataset_hash"]
    assert "Experience Yield" in md.read_text(encoding="utf-8")
