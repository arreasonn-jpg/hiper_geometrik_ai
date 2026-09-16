# -*- coding: utf-8 -*-
"""P0-4: HGA Signature Benchmark testleri."""
import pytest

from hga.evaluation.signature import (
    MEMORY_CRITICAL_TASKS,
    NO,
    TASKS,
    UNCERTAIN,
    YES,
    build_dataset,
    leakage_report,
    symbolic_predict,
)


def _veri(task, seed=1):
    return build_dataset(task, seed, train_size=64, test_size=32,
                         context_slots=8, entity_count=40, relation_count=6)


def test_tum_gorevler_uretilebilir():
    for task in TASKS:
        train, test, _ = _veri(task)
        assert len(train) == 64 and len(test) == 32


def test_bilinmeyen_gorev_acik_hata():
    with pytest.raises(ValueError):
        build_dataset("yok", 1, 8, 8, 4, 20, 4)


def test_held_out_semboller_egitime_sizmaz():
    for task in ("B_unseen_entity", "C_unseen_relation", "D_unseen_both"):
        train, test, world = _veri(task)
        rapor = leakage_report(task, train, test, world)
        assert rapor["clean"], f"{task}: held-out sembol eğitime sızdı"


def test_cold_start_eksenleri_testte_gercekten_gorulmemis_kullanir():
    _, test, world = _veri("B_unseen_entity")
    assert any(o.unseen_entity for o in test)
    _, test_r, _ = _veri("C_unseen_relation")
    assert any(o.unseen_relation for o in test_r)


def test_memory_gorevi_gizli_olgu_tasir():
    """F görevinde zincirin bir kenarı görünür bağlamın DIŞINDA olmalı."""
    _, test, _ = _veri("F_memory_dependent")
    assert any(o.hidden_facts for o in test)
    for ornek in test:
        for gizli in ornek.hidden_facts:
            assert gizli not in ornek.context


def test_bellek_olmadan_f_gorevi_cozulemez():
    """Bellek erişimi kapatılınca YES vakaları çıkarılamaz olmalı."""
    _, test, _ = _veri("F_memory_dependent")
    pozitifler = [o for o in test if o.label == YES]
    assert pozitifler
    for ornek in pozitifler:
        assert symbolic_predict(ornek, use_memory=True) == YES
        assert symbolic_predict(ornek, use_memory=False) != YES


def test_celiski_gorevi_uncertain_uretir():
    _, test, _ = _veri("E_conflict")
    assert any(o.label == UNCERTAIN for o in test)
    for ornek in test:
        if ornek.label == UNCERTAIN:
            assert symbolic_predict(ornek) == UNCERTAIN


def test_epistemik_gorev_uc_sinifi_da_kapsar():
    _, test, _ = _veri("G_epistemic")
    etiketler = {o.label for o in test}
    assert etiketler == {NO, YES, UNCERTAIN}


def test_sembolik_kol_kanit_yoksa_uydurmaz():
    """Hiç kayıt yoksa cevap UNCERTAIN olmalı, NO değil."""
    _, test, _ = _veri("G_epistemic")
    kanitsiz = [o for o in test if o.label == UNCERTAIN]
    assert kanitsiz
    for ornek in kanitsiz:
        assert symbolic_predict(ornek) == UNCERTAIN


def test_distractor_gorevi_gercekten_gurultulu():
    _, test_h, _ = _veri("H_distractor")
    _, test_a, _ = _veri("A_long_chain")
    assert (sum(len(o.context) for o in test_h) / len(test_h)
            > sum(len(o.context) for o in test_a) / len(test_a))


def test_veri_uretimi_deterministik():
    a, _, _ = _veri("A_long_chain", seed=5)
    b, _, _ = _veri("A_long_chain", seed=5)
    assert [o.to_dict() for o in a] == [o.to_dict() for o in b]


def test_etiket_dagilimi_dejenere_degil():
    """Tek sınıf baskınsa çoğunluk kolu her şeyi kazanır, ölçüm ölür."""
    for task in TASKS:
        _, test, _ = _veri(task)
        sayim = {}
        for ornek in test:
            sayim[ornek.label] = sayim.get(ornek.label, 0) + 1
        assert max(sayim.values()) / len(test) <= 0.85, f"{task} dejenere"


def test_memory_kritik_gorev_listesi_tasklarda_var():
    for task in MEMORY_CRITICAL_TASKS:
        assert task in TASKS


@pytest.mark.slow
def test_smoke_kosusu_kapilari_uretir():
    pytest.importorskip("torch")
    from hga.evaluation.signature import run_signature_benchmark, signature_markdown
    rapor = run_signature_benchmark("smoke", seeds=(1,),
                                    tasks=("F_memory_dependent", "G_epistemic"))
    assert rapor.checks["no_held_out_leak"]
    # Parametre bütçesi otomatik oturtulmalı.
    assert rapor.parameter_ratio <= 1.10
    assert rapor.body_parameter_ratio <= 1.10
    # Bellek katkısı yalnız F'de belirgin olmalı.
    assert rapor.signature["F_memory_dependent"]["memory_contribution"] > 0.05
    assert abs(rapor.signature["G_epistemic"]["memory_contribution"]) <= 0.05
    md = signature_markdown(rapor)
    assert "İmza analizi" in md and "Sızıntı denetimi" in md


def test_signature_release_gate_eksik_kanitta_blocked():
    pytest.importorskip("torch")
    from hga.evaluation.signature import (
        run_signature_benchmark,
        run_signature_release_gate,
        signature_release_gate_markdown,
    )
    rapor = run_signature_benchmark("smoke", seeds=(1,),
                                    tasks=("F_memory_dependent", "G_epistemic"))
    gate = run_signature_release_gate(rapor)
    assert gate.status == "BLOCKED"
    assert not gate.release_ready
    assert "profile_is_standard" in gate.failed_checks
    assert "seed_count_at_least_20" in gate.failed_checks
    md = signature_release_gate_markdown(gate)
    assert "Signature Benchmark Release Gate" in md
    assert "BLOCKED" in md


def test_signature_release_gate_tum_kapilar_gecerse_pass():
    from hga.evaluation.signature import ARMS, PROTOCOL, TASKS, run_signature_release_gate
    report = {
        "protocol": PROTOCOL,
        "profile": "standard",
        "seeds": list(range(1, 21)),
        "tasks": list(TASKS),
        "arms": list(ARMS),
        "dataset_hash": "dummyhash",
        "checks": {
            "parameter_budget_within_gate": True,
            "body_parameter_budget_within_gate": True,
            "no_held_out_leak": True,
            "hga_beats_majority_everywhere": True,
            "hga_has_signature_task": True,
            "neural_arms_learn_above_chance": True,
            "memory_gain_is_task_specific": True,
        },
        "signature": {"F_memory_dependent": {"margin": 0.2}},
    }
    gate = run_signature_release_gate(report)
    assert gate.status == "PASS"
    assert gate.release_ready
    assert gate.failed_checks == []


@pytest.mark.slow
def test_bilinmeyen_kol_acik_hata():
    pytest.importorskip("torch")
    from hga.evaluation.signature import run_signature_benchmark
    with pytest.raises(ValueError):
        run_signature_benchmark("smoke", arms=["yok"])
    with pytest.raises(ValueError):
        run_signature_benchmark("yok_profil")
