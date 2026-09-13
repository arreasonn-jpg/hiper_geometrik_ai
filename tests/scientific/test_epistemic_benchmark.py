# -*- coding: utf-8 -*-
"""P0-007 epistemik benchmark testleri.

Buradaki testler yalnız "kod çalışıyor mu" demez; benchmarkın **ölçtüğünü
iddia ettiği şeyi gerçekten ölçtüğünü** sınar:

* dejenere politikalar geçemiyor mu (negatif kontrol),
* yanlış güven metriği gerçekten yanlış güvene tepki veriyor mu,
* fixture kendi kendini bozmaya karşı korunuyor mu.
"""
from __future__ import annotations

import copy
import json

import pytest

from hga.evaluation.epistemic import (
    ABSTENTION_REQUIRED,
    EPISTEMIC_CLASSES,
    EpistemicDataset,
    run_epistemic_baselines,
    run_epistemic_benchmark,
)
from hga.knowledge import DeneyimDurumu


@pytest.fixture(scope="module")
def rapor():
    return run_epistemic_benchmark()


@pytest.fixture(scope="module")
def veri():
    return EpistemicDataset()


# --------------------------------------------------------------------------
# Veri kümesi sözleşmesi
# --------------------------------------------------------------------------

def test_fixture_bes_epistemik_sinifi_da_icerir(veri):
    gorulen = {case["epistemic_class"] for case in veri.cases}
    assert gorulen == set(EPISTEMIC_CLASSES)


def test_fixture_her_sinifta_yeterli_ornek_var(veri):
    for sinif in EPISTEMIC_CLASSES:
        n = sum(1 for case in veri.cases if case["epistemic_class"] == sinif)
        assert n >= 3, f"{sinif} sınıfı çok az örnek içeriyor: {n}"


def test_fixture_hash_deterministik(veri):
    assert veri.dataset_hash() == EpistemicDataset().dataset_hash()
    assert len(veri.dataset_hash()) == 64


def test_fixture_manuel_kurasyon_olarak_isaretli(veri):
    assert veri.document["curation"]["kind"] == "manually_authored_repository_fixture"
    assert veri.document["limitations"]


def test_vaka_ucluleri_benzersiz(veri):
    ucluler = [(c["subject_id"], c["relation_id"], c["object_id"]) for c in veri.cases]
    assert len(ucluler) == len(set(ucluler))


def test_celiski_kaniti_sadece_conflict_vakalarina_yazilir(veri):
    kanit = {(e["subject_id"], e["relation_id"], e["object_id"])
             for e in veri.conflict_evidence}
    for case in veri.cases:
        uclu = (case["subject_id"], case["relation_id"], case["object_id"])
        if uclu in kanit:
            assert case["epistemic_class"] == "CONFLICT"


def test_kirlenmis_fixture_reddedilir(veri):
    """CONFLICT kanıtı bir KNOWN vakasıyla çakışırsa yükleme patlamalı.

    Bu gerçek bir hatadır: ilk sürümde KNOWN-008 ile CONFLICT-003 aynı üçlüydü
    ve KNOWN vakası sessizce CONFLICT'e düşüyordu.
    """
    bozuk = copy.deepcopy(veri.document)
    known = next(c for c in bozuk["cases"] if c["epistemic_class"] == "KNOWN")
    bozuk["conflict_evidence"].append({
        "subject_id": known["subject_id"], "relation_id": known["relation_id"],
        "object_id": known["object_id"], "score": 0.0,
        "confidence": 0.95, "source": "REAL_DATA",
    })
    with pytest.raises(ValueError, match="kirletiyor"):
        EpistemicDataset(bozuk)


def test_yinelenen_uclu_reddedilir(veri):
    bozuk = copy.deepcopy(veri.document)
    bozuk["cases"].append(copy.deepcopy(bozuk["cases"][0]))
    bozuk["cases"][-1]["case_id"] = "KOPYA-001"
    with pytest.raises(ValueError, match="benzersiz"):
        EpistemicDataset(bozuk)


def test_eksik_epistemik_sinif_reddedilir(veri):
    bozuk = copy.deepcopy(veri.document)
    bozuk["cases"] = [c for c in bozuk["cases"] if c["epistemic_class"] != "CONFLICT"]
    with pytest.raises(ValueError, match="Epistemik sınıflar eksik"):
        EpistemicDataset(bozuk)


def test_gecersiz_sema_surumu_reddedilir(veri):
    bozuk = copy.deepcopy(veri.document)
    bozuk["schema_version"] = 99
    with pytest.raises(ValueError, match="şeması"):
        EpistemicDataset(bozuk)


# --------------------------------------------------------------------------
# Asıl metrikler
# --------------------------------------------------------------------------

def test_bilinen_dogrular_kabul_edilir(rapor):
    assert rapor.known_accuracy == 1.0


def test_bilinen_yanlislar_asla_sessizce_kabul_edilmez(rapor):
    """En tehlikeli hata: FALSE örneğine VALID demek."""
    assert rapor.silent_failure_cases == 0
    assert rapor.silent_failure_rate == 0.0


def test_kanit_yoklugunda_yanlis_guven_uretilmez(rapor):
    assert rapor.false_confidence_rate == 0.0
    assert rapor.false_confidence_cases == 0


def test_bilinmeyen_dogrulugu_tam(rapor):
    assert rapor.unknown_accuracy == 1.0


def test_celiskiler_isaretlenir(rapor):
    assert rapor.conflict_accuracy == 1.0


def test_tum_kapilar_gecer(rapor):
    kalanlar = [ad for ad, deger in rapor.checks.items() if not deger]
    assert not kalanlar, f"Başarısız kapılar: {kalanlar}"


def test_cekimser_kalinmasi_gereken_vakalar_taahhut_uretmez(rapor):
    for case in rapor.cases:
        if case["epistemic_class"] in ABSTENTION_REQUIRED:
            assert not case["committed"], case["case_id"]
            assert case["observed_state"] not in ("VALID", "INVALID")


def test_rapor_json_serilestirilebilir(rapor):
    metin = json.dumps(rapor.to_dict(), ensure_ascii=False, sort_keys=True)
    assert json.loads(metin)["protocol"] == rapor.protocol


def test_determinizm(veri):
    a = run_epistemic_benchmark(veri, seed=1).to_dict()
    b = run_epistemic_benchmark(veri, seed=999).to_dict()
    assert a == b, "Protokol deterministik olmalı: seed sonucu değiştirmemeli"


# --------------------------------------------------------------------------
# Negatif kontrol: benchmark kandırılabiliyor mu?
# --------------------------------------------------------------------------

def test_dejenere_politikalar_benchmarki_gecemez(rapor):
    assert rapor.checks["beats_degenerate_baselines"] is True


def test_hep_kabul_eden_politika_maksimum_yanlis_guven_uretir(veri):
    kol = next(a for a in run_epistemic_baselines(veri) if a.arm == "always_valid")
    assert kol.false_confidence_rate == 1.0
    assert kol.silent_failure_rate == 1.0
    assert kol.known_accuracy == 1.0  # tek metrik bakmak yanıltıcı olurdu


def test_hep_cekimser_politika_bilineni_kabul_edemez(veri):
    """unknown_accuracy tek başına yeterli olsaydı bu kol "mükemmel" olurdu."""
    kol = next(a for a in run_epistemic_baselines(veri) if a.arm == "always_abstain")
    assert kol.unknown_accuracy == 1.0
    assert kol.false_confidence_rate == 0.0
    assert kol.known_accuracy == 0.0


def test_gercek_degerlendirici_her_kolu_domine_eder(rapor, veri):
    for kol in run_epistemic_baselines(veri):
        assert rapor.accuracy > kol.accuracy
        assert rapor.known_accuracy >= kol.known_accuracy
        assert rapor.unknown_accuracy >= kol.unknown_accuracy
        assert rapor.false_confidence_rate <= kol.false_confidence_rate


def test_yanlis_guven_metrigi_gercekten_tepki_verir(veri):
    """Metrik sabit 0 döndürmüyor: UNKNOWN'a VALID diyen politika 1.0 almalı."""
    from hga.evaluation.epistemic import _score_policy

    kol = _score_policy(veri.cases, lambda case: DeneyimDurumu.VALID, "x", "y")
    assert kol.false_confidence_rate == 1.0


# --------------------------------------------------------------------------
# Dürüstlük: ölçülmüş mimari sınır
# --------------------------------------------------------------------------

def test_unknown_uncertain_ayrimi_durustce_raporlanir(rapor):
    """Şu an ayrım YAPILAMIYOR; test bunu gizlemek yerine sabitler.

    Ayrı bir durum kodu eklenirse bu test kasıtlı olarak kırılır ve raporun
    güncellenmesi gerektiğini hatırlatır.
    """
    cozunurluk = rapor.epistemic_resolution
    assert cozunurluk["distinguishable"] is False
    assert cozunurluk["unknown_observed_states"] == ["UNCERTAIN"]
    assert cozunurluk["uncertain_observed_states"] == ["UNCERTAIN"]
    assert "mimari sınırdır" in cozunurluk["note"]


def test_sinirlar_raporda_yer_alir(rapor):
    assert len(rapor.limitations) >= 3
    birlesik = " ".join(rapor.limitations)
    assert "gerçek dil anlama" in birlesik


def test_confusion_matrisi_tum_vakalari_kapsar(rapor):
    toplam = sum(sum(d.values()) for d in rapor.confusion.values())
    assert toplam == rapor.total_cases
