# -*- coding: utf-8 -*-
"""P1-005 verim metrikleri testleri.

Kritik soru şudur: bu dört metrik EY'nin yeniden adlandırılması mı, yoksa
gerçekten farklı bir şey mi ölçüyor? Testler bunu sabitler.
"""
from __future__ import annotations

import json

import pytest

from hga.experience.self_learning import _build_domain, _candidate
from hga.experience.verim import (
    _infer_holdout,
    run_yield_experiment,
)
from hga.knowledge import KaynakTuru

KUCUK = dict(cycles=8, batch_size=8, initial_facts=6,
             operands_max=6, negatives_per_fact=3)


@pytest.fixture(scope="module")
def rapor():
    return run_yield_experiment(seed=1, **KUCUK)


# --------------------------------------------------------------------------
# Temel sözleşme
# --------------------------------------------------------------------------

def test_tum_oranlar_gecerli_aralikta(rapor):
    for alan in ("experience_yield", "novelty_yield", "useful_experience_yield",
                 "generalization_yield"):
        deger = getattr(rapor, alan)
        assert 0.0 <= deger <= 1.0, f"{alan}={deger}"
    assert rapor.verified_information_density >= 0.0


def test_determinizm(rapor):
    tekrar = run_yield_experiment(seed=1, **KUCUK)
    assert tekrar.to_dict() == rapor.to_dict()


def test_farkli_tohum_farkli_sonuc():
    a = run_yield_experiment(seed=1, **KUCUK)
    b = run_yield_experiment(seed=7, **KUCUK)
    assert a.to_dict() != b.to_dict()


def test_rapor_json_serilestirilebilir(rapor):
    assert json.loads(json.dumps(rapor.to_dict(), ensure_ascii=False))


def test_gecersiz_parametreler_reddedilir():
    with pytest.raises(ValueError):
        run_yield_experiment(cycles=0)
    with pytest.raises(ValueError):
        run_yield_experiment(batch_size=0)
    with pytest.raises(ValueError):
        run_yield_experiment(duplicate_rate=-1)


def test_bilgi_tabanina_yanlis_olgu_girmez(rapor):
    """Doğrulama hattı çalışıyorsa yanlış olgu sızmamalı."""
    assert rapor.incorrect_facts == 0


# --------------------------------------------------------------------------
# Asıl iddia: metrikler EY'den AYRIŞIYOR
# --------------------------------------------------------------------------

def test_novelty_yield_ey_den_dusuk(rapor):
    """Tekrar üretim varsa NY < EY olmalı; yoksa NY sadece EY'nin takma adıdır."""
    assert rapor.duplicate_generations > 0
    assert rapor.novelty_yield < rapor.experience_yield
    assert rapor.divergence["ey_overstates_novelty"] is True


def test_tekrar_uretim_kapatilirsa_ayrisma_azalir():
    """duplicate_rate=0 iken tekrar üretim olmamalı — kontrol koşusu."""
    r = run_yield_experiment(seed=1, duplicate_rate=0, **KUCUK)
    assert r.duplicate_generations == 0


def test_useful_yield_ey_yi_asamaz(rapor):
    """Kullanışlı bilgi, doğrulanmış bilginin alt kümesidir."""
    assert rapor.useful_experience_yield <= rapor.experience_yield
    assert rapor.useful_facts <= rapor.distinct_new_facts


def test_ayrisma_raporlanir(rapor):
    assert rapor.divergence["ey_vs_ny"] > 0.0
    assert "note" in rapor.divergence


def test_ham_sayimlar_oranlarla_tutarli(rapor):
    """Her oran denetlenebilir olmalı: pay/payda elle yeniden hesaplanır."""
    assert rapor.experience_yield == pytest.approx(
        rapor.verified / rapor.generated, abs=1e-8)
    assert rapor.novelty_yield == pytest.approx(
        rapor.distinct_new_facts / rapor.generated, abs=1e-8)
    assert rapor.useful_experience_yield == pytest.approx(
        rapor.useful_facts / rapor.generated, abs=1e-8)
    assert rapor.verified_information_density == pytest.approx(
        rapor.total_verified_bits / rapor.generated, abs=1e-8)


# --------------------------------------------------------------------------
# Genelleme verimi: ÖLÜ METRİK OLMAMALI
# --------------------------------------------------------------------------

def test_genelleme_verimi_olculebilir(rapor):
    """GY > 0 olabilmeli.

    Regresyon koruması: ilk sürümde holdout kararı ExperienceEvaluator ile
    veriliyordu. R_EQUALS ilişkisinin tip/özellik kısıtı olmadığı için
    evaluator holdout'un tamamına VALID diyordu ve GY yapısal olarak DAİMA
    0.0 çıkıyordu. Sıfır dönen bir metrik hiçbir şey ölçmez.
    """
    assert rapor.generalization_yield > 0.0


def test_genelleme_ogrenmeyle_artar():
    """Daha çok döngü → daha çok holdout kararı. Monotonluk beklenir."""
    kisa = run_yield_experiment(seed=1, cycles=8, batch_size=8, initial_facts=6,
                                operands_max=6, negatives_per_fact=3)
    uzun = run_yield_experiment(seed=1, cycles=24, batch_size=8, initial_facts=6,
                                operands_max=6, negatives_per_fact=3)
    assert uzun.generalization_yield > kisa.generalization_yield
    assert uzun.holdout_after["decided"] > kisa.holdout_after["decided"]


def test_ogrenme_oncesi_holdout_karari_yok(rapor):
    """Başlangıçta holdout özneleri hakkında öğrenilmiş pozitif yoktur."""
    assert rapor.holdout_before["decided"] == 0
    assert rapor.holdout_before["abstained"] == rapor.holdout_before["total"]


def test_holdout_kararlari_dogru(rapor):
    """Fonksiyonel teklik kuralı sağlamdır: karar verilenlerde hata olmamalı."""
    assert rapor.holdout_after["accuracy_on_decided"] == 1.0


def test_cikarim_kurali_dogrulanmis_kaynak_arar():
    """Skor eşiği değil KAYNAK bakılmalı.

    Doğrulama hattı olguyu score=1.0 ile değil (ör. 0.6375) yazar. İlk
    sürümdeki ``score >= 1.0`` eşiği öğrenilen her olguyu eliyordu.
    """
    domain = _build_domain(6, 6, 3, 1)
    uclu = domain.candidate_pool[0]
    assert _infer_holdout(domain.store, uclu) is None  # henüz dayanak yok

    dogru_uclu = next(
        t for t in domain.candidate_pool
        if domain.environment.aday_dogrula(
            domain.store, _candidate(t, "T", 0)) is True
    )
    # Doğrulayıcının yazdığı gibi 1.0'dan KÜÇÜK skorla kaydet.
    domain.store.olgu_kaydet(*dogru_uclu, score=0.6375,
                             source=KaynakTuru.EXTERNAL_VERIFIED, confidence=1.0)
    assert _infer_holdout(domain.store, dogru_uclu) is True

    # Aynı özne, farklı nesne → fonksiyonel teklikten YANLIŞ çıkarılmalı.
    yanlis_uclu = (dogru_uclu[0], dogru_uclu[1], "E_RESULT_00099")
    assert _infer_holdout(domain.store, yanlis_uclu) is False


def test_model_uretimi_tek_basina_cikarim_dayanagi_degildir():
    """MODEL_GENERATED bir olgu 'öğrenilmiş pozitif' sayılmamalı."""
    domain = _build_domain(6, 6, 3, 1)
    uclu = domain.candidate_pool[0]
    domain.store.olgu_kaydet(*uclu, score=0.9,
                             source=KaynakTuru.MODEL_GENERATED, confidence=0.9)
    assert _infer_holdout(domain.store, uclu) is None


# --------------------------------------------------------------------------
# Bilgi yoğunluğu
# --------------------------------------------------------------------------

def test_bit_hesabi_sonuc_uzayiyla_uyumlu(rapor):
    """R=13 sonuç uzayı için olgu başına log2(13) bit."""
    import math
    assert rapor.bits_per_fact == pytest.approx(math.log2(13), abs=1e-6)
    assert rapor.total_verified_bits == pytest.approx(
        rapor.correct_facts * rapor.bits_per_fact, abs=1e-6)


def test_daha_genis_sonuc_uzayi_daha_cok_bit():
    dar = run_yield_experiment(seed=1, cycles=6, batch_size=8, initial_facts=6,
                               operands_max=6, negatives_per_fact=3)
    genis = run_yield_experiment(seed=1, cycles=6, batch_size=8, initial_facts=6,
                                 operands_max=12, negatives_per_fact=3)
    assert genis.bits_per_fact > dar.bits_per_fact


# --------------------------------------------------------------------------
# Dürüstlük
# --------------------------------------------------------------------------

def test_sinirlar_raporlanir(rapor):
    assert len(rapor.limitations) >= 4
    birlesik = " ".join(rapor.limitations)
    assert "Sentetik" in birlesik
    assert "DAR bir genelleme" in birlesik


def test_bulgular_bos_degil(rapor):
    assert rapor.findings
