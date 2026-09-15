# -*- coding: utf-8 -*-
"""P1: Gerçek Türkçe dil modelleme benchmarkı (turkish_lm_v1) testleri.

Torch'suz kısım (korpus, split, n-gram) her ortamda koşar; neural kollar
``importorskip`` ile korunur — twt_results testleriyle aynı desen.
"""
import math

import pytest

from hga.evaluation.turkish_lm import (
    BIGRAM_LAMBDA,
    NGRAM_ALPHA,
    NGRAM_ARMS,
    PROFILES,
    NgramBaselines,
    document_splits_disjoint,
    prepare_lm_corpus,
)


# ── Korpus ve split sözleşmesi (torch gerekmez) ─────────────────────────────
def test_korpus_deterministik_ve_hashli():
    once = prepare_lm_corpus(max_docs=60)
    sonra = prepare_lm_corpus(max_docs=60)
    assert once.dataset_hash == sonra.dataset_hash
    assert once.config_hash == sonra.config_hash
    assert len(once.documents) == 60


def test_max_docs_hashi_degistirir():
    """Alt küme seçimi görünmez olamaz: imza farklı olmalı."""
    assert (prepare_lm_corpus(max_docs=50).dataset_hash
            != prepare_lm_corpus(max_docs=60).dataset_hash)


def test_splitler_belge_ayrik():
    korpus = prepare_lm_corpus(max_docs=200)
    assert document_splits_disjoint(korpus)
    # Her belge tam bir splitte ve üç split de boş değil.
    splitler = {d.split for d in korpus.documents}
    assert splitler == {"train", "dev", "test"}


def test_ayni_belgenin_cumleleri_tek_splitte():
    korpus = prepare_lm_corpus()
    gorulen = {}
    for belge in korpus.documents:
        if belge.doc_id in gorulen:
            assert gorulen[belge.doc_id] == belge.split
        gorulen[belge.doc_id] = belge.split


def test_bos_olmayan_metin_ve_sayimlar():
    korpus = prepare_lm_corpus(max_docs=40)
    for belge in korpus.documents:
        assert belge.text.strip()
        assert belge.word_count == len(belge.text.split())
        assert belge.sentence_count >= 1


# ── tr_corpus_v1: milyon-kelime vendored korpus ─────────────────────────────
def test_tr_corpus_v1_bir_milyon_kelimeyi_asar():
    korpus = prepare_lm_corpus(corpus="tr_corpus_v1")
    assert sum(d.word_count for d in korpus.documents) >= 1_000_000
    assert document_splits_disjoint(korpus)


def test_tr_corpus_v1_hash_dogrulamasi_bozulmayi_yakalar(tmp_path, monkeypatch):
    """Tek bayt oynarsa yükleyici korpusu REDDETMELİ."""
    import gzip
    import shutil

    from hga.evaluation import turkish_lm

    sahte = tmp_path / "tr_corpus_v1"
    shutil.copytree(turkish_lm.TR_CORPUS_DIR, sahte)
    with gzip.open(sahte / "corpus.jsonl.gz", "rb") as h:
        icerik = h.read()
    bozuk = icerik.replace(b"bir", b"iki", 1)
    assert bozuk != icerik
    with gzip.open(sahte / "corpus.jsonl.gz", "wb") as h:
        h.write(bozuk)
    monkeypatch.setattr(turkish_lm, "TR_CORPUS_DIR", sahte)
    with pytest.raises(ValueError):
        prepare_lm_corpus(corpus="tr_corpus_v1")


def test_tr_corpus_v1_kaynak_karisimi_ve_lisans_disiligi():
    """UD + bible + twt hepsi mevcut; NC lisanslı IMST kesinlikle YOK."""
    korpus = prepare_lm_corpus(corpus="tr_corpus_v1")
    kaynaklar = {d.doc_id.split(":")[0] for d in korpus.documents}
    assert {"ud_kenet", "ud_boun", "ud_penn", "bible", "twt"} <= kaynaklar
    assert "ud_imst" not in kaynaklar


def test_tr_corpus_v1_provenance_hashleri_gecerli():
    import json

    from hga.evaluation.turkish_lm import TR_CORPUS_DIR

    provenance = json.loads(
        (TR_CORPUS_DIR / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert provenance["counts"]["words"] >= 1_000_000
    kaynak_adlari = {u["source"] for u in provenance["upstream"]}
    assert "ud_imst" not in kaynak_adlari
    assert all(u["license"] for u in provenance["upstream"])
    assert any(h["source"] == "ud_imst"
               for h in provenance["excluded_sources"])


def test_bilinmeyen_korpus_reddedilir():
    with pytest.raises(ValueError):
        prepare_lm_corpus(corpus="opus")


# ── N-gram kontrolleri gerçek olasılık modeli mi? ───────────────────────────
def test_unigram_dagilimi_bire_toplanir():
    model = NgramBaselines([[4, 5, 4, 6]], vocab_size=8)
    toplam = sum(model._p_unigram(t) for t in range(8))
    assert math.isclose(toplam, 1.0, rel_tol=1e-9)


def test_interpolasyonlu_bigram_dagilimi_bire_toplanir():
    model = NgramBaselines([[4, 5, 4, 6], [5, 6, 7]], vocab_size=8)
    for prev in (2, 4, 7):  # BOS, görülen, görülmemiş önceki token
        toplam = sum(
            BIGRAM_LAMBDA * model._p_bigram(prev, t)
            + (1.0 - BIGRAM_LAMBDA) * model._p_unigram(t)
            for t in range(8))
        assert math.isclose(toplam, 1.0, rel_tol=1e-9), prev


def test_ngram_perplexity_sonlu_ve_pozitif():
    model = NgramBaselines([[4, 5, 4, 6, 5]], vocab_size=8,
                           alpha=NGRAM_ALPHA)
    for kol in NGRAM_ARMS:
        sonuc = model.perplexity([[4, 5, 7]], kol)
        assert math.isfinite(sonuc["perplexity"])
        assert sonuc["perplexity"] > 1.0
        assert sonuc["tokens"] == 3.0


def test_bilinmeyen_ngram_kolu_hata():
    model = NgramBaselines([[4, 5]], vocab_size=8)
    with pytest.raises(ValueError):
        model.perplexity([[4]], "trigram")


def test_bos_degerlendirme_kumesi_hata():
    model = NgramBaselines([[4, 5]], vocab_size=8)
    with pytest.raises(ValueError):
        model.perplexity([], "unigram")


def test_cogunluk_kolu_moda_tokenin_orani():
    model = NgramBaselines([[4, 4, 5]], vocab_size=8)
    assert model.majority_accuracy([[4, 5, 4, 6]]) == 0.5


# ── Profil sözleşmesi ───────────────────────────────────────────────────────
def test_profiller_cok_tohumlu():
    for ad, profil in PROFILES.items():
        assert len(profil["seeds"]) >= 2, ad


# ── Neural kollar (torch gerekli) ───────────────────────────────────────────
torch = pytest.importorskip("torch")

from hga.evaluation.turkish_lm import (  # noqa: E402
    NEURAL_ARMS,
    PARAMETER_TOLERANCE,
    build_lm_models,
    run_turkish_lm_benchmark,
    turkish_lm_markdown,
)

HIZLI = {"max_docs": 60, "steps": 6, "max_vocab": 400,
         "batch_size": 16, "eval_batch_size": 256}


@pytest.fixture(scope="module")
def hizli_rapor():
    return run_turkish_lm_benchmark(profile="smoke", seeds=(1, 2),
                                    overrides=HIZLI)


def test_parametre_butcesi_eslenir():
    kurucular, eslesme = build_lm_models(
        512, {**PROFILES["smoke"], **HIZLI})
    sayilar = {ad: sum(p.numel() for p in kurucu().parameters()
                       if p.requires_grad)
               for ad, kurucu in kurucular.items()}
    assert set(sayilar) == set(NEURAL_ARMS)
    assert max(sayilar.values()) / min(sayilar.values()) <= PARAMETER_TOLERANCE


def test_rapor_alanlari_ve_sonluluk(hizli_rapor):
    rapor = hizli_rapor
    assert rapor.protocol == "turkish_lm_v1"
    assert rapor.checks["document_disjoint_splits"]
    assert rapor.checks["finite_metrics_all_arms"]
    assert rapor.checks["multi_seed_reported"]
    for kol in NEURAL_ARMS:
        for kosu in rapor.arms[kol]["per_seed"]:
            assert math.isfinite(kosu["test"]["perplexity"])
            assert 0.0 <= kosu["test"]["top1_accuracy"] <= 1.0
            assert kosu["test"]["top5_accuracy"] >= kosu["test"]["top1_accuracy"]


def test_milyon_kelime_kapisi_twt_uzerinde_fail(hizli_rapor):
    """TWT ~66K kelimedir; bu kapı geçiyorsa ya korpus büyüdü ya ölçüm bozuk."""
    assert hizli_rapor.checks["corpus_at_least_1m_words"] is False
    assert any("corpus_at_least_1m_words" in n
               for n in hizli_rapor.limitations)


def test_karsilastirmalar_hga_iki_rakibe_karsi(hizli_rapor):
    etiketler = {(k["treatment_label"], k["baseline_label"])
                 for k in hizli_rapor.comparisons}
    assert etiketler == {("hga", "dense"), ("hga", "transformer")}


def test_ayni_tohum_ayni_sonuc():
    a = run_turkish_lm_benchmark(profile="smoke", seeds=(1, 2),
                                 overrides=HIZLI)
    b = run_turkish_lm_benchmark(profile="smoke", seeds=(1, 2),
                                 overrides=HIZLI)
    assert a.dataset_hash == b.dataset_hash
    for kol in NEURAL_ARMS:
        assert (a.arms[kol]["summary"]["perplexity"]["mean"]
                == b.arms[kol]["summary"]["perplexity"]["mean"])


def test_tek_tohum_reddedilir():
    with pytest.raises(ValueError):
        run_turkish_lm_benchmark(profile="smoke", seeds=(1,), overrides=HIZLI)


def test_bilinmeyen_profil_reddedilir():
    with pytest.raises(ValueError):
        run_turkish_lm_benchmark(profile="dev")


def test_markdown_kapilari_ve_ngram_tabanini_gosterir(hizli_rapor):
    md = turkish_lm_markdown(hizli_rapor)
    assert "unigram" in md and "bigram" in md
    assert "PASS" in md and "FAIL" in md
    assert "corpus_at_least_1m_words" in md
    assert "held-out" in md


def test_karne_language_modeling_bolumu_skorlanir(hizli_rapor):
    from hga.evaluation.capability_vector import build_scorecard
    karne = build_scorecard(language_modeling=hizli_rapor.to_dict())
    bolum = karne.sections["language_modeling"]
    assert bolum["score"] is not None
    # Milyon-token kapısı FAIL olduğu sürece tavan 10 olamaz.
    assert bolum["score"] < 10.0
    assert bolum["evidence"] and "turkish_lm_v1" in bolum["evidence"][0]
