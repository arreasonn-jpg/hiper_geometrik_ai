# -*- coding: utf-8 -*-
"""P0-5: Türkçe semantik çıkarım hattı ve benchmarkı testleri."""
import pytest

from hga.evaluation.semantic_extraction import (
    GOLD_SET,
    LAYERS,
    run_semantic_extraction_benchmark,
    semantic_extraction_markdown,
)
from hga.experience.semantik_ayiklayici import (
    cumle_coz,
    fiil_coz,
    kok_ve_durum,
    korpus_coz,
    store_a_aktar,
)
from hga.knowledge import KnowledgeStore


# ── Morfoloji ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("yuzey,kok,durum", [
    ("Ankara'ya", "ankara", "yonelme"),
    ("arabayla", "araba", "vasita"),
    ("okula", "okul", "yonelme"),
    ("kitabı", "kitap", "belirtme"),
    ("ata", "at", "yonelme"),
    ("evde", "ev", "bulunma"),
    ("İstanbul'dan", "istanbul", "ayrilma"),
    ("araba", "araba", "yalin"),
])
def test_kok_ve_durum_cozumlemesi(yuzey, kok, durum):
    assert kok_ve_durum(yuzey) == (kok, durum)


def test_okula_oku_la_olarak_yanlis_cozulmez():
    """Ek belirsizliği: sözlükteki kök tercih edilmeli."""
    assert kok_ve_durum("okula")[0] == "okul"


@pytest.mark.parametrize("yuzey,lemma,zaman,olumsuz", [
    ("gitti", "git", "gecmis", False),
    ("gitmeyecek", "git", "gelecek", True),
    ("gelmedi", "gel", "gecmis", True),
    ("okuyor", "oku", "simdiki", False),
    ("uyuyor", "uyu", "simdiki", False),
    ("binecek", "bin", "gelecek", False),
])
def test_fiil_cozumlemesi(yuzey, lemma, zaman, olumsuz):
    sonuc = fiil_coz(yuzey)
    assert sonuc is not None
    assert (sonuc["lemma"], sonuc["tense"], sonuc["negated"]) == (lemma, zaman, olumsuz)


def test_fiil_olmayan_none_doner():
    assert fiil_coz("ev") is None
    assert fiil_coz("ab") is None


# ── Tam hat ─────────────────────────────────────────────────────────────────
def test_ornek_cumle_tum_katmanlari_uretir():
    """README'deki hedef örnek: Ali dün Ankara'ya arabayla gitti."""
    r = cumle_coz("Ali dün Ankara'ya arabayla gitti.")
    varliklar = {e.lemma for e in r.entities}
    assert {"ali", "ankara", "araba"} <= varliklar
    assert ("ali", "gitmek", "ankara") in {
        (x.subject, x.predicate, x.object) for x in r.relations}
    assert ("ali", "travel_mode", "araba") in {
        (p.entity, p.name, p.value) for p in r.properties}
    assert "dün" in {t["value"] for t in r.temporal if t["kind"] == "adverb"}


def test_olumsuzluk_kutbu_ve_sifir_skor():
    r = cumle_coz("Mehmet yarın okula gitmeyecek.")
    assert r.relations[0].polarity == "NEGATIVE"
    assert r.negations and r.negations[0]["scope"] == "gitmek"


def test_sifat_kendinden_sonraki_isme_baglanir():
    r = cumle_coz("Ayşe kırmızı arabaya bindi.")
    renkler = [(p.entity, p.value) for p in r.properties if p.name == "renk"]
    assert renkler == [("araba", "kırmızı")]


def test_bilinmeyen_fiil_kanitliysa_induklenir_dusuk_guvenle():
    """Sözlükte olmayan fiil, kanonik SOV kanıtı varsa mastar olarak
    İNDÜKLENİR; ilişki induced=True ve düşük güvenle işaretlenir."""
    r = cumle_coz("Ali kitabı inceledi.")
    assert len(r.relations) == 1
    iliski = r.relations[0]
    assert (iliski.subject, iliski.predicate, iliski.object) == \
        ("ali", "incelemek", "kitap")
    assert iliski.induced is True
    assert iliski.confidence <= 0.55, "indüklenen ilişki düşük güven taşımalı"


def test_kanit_yetersizse_iliski_uydurulmaz():
    """İndüksiyon kanıt ister: kısa kök (<4) veya eksik üye yapısında yönlü
    ilişki iddiası ÜRETİLMEZ. 'Bilmiyorum' > yanlış bilgi."""
    # 'at' kökü 2 harf: indüklenemez.
    r = cumle_coz("Ali topu attı.")
    assert r.relations == []
    assert any(n.startswith("bilinmeyen_fiil") for n in r.skipped_reasons)
    # Özne yok: indüklenemez.
    r2 = cumle_coz("Kitap düştü.")
    assert r2.relations == []


def test_cati_ekli_fiilde_cekimser_kalinir():
    """Ettirgen/edilgen çatıda üye yapısı yüzeyden çıkarılamaz; naif
    özne/nesne eşlemesi YANLIŞ bilgi üretirdi ('okutan' okumaz)."""
    r = cumle_coz("Öğretmen öğrencilere kitabı okuttu.")
    assert r.relations == []
    assert any(n.startswith("cati_eki_uye_yapisi_belirsiz")
               for n in r.skipped_reasons)


def test_hafif_fiil_bilesigi_kurulur():
    """'tamir etti' → 'tamir etmek': hafif fiil kendinden önceki yalın adla
    bileşik yüklem kurar; o ad ayrı varlık olarak KALMAZ."""
    r = cumle_coz("Ayşe arabayı tamir etti.")
    assert len(r.relations) == 1
    iliski = r.relations[0]
    assert (iliski.subject, iliski.predicate, iliski.object) == \
        ("ayse", "tamir etmek", "araba")
    assert iliski.induced is True
    assert "tamir" not in {e.lemma for e in r.entities}


def test_bilinmeyen_varlik_atilmaz_dusuk_guvenle_kaydedilir():
    r = cumle_coz("Ali helikoptere bindi.")
    kokler = {e.lemma: e for e in r.entities}
    assert "helikopter" in kokler
    assert kokler["helikopter"].entity_type == "UNKNOWN"
    assert kokler["helikopter"].confidence < 0.5


def test_ontoloji_ozelligi_dusuk_guvenli():
    """Metinde yazmayan tip varsayımı, yazan sıfattan daha düşük güvenli olmalı."""
    r = cumle_coz("Ayşe kırmızı arabaya bindi.")
    sifat = next(p for p in r.properties if p.name == "renk")
    ontoloji = next(p for p in r.properties
                    if p.source_surface.startswith("<ontology:"))
    assert ontoloji.confidence < sifat.confidence


def test_entity_linking_ayni_koku_birlestirir():
    r = cumle_coz("Ali arabayla arabaya bindi.")
    assert sum(1 for e in r.entities if e.lemma == "araba") == 1


def test_belirtisiz_nesne_ozne_sayilmaz():
    r = cumle_coz("Veli okulda kitap okuyor.")
    ucluler = {(x.subject, x.predicate, x.object, x.role) for x in r.relations}
    assert ("veli", "okumak", "kitap", "nesne") in ucluler
    assert ("veli", "okumak", "okul", "konum") in ucluler


def test_bos_cumle_cokmez():
    r = cumle_coz("")
    assert r.bos_mu and "bos_cumle" in r.skipped_reasons


# ── KnowledgeStore aktarımı ─────────────────────────────────────────────────
def test_store_aktarimi_guvenleri_korur():
    store = KnowledgeStore()
    r = cumle_coz("Ali dün Ankara'ya arabayla gitti.")
    ozet = store_a_aktar(store, r)
    assert ozet["entities_written"] == 3
    assert ozet["relations_written"] == 1
    assert store.entities.ad_bul("ali") is not None


def test_olumsuz_iliski_pozitif_olgu_olarak_yazilmaz():
    store = KnowledgeStore()
    store_a_aktar(store, cumle_coz("Mehmet yarın okula gitmeyecek."))
    agrega = store.relations.olgu_agrega("E_MEHMET", "R_GITMEK", "E_OKUL")
    assert agrega is not None
    assert agrega["score"] == 0.0


def test_korpus_verim_metrikleri():
    # "attı" → 'at' kökü kısa (<4): indüklenmez, bilinmeyen_fiil sayılır.
    ozet = korpus_coz(["Ali ata bindi.", "Ayşe arabaya bindi.",
                       "Ali topu attı."])
    assert ozet["sentences"] == 3
    assert 0.0 <= ozet["relation_coverage"] <= 1.0
    assert ozet["entity_yield"] > 0
    assert "bilinmeyen_fiil" in ozet["skipped_reasons"]


# ── Benchmark ───────────────────────────────────────────────────────────────
def test_benchmark_tum_katmanlari_olcer():
    rapor = run_semantic_extraction_benchmark()
    assert set(rapor.layers) == set(LAYERS)
    for m in rapor.layers.values():
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0


def test_kapsam_disi_cumlelerden_iliski_uydurulmaz():
    rapor = run_semantic_extraction_benchmark()
    assert rapor.out_of_scope_false_positive_rate == 0.0
    assert rapor.checks["no_out_of_scope_hallucination"]


def test_altin_set_kapsam_disi_ornek_icerir():
    """Precision ancak negatif örnek varsa ölçülebilir."""
    assert any(not g.in_scope for g in GOLD_SET)


def test_kabul_kapilari_gecer():
    rapor = run_semantic_extraction_benchmark()
    for ad, sonuc in rapor.checks.items():
        assert sonuc, f"kapı düştü: {ad}"


def test_markdown_hatali_cumleleri_listeler():
    md = semantic_extraction_markdown(run_semantic_extraction_benchmark())
    assert "Hatalı cümleler" in md and "Kabul kapıları" in md


def test_bos_altin_set_acik_hata():
    with pytest.raises(ValueError):
        run_semantic_extraction_benchmark(gold_set=[])
