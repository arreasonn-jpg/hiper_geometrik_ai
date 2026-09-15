# -*- coding: utf-8 -*-
"""İnsan değerlendirme paket doldurma/toplama akışının testleri.

Neural üretim pahalıdır; burada sembolik kol ve paket/CSV makineleri sahte
yanıtlarla test edilir. Gerçek üretim ayrı koşularak dağıtım paketine gider.
"""
import csv
import json

import pytest

from hga.evaluation.human_eval_fill import (
    collect_ratings_from_csv,
    fill_and_export_packages,
    symbolic_answer,
)
from hga.evaluation.human_evaluation import (
    ARMS,
    DIMENSIONS,
    _default_prompts,
    analyze_ratings,
)


# ── Sembolik kol ────────────────────────────────────────────────────────────
def test_sembolik_kol_bilmedigini_soyler():
    y = symbolic_answer("Kronecker çarpımı nedir, kısaca açıkla.")
    assert "doğrulanmış" in y and ("yok" in y or "bilmiyorum" in y)


def test_sembolik_kol_belirsizlik_prosedurunu_tarif_eder():
    y = symbolic_answer(
        "epistemik belirsizlik hakkında kesin olmayan bir iddiayı "
        "nasıl işaretlersin?")
    assert "UNVERIFIED" in y


def test_sembolik_kol_celiski_prosedurunu_tarif_eder():
    y = symbolic_answer(
        "olgu doğrulama konusunda iki çelişkili kaynak varsa ne yaparsın?")
    assert "çelişki" in y.lower() or "UNVERIFIED" in y


# ── Paket doldurma + körleme ────────────────────────────────────────────────
@pytest.fixture(scope="module")
def paket_dizini(tmp_path_factory):
    kok = tmp_path_factory.mktemp("insan_paketleri")
    prompts = _default_prompts()
    sahte = {kol: [f"({kol[:1]}{i}) yanıt metni"
                   for i in range(len(prompts))] for kol in ARMS}
    manifest = fill_and_export_packages(kok, sahte, prompts=prompts)
    return kok, manifest


def test_paketler_yazilir_ve_korleme_sizmaz(paket_dizini):
    kok, manifest = paket_dizini
    assert manifest["blinding_leaks"] == []
    paketler = sorted((kok / "paketler").glob("R*.json"))
    assert len(paketler) == manifest["raters"] == 10
    icerik = paketler[0].read_text(encoding="utf-8")
    for kol in ARMS:
        assert f'"{kol}"' not in icerik
    veri = json.loads(icerik)
    assert len(veri["items"]) == manifest["items_per_rater"] == 200
    assert all("response" in oge and oge["response"] for oge in veri["items"])


def test_kor_anahtari_ayri_dizinde(paket_dizini):
    kok, manifest = paket_dizini
    anahtar_dosyasi = kok / "_GIZLI_degerlendiriciye_verme" / "kor_anahtari.json"
    assert anahtar_dosyasi.exists()
    anahtar = json.loads(anahtar_dosyasi.read_text(encoding="utf-8"))
    assert len(anahtar) == 50 * len(ARMS)
    assert set(v["arm"] for v in anahtar.values()) == set(ARMS)


def test_puanlama_sablonlari_bos_ve_tam(paket_dizini):
    kok, _ = paket_dizini
    csvler = sorted((kok / "paketler").glob("*_puanlama.csv"))
    assert len(csvler) == 10
    with csvler[0].open(encoding="utf-8") as f:
        satirlar = list(csv.DictReader(f))
    assert len(satirlar) == 200
    for b in DIMENSIONS:
        assert b in satirlar[0]
        assert satirlar[0][b] == ""


def test_yanit_sayisi_uyusmazsa_hata():
    prompts = _default_prompts()
    eksik = {kol: ["x"] * (len(prompts) - 1) for kol in ARMS}
    with pytest.raises(ValueError):
        fill_and_export_packages("/tmp/olmayan_dizin_hata", eksik,
                                 prompts=prompts)


# ── CSV toplama → α analiz köprüsü ──────────────────────────────────────────
def test_csv_toplama_ve_alfa_koprusu(paket_dizini):
    kok, _ = paket_dizini
    # İki değerlendiricinin CSV'sini programatik doldur (yüksek uyum).
    csvler = sorted((kok / "paketler").glob("*_puanlama.csv"))[:2]
    for dosya in csvler:
        with dosya.open(encoding="utf-8") as f:
            satirlar = list(csv.DictReader(f))
        with dosya.open("w", newline="", encoding="utf-8") as f:
            yazici = csv.writer(f)
            yazici.writerow(["item_id"] + list(DIMENSIONS))
            for satir in satirlar:
                # item_id'ye bağlı deterministik puan → iki rater uyumlu
                taban = int(satir["item_id"][:2], 16) % 5 + 1
                yazici.writerow([satir["item_id"]]
                                + [str(taban)] * 4 + [str(taban % 2)])
    toplanan, ozet = collect_ratings_from_csv(kok)
    assert ozet["raters_found"] == 10  # 8'i boş olsa da dosyalar okunur
    assert ozet["items"] == 200
    sonuc = analyze_ratings(toplanan)
    # İki uyumlu değerlendirici → yüksek α (boş dosyalar None sayılır)
    for boyut in DIMENSIONS:
        assert sonuc[boyut]["alpha"] is None or sonuc[boyut]["alpha"] > 0.9


def test_bos_dizinde_acik_hata(tmp_path):
    with pytest.raises(ValueError):
        collect_ratings_from_csv(tmp_path)
