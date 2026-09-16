# -*- coding: utf-8 -*-
"""Faz 1: legacy kodun aktif mimariye sızmadığını otomatik denetler."""
import ast
import pathlib

KOK = pathlib.Path(__file__).resolve().parents[1]
LEGACY_MODULLER = {"bilgi_katmani", "calistir", "arayuz"}
# hga.ui_runtime bilinçli tek istisnadır: eski sohbet arayüzlerini ayakta tutar.
IZINLI_ISTISNALAR = {"hga/ui_runtime.py"}

AKTIF_DIZINLER = ("hga", "mimari", "egitim", "experiments", "golden_dataset")


def _python_dosyalari():
    for dizin in AKTIF_DIZINLER:
        for yol in (KOK / dizin).rglob("*.py"):
            if "__pycache__" in yol.parts:
                continue
            yield yol


def _import_edilen_modul_adlari(yol: pathlib.Path):
    agac = ast.parse(yol.read_text(encoding="utf-8"))
    for dugum in ast.walk(agac):
        if isinstance(dugum, ast.Import):
            for ad in dugum.names:
                yield ad.name.split(".")[0]
        elif isinstance(dugum, ast.ImportFrom) and dugum.module:
            yield dugum.module.split(".")[0]


def test_legacy_dosyalari_kokte_degil():
    for ad in LEGACY_MODULLER:
        assert not (KOK / f"{ad}.py").exists(), (
            f"{ad}.py hâlâ kökte; legacy/ altına taşınmalı")
        assert (KOK / "legacy" / f"{ad}.py").exists()


def test_legacy_paketi_isaretlenmis():
    init = (KOK / "legacy" / "__init__.py").read_text(encoding="utf-8")
    assert "LEGACY" in init


def test_aktif_katmanlar_legacy_import_etmez():
    sizinti = []
    for yol in _python_dosyalari():
        goreli = yol.relative_to(KOK).as_posix()
        if goreli in IZINLI_ISTISNALAR:
            continue
        for modul in _import_edilen_modul_adlari(yol):
            if modul in LEGACY_MODULLER or modul == "legacy":
                sizinti.append(f"{goreli} → {modul}")
    assert not sizinti, "Aktif katmanda legacy import sızıntısı: " + ", ".join(sizinti)


def test_ui_runtime_istisnasi_legacy_paketinden_import_eder():
    kaynak = (KOK / "hga" / "ui_runtime.py").read_text(encoding="utf-8")
    assert "from legacy.bilgi_katmani import" in kaynak
    assert "from bilgi_katmani import" not in kaynak


def test_tek_bagimlilik_kaynagi_pyproject():
    assert not (KOK / "gereksinimler.txt").exists(), (
        "gereksinimler.txt kaldırıldı; tek kaynak pyproject.toml")
    requirements = (KOK / "requirements.txt").read_text(encoding="utf-8")
    assert "pyproject.toml" in requirements
    assert "--editable ." in requirements
    assert (KOK / "requirements-lock.txt").exists()
    assert (KOK / "pyproject.toml").exists()


def test_legacy_paketi_dagitima_girmez():
    pyproject = (KOK / "pyproject.toml").read_text(encoding="utf-8")
    assert "legacy*" not in pyproject.split("[tool.setuptools.packages.find]")[1]


def test_readme_legacyyi_aktif_mimari_olarak_sunmaz():
    readme = (KOK / "README.md").read_text(encoding="utf-8")
    assert "Legacy 3 Katmanlı Halüsinasyon Prototipi (aktif mimari değil)" in readme
    ilk_bolum = readme.split("---", 1)[0]
    assert "3 katmanlı halüsinasyon kontrol mekanizması" not in ilk_bolum
    ilk_bolum_duz = " ".join(ilk_bolum.lower().split())
    assert "aktif araştırma anlatısı" in ilk_bolum_duz
