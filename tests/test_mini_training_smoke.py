# -*- coding: utf-8 -*-
import os
import sys
import tempfile

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)
MIMARI = os.path.join(KOK, "mimari")
if MIMARI not in sys.path:
    sys.path.insert(0, MIMARI)

from egitim.mini_smoke import mini_egitim_smoke_markdown, mini_egitim_smoke_raporu


def test_mini_egitim_smoke_raporu_dosya_yazar():
    with tempfile.TemporaryDirectory() as tmp:
        rapor_yol = os.path.join(tmp, "mini.json")
        md_yol = os.path.join(tmp, "mini.md")
        rapor = mini_egitim_smoke_raporu(
            rapor_yol=rapor_yol, markdown_yol=md_yol,
            log_dizini=os.path.join(tmp, "logs"),
            checkpoint_dizini=os.path.join(tmp, "ckpt"),
            cag=1, batch=32, tekrar=2)
        assert os.path.exists(rapor_yol)
        assert os.path.exists(md_yol)
        assert rapor["rapor_tipi"] == "hga-mini-training-smoke-v1"
        if rapor["durum"] == "ok":
            assert rapor["egitim"]["loss_sonlu"] is True
            assert rapor["egitim"]["grad_sonlu"] is True
            assert rapor["checkpoint_uyumluluk"]["ok"] is True
        else:
            assert rapor["durum"] == "skipped_dependency"


def test_mini_egitim_smoke_markdown_skipped():
    md = mini_egitim_smoke_markdown({"durum": "skipped_dependency", "rapor_tipi": "x", "cihaz": {}})
    assert "Mini Eğitim Smoke" in md


if __name__ == "__main__":
    test_mini_egitim_smoke_raporu_dosya_yazar()
    test_mini_egitim_smoke_markdown_skipped()
    print("OK")
