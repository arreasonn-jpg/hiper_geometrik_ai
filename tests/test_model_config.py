# -*- coding: utf-8 -*-
"""Model/eğitim config yükleyici testleri."""
import os
import sys
import tempfile

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIMARI = os.path.join(KOK, "mimari")
for _p in [KOK, MIMARI]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_config import ModelConfig, TrainingConfig, yukle  # noqa: E402


def test_model_config_from_dict_cast():
    c = ModelConfig.from_dict({"n": "128", "katman_sayisi": "2",
                               "dropout": "0.25", "checkpoint_kullan": "true",
                               "bilinmeyen": 123})
    assert c.n == 128
    assert c.katman_sayisi == 2
    assert abs(c.dropout - 0.25) < 1e-9
    assert c.checkpoint_kullan is True
    assert not hasattr(c, "bilinmeyen")


def test_training_config_from_dict():
    t = TrainingConfig.from_dict({"ogrenme_hizi": "0.0005", "toplam_cag": "7",
                                  "early_stopping_patience": "3"})
    assert t.ogrenme_hizi == 0.0005
    assert t.toplam_cag == 7
    assert t.early_stopping_patience == 3


def test_yaml_yukle_fallback():
    metin = """
model:
  n: 64
  katman_sayisi: 3
  checkpoint_kullan: true
training:
  toplam_cag: 2
  validation_split: 0.1
"""
    fd, yol = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(metin)
        cfg = yukle(yol)
        assert cfg["model"].n == 64
        assert cfg["model"].katman_sayisi == 3
        assert cfg["model"].checkpoint_kullan is True
        assert cfg["training"].toplam_cag == 2
        assert abs(cfg["training"].validation_split - 0.1) < 1e-9
    finally:
        os.remove(yol)


if __name__ == "__main__":
    testler = [(ad, fn) for ad, fn in sorted(globals().items())
               if ad.startswith("test_") and callable(fn)]
    basarisiz = 0
    for ad, fn in testler:
        try:
            fn()
            print(f"  ✅ {ad}")
        except Exception as e:  # noqa: BLE001
            basarisiz += 1
            print(f"  ❌ {ad}: {type(e).__name__}: {e}")
    print("\nTÜM TESTLER GEÇTİ" if basarisiz == 0 else f"{basarisiz} test başarısız")
    sys.exit(1 if basarisiz else 0)
