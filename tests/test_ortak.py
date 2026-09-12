# -*- coding: utf-8 -*-
"""ortak.py: ağırlık yükleme ve uyuşmazlık raporlama testleri."""
import logging

import pytest
import torch

from ortak import agirlik_yukle, model_olustur


def test_tam_eslesen_agirlik_sessiz(tmp_path, caplog):
    a = model_olustur(sozluk_boyutu=50, n=32, baglam_penceresi=4, emb_dim=16)
    b = model_olustur(sozluk_boyutu=50, n=32, baglam_penceresi=4, emb_dim=16)
    yol = tmp_path / "hiper_model_32.pt"
    torch.save(a.state_dict(), yol)

    with caplog.at_level(logging.WARNING, logger="hiper"):
        agirlik_yukle(b, str(yol))
    assert not [r for r in caplog.records if "Ağırlık uyuşmazlığı" in r.getMessage()]


def test_eksik_ve_beklenmeyen_anahtar_uyarisi(tmp_path, caplog):
    """Eski kontrol noktası senaryosu: bazı anahtarlar eksik/beklenmeyen.

    Uyarı loglanmalı ama eşleşen ağırlıklar yüklenmeye devam etmeli.
    """
    a = model_olustur(sozluk_boyutu=50, n=32, baglam_penceresi=4, emb_dim=16)
    sd = a.state_dict()
    del sd["decoder.anlam_cozucu.weight"]       # eksik anahtar
    sd["eski_mimari.katman"] = torch.zeros(4)   # beklenmeyen anahtar
    yol = tmp_path / "eski_model.pt"
    torch.save(sd, yol)

    b = model_olustur(sozluk_boyutu=50, n=32, baglam_penceresi=4, emb_dim=16)
    with caplog.at_level(logging.WARNING, logger="hiper"):
        bilgi = agirlik_yukle(b, str(yol))

    assert "decoder.anlam_cozucu.weight" in bilgi.missing_keys
    assert "eski_mimari.katman" in bilgi.unexpected_keys
    assert any("Ağırlık uyuşmazlığı" in r.getMessage() for r in caplog.records)
    # Eşleşen ağırlıklar gerçekten yüklendi mi?
    assert torch.equal(b.kelime_gomme.weight, a.kelime_gomme.weight)


def test_hic_eslesme_olursa_hata(tmp_path):
    """Tamamen alakasız bir state_dict: sessiz başlangıç yerine RuntimeError."""
    b = model_olustur(sozluk_boyutu=50, n=32, baglam_penceresi=4, emb_dim=16)
    yol = tmp_path / "baska_model.pt"
    torch.save({"tamamen.baska": torch.zeros(3)}, yol)

    with pytest.raises(RuntimeError, match="Hiçbir ağırlık"):
        agirlik_yukle(b, str(yol))


def test_boyut_uyusmazligi_hata_firlatir(tmp_path):
    """Aynı adlı ama farklı boyutlu ağırlık: PyTorch boyut hatası iletilir."""
    a = model_olustur(sozluk_boyutu=50, n=32, baglam_penceresi=4, emb_dim=16)
    b = model_olustur(sozluk_boyutu=50, n=64, baglam_penceresi=4, emb_dim=16)
    yol = tmp_path / "hiper_model_32.pt"
    torch.save(a.state_dict(), yol)

    with pytest.raises(Exception):
        agirlik_yukle(b, str(yol))
