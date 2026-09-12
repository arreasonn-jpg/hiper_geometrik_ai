# -*- coding: utf-8 -*-
"""TalimatToplayici: 'varsa yükle, yoksa oluştur' davranışı testleri.

REGRESYON: Eski sürüm dosyanın varlığını kontrol etmediği için kullanıcının
elle eklediği talimat örnekleri her eğitimde sessizce siliniyordu.
"""
import json

from egitim.talimat_toplayici import ZENGIN, TalimatToplayici


def test_dosya_yoksa_olusturur(tmp_path):
    yol = tmp_path / "talimat_verisi.json"
    tt = TalimatToplayici(str(yol))
    veri = tt.hazirla_veya_yukle()
    assert yol.exists()
    assert veri == ZENGIN
    with open(yol, encoding="utf-8") as f:
        assert json.load(f) == ZENGIN


def test_mevcut_dosyayi_yukler_ve_ezmez(tmp_path):
    yol = tmp_path / "talimat_verisi.json"
    tt = TalimatToplayici(str(yol))
    tt.hazirla_veya_yukle()  # ilk oluşturma

    # Kullanıcı kendi örneğini ekliyor
    ozel = {"soru": "örnek soru", "cevap": "örnek cevap"}
    with open(yol, encoding="utf-8") as f:
        veri = json.load(f)
    veri.append(ozel)
    with open(yol, "w", encoding="utf-8") as f:
        json.dump(veri, f, ensure_ascii=False)

    # İkinci çağrı dosyayı EZMEMELİ, yüklemeli
    veri2 = tt.hazirla_veya_yukle()
    assert len(veri2) == len(ZENGIN) + 1
    assert veri2[-1] == ozel


def test_sifirla_varsayilana_dondurur(tmp_path):
    yol = tmp_path / "talimat_verisi.json"
    tt = TalimatToplayici(str(yol))

    with open(yol, "w", encoding="utf-8") as f:
        json.dump([{"soru": "x", "cevap": "y"}], f)

    veri = tt.hazirla_veya_yukle(sifirla=True)
    assert veri == ZENGIN


def test_bozuk_dosya_kurtarilir(tmp_path):
    yol = tmp_path / "talimat_verisi.json"
    yol.write_text("{bu geçerli json değil", encoding="utf-8")
    tt = TalimatToplayici(str(yol))
    veri = tt.hazirla_veya_yukle()
    assert veri == ZENGIN
