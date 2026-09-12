# -*- coding: utf-8 -*-
"""
Görev ablasyonu testleri — yazılan bilgi, modelin kendi görevini çözüyor mu?
============================================================================
Çalıştırma (torch gerekir):
    .venv/bin/python tests/test_gorev_ablasyonu.py
    .venv/bin/python -m pytest tests/test_gorev_ablasyonu.py -q
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [KOK, os.path.join(KOK, "mimari")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hga.memory import GorevAblasyonu, torch_var_mi  # noqa: E402


def _model(tohum=0):
    import torch
    torch.manual_seed(tohum)
    from kuresel_model import HiperGeometrikAI
    return HiperGeometrikAI(
        n=32, katman_sayisi=2, baglam_penceresi=4, emb_dim=16,
        num_heads=2, sozluk_boyutu=32, dropout=0.0,
        seyrek_tablo_boyutu=2048, seyrek_boyut=32, bilgilendir=False)


def _tamamlama_ucluleri():
    """(özne, ilişki) → nesne eşlemesi deterministik olan 8 üçlü."""
    return [("E1", "R1", "E5"),
            ("E2", "R1", "E6"),
            ("E3", "R1", "E7"),
            ("E4", "R1", "E8"),
            ("E5", "R1", "E1"),
            ("E6", "R1", "E2"),
            ("E7", "R1", "E3"),
            ("E8", "R1", "E4")]


def test_gorev_ablasyon_torch_yoksa_guvenli_hata():
    if torch_var_mi():
        print("  (torch mevcut — gerçek testler aşağıda)")
        return
    hata = None
    try:
        GorevAblasyonu(object())
    except ImportError as e:
        hata = e
    assert hata is not None


def test_gorev_ablasyon_sekrek_belleksiz_hata():
    """Seyrek bellek kapalı model → açık hata."""
    if not torch_var_mi():
        return
    import torch
    torch.manual_seed(0)
    from kuresel_model import HiperGeometrikAI
    model = HiperGeometrikAI(n=16, katman_sayisi=1, baglam_penceresi=4,
                             emb_dim=8, num_heads=2, sozluk_boyutu=16,
                             dropout=0.0, seyrek_tablo_boyutu=0,
                             bilgilendir=False)
    hata = None
    try:
        GorevAblasyonu(model)
    except ValueError as e:
        hata = e
    assert hata is not None


def test_gorev_ablasyon_sozluk_ve_pencereler():
    """Sözlük kurma + pencere: PAD solda, hedef = nesne id'si."""
    if not torch_var_mi():
        return
    d = GorevAblasyonu(_model(), tohum=0)
    ucluler = [("E1", "R1", "E5"), ("E2", "R1", "E6")]
    sozluk = d.sozluk_kur(ucluler)
    assert sozluk["E1"] == 1 and sozluk["R1"] >= 1
    X, y = d.pencereler(ucluler)
    assert tuple(X.shape) == (2, 4)
    assert X[0, 0].item() == 0 and X[0, 1].item() == 0      # PAD solda
    assert X[0, 2].item() == sozluk["E1"]                   # özne
    assert X[0, 3].item() == sozluk["R1"]                   # ilişki
    assert y[0].item() == sozluk["E5"]                      # hedef = nesne
    assert y[1].item() == sozluk["E6"]


def test_gorev_ablasyon_kontrol_sans():
    """Boş + donuk bellek yolu: yoğun gövde donuk-rastgele → ~şans (< 0.4)."""
    if not torch_var_mi():
        return
    d = GorevAblasyonu(_model(), tohum=0)
    ucluler = _tamamlama_ucluleri()
    d.sozluk_kur(ucluler)
    dogruluk = d.kontrol(ucluler)
    assert dogruluk < 0.4, f"beklenen ~şans (1/8=0.125), gelen: {dogruluk}"


def test_gorev_ablasyon_bilgi_yazili_yuksek():
    """Bellek yolu eğitildikten sonra tamamlama doğruluğu ~%100."""
    if not torch_var_mi():
        return
    d = GorevAblasyonu(_model(), tohum=0)
    ucluler = _tamamlama_ucluleri()
    d.sozluk_kur(ucluler)
    d.bilgi_yaz(ucluler)
    dolu, _ = d.tablo.doluluk_orani()
    assert dolu > 0
    dogruluk = d.tamamlama_dogrulugu(ucluler)
    assert dogruluk >= 0.9, f"beklenen ~%100, gelen: {dogruluk}"


def test_gorev_ablasyon_kos_ve_yol_canliligi():
    """kos(): kontrol ~şans, deney ~%100; ölü-yol → canlı-yol ölçülür."""
    if not torch_var_mi():
        return
    d = GorevAblasyonu(_model(), tohum=0)
    ucluler = _tamamlama_ucluleri()
    sonuc = d.kos(ucluler)
    assert sonuc["kontrol_dogruluk"] < 0.4
    assert sonuc["deney_dogruluk"] >= 0.9
    assert sonuc["yazilan_satir"] > 0
    assert sonuc["yol_canlandi"] is True
    # etki büyüklüğü: deney - kontrol büyük
    assert sonuc["deney_dogruluk"] - sonuc["kontrol_dogruluk"] > 0.5


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
