# -*- coding: utf-8 -*-
"""
Genelleme ablasyonu testleri — bellek, görülmeyen olgulara genelleştiriyor mu?
=============================================================================
Çalıştırma (torch gerekir):
    .venv/bin/python tests/test_genelleme_ablasyonu.py
    .venv/bin/python -m pytest tests/test_genelleme_ablasyonu.py -q
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [KOK, os.path.join(KOK, "mimari")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hga.memory import GenellemeAblasyonu, torch_var_mi  # noqa: E402


def _model(tohum=0):
    import torch
    torch.manual_seed(tohum)
    from kuresel_model import HiperGeometrikAI
    return HiperGeometrikAI(
        n=32, katman_sayisi=2, baglam_penceresi=4, emb_dim=16,
        num_heads=2, sozluk_boyutu=64, dropout=0.0,
        seyrek_tablo_boyutu=2048, seyrek_boyut=32, bilgilendir=False)


def _kategori_ucluleri():
    """4 nesne kategorisi, her biri 2 öznede → 8 olgu (dengeli genelleme)."""
    return [("S1", "R1", "C1"), ("S2", "R1", "C1"),
            ("S3", "R1", "C2"), ("S4", "R1", "C2"),
            ("S5", "R1", "C3"), ("S6", "R1", "C3"),
            ("S7", "R1", "C4"), ("S8", "R1", "C4")]


def test_genelleme_torch_yoksa_guvenli_hata():
    if torch_var_mi():
        print("  (torch mevcut — gerçek testler aşağıda)")
        return
    hata = None
    try:
        GenellemeAblasyonu(object())
    except ImportError as e:
        hata = e
    assert hata is not None


def test_genelleme_sekrek_belleksiz_hata():
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
        GenellemeAblasyonu(model)
    except ValueError as e:
        hata = e
    assert hata is not None


def test_genelleme_nesne_kodlari_ve_bol():
    if not torch_var_mi():
        return
    d = GenellemeAblasyonu(_model(), tohum=0)
    ucluler = _kategori_ucluleri()
    d.sozluk_kur(ucluler)
    K = d.nesne_kodlari(ucluler)
    assert K == 4
    assert d.nesne_indeks == {"C1": 0, "C2": 1, "C3": 2, "C4": 3}
    egitim, heldout = d.bol(ucluler)
    # her nesne eğitimde en az bir kez; held-out boş değil
    egitim_nesneleri = {ucluler[i][2] for i in egitim}
    assert egitim_nesneleri == {"C1", "C2", "C3", "C4"}
    assert len(heldout) == 4
    assert set(egitim).isdisjoint(set(heldout))


def test_genelleme_kos_kontrol_sans_deney_geneller():
    if not torch_var_mi():
        return
    d = GenellemeAblasyonu(_model(), tohum=0)
    s = d.kos(_kategori_ucluleri())
    # kontrol (boş bellek): hem eğitim hem held-out şans düzeyinde
    assert s["kontrol_heldout_dogruluk"] < 0.6
    # deney (bilgi yazılı): okuyucu held-out'a geneller
    assert s["deney_heldout_dogruluk"] >= 0.9
    assert s["deney_egitim_dogruluk"] >= 0.9
    assert s["yazilan_satir"] > 0
    # genelleme etkisi: held-out'ta kontrol → deney belirgin artış
    assert s["deney_heldout_dogruluk"] - s["kontrol_heldout_dogruluk"] > 0.3


def test_genelleme_yetersiz_veri_hata():
    """Nesne başına tek olgu → held-out kalmaz → dürüst hata."""
    if not torch_var_mi():
        return
    d = GenellemeAblasyonu(_model(), tohum=0)
    ucluler = [("S1", "R1", "C1"), ("S2", "R1", "C2"), ("S3", "R1", "C3")]
    hata = None
    try:
        d.kos(ucluler)
    except ValueError as e:
        hata = e
    assert hata is not None


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
