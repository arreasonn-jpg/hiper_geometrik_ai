# -*- coding: utf-8 -*-
"""
Ablasyon deneyi testleri — belleğe yazılan bilgi öğrenmeyi etkiliyor mu?
=========================================================================
Çalıştırma (torch gerekir):
    .venv/bin/python tests/test_ablation.py
    .venv/bin/python -m pytest tests/test_ablation.py -q
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [KOK, os.path.join(KOK, "mimari")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hga.memory import AblasyonDeneyi, torch_var_mi  # noqa: E402


def _model():
    from kuresel_model import HiperGeometrikAI
    return HiperGeometrikAI(
        n=64, katman_sayisi=2, baglam_penceresi=8, emb_dim=32,
        num_heads=4, sozluk_boyutu=256, dropout=0.0,
        seyrek_tablo_boyutu=2048, seyrek_boyut=8, bilgilendir=False)


def _dengeli_ornekler():
    """6 doğru + 6 yanlış üçlü (dengeli → şans %50)."""
    dogru = [("E_001", "R_001", "E_004"),
             ("E_002", "R_001", "E_005"),
             ("E_003", "R_001", "E_006"),
             ("E_004", "R_001", "E_001"),
             ("E_005", "R_001", "E_002"),
             ("E_006", "R_001", "E_003")]
    yanlis = [("E_001", "R_001", "E_005"),
              ("E_002", "R_001", "E_004"),
              ("E_003", "R_001", "E_002"),
              ("E_004", "R_001", "E_006"),
              ("E_005", "R_001", "E_003"),
              ("E_006", "R_001", "E_001")]
    return [(u, 1.0) for u in dogru] + [(u, 0.0) for u in yanlis]


def test_ablation_torch_yoksa_guvenli_hata():
    if torch_var_mi():
        print("  (torch mevcut — gerçek testler aşağıda)")
        return
    hata = None
    try:
        AblasyonDeneyi(object())
    except ImportError as e:
        hata = e
    assert hata is not None


def test_ablation_bos_bellek_sans():
    """Boş bellek: salt okuma kafası sinyal bulamaz → dengeli veride ~%50."""
    if not torch_var_mi():
        return
    d = AblasyonDeneyi(_model(), tohum=0)
    ornekler = _dengeli_ornekler()
    d._sifirla()
    _, dogruluk = d.okuma_egit(ornekler)
    assert abs(dogruluk - 0.5) < 0.12, f"beklenen ~%50, gelen: {dogruluk}"


def test_ablation_bilgi_yazili_yuksek():
    """Bilgi yazıldıktan sonra salt okuma kafası mükemmele yakın ayrıştırır."""
    if not torch_var_mi():
        return
    d = AblasyonDeneyi(_model(), tohum=0)
    ornekler = _dengeli_ornekler()
    d._sifirla()
    d.bilgi_yaz(ornekler)
    dolu, _ = d.kopru.doluluk()
    assert dolu > 0                              # satırlar doldu
    _, dogruluk = d.okuma_egit(ornekler)
    assert dogruluk >= 0.99, f"beklenen ~%100, gelen: {dogruluk}"


def test_ablation_kos_farkli():
    """kos(): kontrol ~%50, deney ~%100 — bilgi etkisi ölçülür."""
    if not torch_var_mi():
        return
    d = AblasyonDeneyi(_model(), tohum=0)
    sonuc = d.kos(_dengeli_ornekler())
    assert abs(sonuc["kontrol_dogruluk"] - 0.5) < 0.12
    assert sonuc["deney_dogruluk"] >= 0.99
    assert sonuc["yazilan_satir"] > 0
    # etki büyüklüğü: deney - kontrol ≈ +%50
    assert sonuc["deney_dogruluk"] - sonuc["kontrol_dogruluk"] > 0.3


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
