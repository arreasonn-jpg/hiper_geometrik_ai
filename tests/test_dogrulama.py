# -*- coding: utf-8 -*-
"""
Doğrulama hattı testleri — deterministik kanıtla VERIFIED / INVALID
====================================================================
Çalıştırma:
    python tests/test_dogrulama.py
    pytest tests/test_dogrulama.py -q
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.experience import (  # noqa: E402
    AritmetikOrtam,
    DogrulamaHatti,
    ExperienceEvaluator,
    ExperienceGenerator,
    aritmetik_etki_alani,
)
from hga.knowledge import DeneyimDurumu, KaynakTuru  # noqa: E402


def _adaylar():
    k = aritmetik_etki_alani()
    ev = ExperienceEvaluator()
    adaylar = ExperienceGenerator(tip_filtresi=False).uret(k)
    for a in adaylar:
        ev.degerlendir(a, k)
    return k, adaylar


def test_dogrulama_hatti_sifir_false_accept():
    """30 eşitlik: 6 doğrulanır, 24 çürütülür → false accept 24 → 0."""
    k, adaylar = _adaylar()
    assert len(adaylar) == 30
    assert all(a.state == DeneyimDurumu.VALID for a in adaylar)

    h = DogrulamaHatti(AritmetikOrtam().aday_dogrula)
    rapor = h.isle(k, adaylar)

    assert rapor.islenen == 30
    assert rapor.dogrulanan == 6
    assert rapor.reddedilen == 24
    assert rapor.belirsiz == 0
    assert rapor.yanlis_kabul_oncesi == 24
    assert rapor.yanlis_kabul_sonrasi == 0
    # 6 doğru kalıcı bilgiye yazıldı (her biri sürüm artırır)
    assert rapor.bilgi_buyumesi == 6


def test_dogrulama_kaynak_degisir():
    """Doğrulanan adayın kaynağı EXTERNAL_VERIFIED olur (bağımsız kanıt)."""
    k, adaylar = _adaylar()
    h = DogrulamaHatti(AritmetikOrtam().aday_dogrula)
    h.isle(k, adaylar)
    dogrulanan = [a for a in adaylar if a.state == DeneyimDurumu.VERIFIED]
    assert len(dogrulanan) == 6
    for a in dogrulanan:
        assert a.source == KaynakTuru.EXTERNAL_VERIFIED
        assert a.verified_by == "deterministik-dogrulayici"
        assert a.source != KaynakTuru.MODEL_GENERATED


def test_dogrulama_belirsiz_uncertain_olur():
    """Doğrulayıcı None dönerse kanıt eksikliği açıkça UNCERTAIN olur."""
    k = aritmetik_etki_alani()
    ev = ExperienceEvaluator()
    adaylar = ExperienceGenerator(tip_filtresi=False).uret(k)
    for a in adaylar:
        ev.degerlendir(a, k)
    h = DogrulamaHatti(lambda store, a: None)   # her şey belirsiz
    rapor = h.isle(k, adaylar)
    assert rapor.dogrulanan == 0
    assert rapor.reddedilen == 0
    assert rapor.belirsiz == 30
    assert all(a.state == DeneyimDurumu.UNCERTAIN for a in adaylar)
    assert all(a.source == KaynakTuru.MODEL_GENERATED for a in adaylar)
    # belirsiz aday kalıcı bilgiye yazılmadı
    assert rapor.bilgi_buyumesi == 0


def test_dogrulama_curutulen_celiski_gunlugune():
    k, adaylar = _adaylar()
    h = DogrulamaHatti(AritmetikOrtam().aday_dogrula)
    h.isle(k, adaylar)
    # çürütülenler çelişki günlüğüne "deterministik-curutme" olarak düştü
    assert any(g["neden"] == "deterministik-curutme" for g in k.celiski_gunlugu)
    assert len(k.celiski_gunlugu) == 24


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
