# -*- coding: utf-8 -*-
"""
Korpus üretici testleri — çevrimdışı sentetik Türkçe korpus (torch'suz)
========================================================================
Çalıştırma:
    python tests/test_korpus_uretici.py
    pytest tests/test_korpus_uretici.py -q
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.experience import (sentetik_korpus_uret, korpus_borusu,  # noqa: E402
                            sozlugu_buyut, VARSAYILAN_SOZLUK,
                            yonelme_eki, belirtme_eki)
from hga.experience.korpus_uretici import _KALIPLAR  # noqa: E402
from hga.experience.sozluk_buyutme import _kok_bul  # noqa: E402
from hga.knowledge import KnowledgeStore  # noqa: E402


def _gercek_sayisi(ozne_sayisi, nesne_sayisi):
    """Gürültüsüz üretilen gerçek cümle sayısı (kalıplardan hesaplanır)."""
    return sum(ozne_sayisi * min(nesne_sayisi, len(k["nesneler"]))
               for k in _KALIPLAR)


def _tum_fiiller():
    return {y for k in _KALIPLAR for y in k["yuklemler"]}


def test_determinizm():
    a = sentetik_korpus_uret(ozne_sayisi=10, nesne_sayisi=4, tohum=0)
    b = sentetik_korpus_uret(ozne_sayisi=10, nesne_sayisi=4, tohum=0)
    c = sentetik_korpus_uret(ozne_sayisi=10, nesne_sayisi=4, tohum=1)
    assert a == b                       # aynı tohum → aynı korpus
    assert a != c                       # farklı tohum → farklı korpus


def test_gercek_cumle_sayisi():
    # gürültüsüz: özne_sayısı × her kalıbın seçtiği nesne sayısı
    ozne, nesne = 7, 5
    c = sentetik_korpus_uret(ozne_sayisi=ozne, nesne_sayisi=nesne,
                             gurultu_orani=0.0, tohum=0)
    assert len(c) == _gercek_sayisi(ozne, nesne)
    # her cümle noktayla biter ve bilinen bir fiil yüzey biçimi taşır
    fiiller = _tum_fiiller()
    for cumle in c:
        assert cumle.endswith(".")
        assert cumle.split()[-1][:-1] in fiiller


def test_nesneler_guvenli_kok():
    # üreticinin nesne listeleri durum hâline geçince TEMİZ kök verir
    # (yumuşama geri çevrilir; ğ-ambigua/ünlü düşmesi/sesli-belirsizliği dışlanır).
    for kalip in _KALIPLAR:
        durum = kalip["durum"]
        cekimle = yonelme_eki if durum == "yonelme" else belirtme_eki
        for n in kalip["nesneler"]:
            cekim = cekimle(n)
            kok = _kok_bul(cekim, durum)
            assert kok.lower() == n, f"{n!r} → {cekim!r} → kök {kok!r}"


def test_aktarim_ve_atlama():
    # 6 özne × kalıpların seçtiği nesneler; gerisi kasıtlı gürültü.
    ozne, nesne = 6, 3
    c = sentetik_korpus_uret(ozne_sayisi=ozne, nesne_sayisi=nesne,
                             gurultu_orani=0.1, tohum=0)
    k = KnowledgeStore()
    rapor = korpus_borusu(k, "\n".join(c))
    assert rapor.aktarilan_uclu == _gercek_sayisi(ozne, nesne)  # her gerçek cümle bir üçlü
    assert rapor.cumle_sayisi > rapor.aktarilan_uclu  # gürültü atlandı
    assert rapor.buyutme.yeni_iliski == 0          # ilişki asla uydurulmaz
    # genişletilmiş sözlük: üretilen her ilişki bilgi tabanında tanımlı
    iliskiler = {k["iliski"] for k in _KALIPLAR}
    taban_iliskileri = {r.token for r in k.relations.iliskiler()}
    assert iliskiler <= taban_iliskileri


def test_gurultu_dogrulanabilir():
    # gürültü cümleleri bilinen kalıplara uymaz → sozlugu_buyut atlar.
    c = sentetik_korpus_uret(ozne_sayisi=5, nesne_sayisi=2,
                             gurultu_orani=0.2, tohum=0)
    _, rapor, _ = sozlugu_buyut(VARSAYILAN_SOZLUK, c)
    gurultu = [x for x in c
               if "kuantum" in x or "kamyon denize" in x or x.endswith("kamyon bindi.")]
    assert rapor.atlanan_cumle >= len(gurultu) - 1
    # hiçbir gürültü cümlesi üçlü ÜRETMEMELİ
    assert rapor.eslesen_cumle == rapor.taranan_cumle - rapor.atlanan_cumle


def test_iliski_icat_edilmez_buyuk_olcekte():
    # ölçekte de yeni ilişki asla icat edilmez.
    c = sentetik_korpus_uret(ozne_sayisi=20, nesne_sayisi=5,
                             gurultu_orani=0.05, tohum=3)
    _, rapor, _ = sozlugu_buyut(VARSAYILAN_SOZLUK, c)
    assert rapor.yeni_iliski == 0


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
