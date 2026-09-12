# -*- coding: utf-8 -*-
"""
HGA Experience Engine — Çevrimdışı Korpus Ölçeği Demosu (v1.0+)
===============================================================
"Dış korpusu milyon-kelime ölçeğine taşımak" maddesinin AĞSIZ provası (rapor §8).

Canlı korpus çekimi (Wikipedia/HF) `egitim/veri_toplayici.py`'nin işidir ve ağ
gerektirir. Bu demo, aynı boruyu (`korpus_boru.py`) belirleyici SENTETİK Türkçe
cümlelerle (sözlük büyütme + REAL_DATA aktarımı) tekrarlanabilir biçimde sınar:

    sentetik_korpus_uret → korpus_borusu → sözlük büyümesi + bilgi tabanı + hız

Çalıştırma:
    python experiments/experience_loop/run_korpus_olcegi.py [--ozne N] [--nesne N] [--gurultu X]
"""
import os
import sys
import time

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.knowledge import KnowledgeStore  # noqa: E402
from hga.experience import (sentetik_korpus_uret, korpus_borusu,  # noqa: E402
                            sozlugu_buyut, VARSAYILAN_SOZLUK)
from hga.experience.korpus_uretici import TURKCE_ADLAR  # noqa: E402

CIZGI = "=" * 74


def _arg_int(ad: str, varsayilan: int) -> int:
    if ad in sys.argv:
        return int(sys.argv[sys.argv.index(ad) + 1])
    return varsayilan


def _arg_float(ad: str, varsayilan: float) -> float:
    if ad in sys.argv:
        return float(sys.argv[sys.argv.index(ad) + 1])
    return varsayilan


def _raporla(k, cumleler, baslik):
    t0 = time.perf_counter()
    r = korpus_borusu(k, "\n".join(cumleler))
    dt = time.perf_counter() - t0
    print(f"\n{baslik}:")
    print(f"   cümle={r.cumle_sayisi}  aktarılan üçlü={r.aktarilan_uclu}  "
          f"atlanan={r.cumle_sayisi - r.aktarilan_uclu}")
    b = r.buyutme
    print(f"   sözlük büyüme: yeni özne={b.yeni_ozne}  yeni nesne={b.yeni_nesne}  "
          f"yeni ilişki={b.yeni_iliski}")
    print(f"   bilgi tabanı: {k.ozet()}")
    print(f"   süre={dt:.3f}s  (≈{r.cumle_sayisi / max(dt, 1e-9):,.0f} cümle/sn)")


def main():
    ozne = _arg_int("--ozne", 24)
    nesne = _arg_int("--nesne", 8)
    gurultu = _arg_float("--gurultu", 0.05)

    print(CIZGI)
    print("Çevrimdışı Korpus Ölçeği: sentetik Türkçe cümleler → boru → REAL_DATA")
    print(CIZGI)

    # 1. Belirleyici korpus üretimi
    cumleler = sentetik_korpus_uret(ozne_sayisi=ozne, nesne_sayisi=nesne,
                                    gurultu_orani=gurultu, tohum=0)
    print(f"\n1) Üretilen korpus: {len(cumleler)} cümle "
          f"({ozne} özne × {nesne} nesne × 2 ilişki + %{gurultu * 100:.0f} gürültü)")
    print(f"   örnek: {cumleler[0]}")
    gurultu_ornek = next((c for c in cumleler
                          if "kuantum" in c or c.endswith("kamyon bindi.")
                          or "kamyon denize" in c), None)
    if gurultu_ornek:
        print(f"   kasıtlı gürültü örneği: {gurultu_ornek!r}")

    # 2. Salt sözlük büyümesi (bilgi tabanına dokunmadan)
    _, buyutme, _ = sozlugu_buyut(VARSAYILAN_SOZLUK, cumleler)
    print(f"\n2) Sözlük büyüme: taranan={buyutme.taranan_cumle}  "
          f"eşleşen={buyutme.eslesen_cumle}  atlanan={buyutme.atlanan_cumle}")
    print(f"   yeni özne={buyutme.yeni_ozne}  yeni nesne={buyutme.yeni_nesne}  "
          f"yeni ilişki={buyutme.yeni_iliski}  (ilişki ASLA uydurulmaz)")

    # 3. Uçtan uca boru
    k = KnowledgeStore()
    _raporla(k, cumleler, "3) Uçtan uca boru (REAL_DATA aktarımı)")

    # 4. Ölçek: aynı boruyu tüm ad listesi × 12 nesne ile dene
    buyuk = sentetik_korpus_uret(ozne_sayisi=len(TURKCE_ADLAR),
                                 nesne_sayisi=12, gurultu_orani=gurultu, tohum=1)
    k2 = KnowledgeStore()
    _raporla(k2, buyuk, "4) Ölçek provası (tüm ad listesi × 12 nesne)")

    print("\n" + CIZGI)
    print("SONUÇ: Boru, belirleyici sentetik korpusta yüzlerce/bini aşkın cümleyi")
    print("saniyenin altında REAL_DATA olarak akıttı; yeni ilişki icat etmedi,")
    print("kuşkulu cümleleri atladı. Bu bir SENTETİK stres testidir — gerçek")
    print("Wikipedia/OSCAR ölçeği ağ gerektirir ve veri_toplayici.py'nin işidir.")
    print(CIZGI)


if __name__ == "__main__":
    main()
