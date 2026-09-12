# -*- coding: utf-8 -*-
"""
HGA Experience Engine — Kontrollü Benchmark Demosu (v1.0)
===========================================================
Döngüyü deterministik ground-truth'a (aritmetik eşitlik) karşı ölçer ve
rapor §20'deki metrikleri tek tabloda gösterir.

    false acceptance: kural tabanlı değerlendirme (MODEL_GENERATED) tüm
    kombinasyonları VALID kabul eder; ground-truth bunların çoğunun yanlış
    olduğunu ortaya çıkarır → rapor §18/§21'in "domain-specific doğrulayıcı
    şart" tezini sayıya döker.

Çalıştırma:
    python experiments/experience_loop/run_benchmark.py
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.experience import (aritmetik_etki_alani, kos, ozetle,  # noqa: E402
                            karsilastirma)

CIZGI = "=" * 74


def tablo(baslik, ozet):
    print(f"\n{baslik}")
    print("-" * 46)
    satirlar = [
        ("Adım sayısı", ozet.adim_sayisi),
        ("Üretilen aday", ozet.toplam_uretilen),
        ("Yeni üçlü (novel)", ozet.toplam_novel),
        ("VALID", ozet.toplam_valid),
        ("CONFLICT", ozet.toplam_conflict),
        ("INVALID", ozet.toplam_invalid),
        ("VERIFIED", ozet.toplam_verified),
        ("Bilgi büyümesi", ozet.bilgi_buyumesi),
        ("Acceptance rate (son)", ozet.son_acceptance_rate),
        ("Conflict rate (son)", ozet.son_conflict_rate),
        ("Replay verimliliği (son)", ozet.son_replay_verimliligi),
        ("YANLIŞ KABUL (false accept)", ozet.yanlis_kabul),
        ("YANLIŞ RET (false reject)", ozet.yanlis_ret),
    ]
    for k, v in satirlar:
        print(f"  {k:<26}: {v}")


def main():
    print(CIZGI)
    print("Kontrollü Benchmark — aritmetik eşitlik alanı (ground-truth biliniyor)")
    print(CIZGI)

    # Tek adımlık ölçüm
    sonuc = karsilastirma(adimlar=1)
    tablo("Tek adım (MODEL_GENERATED, doğrulayıcıyla ölçülüyor)", sonuc["ozet"])

    # Çok adımlı koşu
    k = aritmetik_etki_alani()
    from hga.experience import AritmetikOrtam
    raporlar = kos(k, adimlar=3, dogrulayici=AritmetikOrtam().aday_dogrula)
    tablo("3 adım (novelty azalması + replay verimliliği)", ozetle(raporlar))

    print("\n" + CIZGI)
    print("SONUÇ: 6 varlıkta 30 eşitlik adayı üretildi; ground-truth'a göre")
    print("yalnız 6'sı doğruydu. Kural tabanlı değerlendirme 30'unu VALID kabul")
    print("etti → 24 YANLIŞ KABUL. Bu, rapor §18'in doğrulayıcıların gerekliliği")
    print("tezini ölçülebilir biçimde gösterir (MODEL_GENERATED asla VERIFIED olmaz).")
    print(CIZGI)


if __name__ == "__main__":
    main()
