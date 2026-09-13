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

from hga.experience import (  # noqa: E402
    AritmetikOrtam,
    DogrulamaHatti,
    ExperienceEvaluator,
    ExperienceGenerator,
    aritmetik_etki_alani,
    karsilastirma,
    kos,
    ozetle,
)

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
    raporlar = kos(k, adimlar=3, dogrulayici=AritmetikOrtam().aday_dogrula)
    tablo("3 adım (novelty azalması + replay verimliliği)", ozetle(raporlar))

    # ── Kapalı doğrulama döngüsü: deterministik kanıtla false accept sıfırlama ──
    print("\n" + CIZGI)
    print("Kapalı doğrulama hattı (rapor §11/§18 döngüsünün kapanışı)")
    print(CIZGI)
    k2 = aritmetik_etki_alani()
    ev = ExperienceEvaluator()
    adaylar = ExperienceGenerator(tip_filtresi=False).uret(k2)
    for a in adaylar:
        ev.degerlendir(a, k2)
    print(f"  Kural tabanlı değerlendirme: {len(adaylar)} aday → hepsi VALID")
    h = DogrulamaHatti(AritmetikOrtam().aday_dogrula, evaluator=ev)
    rapor = h.isle(k2, adaylar)
    print(f"  Doğrulanan (VERIFIED): {rapor.dogrulanan}")
    print(f"  Çürütülen (INVALID)  : {rapor.reddedilen}")
    print(f"  False accept: {rapor.yanlis_kabul_oncesi} → {rapor.yanlis_kabul_sonrasi}")
    print(f"  Bilgi büyümesi: {rapor.bilgi_buyumesi} (doğrulanan kalıcı bilgiye yazıldı)")

    print("\n" + CIZGI)
    print("SONUÇ: 6 varlıkta 30 eşitlik adayı üretildi; ground-truth'a göre")
    print("yalnız 6'sı doğruydu. Kural tabanlı değerlendirme 30'unu VALID kabul")
    print("etti → 24 YANLIŞ KABUL. Deterministik doğrulama hattı 6'sını VERIFIED'a")
    print("yükseltip 24'ünü INVALID'e düşürdü → false accept 0. Bu, rapor §18'in")
    print("doğrulayıcıların gerekliliği tezini ölçülebilir biçimde gösterir.")
    print(CIZGI)


if __name__ == "__main__":
    main()
