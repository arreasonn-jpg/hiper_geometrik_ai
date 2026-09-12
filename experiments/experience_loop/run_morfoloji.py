# -*- coding: utf-8 -*-
"""
HGA Experience Engine — Türkçe Morfoloji Demosu (v1.0+)
========================================================
`hga/experience/turkce.py`'nin tamamlanan morfoloji parçalarını sergiler:

    1. Ünlü düşmesi        — burun→burna, şehir→şehri (sesliyle başlayan ekte)
    2. İyelik ekleri        — 6 kişi (benim/senin/onun/bizim/sizin/onların)
    3. İyelik + durum       — evi→evine (3. tekil sonrası 'n' ara harfi)
    4. Fiil çekimleri       — geçmiş/şimdiki/gelecek/geniş zaman 3. tekil

Çalıştırma (torch gerekmez):
    python experiments/experience_loop/run_morfoloji.py
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.experience import (yonelme_eki, belirtme_eki, bulunma_eki,  # noqa: E402
                            ayrilma_eki, iyelik_eki, iyelik_li_durum,
                            gecmis_zaman_3tekil, simdiki_zaman_3tekil,
                            gelecek_zaman_3tekil, genis_zaman_3tekil)

CIZGI = "=" * 74


def main():
    print(CIZGI)
    print("Türkçe Morfoloji — tamamlanan parçalar (ünlü düşmesi + iyelik + fiil)")
    print(CIZGI)

    print("\n1) Ünlü düşmesi (yalnız sesliyle başlayan eklerde):")
    print(f"   {'sözcük':<10} {'yönelme':<12} {'belirtme':<12} {'bulunma':<12}")
    for w in ["burun", "şehir", "isim", "vakit", "kayıp", "fesat"]:
        print(f"   {w:<10} {yonelme_eki(w):<12} {belirtme_eki(w):<12} "
              f"{bulunma_eki(w):<12}")
    print("   (bulunma/ayrılma ünsüzle başlar → düşme YOK: burunda, vakitte)")

    print("\n2) İyelik ekleri (6 kişi):")
    kisiler = [("1t", "benim"), ("2t", "senin"), ("3t", "onun"),
               ("1c", "bizim"), ("2c", "sizin"), ("3c", "onların")]
    print(f"   {'sözcük':<10} " + " ".join(f"{k:<4}" for k, _ in kisiler))
    for w in ["ev", "araba", "okul", "göz", "kitap", "burun"]:
        print(f"   {w:<10} " + " ".join(f"{iyelik_eki(w, k):<4}"
                                        for k, _ in kisiler))

    print("\n3) İyelik + durum zinciri (3. tekil sonrası 'n' ara harfi):")
    print(f"   {'kök':<10} {'iyelik':<10} {'yönelme':<14} {'belirtme':<14} "
          f"{'bulunma':<14} {'ayrılma':<14}")
    for w in ["ev", "araba"]:
        print(f"   {w:<10} {iyelik_eki(w, '3t'):<10} "
              f"{iyelik_li_durum(w, '3t', 'yonelme'):<14} "
              f"{iyelik_li_durum(w, '3t', 'belirtme'):<14} "
              f"{iyelik_li_durum(w, '3t', 'bulunma'):<14} "
              f"{iyelik_li_durum(w, '3t', 'ayrilma'):<14}")

    print("\n4) Fiil çekimleri (geçmiş/şimdiki/gelecek/geniş zaman 3. tekil):")
    kokler = ["bin", "bak", "git", "sev", "gel", "gör", "dur", "oku",
              "bekle", "ye", "et", "otur"]
    print(f"   {'kök':<8} {'geçmiş':<10} {'şimdiki':<10} {'gelecek':<10} {'geniş':<10}")
    for k in kokler:
        print(f"   {k:<8} {gecmis_zaman_3tekil(k):<10} "
              f"{simdiki_zaman_3tekil(k):<10} {gelecek_zaman_3tekil(k):<10} "
              f"{genis_zaman_3tekil(k):<10}")

    print("\n" + CIZGI)
    print("NOT: Ünsüz yumuşaması, ünlü düşmesi ve aorist düzensizlikleri")
    print("küratörlü listelerle sınırlıdır (kural-genelleme değil). İsim tamlaması")
    print("ve kişi ekli çekimler (biniyorum/biniyorsun/…) bilinçli olarak kapsam")
    print("dışıdır.")
    print(CIZGI)


if __name__ == "__main__":
    main()
