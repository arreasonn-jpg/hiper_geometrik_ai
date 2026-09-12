# -*- coding: utf-8 -*-
"""
HGA Experience Engine — Dış Korpus Borusu Demosu (v1.0+)
=========================================================
"veri_toplayici.py çıktısını corpus.py üzerinden REAL_DATA olarak akıtmak +
sözlüğü gerçek korpustan çıkan desenlerle büyütmek" maddesinin gösterimi
(rapor §8, §12).

Akış (ağ/requests/pyarrow GEREKTİRMEZ — yalnız dosya sözleşmesi):
    1. `OtomatikVeriToplayici.metni_kaydet`'in yazdığı `turkce_metin.txt`
       sözleşmesiyle örnek bir korpus dosyası üretilir (gerçek ortamda bu
       dosyayı egitim/veri_toplayici.py üretir).
    2. `veri_toplayici_ciktisindan` dosyayı okur: cümlelere böler, sözlüğü
       büyütür (yeni özne/nesne varlıkları), üçlüleri ayıklar ve REAL_DATA
       olarak KnowledgeStore'a yazar.
    3. Büyüme raporu + bilgi tabanı özeti yazdırılır.

Çalıştırma:
    python experiments/experience_loop/run_korpus_boru.py
"""
import os
import sys
import tempfile

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.knowledge import KnowledgeStore  # noqa: E402
from hga.experience import (VARSAYILAN_SOZLUK, sozlugu_buyut,  # noqa: E402
                            veri_toplayici_ciktisindan)

CIZGI = "=" * 74

# Örnek korpus: veri toplayıcının üreteceği türden yalın Türkçe cümleler.
ORNEK_KORPUS = (
    "Ali ataya bindi.\n"
    "Ali arabaya bindi.\n"
    "Ayşe gökyüzüne baktı.\n"
    "Mehmet kamyona bindi.\n"
    "Zeynep dağa baktı.\n"
    "Fatma yıldıza baktı.\n"
    "Ahmet kamyona bindi.\n"
    "kuantum bilgisayar nedir\n"   # sözlük dışı → atlanır (asla uydurulmaz)
)


def main():
    print(CIZGI)
    print("Dış Korpus Borusu: veri toplayıcı çıktısı → sözlük büyütme → REAL_DATA")
    print(CIZGI)

    # 1. veri_toplayici.py'nin dosya sözleşmesi (metni_kaydet → turkce_metin.txt)
    with tempfile.TemporaryDirectory() as d:
        yol = os.path.join(d, "turkce_metin.txt")
        with open(yol, "w", encoding="utf-8") as f:
            f.write(ORNEK_KORPUS)
        print(f"1) Korpus dosyası (veri toplayıcı sözleşmesi): {yol}")
        print(f"   {len(ORNEK_KORPUS.splitlines())} cümle\n")

        # 2. Sözlük büyüme (bilgi tabanına dokunmadan, salt sözlük)
        onceki_ozne = set(VARSAYILAN_SOZLUK["binmek"]["ozneler"])
        buyumus, buyutme, ucluler = sozlugu_buyut(
            VARSAYILAN_SOZLUK, ORNEK_KORPUS.splitlines())
        print("2) Sözlük büyüme raporu:")
        print(f"   taranan={buyutme.taranan_cumle}  eşleşen={buyutme.eslesen_cumle}  "
              f"atlanan={buyutme.atlanan_cumle}")
        print(f"   yeni özne={buyutme.yeni_ozne}  yeni nesne={buyutme.yeni_nesne}  "
              f"yeni ilişki={buyutme.yeni_iliski}  (ilişki asla uydurulmaz)")
        yeni_ozneler = []
        for il in ("binmek", "bakmak"):
            for n in sorted(set(buyumus[il]["ozneler"]) -
                            set(VARSAYILAN_SOZLUK[il]["ozneler"])):
                yeni_ozneler.append(buyumus[il]["ozneler"][n]["token"])
        print(f"   örnek yeni özneler: {yeni_ozneler}")
        yeni_nesne_tokenleri = []
        for il in ("binmek", "bakmak"):
            for n in sorted(set(buyumus[il]["nesneler"]) -
                            set(VARSAYILAN_SOZLUK[il]["nesneler"])):
                yeni_nesne_tokenleri.append(buyumus[il]["nesneler"][n]["token"])
        print(f"   örnek yeni nesneler: {yeni_nesne_tokenleri}")
        print(f"   (orijinal VARSAYILAN_SOZLUK değişmedi: "
              f"{set(VARSAYILAN_SOZLUK['binmek']['ozneler']) == onceki_ozne})\n")

        # 3. Uçtan uca boru: dosya → sözlük büyütme → REAL_DATA
        k = KnowledgeStore()
        rapor = veri_toplayici_ciktisindan(k, d)
        print("3) Uçtan uca boru raporu:")
        print(f"   cümle={rapor.cumle_sayisi}  aktarılan üçlü={rapor.aktarilan_uclu}")
        print(f"   bilgi tabanı: {k.ozet()}")

        # 4. Yeni varlıkların kaynağı REAL_DATA
        print("\n4) Yeni varlıklar (REAL_DATA kaynaklı):")
        for ad in ["Mehmet", "Zeynep", "Fatma", "Ahmet", "Kamyon", "Dağ"]:
            eid = k.entities.ad_bul(ad)
            e = k.entities.getir(eid)
            print(f"   {e.token:<10} tip={e.entity_type:<8} kaynak={e.source.value}")

    print("\n" + CIZGI)
    print("SONUÇ: veri toplayıcının yazdığı dosya, sözlüğü büyüterek ve hiçbir")
    print("şey uydurmadan REAL_DATA olarak bilgi tabanına aktı; yeni ilişki icat")
    print("edilmedi, kuşkulu cümleler atlandı.")
    print(CIZGI)


if __name__ == "__main__":
    main()
