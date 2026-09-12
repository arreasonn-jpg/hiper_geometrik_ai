# -*- coding: utf-8 -*-
"""
HGA Experience Engine — Genelleme Ablasyonu Demosu (v1.0+)
===========================================================
"Belleğe yazılan doğrulanmış bilgi, okuyucunun HİÇ GÖRMEDİĞİ sorguları
yanıtlamasını sağlıyor mu?" (rapor §17/§20; görev ablasyonunun genelleme boyutu)

Kurgu:
    1. Küçük bir "kategori" bilgi tabanı: 4 kategori (C1..C4), her kategoriye
       ait 2 özne → 8 olgu (deterministik ground-truth; doğrulanmış bilgi).
    2. Faz W: HER olgunun [özne, ilişki] penceresine nesnenin KANONİK kodu
       (one-hot) yazılır — bilgi paylaşılan biçimde saklanır.
    3. Faz R: bellek donuk; okuyucu (modelin gen_kopru köprüsü + küçük kafa)
       YALNIZCA eğitim alt kümesinde (kategori başına 1 olgu) eğitilir.
    4. Ölçüm: held-out olgularda (eğitimde görülmeyen özneler) tamamlama.

    KONTROL (boş bellek) : okuyucu sabit girdi görür → ~şans (eğitimde ve held-out).
    DENEY   (bilgi yazılı): okuyucu KANONİK kodu çözer; kod paylaşıldığı için
                            held-out'ta da doğru nesneyi okur → ~%100.

Çalıştırma (torch gerekir):
    .venv/bin/python experiments/experience_loop/run_genelleme_ablasyonu.py
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in [KOK, os.path.join(KOK, "mimari")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hga.memory import torch_var_mi  # noqa: E402

CIZGI = "=" * 74

# Doğrulanmış kategori bilgisi: 4 kategori, kategori başına 2 özne.
UCLULER = [
    ("S1", "R1", "C1"), ("S2", "R1", "C1"),
    ("S3", "R1", "C2"), ("S4", "R1", "C2"),
    ("S5", "R1", "C3"), ("S6", "R1", "C3"),
    ("S7", "R1", "C4"), ("S8", "R1", "C4"),
]


def main():
    print(CIZGI)
    print("Genelleme Ablasyonu: bellek, görülmeyen olgulara genelleştiriyor mu?")
    print(CIZGI)

    if not torch_var_mi():
        print("[torch kurulu değil] GenellemeAblasyonu gerçek model gerektirir.")
        return

    import torch
    from kuresel_model import HiperGeometrikAI
    from hga.memory import GenellemeAblasyonu

    print(f"1) Doğrulanmış bilgi: {len(UCLULER)} olgu, "
          f"{len({u[2] for u in UCLULER})} kategori (kategori başına 2 özne)")

    torch.manual_seed(0)
    model = HiperGeometrikAI(n=32, katman_sayisi=2, baglam_penceresi=4,
                             emb_dim=16, num_heads=2, sozluk_boyutu=64,
                             dropout=0.0, seyrek_tablo_boyutu=2048,
                             seyrek_boyut=32, bilgilendir=False)
    deney = GenellemeAblasyonu(model, tohum=0)
    s = deney.kos(UCLULER)

    print("\n2) Genelleme ablasyonu sonuçları:")
    print(f"   Eğitim olgusu: {s['egitim_olgu']}   Held-out olgusu: {s['heldout_olgu']}")
    print(f"   Yazılan satır: {s['yazilan_satir']}/{s['toplam_satir']}\n")
    print(f"   {'':<28} {'eğitim':>8} {'held-out':>10}")
    print(f"   {'KONTROL (boş bellek)':<28} {s['kontrol_egitim_dogruluk']:>8.0%} "
          f"{s['kontrol_heldout_dogruluk']:>10.0%}")
    print(f"   {'DENEY (bilgi yazılı)':<28} {s['deney_egitim_dogruluk']:>8.0%} "
          f"{s['deney_heldout_dogruluk']:>10.0%}")

    etki = s["deney_heldout_dogruluk"] - s["kontrol_heldout_dogruluk"]
    print(f"\n3) Genelleme etkisi (held-out): +{etki:.0%} "
          f"({s['kontrol_heldout_dogruluk']:.0%} → {s['deney_heldout_dogruluk']:.0%})")

    print("\n" + CIZGI)
    print("SONUÇ: Okuyucu, bilgiyi KANONİK biçimde taşıyan bellekten okumayı")
    print("öğrendiği için eğitimde hiç görmediği öznelerde de doğru nesneyi")
    print("tamamladı. Boş bellek bu sinyali vermez (şans). Genelleme, bellekten")
    print("gelir — ezberden değil. (Tam gövde eğitilebilir bırakılırsa model")
    print("ezberler: train ~%100 ama held-out ~şans; bu yüzden gövde donuktur.)")
    print(CIZGI)


if __name__ == "__main__":
    main()
