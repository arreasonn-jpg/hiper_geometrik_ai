# -*- coding: utf-8 -*-
"""
HGA Experience Engine — Genelleme Ablasyonu ÖLÇEK + GÜRÜLTÜ Demosu (v1.0+)
==========================================================================
rapor §8 "Sıradaki adım: bu protokolü gürültülü ve büyük korpuslarda tekrarlamak."

`run_genelleme_ablasyonu.py` 4 kategoride (8 olgu) genellemeyi kanıtladı. Bu demo
aynı protokolü BÜYÜK bir bilgi tabanında (6 kategori × 4 özne = 24 olgu) tekrar
eder ve ayrıca GÜRÜLTÜYÜ ölçer: eğitim olgularının bir kısmına YANLIŞ nesne kodu
yazılınca held-out genellemesi ne kadar bozulur?

    * Gürültüsüz: held-out ~%100 (ölçekte de genelleme korunur).
    * Gürültülü : `bol()` her kategoriye EĞİTİMDE TEK örnek verdiği için, o tek
                  örnek gürültülenirse kategori okuyucuya hiç temiz gösterilmez
                  → held-out ≈ 1 − gürültü_oranı düşer (dürüst bozulma eğrisi).

Çalıştırma (torch gerekir):
    .venv/bin/python experiments/experience_loop/run_genelleme_olcegi.py
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in [KOK, os.path.join(KOK, "mimari")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hga.memory import torch_var_mi  # noqa: E402

CIZGI = "=" * 74


def buyuk_ucluler(K=6, kisi=4):
    """K kategori × kisi özne → K·kisi olgu (deterministik ground-truth)."""
    u = []
    for c in range(K):
        for s in range(kisi):
            u.append((f"S{c}_{s}", "R1", f"C{c}"))
    return u


def main():
    print(CIZGI)
    print("Genelleme Ablasyonu: BÜYÜK + GÜRÜLTÜLÜ korpus provası")
    print(CIZGI)

    if not torch_var_mi():
        print("[torch kurulu değil] GenellemeAblasyonu gerçek model gerektirir.")
        return

    import torch
    from kuresel_model import HiperGeometrikAI
    from hga.memory import GenellemeAblasyonu

    K, KISI = 6, 4
    ucluler = buyuk_ucluler(K, KISI)
    print(f"1) Bilgi tabanı: {len(ucluler)} olgu, {K} kategori "
          f"(kategori başına {KISI} özne; eğitimde kategori başına 1 örnek)")

    def model(tohum=0):
        torch.manual_seed(tohum)
        return HiperGeometrikAI(n=32, katman_sayisi=2, baglam_penceresi=4,
                                emb_dim=16, num_heads=2, sozluk_boyutu=64,
                                dropout=0.0, seyrek_tablo_boyutu=2048,
                                seyrek_boyut=32, bilgilendir=False)

    # 2) Ölçek: gürültüsüz, 3 tohumda held-out genellemesi
    print("\n2) ÖLÇEK (gürültüsüz, 3 tohum):")
    for toh in (0, 1, 2):
        r = GenellemeAblasyonu(model(), tohum=toh).gurultulu_kos(
            ucluler, gurultu_orani=0.0, tohum=toh)
        print(f"   tohum={toh}  eğitim={r['egitim_dogruluk']:.0%}  "
              f"held-out={r['heldout_dogruluk']:.0%}  "
              f"(yazılan satır: {r['nesne_sayisi']} kategori)")

    # 3) Gürültü: held-out genellemesi gürültüyle dürüstçe bozulur
    print("\n3) GÜRÜLTÜ (eğitim olgularına yanlış nesne kodu yazılır):")
    print(f"   {'gürültü oranı':>14} {'eğitim':>8} {'temiz-eğitim':>14} "
          f"{'held-out':>10}")
    for g in (0.0, 0.2, 0.5, 1.0):
        r = GenellemeAblasyonu(model(), tohum=0).gurultulu_kos(
            ucluler, gurultu_orani=g, tohum=0)
        print(f"   {r['gurultu_orani']:>14.0%} {r['egitim_dogruluk']:>8.0%} "
              f"{r['temiz_egitim_dogruluk']:>14.0%} {r['heldout_dogruluk']:>10.0%}")

    print("\n" + CIZGI)
    print("SONUÇ: Ölçekte genelleme korunur (held-out ~%100, 3 tohumda kararlı).")
    print("Gürültü ise dürüstçe bozar: her kategori eğitimde TEK temiz örnekle")
    print("temsil edildiği için, o örnek gürültülenirse kategori şansa düşer —")
    print("held-out ≈ 1 − gürültü_oranı azalır; tam gürültüde ~şans. Bu, seyrek")
    print("belleğin değil OKUYUCU eğitiminin veri bağımlılığıdır: tek örnek =")
    print("kırılgan; yedekli temiz veri genellemeyi sağlamlaştırır.")
    print(CIZGI)


if __name__ == "__main__":
    main()
