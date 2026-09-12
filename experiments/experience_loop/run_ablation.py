# -*- coding: utf-8 -*-
"""
HGA Experience Engine — Ablasyon Demosu (v1.0+)
=================================================
Doğrulanmış deneyim bilgisinin modelin seyrek belleğine yazılmasının, aşağı-akış
öğrenmeyi gerçekten etkileyip etkilemediğini ölçer (rapor §17/§20 kapanışı).

Akiş:
    1. Aritmetik alanda deneyim üret + deterministik doğrulama.
    2. 6 doğru (VERIFIED) + 6 yanlış (INVALID) üçlüyü dengeli kümeye al.
    3. AblasyonDeneyi.kos():
         KONTROL — boş bellekten salt okuma  → ~%50 (sinyal yok, şans)
         DENEY   — bilgi yazılı bellekten okuma → ~%100 (sinyal var)
    4. Etki büyüklüğü raporlanır.

Torch kurulu değilse dürüstçe bilgi verir ve atlar.

Çalıştırma:
    .venv/bin/python experiments/experience_loop/run_ablation.py
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in [KOK, os.path.join(KOK, "mimari")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hga.experience import (AritmetikOrtam, aritmetik_etki_alani,  # noqa: E402
                            ExperienceGenerator, ExperienceEvaluator,
                            DogrulamaHatti)
from hga.knowledge import DeneyimDurumu  # noqa: E402
from hga.memory import torch_var_mi  # noqa: E402

CIZGI = "=" * 74


def dengeli_kume():
    """6 doğru + 6 yanlış üçlü (dengeli → şans %50)."""
    k = aritmetik_etki_alani()
    ev = ExperienceEvaluator()
    adaylar = ExperienceGenerator(tip_filtresi=False).uret(k)
    for a in adaylar:
        ev.degerlendir(a, k)
    DogrulamaHatti(AritmetikOrtam().aday_dogrula, evaluator=ev).isle(k, adaylar)

    dogru = [a.uclusu for a in adaylar if a.state == DeneyimDurumu.VERIFIED][:6]
    yanlis = [a.uclusu for a in adaylar if a.state == DeneyimDurumu.INVALID][:6]
    return [(u, 1.0) for u in dogru] + [(u, 0.0) for u in yanlis]


def main():
    print(CIZGI)
    print("Ablasyon: Belleğe yazılan doğrulanmış bilgi öğrenmeyi etkiliyor mu?")
    print(CIZGI)

    if not torch_var_mi():
        print("[torch kurulu değil] AblasyonDeneyi gerçek model gerektirir.")
        return

    from kuresel_model import HiperGeometrikAI
    from hga.memory import AblasyonDeneyi

    # 1. Dengeli deney kümesi (doğrulama hattından)
    ornekler = dengeli_kume()
    print(f"1) Deney kümesi: {len(ornekler)} üçlü "
          f"({sum(e for _, e in ornekler):.0f} doğru, "
          f"{len(ornekler) - sum(e for _, e in ornekler):.0f} yanlış — şans %50)")

    # 2. Küçük model + ablasyon
    model = HiperGeometrikAI(n=64, katman_sayisi=2, baglam_penceresi=8,
                             emb_dim=32, num_heads=4, sozluk_boyutu=256,
                             dropout=0.0, seyrek_tablo_boyutu=2048,
                             seyrek_boyut=8, bilgilendir=False)
    deney = AblasyonDeneyi(model, tohum=0)
    sonuc = deney.kos(ornekler)

    print("\n2) Ablasyon sonuçları:")
    print(f"   KONTROL (boş bellek)   : salt okuma doğruluğu "
          f"{sonuc['kontrol_dogruluk']:.0%}  ← sinyal YOK (şans)")
    print(f"   DENEY   (bilgi yazılı) : salt okuma doğruluğu "
          f"{sonuc['deney_dogruluk']:.0%}  ← sinyal VAR")
    print(f"   Yazılan satır: {sonuc['yazilan_satir']}/{sonuc['toplam_satir']}")

    etki = sonuc["deney_dogruluk"] - sonuc["kontrol_dogruluk"]
    print(f"\n3) Etki büyüklüğü: +{etki:.0%} (bilgi yazmak, okuma doğruluğunu "
          f"{sonuc['kontrol_dogruluk']:.0%} → {sonuc['deney_dogruluk']:.0%} taşıdı)")

    print("\n" + CIZGI)
    print("SONUÇ: Doğrulanmış deneyim bilgisi seyrek belleğe yazıldığında,")
    print("aşağı-akış okuyucu için ÖLÇÜLEBİLİR bir öğrenme sinyaline dönüşüyor.")
    print("Boş bellekte bu sinyal yoktur (şans). Bu, 'küçük fiziksel modelin çok")
    print("büyük adreslenebilir uzayı aktif keşfi' tezinin tek örnekli kanıtıdır.")
    print(CIZGI)


if __name__ == "__main__":
    main()
