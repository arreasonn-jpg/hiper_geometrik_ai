# -*- coding: utf-8 -*-
"""
HGA Experience Engine — Görev Ablasyonu Demosu (v1.0+)
========================================================
Doğrulanmış deneyim bilgisi, modelin KENDİ dizisel tamamlama görevini çözmesini
sağlıyor mu? (rapor §17/§20 kapanışının görev boyutu; §19 v0.6 + EK-B)

Akış:
    1. Aritmetik alanda deneyim üret + deterministik doğrulama.
    2. VERIFIED üçlüleri (özne, ilişki, nesne) tamamlama örnekleri olarak al.
    3. GorevAblasyonu.kos():  YOĞUN GÖVDE DONUKKEN
         KONTROL — bellek yolu boş+donuk   → ~şans (model çözemez)
         DENEY   — bellek yolu eğitilir    → ~%100 (bellek görevi taşır)
       + ölü-yol → canlı-yol geçişi (gen_kopru gradyanı 0 → >0).
    4. tablo_sifir(): TABLO-SIFIR — tablo boşaltılınca %100 → %0
       (bilgi köprüde değil, tablo SATIRLARINDADIR).
    5. heldout_siniri(): HELD-OUT — yazılmamış olgular genellemez (~şans).

Torch kurulu değilse dürüstçe bilgi verir ve atlar.

Çalıştırma:
    .venv/bin/python experiments/experience_loop/run_gorev_ablasyonu.py
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in [KOK, os.path.join(KOK, "mimari")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hga.experience import (  # noqa: E402
    AritmetikOrtam,
    DogrulamaHatti,
    ExperienceEvaluator,
    ExperienceGenerator,
    aritmetik_etki_alani,
)
from hga.knowledge import DeneyimDurumu  # noqa: E402
from hga.memory import torch_var_mi  # noqa: E402

CIZGI = "=" * 74


def dogrulanmis_ucluler(adet=6):
    """Doğrulama hattından VERIFIED üçlüler (özne, ilişki, nesne)."""
    k = aritmetik_etki_alani()
    ev = ExperienceEvaluator()
    adaylar = ExperienceGenerator(tip_filtresi=False).uret(k)
    for a in adaylar:
        ev.degerlendir(a, k)
    DogrulamaHatti(AritmetikOrtam().aday_dogrula, evaluator=ev).isle(k, adaylar)
    return [a.uclusu for a in adaylar if a.state == DeneyimDurumu.VERIFIED][:adet]


def main():
    print(CIZGI)
    print("Görev Ablasyonu: Yazılan bilgi, modelin KENDİ görevini çözüyor mu?")
    print(CIZGI)

    if not torch_var_mi():
        print("[torch kurulu değil] GorevAblasyonu gerçek model gerektirir.")
        return

    import torch
    from kuresel_model import HiperGeometrikAI

    from hga.memory import GorevAblasyonu

    # 1. Doğrulama hattından tamamlama örnekleri (özne, ilişki → nesne)
    ucluler = dogrulanmis_ucluler(adet=6)
    print(f"1) Doğrulanmış (VERIFIED) üçlü: {len(ucluler)} adet "
          f"(tamamlama: özne + ilişki → nesne)")

    # 2. Küçük model; yoğun gövde deney içinde dondurulur
    torch.manual_seed(0)
    model = HiperGeometrikAI(n=32, katman_sayisi=2, baglam_penceresi=4,
                             emb_dim=16, num_heads=2, sozluk_boyutu=32,
                             dropout=0.0, seyrek_tablo_boyutu=2048,
                             seyrek_boyut=32, bilgilendir=False)
    deney = GorevAblasyonu(model, tohum=0)
    sonuc = deney.kos(ucluler)

    print("\n2) Görev ablasyonu sonuçları (yoğun gövde donuk):")
    print(f"   KONTROL (bellek boş+donuk) : tamamlama doğruluğu "
          f"{sonuc['kontrol_dogruluk']:.0%}  ← model görevi çözemez (şans)")
    print(f"   DENEY   (bellek eğitildi)  : tamamlama doğruluğu "
          f"{sonuc['deney_dogruluk']:.0%}  ← bellek görevi TAŞIYOR")
    print(f"   Yazılan satır: {sonuc['yazilan_satir']}/{sonuc['toplam_satir']}")
    print(f"   Ölü-yol → canlı-yol: {sonuc['yol_canlandi']} "
          f"(gen_kopru gradyanı 0 → >0)")

    etki = sonuc["deney_dogruluk"] - sonuc["kontrol_dogruluk"]
    print(f"\n3) Etki büyüklüğü: +{etki:.0%} (tamamlama doğruluğunu "
          f"{sonuc['kontrol_dogruluk']:.0%} → {sonuc['deney_dogruluk']:.0%} taşıdı)")

    # 4. TABLO-SIFIR: bilgiyi geri ÇEKİNCE kazanım da geri gider mi?
    ts = GorevAblasyonu(HiperGeometrikAI(n=32, katman_sayisi=2,
                                         baglam_penceresi=4, emb_dim=16,
                                         num_heads=2, sozluk_boyutu=32,
                                         dropout=0.0, seyrek_tablo_boyutu=2048,
                                         seyrek_boyut=32, bilgilendir=False),
                        tohum=0).tablo_sifir(ucluler)
    print("\n4) TABLO-SIFIR (bilgi köprüde değil, tabloda mı?):")
    print(f"   DENEY  (bellek eğitildi)  : {ts['deney_dogruluk']:.0%}")
    print(f"   SIFIR  (tablo boşaltıldı) : {ts['tablo_sifir_dogruluk']:.0%}  "
          f"← köprü eğitimli kaldı ama görev çöktü")

    # 5. HELD-OUT: yazılmamış olgular genellemez (dürüst sınır)
    tren = [("E1", "R1", "E5"), ("E2", "R1", "E6"),
            ("E3", "R1", "E7"), ("E4", "R1", "E8")]
    test = [("F1", "R1", "E5"), ("F2", "R1", "E6"),
            ("F3", "R1", "E7"), ("F4", "R1", "E8")]
    ho = GorevAblasyonu(HiperGeometrikAI(n=32, katman_sayisi=2,
                                         baglam_penceresi=4, emb_dim=16,
                                         num_heads=2, sozluk_boyutu=32,
                                         dropout=0.0, seyrek_tablo_boyutu=2048,
                                         seyrek_boyut=32, bilgilendir=False),
                        tohum=0).heldout_siniri(tren, test)
    print("\n5) HELD-OUT SINIRI (yazılmamış olgu genellemez):")
    print(f"   eğitim üçlüleri : {ho['egitim_dogruluk']:.0%}")
    print(f"   yeni üçlüler    : {ho['heldout_dogruluk']:.0%}  "
          f"← bellekte satırı olmayan olgu yoktan var olmaz")

    print("\n" + CIZGI)
    print("SONUÇ: Doğrulanmış bilgi seyrek belleğe yazıldığında, modelin kendi")
    print("ileri geçiş yolu (gen_kopru → geometrik çekirdek → decoder) üzerinden")
    print("gerçek bir tamamlama görevini çözebilir hâle geliyor. Yoğun gövde")
    print("donukken kontrol şans düzeyinde kalır; bilgiyi tablodan geri çekince")
    print("(TABLO-SIFIR) kazanım kaybolur; yazılmamış olgular (HELD-OUT) ise")
    print("genellemez — bellek ölçülebilir, göreve yönelik ve dürüst bir kanaldır.")
    print(CIZGI)


if __name__ == "__main__":
    main()
