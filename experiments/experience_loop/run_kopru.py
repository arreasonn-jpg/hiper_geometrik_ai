# -*- coding: utf-8 -*-
"""
HGA Experience Engine — Deneyim ↔ Seyrek Bellek Köprüsü Demosu (v0.6)
=====================================================================
Doğrulanmış deneyimlerin, mevcut geometrik çekirdekteki torch seyrek tabloya
(`mimari/seyrek_tablo.py`) nasıl yazıldığını gösterir (rapor §22 commit 6, v0.6):

    1. Aritmetik alanda deneyim üret + deterministik doğrulama → 6 VERIFIED.
    2. Her doğrulanmış üçlüyü TorchKoprusu ile seyrek tabloya yaz.
    3. Tablo doluluğunu önce/sonra raporla (boş küme → dolu küme).

Torch kurulu değilse dürüstçe bilgi verir ve atlar (saf-Python katmanı etkilenmez).

Çalıştırma:
    python experiments/experience_loop/run_kopru.py
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
)
from hga.knowledge import DeneyimDurumu  # noqa: E402
from hga.memory import torch_var_mi  # noqa: E402

CIZGI = "=" * 74


def main():
    print(CIZGI)
    print("Deneyim ↔ Geometrik Seyrek Bellek Köprüsü (v0.6)")
    print(CIZGI)

    # 1. Doğrulanmış deneyim üret
    k = aritmetik_etki_alani()
    ev = ExperienceEvaluator()
    adaylar = ExperienceGenerator(tip_filtresi=False).uret(k)
    for a in adaylar:
        ev.degerlendir(a, k)
    DogrulamaHatti(AritmetikOrtam().aday_dogrula, evaluator=ev).isle(k, adaylar)
    dogrulananlar = [a for a in adaylar if a.state == DeneyimDurumu.VERIFIED]
    print(f"1) Doğrulanan deneyim sayısı: {len(dogrulananlar)} (VERIFIED)")

    # 2. Köprüyle seyrek tabloya yaz
    if not torch_var_mi():
        print("\n[torch kurulu değil] Köprü hazır ama tabloya gerçek yazım yapılamadı.")
        print("Saf-Python katmanı (DeneyimSlotlari) aynı adreslemeyi zaten sağlıyor.")
        return

    import torch

    from hga.memory import TorchKoprusu
    kopru = TorchKoprusu(tablo_boyutu=4096, boyut=32)
    print(f"2) Torch köprüsü kuruldu: {kopru.tablo.__class__.__name__}")
    once = kopru.doluluk()
    print(f"   Yazmadan önce doluluk: {once[0]}/{once[1]} (tümü boş küme)")

    # Her doğrulanmış üçlünün adreslediği satırı oku; aynı üçlü → aynı anahtar.
    vektorler = []
    for a in dogrulananlar:
        anahtar = kopru.anahtar(a.uclusu)[0].item()
        vektorler.append(kopru.dokun(a.uclusu))
        print(f"   {a.uclusu} → anahtar={anahtar}")

    # Çekirdeğin "boş küme" kuralı: satır yalnızca gradyan aldığında dolar.
    # Dokunulan satırlara bir gradyan adımı uygula (eğitimde olan budur).
    toplam = torch.cat(vektorler).sum()
    toplam.backward()
    torch.optim.AdamW(kopru.tablo.parameters(), lr=1e-2).step()

    sonra = kopru.doluluk()
    print(f"3) Gradyan adımından sonra doluluk: {sonra[0]}/{sonra[1]}")
    print(f"   Boş kümeler dolu kümeye dönüştü: {once[0]} → {sonra[0]}")

    print("\n" + CIZGI)
    print("Köprü tamam: doğrulanmış deneyimler geometrik çekirdeğin seyrek")
    print("belleğine adreslendi (aynı üçlü → her zaman aynı satır).")
    print(CIZGI)


if __name__ == "__main__":
    main()
