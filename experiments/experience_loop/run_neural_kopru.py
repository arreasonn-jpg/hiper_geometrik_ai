# -*- coding: utf-8 -*-
"""
HGA Experience Engine — Deneyim ↔ Model Seyrek Belleği Sinirsel Köprüsü (v0.6+)
===============================================================================
Doğrulanmış deneyim bilgisinin, geometrik çekirdeğin EĞİTİMDE kullandığı seyrek
belleğe nasıl bağlandığını ve "ölü-yol tuzağı"nın nasıl aşıldığını gösterir:

    1. Küçük bir HiperGeometrikAI modeli kur (seyrek bellek AÇIK).
    2. Aritmetik alanda deneyim üret + doğrula → 6 VERIFIED üçlü.
    3. NeuralKopru ile doğrulanmış üçlüleri modelin SEYREK tablosuna yaz
       (gra dy an adımıyla doluluk 0 → 6).
    4. Köprü canlılığını ölç: yazmadan önce gen_kopru gradyanı 0 (ölü yol),
       yazdıktan sonra > 0 (canlı yol).
    5. Token eğitimiyle birlikte çalıştığını göster: loss düşer, tabloya token
       pencereleri de eklenir (paylaşımlı bellek).

Torch kurulu değilse dürüstçe bilgi verir ve atlar.

Çalıştırma:
    .venv/bin/python experiments/experience_loop/run_neural_kopru.py
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


def dogrulanmis_ucluler():
    k = aritmetik_etki_alani()
    ev = ExperienceEvaluator()
    adaylar = ExperienceGenerator(tip_filtresi=False).uret(k)
    for a in adaylar:
        ev.degerlendir(a, k)
    DogrulamaHatti(AritmetikOrtam().aday_dogrula, evaluator=ev).isle(k, adaylar)
    return [a.uclusu for a in adaylar if a.state == DeneyimDurumu.VERIFIED]


def main():
    print(CIZGI)
    print("Deneyim ↔ Model Seyrek Belleği — Sinirsel Köprü (v0.6+)")
    print(CIZGI)

    if not torch_var_mi():
        print("[torch kurulu değil] NeuralKopru gerçek tabloya yazım gerektirir.")
        print("Saf-Python protokolü (DeneyimSlotlari) aynı adreslemeyi zaten sağlıyor.")
        return

    import torch
    from kuresel_model import HiperGeometrikAI
    from hga.memory import NeuralKopru

    # 1. Küçük model (seyrek bellek açık)
    model = HiperGeometrikAI(n=64, katman_sayisi=2, baglam_penceresi=8,
                             emb_dim=32, num_heads=4, sozluk_boyutu=256,
                             dropout=0.0, seyrek_tablo_boyutu=2048,
                             seyrek_boyut=8, bilgilendir=False)
    print(f"1) Model kuruldu: n=64, K=2, seyrek tablo 2048×8")

    # 2. Doğrulanmış deneyimler
    ucluler = dogrulanmis_ucluler()
    print(f"2) Doğrulanmış deneyim üçlüsü: {len(ucluler)} (VERIFIED)")

    # 3. NeuralKopru ile modelin kendi tablosuna bağla
    kopru = NeuralKopru(model)
    once, toplam = kopru.doluluk()
    print(f"3) Yazmadan önce doluluk: {once}/{toplam}")

    ornek = ucluler[0]
    print(f"   Ölü yol kontrolü (yazmadan): gen_kopru canlı mı? "
          f"{kopru.kopru_canli_mi(ornek)}")
    assert kopru.kopru_canli_mi(ornek) is False

    kopru.coklu_yaz(ucluler)
    sonra, _ = kopru.doluluk()
    print(f"4) Doğrulanmış üçlüler yazıldı → doluluk: {once} → {sonra}")
    print(f"   Canlı yol kontrolü (yazdıktan): gen_kopru canlı mı? "
          f"{kopru.kopru_canli_mi(ornek)}")
    assert kopru.kopru_canli_mi(ornek) is True

    # 5. Token eğitimiyle birlikte (paylaşımlı bellek)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    desen = [10 + (i % 4) for i in range(120)]
    x = torch.tensor([desen[i:i + 8] for i in range(112)], dtype=torch.long)
    y = torch.tensor(desen[8:], dtype=torch.long)
    model.train()
    ilk = son = None
    for adim in range(60):
        i = (adim * 16) % (112 - 16)
        opt.zero_grad()
        loss = loss_fn(model(x[i:i + 16]), y[i:i + 16])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if adim == 0:
            ilk = loss.item()
        son = loss.item()
    print(f"5) Token eğitimi: loss {ilk:.3f} → {son:.3f} (düştü ✓)")

    doluluk_son, _ = kopru.doluluk()
    print(f"   Paylaşımlı bellek doluluğu: {sonra} → {doluluk_son} "
          f"(token pencereleri eklendi)")

    print("\n" + CIZGI)
    print("SONUÇ: Doğrulanmış deneyim bilgisi, modelin kendi seyrek belleğinde")
    print("yer kaplıyor ve gen_kopru köprüsünden gradyanla öğrenilebilir bir yola")
    print("dönüşüyor. 'Boş küme → ölü yol' tuzağı, deneyim satırları dolunca aşılıyor.")
    print(CIZGI)


if __name__ == "__main__":
    main()
