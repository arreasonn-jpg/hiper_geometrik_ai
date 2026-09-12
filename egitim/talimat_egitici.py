# -*- coding: utf-8 -*-
"""Instruction fine-tuning: temel modelin üzerine eğitilmiş soru–cevap modeli üretir.

ÖNEMLİ davranış değişikliği: Eski sürüm, fine-tuning sonucunu hem
``hiper_model_1000_talimat.pt`` hem de TEMEL model dosyası ``hiper_model_1000.pt``
olarak kaydediyordu; yani her çalıştırmada ön-eğitim ağırlığı geri dönülemez
biçimde eziliyordu. Artık yalnızca ``hiper_model_<n>_talimat.pt`` yazılır.
"""
import argparse
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from ortak import (  # noqa: E402
    KOK_DIZIN,
    agirlik_yukle,
    log_kur,
    model_olustur,
    model_yolu,
    tokenizer_hazirla,
)
from egitim.talimat_toplayici import TalimatToplayici  # noqa: E402
from mimari.kuresel_loss import KureselGeometrikLoss  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Instruction fine-tuning")
    parser.add_argument("--n", type=int, default=1000, help="Model boyutu n (varsayılan 1000)")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--talimat-sifirla", action="store_true",
                        help="talimat_verisi.json'u varsayılan setle yeniden yaz")
    args = parser.parse_args()

    log_kur()
    print(f"\n🎯 Instruction FT (n={args.n}, kilitli sözlük)")

    # 1) Talimat verisi: VARSA dosyadan yükle (elle eklenen örnekler korunur)
    tt = TalimatToplayici(os.path.join(KOK_DIZIN, "talimat_verisi.json"))
    talimatlar = tt.hazirla_veya_yukle(sifirla=args.talimat_sifirla)

    # 2) Sözlük: sozluk.json varsa YÜKLE (kimlikler kaymasın), yoksa kur
    tok = tokenizer_hazirla(8000)

    # 3) Model + varsa temel ağırlık
    model = model_olustur(len(tok.sozluk), n=args.n, baglam_penceresi=8)
    temel_yol = model_yolu(args.n)
    if os.path.exists(temel_yol):
        agirlik_yukle(model, temel_yol)
        print("✅ Temel ağırlık yüklendi:", os.path.basename(temel_yol))
    else:
        print("⚠️ Temel model bulunamadı; sıfırdan (rastgele) fine-tuning yapılıyor.")

    # 4) Eğitim örneği üretimi
    X, Y = [], []
    for it in talimatlar:
        ids = tok.encode(f"soru {it['soru']} cevap {it['cevap']} son")
        cid = tok.encode("cevap")
        ctok = cid[0] if cid else -1
        for i in range(1, len(ids)):
            if ctok not in ids[:i]:
                continue
            x = ids[max(0, i - 8):i]
            x = [0] * (8 - len(x)) + x
            X.append(x)
            Y.append(ids[i])

    if not X:
        print("❌ Eğitim örneği üretilemedi (sözlük çok küçük olabilir).")
        sys.exit(1)

    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    print(f"✨ Eğitim adımı: {len(X)}")

    # 5) Eğitim döngüsü
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = KureselGeometrikLoss(label_smoothing=0.02)
    model.train()

    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(len(X))
        toplam, adim = 0.0, 0
        for b in range(0, len(X), args.batch):
            idx = perm[b:b + args.batch]
            opt.zero_grad()
            logits = model(X[idx])
            loss = loss_fn(logits, Y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            toplam += loss.item()
            adim += 1
        if ep % 20 == 0 or ep == args.epochs:
            print(f"  🌟 Epoch {ep:03d}/{args.epochs}  loss={toplam / max(adim, 1):.4f}")

    # 6) SADECE talimat checkpoint'ini yaz; temel model (hiper_model_<n>.pt) DOKUNULMAZ
    out = model_yolu(args.n, talimat=True)
    torch.save(model.state_dict(), out)
    print(f"💾 Kaydedildi: {out}")
    print(f"🛡️ Temel model korunuyor: {temel_yol} (üzerine yazılmadı)")


if __name__ == "__main__":
    main()
