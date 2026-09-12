# -*- coding: utf-8 -*-
"""
Talimat (Instruction) Fine-Tuning Motoru
========================================
Bu sürümdeki değişiklikler (rapor 8.1 / 8.4):
  - model_olustur kopyası KALDIRILDI → mimari/kuresel_model.py'den gelir
    (tek doğruluk kaynağı; 'n' artık gerçekten modele iletilir).
  - Tokenizer artık BPE; sözlük bir kez kurulup KİLİTLENİR (bpe_sozluk.json)
    — her çalıştırmada sessizce yeniden kurulmaz.
  - Temel ağırlık yükleme strict=True — uyumsuzluk açıkça raporlanır.
  - Bağlam penceresi modelden okunur (eski sabit 8 değildir).
"""
import sys
import os
import argparse

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [KOK, os.path.join(KOK, "mimari"), os.path.dirname(__file__)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

from kuresel_model import (model_olustur, agirlik_yukle, agirlik_kaydet,
                           VARSAYILAN_N, VARSAYILAN_KATMAN, VARSAYILAN_BAGLAM,
                           VARSAYILAN_SEYREK_SATIR, VARSAYILAN_SEYREK_BOYUT)
from bpe_tokenizer import BPETokenizer
from talimat_toplayici import TalimatToplayici


def bpe_tokenizer_hazirla(talimat_metni: str) -> BPETokenizer:
    """BPE sözlüğü: diskte varsa KİLİTLİ olarak yükle, yoksa kur ve kaydet."""
    tok = BPETokenizer(baglam_penceresi=VARSAYILAN_BAGLAM, max_vocab_size=8000)
    yol = os.path.join(KOK, "bpe_sozluk.json")
    if os.path.exists(yol):
        tok.yukle(yol)
        print(f"🔒 Kilitli BPE sözlüğü yüklendi: {tok.sozluk_boyutu} parça")
    else:
        korpus = os.path.join(KOK, "turkce_metin.txt")
        parcalar = [talimat_metni]
        if os.path.exists(korpus):
            with open(korpus, "r", encoding="utf-8") as f:
                parcalar.append(f.read(800000))
        tok.fit_on_text("\n".join(parcalar))
        tok.kaydet(yol)
        print(f"🔧 BPE sözlüğü kuruldu ve kilitlendi: {tok.sozluk_boyutu} parça "
              f"→ bpe_sozluk.json")
    return tok


def main():
    ap = argparse.ArgumentParser(description="Hiper-Geometrik AI talimat fine-tuning")
    ap.add_argument("--cag", type=int, default=120)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n", type=int, default=VARSAYILAN_N)
    ap.add_argument("--katman", type=int, default=VARSAYILAN_KATMAN)
    # Seyrek bellek parametreleri — TEMEL EĞİTİMLE AYNI OLMALI (strict yükleme)
    ap.add_argument("--seyrek-satir", type=int, default=VARSAYILAN_SEYREK_SATIR)
    ap.add_argument("--seyrek-boyut", type=int, default=VARSAYILAN_SEYREK_BOYUT)
    ap.add_argument("--seyrek-yok", action="store_true")
    args = ap.parse_args()

    print("\n🎯 Talimat (Instruction) Fine-Tuning — Kronecker zinciri sürümü")
    tt = TalimatToplayici(os.path.join(KOK, "talimat_verisi.json"))
    talimatlar = tt.hazirla_veya_yukle()

    talimat_metni = " ".join(f"{it['soru']} {it['cevap']}" for it in talimatlar)
    tok = bpe_tokenizer_hazirla(talimat_metni)

    model = model_olustur(sozluk_boyutu=max(len(tok.sozluk), 64), n=args.n,
                          baglam_penceresi=VARSAYILAN_BAGLAM,
                          katman_sayisi=args.katman,
                          seyrek_tablo_boyutu=(0 if args.seyrek_yok else args.seyrek_satir),
                          seyrek_boyut=args.seyrek_boyut)

    baglam = model.baglam_penceresi
    temel_yol = os.path.join(KOK, f"hiper_model_{args.n}.pt")
    if os.path.exists(temel_yol):
        try:
            agirlik_yukle(model, temel_yol, strict=True)
            print("✅ Temel ağırlık strict=True ile yüklendi")
        except Exception as e:
            print(f"⚠️ Temel ağırlık yüklenemedi (mimari uyumsuz olabilir) — "
                  f"rastgele ağırlıkla devam ediliyor: {e}")

    # 'soru ... cevap ... son' biçiminde pencereler kur:
    # yalnızca 'cevap' işaretinden SONRAKİ tokenlar hedef olur
    X, Y = [], []
    for it in talimatlar:
        ids = tok.encode(f"soru {it['soru']} cevap {it['cevap']} son")
        cevap_ids = tok.encode("cevap")
        isaret = cevap_ids[0] if cevap_ids else -1
        for i in range(1, len(ids)):
            if isaret not in ids[:i]:
                continue
            x = ids[max(0, i - baglam):i]
            x = [0] * (baglam - len(x)) + x
            X.append(x)
            Y.append(ids[i])

    if not X:
        print("❌ Eğitim örneği üretilemedi (sözlük çok küçük olabilir).")
        sys.exit(1)

    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    print(f"✨ Eğitim adımı: {len(X):,} (bağlam={baglam})")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)
    model.train()

    for ep in range(1, args.cag + 1):
        perm = torch.randperm(len(X))
        tl, n = 0.0, 0
        for b in range(0, len(X), args.batch):
            idx = perm[b:b + args.batch]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), Y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
            n += 1
        if ep % 20 == 0 or ep == args.cag:
            print(f"  🌟 Epoch {ep:03d}/{args.cag}  loss={tl / max(n, 1):.4f}")

    talimat_yol = os.path.join(KOK, f"hiper_model_{args.n}_talimat.pt")
    agirlik_kaydet(model, talimat_yol)
    agirlik_kaydet(model, temel_yol)
    if model.seyrek_tablo is not None:
        print(f"🧠 {model.seyrek_doluluk_metni()}")
    print(f"💾 Kaydedildi: {talimat_yol}")


if __name__ == "__main__":
    main()
