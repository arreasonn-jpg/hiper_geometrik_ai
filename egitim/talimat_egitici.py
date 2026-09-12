# -*- coding: utf-8 -*-
import sys, os, inspect
KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in [KOK, os.path.join(KOK, "mimari")]:
    if p not in sys.path: sys.path.insert(0, p)

import torch, torch.nn as nn
from kuresel_model import HiperGeometrikAI
from tokenizer import GeometrikTokenizer
from talimat_toplayici import TalimatToplayici

def model_olustur(vocab_size, n_gen=1000, baglam=8):
    sig = inspect.signature(HiperGeometrikAI.__init__)
    names = list(sig.parameters.keys())
    kw = {}
    for k in ["sozluk_boyutu", "vocab_size", "sozluk_boyut"]:
        if k in names: kw[k] = vocab_size; break
    for k in ["n_gen", "gen_sayisi", "boyut"]:
        if k in names: kw[k] = n_gen; break
    for k in ["baglam_penceresi", "baglam", "context_length"]:
        if k in names: kw[k] = baglam; break
    for k in ["emb_dim", "embedding_dim"]:
        if k in names: kw[k] = 64; break
    for k in ["num_heads", "kafa_sayisi"]:
        if k in names: kw[k] = 4; break
    return HiperGeometrikAI(**kw)

def main():
    print("\n🎯 v13.0 Instruction FT + Kilitli Sözlük")
    tt = TalimatToplayici(os.path.join(KOK, "talimat_verisi.json"))
    talimatlar = tt.hazirla_veya_yukle()

    tok = GeometrikTokenizer(8000)
    sozluk_yol = os.path.join(KOK, "sozluk.json")
    korpus = os.path.join(KOK, "turkce_metin.txt")

    # Sözlüğü bir kez oluştur ve KLTLE
    if os.path.exists(korpus):
        with open(korpus, "r", encoding="utf-8") as f:
            tok.fit(f.read(800000))
    tok.kaydet(sozluk_yol)
    print(f"🔒 Sözlük kilitlendi: {len(tok.sozluk)} kelime → sozluk.json")

    model = model_olustur(len(tok.sozluk), 1000, 8)
    pt = os.path.join(KOK, "hiper_model_1000.pt")
    if os.path.exists(pt):
        try:
            model.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True), strict=False)
            print("✅ Temel ağırlık yüklendi")
        except Exception as e:
            print("⚠️", e)

    X, Y = [], []
    for it in talimatlar:
        ids = tok.encode(f"soru {it['soru']} cevap {it['cevap']} son")
        cid = tok.encode("cevap")
        ctok = cid[0] if cid else -1
        for i in range(1, len(ids)):
            if ctok not in ids[:i]:
                continue
            x = ids[max(0, i-8):i]
            x = [0]*(8-len(x)) + x
            X.append(x); Y.append(ids[i])

    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    print(f"✨ Eğitim adımı: {len(X)}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)
    model.train()
    bs, epochs = 16, 120

    for ep in range(1, epochs+1):
        perm = torch.randperm(len(X)); tl, n = 0.0, 0
        for b in range(0, len(X), bs):
            idx = perm[b:b+bs]
            opt.zero_grad()
            logits = model(X[idx])
            loss = loss_fn(logits, Y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item(); n += 1
        if ep % 20 == 0 or ep == epochs:
            print(f"  🌟 Epoch {ep:03d}/{epochs}  loss={tl/max(n,1):.4f}")

    out = os.path.join(KOK, "hiper_model_1000_talimat.pt")
    torch.save(model.state_dict(), out)
    torch.save(model.state_dict(), pt)
    print(f"💾 Kaydedildi: {out}")

if __name__ == "__main__":
    main()