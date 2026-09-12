# -*- coding: utf-8 -*-
"""
Hiper-Geometrik AI — Terminal Chatbot (Instruction Fine-Tuned)
"""

import sys
import os

KOK_DIZIN = os.path.abspath(os.path.dirname(__file__))
MIMARI_DIZIN = os.path.join(KOK_DIZIN, "mimari")
EGITIM_DIZIN = os.path.join(KOK_DIZIN, "egitim")

for p in [KOK_DIZIN, MIMARI_DIZIN, EGITIM_DIZIN]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import inspect
from kuresel_model import HiperGeometrikAI
from tokenizer import GeometrikTokenizer
from bpe_tokenizer import BPETokenizer

SORU_ETIKETI = "### Soru:"
CEVAP_ETIKETI = "### Cevap:"
SON_ETIKET = "### Son"

def model_olustur(vocab_size, n_gen, baglam):
    sig = inspect.signature(HiperGeometrikAI.__init__)
    param_names = list(sig.parameters.keys())
    kwargs = {}
    for k in ["sozluk_boyutu", "vocab_size", "sozluk_boyut"]:
        if k in param_names: kwargs[k] = vocab_size; break
    for k in ["n_gen", "gen_sayisi", "boyut"]:
        if k in param_names: kwargs[k] = n_gen; break
    for k in ["baglam_penceresi", "baglam", "context_length"]:
        if k in param_names: kwargs[k] = baglam; break
    for k in ["emb_dim", "embedding_dim"]:
        if k in param_names: kwargs[k] = 64; break
    for k in ["num_heads", "kafa_sayisi"]:
        if k in param_names: kwargs[k] = 4; break
    return HiperGeometrikAI(**kwargs)

def to_token_ids(tokenizer, metin):
    if hasattr(tokenizer, "encode"):
        res = tokenizer.encode(metin)
    elif hasattr(tokenizer, "text_to_ids"):
        res = tokenizer.text_to_ids(metin)
    elif hasattr(tokenizer, "metin_to_id"):
        res = tokenizer.metin_to_id(metin)
    else:
        res = [tokenizer.sozluk.get(w.lower(), 1) for w in metin.split()]

    if isinstance(res, torch.Tensor):
        return res.detach().cpu().flatten().tolist()
    return list(res)

def id_to_metin(tokenizer, token_ids):
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids)
    elif hasattr(tokenizer, "ids_to_text"):
        return tokenizer.ids_to_text(token_ids)
    else:
        id_map = getattr(tokenizer, "id_to_kelime", {v: k for k, v in getattr(tokenizer, "sozluk", {}).items()})
        return " ".join([id_map.get(idx, "<UNK>") for idx in token_ids if idx > 2])

def main():
    print("═"*65)
    print("🌀 HPER-GEOMETRK FRAKTAL AI — TERMINAL SOHBET (Sürüm 11.5)")
    print("═"*65)

    n_gen = 1000
    baglam = 8
    tokenizer = GeometrikTokenizer(max_vocab_size=8000)
    
    korpus_yolu = os.path.join(KOK_DIZIN, "turkce_metin.txt")
    if os.path.exists(korpus_yolu):
        with open(korpus_yolu, "r", encoding="utf-8") as f:
            ham_metin = f.read(500000)
            for m in ["sozluk_olustur", "egit", "ogren", "fit"]:
                if hasattr(tokenizer, m):
                    getattr(tokenizer, m)(ham_metin)
                    break
    
    vocab_size = len(getattr(tokenizer, "sozluk", {}))
    if vocab_size < 100: vocab_size = 8003

    model = model_olustur(vocab_size, n_gen, baglam)
    model_yolu = os.path.join(KOK_DIZIN, f"hiper_model_{n_gen}.pt")
    
    if os.path.exists(model_yolu):
        model.load_state_dict(torch.load(model_yolu, map_location="cpu", weights_only=True), strict=False)
        print(f"✅ Model ağırlıkları yüklendi: {os.path.basename(model_yolu)}")
    else:
        print("⚠️ Model ağırlığı bulunamadı!")

    model.eval()
    print("\n💡 Komutlar: 'q' / 'exit' (Çıkış), '/mod' (Mod Değiştir), '/temizle'")
    print("⚡ Mod: [Sohbet Asistanı (Instruction FT)]\n")

    talimat_modu = True

    while True:
        try:
            kullanici = input("👤 Sen: ").strip()
            if not kullanici:
                continue
            if kullanici.lower() in ["q", "exit", "cikis"]:
                print("👋 Görüşmek üzere!")
                break
            if kullanici.lower() == "/mod":
                talimat_modu = not talimat_modu
                print(f"🔄 Aktif Mod: {'[Sohbet Asistanı]' if talimat_modu else '[Serbest Metin Tamamlama]'}")
                continue

            if talimat_modu:
                prompt = f"{SORU_ETIKETI} {kullanici} {CEVAP_ETIKETI}"
            else:
                prompt = kullanici

            giris_ids = to_token_ids(tokenizer, prompt)
            uretilen_ids = list(giris_ids)

            print("🤖 AI: ", end="", flush=True)

            yeni_kelimeler = []
            for _ in range(40):
                pencere = uretilen_ids[-baglam:]
                if len(pencere) < baglam:
                    pencere = [0] * (baglam - len(pencere)) + pencere

                inp_tensor = torch.tensor([pencere], dtype=torch.long)
                with torch.no_grad():
                    logits = model(inp_tensor)
                    logits[0, :3] = -float("inf") # UNK, PAD engelle
                    probs = torch.softmax(logits / 0.7, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()

                uretilen_ids.append(next_token)
                kelime = id_to_metin(tokenizer, [next_token]).strip()

                if "###" in kelime or "Son" in kelime:
                    break

                yeni_kelimeler.append(kelime)
                print(kelime + " ", end="", flush=True)

            print("\n")

        except KeyboardInterrupt:
            print("\n👋 Çıkış yapıldı.")
            break
        except Exception as e:
            print(f"\n⚠️ Hata: {e}")

if __name__ == "__main__":
    main()