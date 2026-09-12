# -*- coding: utf-8 -*-
"""
Hiper-Geometrik AI — Terminal Chatbot (Kronecker Zinciri Sürümü, v14)
=====================================================================

Bu sürümdeki değişiklikler (rapor 8.1 / 8.4 / 8.6):
  - model_olustur artık bu dosyada DEĞİL: tek doğruluk kaynağı
    mimari/kuresel_model.py'dir ('n' parametresi gerçekten modele iletilir).
  - Varsayılan tokenizer BPE (alt-kelime); sözlük bpe_sozluk.json'a kilitlenir.
  - Varsayılan mimari: n=256, K=4 bilinear (A@X@B) zinciri, bağlam=16,
    nedensel dikkat.
  - Ağırlık yükleme strict=True; uyumsuz checkpoint açıkça raporlanır.
  - Sohbet istem biçimi, talimat eğitimiyle AYNI: 'soru ... cevap ... son'.
"""

import sys
import os

KOK_DIZIN = os.path.abspath(os.path.dirname(__file__))
MIMARI_DIZIN = os.path.join(KOK_DIZIN, "mimari")
EGITIM_DIZIN = os.path.join(KOK_DIZIN, "egitim")

for _p in [KOK_DIZIN, MIMARI_DIZIN, EGITIM_DIZIN]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from kuresel_model import (model_olustur, agirlik_yukle,
                           VARSAYILAN_N, VARSAYILAN_BAGLAM, VARSAYILAN_SOZLUK)
from bpe_tokenizer import BPETokenizer

# Talimat eğitimiyle (egitim/talimat_egitici.py) aynı istem biçimi
SORU_ETIKETI = "soru"
CEVAP_ETIKETI = "cevap"

# Üretimde karşılaşınca durulacak kelimeler (birleştirilmiş kelime üzerinden
# kontrol edilir: "so"+"n" gibi bölünmüş parçalar da doğru yakalanır)
DURDURMA_KELIMELERI = {"son", "soru", "cevap"}


def tokenizer_hazirla() -> BPETokenizer:
    """BPE sözlüğü: kilitli dosya varsa yükle, yoksa korpustan kur ve kilitle."""
    tok = BPETokenizer(baglam_penceresi=VARSAYILAN_BAGLAM,
                       max_vocab_size=VARSAYILAN_SOZLUK)
    sozluk_yolu = os.path.join(KOK_DIZIN, "bpe_sozluk.json")
    korpus_yolu = os.path.join(KOK_DIZIN, "turkce_metin.txt")
    if os.path.exists(sozluk_yolu):
        tok.yukle(sozluk_yolu)
        print(f"🔒 Kilitli BPE sözlüğü yüklendi: {tok.sozluk_boyutu} parça")
    elif os.path.exists(korpus_yolu):
        with open(korpus_yolu, "r", encoding="utf-8") as f:
            tok.fit_on_text(f.read(800000))
        tok.kaydet(sozluk_yolu)
        print(f"🔧 BPE sözlüğü kuruldu ve kilitlendi: {tok.sozluk_boyutu} parça "
              f"→ bpe_sozluk.json")
    else:
        print("⚠️ bpe_sozluk.json / turkce_metin.txt bulunamadı — sözlüksüz (demo) mod.")
    return tok


def to_token_ids(tokenizer, metin):
    if hasattr(tokenizer, "encode"):
        res = tokenizer.encode(metin)
    elif hasattr(tokenizer, "text_to_ids"):
        res = tokenizer.text_to_ids(metin)
    else:
        res = [tokenizer.sozluk.get(w.lower(), 1) for w in metin.split()]
    if isinstance(res, torch.Tensor):
        return res.detach().cpu().flatten().tolist()
    return list(res)


def id_to_metin(tokenizer, token_ids):
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids)
    if hasattr(tokenizer, "ids_to_text"):
        return tokenizer.ids_to_text(token_ids)
    id_map = getattr(tokenizer, "id_to_kelime", {})
    return " ".join(id_map.get(int(i), "<UNK>") for i in token_ids if int(i) > 2)


def main():
    print("═" * 65)
    print("🌀 HİPER-GEOMETRİK AI — TERMINAL SOHBET (Kronecker Zinciri v14)")
    print("═" * 65)

    n_gen = VARSAYILAN_N
    baglam = VARSAYILAN_BAGLAM
    tokenizer = tokenizer_hazirla()

    vocab_size = len(getattr(tokenizer, "sozluk", {}))
    if vocab_size < 100:
        vocab_size = VARSAYILAN_SOZLUK

    model = model_olustur(vocab_size, n=n_gen, baglam_penceresi=baglam)

    yuklendi = False
    for ad in [f"hiper_model_{n_gen}_talimat.pt", f"hiper_model_{n_gen}.pt"]:
        yol = os.path.join(KOK_DIZIN, ad)
        if os.path.exists(yol):
            try:
                agirlik_yukle(model, yol, strict=True)
                print(f"✅ Model ağırlıkları yüklendi: {ad}")
                yuklendi = True
                break
            except Exception as e:
                print(f"⚠️ {ad} yüklenemedi (mimari uyumsuz?): {e}")
    if not yuklendi:
        print("⚠️ Uyumlu ağırlık yok — model rastgele başlatıldı.")
        print("   Eğitmek için: python egitim/egitici.py  ve  "
              "python egitim/talimat_egitici.py")

    model.eval()
    print("\n💡 Komutlar: 'q' / 'exit' (Çıkış), '/mod' (Mod Değiştir)")
    print("⚡ Mod: [Sohbet Asistanı (Instruction FT)]\n")

    talimat_modu = True
    eos_id = getattr(tokenizer, "EOS_ID", 3)

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

            uretilen_ids = list(to_token_ids(tokenizer, prompt))
            print("🤖 AI: ", end="", flush=True)
            parca = ""  # BPE: kelime içi parçaları biriktir

            for _ in range(40):
                pencere = uretilen_ids[-baglam:]
                if len(pencere) < baglam:
                    pencere = [0] * (baglam - len(pencere)) + pencere

                inp = torch.tensor([pencere], dtype=torch.long)
                with torch.no_grad():
                    logits = model(inp)
                    logits[0, :3] = -float("inf")   # PAD/UNK/BOS engelle (EOS serbest)
                    probs = torch.softmax(logits / 0.7, dim=-1)
                    next_token = int(torch.multinomial(probs, num_samples=1).item())

                uretilen_ids.append(next_token)
                if next_token == eos_id:
                    break

                ham = getattr(tokenizer, "id_to_kelime", {}).get(next_token, "")
                if not ham or ham in ("<pad>", "<unk>", "<bos>"):
                    continue
                if hasattr(tokenizer, "subword_set"):       # BPE
                    if ham.endswith("</w>"):                # kelime sonu → kontrol + yazdır
                        kelime = parca + ham[:-4]
                        parca = ""
                        if kelime in DURDURMA_KELIMELERI:
                            break
                        if kelime:
                            print(kelime + " ", end="", flush=True)
                    else:                                   # kelime içi → biriktir
                        parca += ham
                else:                                       # kelime-bazlı tokenizer
                    if ham.strip() in DURDURMA_KELIMELERI:
                        break
                    print(ham + " ", end="", flush=True)

            if parca:
                print(parca + " ", end="", flush=True)
            print("\n")

        except KeyboardInterrupt:
            print("\n👋 Çıkış yapıldı.")
            break
        except Exception as e:
            print(f"\n⚠️ Hata: {e}")


if __name__ == "__main__":
    main()
