# -*- coding: utf-8 -*-
"""
Hiper-Geometrik AI — Terminal Chatbot
=====================================
(Kronecker zinciri + seyrek bellek + 3 katmanlı halüsinasyon kontrolü, v15)

Bu sürümdeki değişiklikler (rapor 8.1 / 8.4 / 8.6 / 9 / 10):
  - model_olustur tek doğruluk kaynağı: mimari/kuresel_model.py ('n' iletilir).
  - Varsayılan tokenizer BPE; sözlük bpe_sozluk.json'a kilitlenir.
  - Mimari: n=256, K=4 bilinear (A@X@B) zinciri + seyrek 'boş küme' belleği
    (HashlenmisKureselTablo) + nedensel dikkat.
  - Ağırlık yükleme strict=True; uyumsuz checkpoint açıkça raporlanır.
  - 3 KATMANLI KARAR MEKANİZMASI (rapor 10.5, bilgi_katmani.py):
      1. Tam eşleşme     → kayıtlı cevap doğrudan [KAYITLI BİLGİ ✓]
      2. Kısmi eşleşme   → eşleşen kaydın kelimeleriyle BEYAZ LİSTELİ üretim
      3. Eşleşme yok     → serbest sinir ağı üretimi [DOĞRULANMAMIŞ]
"""

import json
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
from bilgi_katmani import (BilgiKatmani, KATMAN_TAM, KATMAN_KISMI, KATMAN_ACIK,
                           KATMAN_ETIKET, beyaz_liste_olustur)
from talimat_toplayici import ZENGIN

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


def bilgi_katmani_hazirla() -> BilgiKatmani:
    """Kayıtlı bilgi: talimat_verisi.json (varsa) + yerleşik talimat seti."""
    talimatlar = []
    yol = os.path.join(KOK_DIZIN, "talimat_verisi.json")
    if os.path.exists(yol):
        try:
            with open(yol, "r", encoding="utf-8") as f:
                talimatlar = json.load(f)
        except Exception as e:
            print(f"⚠️ talimat_verisi.json okunamadı ({e}) — yerleşik set kullanılacak.")
    if not talimatlar:
        talimatlar = ZENGIN
    return BilgiKatmani(talimatlar)


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


def uretim_yap(model, tokenizer, prompt, baglam, izinli=None, max_token=40):
    """Akışkan üretim. izinli (bool maske) verilirse yalnız beyaz listedeki
    token'lar örneklenebilir (KATMAN_KISMI kısıtlı üretimi, rapor 10.5.2)."""
    eos_id = getattr(tokenizer, "EOS_ID", 3)
    uretilen_ids = list(to_token_ids(tokenizer, prompt))
    parca = ""  # BPE: kelime içi parçaları biriktir

    for _ in range(max_token):
        pencere = uretilen_ids[-baglam:]
        if len(pencere) < baglam:
            pencere = [0] * (baglam - len(pencere)) + pencere

        inp = torch.tensor([pencere], dtype=torch.long)
        with torch.no_grad():
            logits = model(inp)
            logits[0, :3] = -float("inf")   # PAD/UNK/BOS engelle (EOS serbest)
            if izinli is not None:
                logits[0, ~izinli] = -float("inf")   # beyaz liste dışını engelle
            probs = torch.softmax(logits / 0.7, dim=-1)
            next_token = int(torch.multinomial(probs, num_samples=1).item())

        uretilen_ids.append(next_token)
        if next_token == eos_id:
            break

        ham = getattr(tokenizer, "id_to_kelime", {}).get(next_token, "")
        if not ham or ham in ("<pad>", "<unk>", "<bos>"):
            continue
        if hasattr(tokenizer, "subword_set"):           # BPE
            if ham.endswith("</w>"):                    # kelime sonu → kontrol + yazdır
                kelime = parca + ham[:-4]
                parca = ""
                if kelime in DURDURMA_KELIMELERI:
                    break
                if kelime:
                    print(kelime + " ", end="", flush=True)
            else:                                       # kelime içi → biriktir
                parca += ham
        else:                                           # kelime-bazlı tokenizer
            if ham.strip() in DURDURMA_KELIMELERI:
                break
            print(ham + " ", end="", flush=True)

    if parca:
        print(parca + " ", end="", flush=True)


def main():
    print("═" * 65)
    print("🌀 HİPER-GEOMETRİK AI — TERMINAL SOHBET (v15: Kronecker + Seyrek Bellek)")
    print("═" * 65)

    n_gen = VARSAYILAN_N
    baglam = VARSAYILAN_BAGLAM
    tokenizer = tokenizer_hazirla()
    bilgi = bilgi_katmani_hazirla()

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

    if hasattr(model, "seyrek_doluluk_metni"):
        print(f"🧠 {model.seyrek_doluluk_metni()}")

    model.eval()
    print("\n💡 Komutlar: 'q' / 'exit' (Çıkış), '/mod' (Mod Değiştir)")
    print("⚡ Mod: [Sohbet Asistanı (Instruction FT)]")
    print("🛡️ Halüsinasyon kontrolü: 1) kayıtlı bilgi 2) beyaz listeli üretim "
          "3) doğrulanmamış üretim\n")

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

            # ── 3 KATMANLI KARAR MEKANİZMASI (rapor 10.5) ──────────────
            izinli = None
            if talimat_modu:
                katman, cevap, skor, eslesme = bilgi.ara(kullanici)
            else:
                katman, cevap, skor, eslesme = KATMAN_ACIK, None, 0.0, None

            if katman == KATMAN_TAM:
                # 1. katman: hiçbir şey üretilmez — yalnızca hatırlanır
                print(f"🤖 AI {KATMAN_ETIKET[katman]} (güven 1.00): {cevap}\n")
                continue

            if katman == KATMAN_KISMI:
                # 2. katman: eşleşen kaydın kelimeleriyle kısıtlı üretim
                izinli = beyaz_liste_olustur(tokenizer, eslesme, model.sozluk_boyutu)
                etiket = f"{KATMAN_ETIKET[katman]} skor={skor:.2f}"
            elif talimat_modu:
                # 3. katman: açık genelleme — açıkça işaretlenmiş
                etiket = KATMAN_ETIKET[KATMAN_ACIK]
            else:
                etiket = "[SERBEST ÜRETİM]"

            prompt = f"{SORU_ETIKETI} {kullanici} {CEVAP_ETIKETI}" if talimat_modu else kullanici
            print(f"🤖 AI {etiket}: ", end="", flush=True)
            uretim_yap(model, tokenizer, prompt, baglam, izinli=izinli)
            print("\n")

        except KeyboardInterrupt:
            print("\n👋 Çıkış yapıldı.")
            break
        except Exception as e:
            print(f"\n⚠️ Hata: {e}")


if __name__ == "__main__":
    main()
