# -*- coding: utf-8 -*-
"""
Hiper-Geometrik AI — Terminal Chatbot (Instruction Fine-Tuned)

Kullanım:
    python calistir.py            # varsayılan n=1000
    python calistir.py --n 500    # farklı model boyutu (ağırlık dosyası varsa)
"""
import argparse
import os
import sys

KOK_DIZIN = os.path.abspath(os.path.dirname(__file__))
if KOK_DIZIN not in sys.path:
    sys.path.insert(0, KOK_DIZIN)

from ortak import (  # noqa: E402  (KOK_DIZIN sys.path'e eklendikten sonra)
    agirlik_yukle,
    log_kur,
    metin_uret,
    model_olustur,
    model_yolu,
    tokenizer_hazirla,
)

SORU_ETIKETI = "### Soru:"
CEVAP_ETIKETI = "### Cevap:"


def main():
    parser = argparse.ArgumentParser(description="Hiper-Geometrik AI terminal sohbet")
    parser.add_argument("--n", type=int, default=1000, help="Model boyutu n (varsayılan 1000)")
    parser.add_argument("--temel", action="store_true",
                        help="Talimat yerine temel model ağırlığını yükle")
    args = parser.parse_args()

    log_kur()
    print("═" * 65)
    print(f"🌀 HİPER-GEOMETRİK AI — TERMINAL SOHBET (n={args.n})")
    print("═" * 65)

    baglam = 8
    tokenizer = tokenizer_hazirla(8000)

    # model_olustur artık gerçek 'n' parametresini iletiyor (bkz. ortak.py)
    model = model_olustur(len(tokenizer.sozluk), args.n, baglam)

    talimat_yol = model_yolu(args.n, talimat=True)
    temel_yol = model_yolu(args.n)
    yol = temel_yol if args.temel else talimat_yol
    if not os.path.exists(yol):  # talimat yoksa temele, temel yoksa talimata düş
        yedek = temel_yol if not args.temel else talimat_yol
        if os.path.exists(yedek):
            yol = yedek

    if os.path.exists(yol):
        agirlik_yukle(model, yol)
        print(f"✅ Model ağırlıkları yüklendi: {os.path.basename(yol)}")
    else:
        print("⚠️ Model ağırlığı bulunamadı! Eğitimsiz (rastgele) modelle sohbet.")
        print("   Eğitim için bkz. README → 'Eğitim Akışı'.")

    model.eval()
    print("\n💡 Komutlar: 'q' / 'exit' (Çıkış), '/mod' (Mod Değiştir)")
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

            prompt = f"{SORU_ETIKETI} {kullanici} {CEVAP_ETIKETI}" if talimat_modu else kullanici
            giris_ids = tokenizer.encode(prompt)

            print("🤖 AI: ", end="", flush=True)
            kelimeler = metin_uret(model, tokenizer, giris_ids, baglam, max_token=40)
            print(" ".join(kelimeler) + "\n")

        except KeyboardInterrupt:
            print("\n👋 Çıkış yapıldı.")
            break
        except Exception as e:
            print(f"\n⚠️ Hata: {e}")


if __name__ == "__main__":
    main()
