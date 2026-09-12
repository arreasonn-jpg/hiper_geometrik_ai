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

import sys
import os

KOK_DIZIN = os.path.abspath(os.path.dirname(__file__))
MIMARI_DIZIN = os.path.join(KOK_DIZIN, "mimari")
EGITIM_DIZIN = os.path.join(KOK_DIZIN, "egitim")

for _p in [KOK_DIZIN, MIMARI_DIZIN, EGITIM_DIZIN]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bilgi_katmani import (KATMAN_TAM, KATMAN_KISMI, KATMAN_ACIK,
                           KATMAN_ETIKET, beyaz_liste_olustur)
from hga.ui_runtime import (
    SORU_ETIKETI, CEVAP_ETIKETI,
    config_yukle,
    tokenizer_hazirla as ortak_tokenizer_hazirla,
    bilgi_katmani_hazirla as ortak_bilgi_katmani_hazirla,
    model_ve_agirlik_yukle,
    metin_uret,
)


def tokenizer_hazirla():
    """BPE sözlüğünü ortak UI runtime üzerinden hazırla."""
    return ortak_tokenizer_hazirla(KOK_DIZIN, logger=print)


def bilgi_katmani_hazirla():
    """Kayıtlı bilgiyi ortak UI runtime üzerinden hazırla."""
    bilgi, _ = ortak_bilgi_katmani_hazirla(KOK_DIZIN, logger=print)
    return bilgi


def uretim_yap(model, tokenizer, prompt, baglam, izinli=None, max_token=40):
    """Terminal üretimi; gerçek BPE döngüsü hga.ui_runtime.metin_uret'tedir."""
    metin = metin_uret(
        model, tokenizer, prompt, baglam, izinli=izinli, max_token=max_token,
        strateji="sample", temperature=0.7, special_token_siniri=3,
    )
    if metin:
        print(metin + " ", end="", flush=True)
    return metin


def main():
    print("═" * 65)
    print("🌀 HİPER-GEOMETRİK AI — TERMINAL SOHBET (v15: Kronecker + Seyrek Bellek)")
    print("═" * 65)

    cfg = config_yukle(KOK_DIZIN)
    mcfg = cfg["model"]
    n_gen = mcfg.n
    baglam = mcfg.baglam_penceresi
    tokenizer = tokenizer_hazirla()
    bilgi = bilgi_katmani_hazirla()

    model, yuklenen, _ = model_ve_agirlik_yukle(
        KOK_DIZIN, tokenizer, n=n_gen, baglam=baglam, logger=print,
    )
    if yuklenen is None:
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
