# -*- coding: utf-8 -*-
"""
Küresel Eğitim Motoru
=====================
Önceki sürümün güçlü yanları korunuyor (DataLoader + batch, AdamW, gradyan
kırpma, cosine LR, kayan pencere korpus taraması) ve iki yeni yetenek
ekleniyor:

  - Karışık hassasiyet (AMP): CUDA + bf16 destekliyorsa otomatik açılır
    (rapor 8.4.1). CPU'da kararlılık için varsayılan kapalıdır.
  - Girdi artık token id listesi YA DA ham metin olabilir; BPE tokenizer
    ile birlikte kullanılır (rapor 8.4.6).

Kullanım:
    python egitim/egitici.py --korpus turkce_metin.txt --cag 5
    python egitim/egitici.py --n 128 --katman 2 --batch 32   (küçük/deneysel)
"""
import os
import sys
import time
import argparse

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [KOK, os.path.join(KOK, "mimari")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from kuresel_model import (model_olustur, agirlik_kaydet,
                           VARSAYILAN_N, VARSAYILAN_KATMAN, VARSAYILAN_BAGLAM,
                           VARSAYILAN_SEYREK_SATIR, VARSAYILAN_SEYREK_BOYUT)
from bpe_tokenizer import BPETokenizer


class NgramDataset(Dataset):
    def __init__(self, token_ids, baglam_penceresi):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.baglam_penceresi = baglam_penceresi

    def __len__(self):
        return max(0, len(self.token_ids) - self.baglam_penceresi)

    def __getitem__(self, idx):
        x = self.token_ids[idx: idx + self.baglam_penceresi]
        y = self.token_ids[idx + self.baglam_penceresi]
        return x, y


class KureselEgitimMotoru:
    def __init__(self, model, ogrenme_hizi=1e-3, toplam_cag=15,
                 karisik_hassasiyet="auto"):
        """
        karisik_hassasiyet:
            "auto" → CUDA + bf16 destekliyse aç (rapor 8.4.1)
            True   → CUDA'da zorla aç
            False  → kapalı
        """
        self.model = model
        self.aygit = next(model.parameters()).device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=ogrenme_hizi,
                                           weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, toplam_cag), eta_min=1e-5)
        self.karisik_hassasiyet = self._amp_karar(karisik_hassasiyet)

    def _amp_karar(self, tercih):
        if tercih == "auto":
            return (self.aygit.type == "cuda" and torch.cuda.is_available()
                    and torch.cuda.is_bf16_supported())
        return bool(tercih) and self.aygit.type == "cuda"

    def _autocast(self):
        if self.karisik_hassasiyet:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        import contextlib
        return contextlib.nullcontext()

    def ngram_egitim_dongusu(self, veri, tokenizer=None, cag_sayisi=15,
                             batch_size=64):
        """Kayan pencereli n-gram eğitim döngüsü.

        veri      : List[int] token id'leri (tercih edilen) veya ham metin (str)
        tokenizer : veri str ise zorunlu ('encode' metodu olan her tokenizer)
        """
        if isinstance(veri, str):
            if tokenizer is None or not hasattr(tokenizer, "encode"):
                raise ValueError("Ham metin eğitimi için 'encode' metodu olan "
                                 "bir tokenizer gerekli.")
            tum_ids = tokenizer.encode(veri)
        else:
            tum_ids = [int(i) for i in veri]

        baglam = self.model.baglam_penceresi
        toplam_token = len(tum_ids)
        if toplam_token <= baglam + 1:
            raise ValueError(f"Korpus çok küçük ({toplam_token} token; "
                             f"en az {baglam + 2} gerekli).")

        # Çağ başı dilim: RAM dostu kayan pencere (önceki sürümden)
        DILIM_BOYUTU = min(60000, toplam_token)
        stride = (max(1, (toplam_token - DILIM_BOYUTU) // max(1, cag_sayisi - 1))
                  if toplam_token > DILIM_BOYUTU else 0)

        print("\n" + "═" * 60)
        print("   🚀 KAYAN PENCERELİ EĞİTİM MOTORU (BPE + bilinear Kronecker zinciri)")
        print(f"   📊 Korpus: {toplam_token:,} token | Dilim: {DILIM_BOYUTU:,} | "
              f"Batch: {batch_size} | AMP: {self.karisik_hassasiyet} | Aygıt: {self.aygit}")
        print("═" * 60)

        for cag in range(cag_sayisi):
            self.model.train()
            baslangic = (cag * stride) if stride > 0 else 0
            bitis = min(toplam_token, baslangic + DILIM_BOYUTU)
            dataset = NgramDataset(tum_ids[baslangic:bitis], baglam)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                drop_last=len(dataset) >= batch_size)

            toplam_loss, adim, cag_basla = 0.0, 0, time.time()
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.aygit)
                batch_y = batch_y.to(self.aygit)
                self.optimizer.zero_grad()
                with self._autocast():
                    logits = self.model(batch_x)
                    loss = self.criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                toplam_loss += loss.item()
                adim += 1

            self.scheduler.step()
            ortalama = toplam_loss / max(1, adim)
            bar = "█" * int((cag + 1) / cag_sayisi * 20)
            # Seyrek 'boş küme' doluluk izleme (rapor 9.4.4) — çağ başı bir kez
            seyrek_bilgi = ""
            tablo = getattr(self.model, "seyrek_tablo", None)
            if tablo is not None:
                dolu, toplam_satir = tablo.doluluk_orani()
                seyrek_bilgi = f" | Seyrek: {dolu:,}/{toplam_satir:,}"
            print(f"  Çağ {cag + 1:02d}/{cag_sayisi:02d} | Loss: {ortalama:.4f} | "
                  f"Süre: {time.time() - cag_basla:.1f}s | "
                  f"Dilim: [{baslangic:,}-{bitis:,}]{seyrek_bilgi} |{bar:<20}|")

        print("═" * 60)
        print("  [🎉 TAMAMLANDI] Eğitim bitti.")
        print("═" * 60 + "\n")


def main():
    ap = argparse.ArgumentParser(description="Hiper-Geometrik AI temel eğitimi")
    ap.add_argument("--korpus", default=os.path.join(KOK, "turkce_metin.txt"))
    ap.add_argument("--cag", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ogrenme-hizi", type=float, default=1e-3)
    ap.add_argument("--n", type=int, default=VARSAYILAN_N,
                    help="küresel bağ boyutu (128-256 önerilir)")
    ap.add_argument("--katman", type=int, default=VARSAYILAN_KATMAN,
                    help="bilinear katman sayısı K (4-8 önerilir)")
    ap.add_argument("--baglam", type=int, default=VARSAYILAN_BAGLAM)
    ap.add_argument("--checkpoint", action="store_true",
                    help="Gradient checkpointing (derin zincir, rapor 8.4.2)")
    ap.add_argument("--seyrek-satir", type=int, default=VARSAYILAN_SEYREK_SATIR,
                    help="seyrek 'boş küme' tablosu satır sayısı (rapor 9; "
                         "1.048.576 satır × 32 boyut ≈ 128 MB)")
    ap.add_argument("--seyrek-boyut", type=int, default=VARSAYILAN_SEYREK_BOYUT,
                    help="her kümenin vektör boyutu")
    ap.add_argument("--seyrek-yok", action="store_true",
                    help="seyrek belleği tamamen kapat")
    ap.add_argument("--amp", default="auto", choices=["auto", "evet", "hayir"])
    args = ap.parse_args()

    if not os.path.exists(args.korpus):
        print(f"❌ Korpus bulunamadı: {args.korpus}")
        print("   Önce egitim/veri_toplayici.py ile korpus toplayın (ör. geniş")
        print("   Wikipedia korpusu: OtomatikVeriToplayici.genis_korpus_cek)")
        print("   veya turkce_metin.txt dosyasını elle oluşturun.")
        sys.exit(1)

    with open(args.korpus, "r", encoding="utf-8") as f:
        metin = f.read()

    # BPE sözlüğü: kilitliyse yükle, yoksa kur (rapor 8.4.6)
    tok = BPETokenizer(baglam_penceresi=args.baglam, max_vocab_size=8000)
    sozluk_yolu = os.path.join(KOK, "bpe_sozluk.json")
    if os.path.exists(sozluk_yolu):
        tok.yukle(sozluk_yolu)
        print(f"🔒 Kilitli BPE sözlüğü yüklendi: {tok.sozluk_boyutu} parça")
    else:
        tok.fit_on_text(metin)
        tok.kaydet(sozluk_yolu)
        print(f"🔧 BPE sözlüğü kuruldu: {tok.sozluk_boyutu} parça → bpe_sozluk.json")

    model = model_olustur(sozluk_boyutu=max(len(tok.sozluk), 64), n=args.n,
                          baglam_penceresi=args.baglam,
                          katman_sayisi=args.katman,
                          checkpoint_kullan=args.checkpoint,
                          seyrek_tablo_boyutu=(0 if args.seyrek_yok else args.seyrek_satir),
                          seyrek_boyut=args.seyrek_boyut)

    motor = KureselEgitimMotoru(
        model, ogrenme_hizi=args.ogrenme_hizi, toplam_cag=args.cag,
        karisik_hassasiyet={"evet": True, "hayir": False, "auto": "auto"}[args.amp])
    motor.ngram_egitim_dongusu(tok.encode(metin), cag_sayisi=args.cag,
                               batch_size=args.batch)

    cikis = os.path.join(KOK, f"hiper_model_{args.n}.pt")
    agirlik_kaydet(model, cikis)
    print(f"💾 Model kaydedildi: {cikis}")


if __name__ == "__main__":
    main()
