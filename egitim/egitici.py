# -*- coding: utf-8 -*-
"""Ön-eğitim motoru: korpus üzerinde next-token (n-gram) eğitimi.

Kullanım:
    python -m egitim.egitici --n 1000 --cag 15 --batch 256

Önce ``egitim/veri_toplayici.py`` ile ``turkce_metin.txt`` oluşturulmalıdır.
"""
import argparse
import logging
import os
import re
import sys
import time

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from mimari.kuresel_loss import KureselGeometrikLoss  # noqa: E402
from ortak import (  # noqa: E402
    agirlik_yukle,
    korpus_yolu,
    log_kur,
    model_olustur,
    model_yolu,
    sozluk_yolu,
    tokenizer_hazirla,
)

log = logging.getLogger("hiper.egitici")


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


def _kelime_id(tokenizer, kelime):
    """Tokenizer'dan kelime kimliği al (sozluk VEYA kelime_to_id özniteliğiyle).

    Eski kod doğrudan ``tokenizer.kelime_to_id`` kullanıyordu; GeometrikTokenizer'da
    bu öznitelik yoktur (sözlük ``sozluk`` adındadır) ve kod ilk çağrıda
    AttributeError ile düşerdi.
    """
    sozluk = getattr(tokenizer, "sozluk", None) or getattr(tokenizer, "kelime_to_id", {})
    unk_id = getattr(tokenizer, "UNK_ID", 1)
    return sozluk.get(kelime, unk_id)


class KureselEgitimMotoru:
    def __init__(self, model, ogrenme_hizi=0.002, toplam_cag=15):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=ogrenme_hizi, weight_decay=0.01)
        self.criterion = KureselGeometrikLoss(label_smoothing=0.05, ignore_index=0)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=toplam_cag, eta_min=1e-5
        )

    def ngram_egitim_dongusu(self, kelimeler, tokenizer, cag_sayisi=15, batch_size=256):
        baglam = self.model.baglam_penceresi
        toplam_kelime = len(kelimeler)

        # 🚀 CPU LIMITI: Çağ başı 60.000 kelimelik kayan pencere
        dilim_boyutu = min(60000, toplam_kelime)
        stride = (
            max(1, (toplam_kelime - dilim_boyutu) // max(1, cag_sayisi - 1))
            if toplam_kelime > dilim_boyutu
            else 0
        )

        log.info("=" * 60)
        log.info("🚀 AKILLI SLIDING-WINDOW MOTORU (TÜM KORPUS TARANIYOR)")
        log.info("📊 Toplam Korpus: %s kelime | Çağ Dilimi: %s | Batch: %s",
                 f"{toplam_kelime:,}", f"{dilim_boyutu:,}", batch_size)
        log.info("=" * 60)

        t_start = time.time()
        tum_ids = [_kelime_id(tokenizer, w) for w in kelimeler]
        log.info("[✅ İNDEKS] %s kelime %.2fs'de belleğe hazırlandı.",
                 f"{toplam_kelime:,}", time.time() - t_start)

        for cag in range(cag_sayisi):
            self.model.train()

            # Korpusta kayan pencere (Sliding Window Index)
            baslangic_idx = (cag * stride) if stride > 0 else 0
            bitis_idx = min(toplam_kelime, baslangic_idx + dilim_boyutu)
            cag_ids = tum_ids[baslangic_idx:bitis_idx]

            dataset = NgramDataset(cag_ids, baglam)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            toplam_loss = 0.0
            adim = 0
            cag_basla = time.time()

            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                toplam_loss += loss.item()
                adim += 1

            self.scheduler.step()
            ortalama_loss = toplam_loss / max(1, adim)
            gecen_sure = time.time() - cag_basla

            ilerleme = int((cag + 1) / cag_sayisi * 20)
            bar = "█" * ilerleme + " " * (20 - ilerleme)
            log.info("Çağ %02d/%02d | Loss: %.4f | Süre: %.1fs | Dilim: [%s-%s] |%s|",
                     cag + 1, cag_sayisi, ortalama_loss, gecen_sure,
                     f"{baslangic_idx:,}", f"{bitis_idx:,}", bar)

        log.info("=" * 60)
        log.info("[🎉 TAMAMLANDI] Akıllı Tüm Korpus Eğitimi Bitti!")
        log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Ön-eğitim (next-token / n-gram)")
    parser.add_argument("--n", type=int, default=1000, help="Model boyutu n (varsayılan 1000)")
    parser.add_argument("--cag", type=int, default=15, help="Çağ (epoch) sayısı")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--korpus", default=korpus_yolu(), help="Korpus dosyası")
    parser.add_argument("--sozluk", default=sozluk_yolu(), help="Sözlük dosyası")
    parser.add_argument("--devam", action="store_true",
                        help="Var olan hiper_model_<n>.pt ağırlığından devam et")
    args = parser.parse_args()

    log_kur()

    if not os.path.exists(args.korpus):
        log.error("Korpus bulunamadı: %s", args.korpus)
        log.error("Önce veri toplayın:  python -m egitim.veri_toplayici --kelime 40000")
        sys.exit(1)

    # Sözlük: sozluk.json varsa yükle (kilitli), yoksa korpusdan kur ve kaydet
    tok = tokenizer_hazirla(8000)

    model = model_olustur(len(tok.sozluk), n=args.n, baglam_penceresi=8)
    yol = model_yolu(args.n)
    if args.devam and os.path.exists(yol):
        agirlik_yukle(model, yol)
        log.info("Mevcut ağırlıktan devam ediliyor: %s", yol)

    log.info("Model: n=%d | sözlük=%d kelime | parametre=%s",
             args.n, len(tok.sozluk), f"{sum(p.numel() for p in model.parameters()):,}")

    with open(args.korpus, "r", encoding="utf-8") as f:
        metin = f.read()
    kelimeler = re.findall(r"\b\w+\b", metin.lower())
    if len(kelimeler) <= 8:
        log.error("Korpus çok küçük (%d kelime); veri toplayıcıyı çalıştırın.", len(kelimeler))
        sys.exit(1)

    motor = KureselEgitimMotoru(model, ogrenme_hizi=args.lr, toplam_cag=args.cag)
    motor.ngram_egitim_dongusu(kelimeler, tok, cag_sayisi=args.cag, batch_size=args.batch)

    torch.save(model.state_dict(), yol)
    log.info("💾 Temel model kaydedildi: %s", yol)


if __name__ == "__main__":
    main()
