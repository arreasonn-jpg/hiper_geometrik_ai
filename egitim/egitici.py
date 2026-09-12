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
import math
import glob
import json
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
from model_config import yukle as model_config_yukle
from bpe_tokenizer import BPETokenizer
try:
    from egitim.scheduler import warmup_cosine_scheduler
except Exception:  # doğrudan script çalıştırma
    from scheduler import warmup_cosine_scheduler


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


@dataclass
class GradientRaporu:
    """Bir eğitim adımındaki gradient sağlığı özeti."""

    toplam_norm: float
    max_katman: str = ""
    max_katman_norm: float = 0.0
    sonlu: bool = True
    katman_normlari: Dict[str, float] = field(default_factory=dict)


@dataclass
class EgitimCagRaporu:
    cag: int
    train_loss: float
    val_loss: Optional[float] = None
    perplexity: Optional[float] = None
    lr: float = 0.0
    grad_norm: float = 0.0
    max_grad_katman: str = ""
    max_grad_norm: float = 0.0


class KureselEgitimMotoru:
    def __init__(self, model, ogrenme_hizi=1e-3, toplam_cag=15,
                 karisik_hassasiyet="auto", grad_clip: float = 1.0,
                 warmup_cag: int = 0, lr_min_factor: float = 0.01,
                 log_dizini: Optional[str] = None,
                 checkpoint_dizini: Optional[str] = None,
                 son_checkpoint_sayisi: int = 3):
        """
        karisik_hassasiyet:
            "auto" → CUDA + bf16 destekliyse aç (rapor 8.4.1)
            True   → CUDA'da zorla aç
            False  → kapalı

        Sağlamlaştırma:
            grad_clip             → gradient norm clipping üst sınırı
            warmup_cag/lr_min_factor → warmup + cosine decay LR takvimi
            log_dizini            → metrics.csv + varsa TensorBoard logları
            checkpoint_dizini     → best/latest/epoch checkpoint yönetimi
            son_checkpoint_sayisi → best/latest hariç tutulacak son N epoch
        """
        self.model = model
        self.aygit = next(model.parameters()).device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=ogrenme_hizi,
                                           weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.scheduler = warmup_cosine_scheduler(
            self.optimizer,
            total_steps=max(1, toplam_cag),
            warmup_steps=warmup_cag,
            min_factor=lr_min_factor,
        )
        self.karisik_hassasiyet = self._amp_karar(karisik_hassasiyet)
        self.grad_clip = float(grad_clip)
        self.gradient_gecmisi: List[GradientRaporu] = []
        self.cag_gecmisi: List[EgitimCagRaporu] = []
        self.log_dizini = log_dizini
        self.checkpoint_dizini = checkpoint_dizini
        self.son_checkpoint_sayisi = int(son_checkpoint_sayisi)
        self.baslangic_cag = 0
        self._tb_writer = None
        if self.log_dizini:
            os.makedirs(self.log_dizini, exist_ok=True)
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(self.log_dizini)
            except Exception:
                self._tb_writer = None
            self._csv_baslat()

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

    @staticmethod
    def _sonlu_mu(t: torch.Tensor) -> bool:
        return bool(torch.isfinite(t).all().item())

    def _loss_kontrol(self, loss: torch.Tensor, logits: torch.Tensor):
        """AMP açık/kapalı NaN/Inf üretimini anında yakala."""
        if not self._sonlu_mu(logits):
            raise FloatingPointError("Model logitleri NaN/Inf içeriyor (AMP/geometrik katman kararsızlığı)")
        if not self._sonlu_mu(loss):
            raise FloatingPointError(f"Loss NaN/Inf: {loss.item()}")

    def _gradient_raporu(self, toplam_norm: float) -> GradientRaporu:
        normlar: Dict[str, float] = {}
        sonlu = True
        max_ad, max_norm = "", 0.0
        for ad, p in self.model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            if not self._sonlu_mu(g):
                sonlu = False
            n = float(g.norm(2).item())
            normlar[ad] = n
            if n > max_norm:
                max_ad, max_norm = ad, n
        rapor = GradientRaporu(
            toplam_norm=float(toplam_norm), max_katman=max_ad,
            max_katman_norm=max_norm, sonlu=sonlu, katman_normlari=normlar)
        self.gradient_gecmisi.append(rapor)
        if not sonlu:
            raise FloatingPointError("Gradient NaN/Inf içeriyor; eğitim adımı durduruldu")
        return rapor

    @staticmethod
    def train_val_bol(token_ids: List[int], baglam: int,
                      validation_split: float = 0.0) -> Tuple[List[int], Optional[List[int]]]:
        """Data leakage'i azaltan deterministik train/validation ayrımı.

        Kayan pencere LM'de train'in son penceresi ile validation'ın ilk
        penceresi arasında ``baglam`` tokenlık güvenli boşluk bırakılır. Veri
        küçükse validation devre dışı kalır.
        """
        ids = [int(i) for i in token_ids]
        if validation_split <= 0.0:
            return ids, None
        if not 0.0 < validation_split < 0.5:
            raise ValueError("validation_split 0 ile 0.5 arasında olmalı")
        min_len = baglam + 2
        if len(ids) < 3 * min_len:
            return ids, None
        val_len = max(min_len, int(len(ids) * validation_split))
        val_start = len(ids) - val_len
        train_end = max(0, val_start - baglam)  # boundary leakage boşluğu
        if train_end < min_len or len(ids[val_start:]) < min_len:
            return ids, None
        return ids[:train_end], ids[val_start:]

    @torch.no_grad()
    def degerlendir_loss(self, token_ids: List[int], batch_size: int = 64) -> Tuple[float, float]:
        """Validation loss + perplexity ölçümü."""
        dataset = NgramDataset(token_ids, self.model.baglam_penceresi)
        if len(dataset) == 0:
            return float("nan"), float("nan")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        onceki_train = self.model.training
        self.model.eval()
        toplam, adim = 0.0, 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.aygit)
            batch_y = batch_y.to(self.aygit)
            with self._autocast():
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
            self._loss_kontrol(loss, logits)
            toplam += float(loss.item())
            adim += 1
        if onceki_train:
            self.model.train()
        ort = toplam / max(1, adim)
        return ort, math.exp(min(50.0, ort))

    def _csv_baslat(self):
        yol = os.path.join(self.log_dizini, "metrics.csv")
        if not os.path.exists(yol):
            with open(yol, "w", encoding="utf-8") as f:
                f.write("cag,train_loss,val_loss,perplexity,lr,grad_norm,max_grad_katman,max_grad_norm\n")

    def _metrik_logla(self, rapor: EgitimCagRaporu):
        if self.log_dizini:
            yol = os.path.join(self.log_dizini, "metrics.csv")
            with open(yol, "a", encoding="utf-8") as f:
                f.write(
                    f"{rapor.cag},{rapor.train_loss},{rapor.val_loss},{rapor.perplexity},"
                    f"{rapor.lr},{rapor.grad_norm},{rapor.max_grad_katman},{rapor.max_grad_norm}\n")
            jsonl = os.path.join(self.log_dizini, "metrics.jsonl")
            with open(jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps({"run": "temel", **rapor.__dict__}, ensure_ascii=False) + "\n")
        if self._tb_writer is not None:
            self._tb_writer.add_scalar("loss/train", rapor.train_loss, rapor.cag)
            if rapor.val_loss is not None:
                self._tb_writer.add_scalar("loss/validation", rapor.val_loss, rapor.cag)
            if rapor.perplexity is not None:
                self._tb_writer.add_scalar("eval/perplexity", rapor.perplexity, rapor.cag)
            self._tb_writer.add_scalar("train/lr", rapor.lr, rapor.cag)
            self._tb_writer.add_scalar("grad/total_norm", rapor.grad_norm, rapor.cag)
            self._tb_writer.add_scalar("grad/max_layer_norm", rapor.max_grad_norm, rapor.cag)

    def _checkpoint_meta(self) -> Dict[str, object]:
        """Checkpoint uyumluluğu için küçük, insan okunur mimari özeti."""
        return {
            "baglam_penceresi": int(getattr(self.model, "baglam_penceresi", 0)),
            "sozluk_boyutu": int(getattr(self.model, "sozluk_boyutu", 0)),
            "n": int(getattr(self.model, "n", 0)),
            "katman_sayisi": int(getattr(getattr(self.model, "kuresel_bag", None), "katman_sayisi",
                                         getattr(getattr(self.model, "zincir", None), "katman_sayisi", 0))),
            "seyrek_var": bool(getattr(self.model, "seyrek_tablo", None) is not None),
        }

    def _checkpoint_kaydet(self, ad: str, cag: int, train_loss: float,
                           val_loss: Optional[float] = None):
        if not self.checkpoint_dizini:
            return None
        os.makedirs(self.checkpoint_dizini, exist_ok=True)
        yol = os.path.join(self.checkpoint_dizini, ad)
        torch.save({
            "checkpoint_version": "hga-train-v1",
            "model_meta": self._checkpoint_meta(),
            "cag": cag,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, yol)
        return yol

    def _eski_checkpoint_temizle(self):
        if not self.checkpoint_dizini or self.son_checkpoint_sayisi <= 0:
            return
        yollar = sorted(glob.glob(os.path.join(self.checkpoint_dizini, "epoch_*.pt")),
                       key=os.path.getmtime)
        for yol in yollar[:-self.son_checkpoint_sayisi]:
            try:
                os.remove(yol)
            except OSError:
                pass

    def checkpoint_yukle(self, yol: str, strict: bool = True) -> int:
        """Eğitime checkpoint'ten devam et; dönen değer kayıtlı çağ numarasıdır."""
        ckpt = torch.load(yol, map_location=self.aygit)
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"], strict=strict)
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.baslangic_cag = int(ckpt.get("cag", 0))
            return self.baslangic_cag
        # Geriye dönük uyumluluk: salt model state_dict dosyası.
        self.model.load_state_dict(ckpt, strict=strict)
        self.baslangic_cag = 0
        return 0

    def ngram_egitim_dongusu(self, veri, tokenizer=None, cag_sayisi=15,
                             batch_size=64, validation_split: float = 0.0,
                             early_stopping_patience: Optional[int] = None,
                             early_stopping_min_delta: float = 0.0):
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

        egitim_ids, val_ids = self.train_val_bol(tum_ids, baglam, validation_split)
        egitim_token = len(egitim_ids)
        val_token = len(val_ids) if val_ids is not None else 0

        # Çağ başı dilim: RAM dostu kayan pencere (önceki sürümden)
        DILIM_BOYUTU = min(60000, egitim_token)
        hedef_cag = int(cag_sayisi)
        baslangic_cag = max(0, min(int(self.baslangic_cag), hedef_cag))
        kalan_cag = max(0, hedef_cag - baslangic_cag)
        stride = (max(1, (egitim_token - DILIM_BOYUTU) // max(1, hedef_cag - 1))
                  if egitim_token > DILIM_BOYUTU else 0)

        print("\n" + "═" * 60)
        print("   🚀 KAYAN PENCERELİ EĞİTİM MOTORU (BPE + bilinear Kronecker zinciri)")
        print(f"   📊 Korpus: {toplam_token:,} token | Train: {egitim_token:,} | "
              f"Val: {val_token:,} | Dilim: {DILIM_BOYUTU:,} | Batch: {batch_size} | "
              f"AMP: {self.karisik_hassasiyet} | GradClip: {self.grad_clip} | Aygıt: {self.aygit}")
        if baslangic_cag:
            print(f"   ↩️ Resume: checkpoint çağ {baslangic_cag}; hedef toplam çağ {hedef_cag}")
        print("═" * 60)

        if kalan_cag == 0:
            print("   ✅ Checkpoint zaten hedef çağ sayısına ulaşmış; eğitim atlandı.")
            return

        en_iyi_val = float("inf")
        sabir_sayac = 0
        for cag in range(baslangic_cag, hedef_cag):
            self.model.train()
            baslangic = (cag * stride) if stride > 0 else 0
            bitis = min(egitim_token, baslangic + DILIM_BOYUTU)
            dataset = NgramDataset(egitim_ids[baslangic:bitis], baglam)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                drop_last=len(dataset) >= batch_size)

            toplam_loss, adim, cag_basla = 0.0, 0, time.time()
            son_grad = GradientRaporu(toplam_norm=0.0)
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.aygit)
                batch_y = batch_y.to(self.aygit)
                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    logits = self.model(batch_x)
                    loss = self.criterion(logits, batch_y)
                self._loss_kontrol(loss, logits)
                loss.backward()
                try:
                    toplam_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip,
                        error_if_nonfinite=True)
                except TypeError:  # eski torch sürümlerinde argüman yok
                    toplam_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip)
                son_grad = self._gradient_raporu(float(toplam_norm))
                self.optimizer.step()
                toplam_loss += float(loss.item())
                adim += 1

            self.scheduler.step()
            ortalama = toplam_loss / max(1, adim)
            val_loss = ppl = None
            if val_ids is not None:
                val_loss, ppl = self.degerlendir_loss(val_ids, batch_size=batch_size)

            lr = float(self.optimizer.param_groups[0]["lr"])
            rapor = EgitimCagRaporu(
                cag=cag + 1, train_loss=ortalama, val_loss=val_loss,
                perplexity=ppl, lr=lr, grad_norm=son_grad.toplam_norm,
                max_grad_katman=son_grad.max_katman,
                max_grad_norm=son_grad.max_katman_norm)
            self.cag_gecmisi.append(rapor)
            self._metrik_logla(rapor)

            # Checkpoint yönetimi: latest + epoch_N + validation varsa best.
            self._checkpoint_kaydet("latest.pt", cag + 1, ortalama, val_loss)
            self._checkpoint_kaydet(f"epoch_{cag + 1:04d}.pt", cag + 1, ortalama, val_loss)
            if val_loss is not None and val_loss + early_stopping_min_delta < en_iyi_val:
                en_iyi_val = val_loss
                sabir_sayac = 0
                self._checkpoint_kaydet("best.pt", cag + 1, ortalama, val_loss)
            elif val_loss is not None and early_stopping_patience is not None:
                sabir_sayac += 1
            self._eski_checkpoint_temizle()

            bar = "█" * int((cag + 1) / max(1, hedef_cag) * 20)
            # Seyrek 'boş küme' doluluk + collision izleme (rapor 9.4.4)
            seyrek_bilgi = ""
            tablo = getattr(self.model, "seyrek_tablo", None)
            if tablo is not None:
                dolu, toplam_satir = tablo.doluluk_orani()
                seyrek_bilgi = f" | Seyrek: {dolu:,}/{toplam_satir:,}"
                try:
                    # Çağ başı ucuz örneklem: ilk 1024 pencere üzerinde collision oranı.
                    sinir = min(len(dataset), 1024)
                    if sinir > 0:
                        pencereler = [egitim_ids[baslangic + i: baslangic + i + baglam]
                                      for i in range(sinir)]
                        carpisma = tablo.carpisma_istatistigi(torch.tensor(pencereler, dtype=torch.long))
                        if carpisma.get("uyari"):
                            seyrek_bilgi += f" | ⚠️ collision %{100 * carpisma['collision_rate']:.2f}"
                except Exception:
                    pass
            val_bilgi = "" if val_loss is None else f" | Val: {val_loss:.4f} | PPL: {ppl:.2f}"
            print(f"  Çağ {cag + 1:02d}/{cag_sayisi:02d} | Loss: {ortalama:.4f}{val_bilgi} | "
                  f"LR: {lr:.2e} | GradNorm: {son_grad.toplam_norm:.3f} "
                  f"(max {son_grad.max_katman}:{son_grad.max_katman_norm:.3f}) | "
                  f"Süre: {time.time() - cag_basla:.1f}s | "
                  f"Dilim: [{baslangic:,}-{bitis:,}]{seyrek_bilgi} |{bar:<20}|")

            if (val_loss is not None and early_stopping_patience is not None and
                    sabir_sayac >= early_stopping_patience):
                print(f"  ⏹️ Early stopping: {early_stopping_patience} çağ boyunca "
                      "validation iyileşmedi.")
                break

        if self._tb_writer is not None:
            self._tb_writer.flush()
        print("═" * 60)
        print("  [🎉 TAMAMLANDI] Eğitim bitti.")
        print("═" * 60 + "\n")


def main():
    ap = argparse.ArgumentParser(description="Hiper-Geometrik AI temel eğitimi")
    ap.add_argument("--config", default=None,
                    help="model/eğitim YAML config yolu (varsayılan: hga/config/model_config.yaml)")
    ap.add_argument("--korpus", default=os.path.join(KOK, "turkce_metin.txt"))
    ap.add_argument("--cag", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--ogrenme-hizi", type=float, default=None)
    ap.add_argument("--n", type=int, default=None,
                    help=f"küresel bağ boyutu (varsayılan config: {VARSAYILAN_N})")
    ap.add_argument("--katman", type=int, default=None,
                    help=f"bilinear katman sayısı K (varsayılan config: {VARSAYILAN_KATMAN})")
    ap.add_argument("--baglam", type=int, default=None,
                    help=f"bağlam penceresi (varsayılan config: {VARSAYILAN_BAGLAM})")
    ap.add_argument("--checkpoint", action="store_true",
                    help="Gradient checkpointing (derin zincir, rapor 8.4.2)")
    ap.add_argument("--seyrek-satir", type=int, default=None,
                    help=f"seyrek 'boş küme' tablosu satır sayısı (varsayılan config: {VARSAYILAN_SEYREK_SATIR})")
    ap.add_argument("--seyrek-boyut", type=int, default=None,
                    help=f"her kümenin vektör boyutu (varsayılan config: {VARSAYILAN_SEYREK_BOYUT})")
    ap.add_argument("--seyrek-yok", action="store_true",
                    help="seyrek belleği tamamen kapat")
    ap.add_argument("--amp", default=None, choices=["auto", "evet", "hayir"])
    ap.add_argument("--grad-clip", type=float, default=None,
                    help="gradient norm clipping üst sınırı")
    ap.add_argument("--validation-split", type=float, default=None,
                    help="validation oranı (0 kapalı; train/val arasında bağlam kadar boşluk bırakır)")
    ap.add_argument("--early-stopping-patience", type=int, default=None,
                    help="validation iyileşmezse kaç çağ sonra durulsun")
    ap.add_argument("--early-stopping-min-delta", type=float, default=None,
                    help="early stopping için minimum validation iyileşmesi")
    ap.add_argument("--warmup-cag", type=int, default=None,
                    help="LR warmup çağ sayısı")
    ap.add_argument("--lr-min-factor", type=float, default=None,
                    help="cosine decay sonunda öğrenme hızı katsayısı")
    ap.add_argument("--log-dizini", default=None,
                    help="metrics.csv ve varsa TensorBoard log dizini")
    ap.add_argument("--checkpoint-dizini", default=None,
                    help="best.pt/latest.pt/epoch_N checkpoint dizini")
    ap.add_argument("--resume", default=None,
                    help="eğitime devam edilecek checkpoint yolu (best/latest/epoch veya salt state_dict)")
    ap.add_argument("--son-checkpoint", type=int, default=None,
                    help="best/latest hariç tutulacak son epoch checkpoint sayısı")
    args = ap.parse_args()

    cfg = model_config_yukle(args.config)
    mcfg, tcfg = cfg["model"], cfg["training"]
    cag = args.cag if args.cag is not None else tcfg.toplam_cag
    batch = args.batch if args.batch is not None else tcfg.batch_size
    ogrenme_hizi = args.ogrenme_hizi if args.ogrenme_hizi is not None else tcfg.ogrenme_hizi
    n = args.n if args.n is not None else mcfg.n
    katman = args.katman if args.katman is not None else mcfg.katman_sayisi
    baglam = args.baglam if args.baglam is not None else mcfg.baglam_penceresi
    seyrek_satir = args.seyrek_satir if args.seyrek_satir is not None else mcfg.seyrek_tablo_boyutu
    seyrek_boyut = args.seyrek_boyut if args.seyrek_boyut is not None else mcfg.seyrek_boyut
    amp = args.amp if args.amp is not None else tcfg.amp
    grad_clip = args.grad_clip if args.grad_clip is not None else tcfg.grad_clip
    validation_split = (args.validation_split if args.validation_split is not None
                        else tcfg.validation_split)
    early_stopping_patience = (args.early_stopping_patience if args.early_stopping_patience is not None
                               else tcfg.early_stopping_patience)
    early_stopping_min_delta = (args.early_stopping_min_delta if args.early_stopping_min_delta is not None
                                else tcfg.early_stopping_min_delta)
    warmup_cag = args.warmup_cag if args.warmup_cag is not None else tcfg.warmup_cag
    lr_min_factor = args.lr_min_factor if args.lr_min_factor is not None else tcfg.lr_min_factor
    log_dizini = args.log_dizini if args.log_dizini is not None else tcfg.log_dizini
    checkpoint_dizini = (args.checkpoint_dizini if args.checkpoint_dizini is not None
                         else tcfg.checkpoint_dizini)
    son_checkpoint = (args.son_checkpoint if args.son_checkpoint is not None
                      else tcfg.son_checkpoint_sayisi)

    if not os.path.exists(args.korpus):
        print(f"❌ Korpus bulunamadı: {args.korpus}")
        print("   Önce egitim/veri_toplayici.py ile korpus toplayın (ör. geniş")
        print("   Wikipedia korpusu: OtomatikVeriToplayici.genis_korpus_cek)")
        print("   veya turkce_metin.txt dosyasını elle oluşturun.")
        sys.exit(1)

    with open(args.korpus, "r", encoding="utf-8") as f:
        metin = f.read()

    # BPE sözlüğü: kilitliyse yükle, yoksa kur (rapor 8.4.6)
    tok = BPETokenizer(baglam_penceresi=baglam, max_vocab_size=mcfg.sozluk_boyutu)
    sozluk_yolu = os.path.join(KOK, "bpe_sozluk.json")
    if os.path.exists(sozluk_yolu):
        tok.yukle(sozluk_yolu)
        print(f"🔒 Kilitli BPE sözlüğü yüklendi: {tok.sozluk_boyutu} parça")
    else:
        tok.fit_on_text(metin)
        tok.kaydet(sozluk_yolu)
        print(f"🔧 BPE sözlüğü kuruldu: {tok.sozluk_boyutu} parça → bpe_sozluk.json")
    tok.vocab_tutarliligi(strict=True)

    model_vocab = max(len(tok.sozluk), 64)
    tok.vocab_tutarliligi(embedding_boyutu=model_vocab, strict=True)
    model = model_olustur(sozluk_boyutu=model_vocab, n=n,
                          baglam_penceresi=baglam,
                          emb_dim=mcfg.emb_dim,
                          num_heads=mcfg.num_heads,
                          katman_sayisi=katman,
                          dropout=mcfg.dropout,
                          encoder_aktivasyon=mcfg.encoder_aktivasyon,
                          zincir_aktivasyon=mcfg.zincir_aktivasyon,
                          checkpoint_kullan=bool(args.checkpoint or mcfg.checkpoint_kullan),
                          seyrek_tablo_boyutu=(0 if args.seyrek_yok else seyrek_satir),
                          seyrek_boyut=seyrek_boyut,
                          seyrek_tablo_sayisi=mcfg.seyrek_tablo_sayisi,
                          seyrek_erisim_izleme=mcfg.seyrek_erisim_izleme)

    amp_tercihi = {"evet": True, "hayir": False, "auto": "auto"}[amp]
    motor = KureselEgitimMotoru(
        model, ogrenme_hizi=ogrenme_hizi, toplam_cag=cag,
        karisik_hassasiyet=amp_tercihi,
        grad_clip=grad_clip,
        warmup_cag=warmup_cag,
        lr_min_factor=lr_min_factor,
        log_dizini=log_dizini,
        checkpoint_dizini=checkpoint_dizini,
        son_checkpoint_sayisi=son_checkpoint)
    if args.resume:
        baslanan = motor.checkpoint_yukle(args.resume)
        print(f"↩️ Checkpoint yüklendi: {args.resume} (cag={baslanan})")
    motor.ngram_egitim_dongusu(
        tok.encode(metin), cag_sayisi=cag, batch_size=batch,
        validation_split=validation_split,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta)

    cikis = os.path.join(KOK, f"hiper_model_{n}.pt")
    agirlik_kaydet(model, cikis)
    print(f"💾 Model kaydedildi: {cikis}")


if __name__ == "__main__":
    main()
