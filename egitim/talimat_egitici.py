# -*- coding: utf-8 -*-
"""
Talimat (Instruction) Fine-Tuning Motoru
========================================

Temel eğitim motorundaki sağlamlaştırma kararları burada da uygulanır:
  * merkezi ``model_config.yaml`` desteği,
  * tokenizer vocab/special-token doğrulaması,
  * loss/logit/gradient NaN/Inf kontrolü,
  * gradient clipping, scheduler, validation/perplexity, early stopping,
  * best/latest/epoch checkpoint ve resume desteği.

Torch kurulu değilse modül import edilebilir; gerçek eğitim çağrısında açık
ImportError verilir. Bu sayede helper fonksiyonlar hafif testlerde çalışır.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [KOK, os.path.join(KOK, "mimari"), os.path.dirname(__file__)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    from torch.utils.data import DataLoader, TensorDataset  # type: ignore
except Exception:  # pragma: no cover - torch'suz CI/sandbox
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore

from model_config import VARSAYILAN_MODEL_CONFIG
from model_config import yukle as model_config_yukle

try:
    from kuresel_model import (
        VARSAYILAN_KATMAN,
        VARSAYILAN_N,
        VARSAYILAN_SEYREK_BOYUT,
        VARSAYILAN_SEYREK_SATIR,
        agirlik_kaydet,
        agirlik_yukle,
        model_olustur,
    )
except Exception:  # torch yoksa kuresel_model import edilemez; helper'lar yine çalışsın
    model_olustur = agirlik_yukle = agirlik_kaydet = None  # type: ignore
    VARSAYILAN_N = VARSAYILAN_MODEL_CONFIG.n
    VARSAYILAN_KATMAN = VARSAYILAN_MODEL_CONFIG.katman_sayisi
    VARSAYILAN_SEYREK_SATIR = VARSAYILAN_MODEL_CONFIG.seyrek_tablo_boyutu
    VARSAYILAN_SEYREK_BOYUT = VARSAYILAN_MODEL_CONFIG.seyrek_boyut
from bpe_tokenizer import BPETokenizer
from talimat_toplayici import TalimatToplayici

try:
    from egitim.scheduler import warmup_cosine_scheduler
except Exception:
    from scheduler import warmup_cosine_scheduler  # type: ignore[no-redef]


def _torch_gerekli():
    if torch is None:
        raise ImportError("Talimat fine-tuning için torch gerekli. "
                          "Kurulum: pip install -r requirements-lock.txt")


def bpe_tokenizer_hazirla(talimat_metni: str, baglam: int,
                          max_vocab_size: int = 8000) -> BPETokenizer:
    """BPE sözlüğü: diskte varsa KİLİTLİ olarak yükle, yoksa kur ve kaydet."""
    tok = BPETokenizer(baglam_penceresi=baglam, max_vocab_size=max_vocab_size)
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
    tok.vocab_tutarliligi(strict=True)
    return tok


def talimat_ornekleri_olustur(tokenizer: BPETokenizer,
                              talimatlar: Sequence[Dict[str, str]],
                              baglam: int) -> Tuple[List[List[int]], List[int]]:
    """'soru ... cevap ... son' biçiminden supervised next-token örnekleri üret.

    Yalnız ``cevap`` etiketinden sonraki tokenlar hedeflenir; böylece model
    soruyu ezberlemek yerine cevap bölümünü tamamlamaya eğitilir.
    """
    X: List[List[int]] = []
    Y: List[int] = []
    cevap_ids = tokenizer.encode("cevap")
    isaret = cevap_ids[0] if cevap_ids else -1
    for it in talimatlar:
        soru = str(it.get("soru", "")).strip()
        cevap = str(it.get("cevap", "")).strip()
        if not soru or not cevap:
            continue
        ids = tokenizer.encode(f"soru {soru} cevap {cevap} son")
        for i in range(1, len(ids)):
            if isaret not in ids[:i]:
                continue
            x = ids[max(0, i - baglam):i]
            x = [tokenizer.PAD_ID] * (baglam - len(x)) + x
            X.append(x)
            Y.append(ids[i])
    return X, Y


def train_val_bol(X: List[List[int]], Y: List[int], validation_split: float = 0.0):
    """Talimat örneklerini deterministik train/validation olarak ayır."""
    if validation_split <= 0.0 or len(X) < 4:
        return X, Y, None, None
    if not 0.0 < validation_split < 0.5:
        raise ValueError("validation_split 0 ile 0.5 arasında olmalı")
    val_n = max(1, int(len(X) * validation_split))
    if len(X) - val_n < 1:
        return X, Y, None, None
    return X[:-val_n], Y[:-val_n], X[-val_n:], Y[-val_n:]


def _autocast(aygit, amp_acik: bool):
    if amp_acik:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    import contextlib
    return contextlib.nullcontext()


def _sonlu(t) -> bool:
    return bool(torch.isfinite(t).all().item())


def _loss_kontrol(loss, logits):
    if not _sonlu(logits):
        raise FloatingPointError("Talimat logitleri NaN/Inf içeriyor")
    if not _sonlu(loss):
        raise FloatingPointError("Talimat loss NaN/Inf")


def _grad_ozet(model) -> Tuple[str, float, bool]:
    max_ad, max_norm, sonlu = "", 0.0, True
    for ad, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not _sonlu(g):
            sonlu = False
        n = float(g.norm(2).item())
        if n > max_norm:
            max_ad, max_norm = ad, n
    return max_ad, max_norm, sonlu


@torch.no_grad() if torch is not None else (lambda f: f)
def _degerlendir(model, X, Y, batch: int, loss_fn, aygit, amp_acik: bool):
    if X is None or Y is None or len(X) == 0:
        return None, None
    dataset = TensorDataset(torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch, shuffle=False)
    onceki = model.training
    model.eval()
    toplam, adim = 0.0, 0
    for bx, by in loader:
        bx, by = bx.to(aygit), by.to(aygit)
        with _autocast(aygit, amp_acik):
            logits = model(bx)
            loss = loss_fn(logits, by)
        _loss_kontrol(loss, logits)
        toplam += float(loss.item())
        adim += 1
    if onceki:
        model.train()
    ort = toplam / max(1, adim)
    return ort, math.exp(min(50.0, ort))


def _checkpoint_meta(model) -> Dict[str, object]:
    return {
        "baglam_penceresi": int(getattr(model, "baglam_penceresi", 0)),
        "sozluk_boyutu": int(getattr(model, "sozluk_boyutu", 0)),
        "n": int(getattr(model, "n", 0)),
        "katman_sayisi": int(getattr(getattr(model, "kuresel_bag", None), "katman_sayisi",
                                     getattr(getattr(model, "zincir", None), "katman_sayisi", 0))),
        "seyrek_var": bool(getattr(model, "seyrek_tablo", None) is not None),
    }


def _checkpoint_kaydet(dizin: Optional[str], ad: str, model, opt, scheduler,
                       ep: int, train_loss: float, val_loss: Optional[float]):
    if not dizin:
        return
    os.makedirs(dizin, exist_ok=True)
    torch.save({
        "checkpoint_version": "hga-instruction-v1",
        "model_meta": _checkpoint_meta(model),
        "epoch": ep,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }, os.path.join(dizin, ad))


def _eski_checkpoint_temizle(dizin: Optional[str], son_n: int):
    if not dizin or son_n <= 0:
        return
    yollar = sorted(glob.glob(os.path.join(dizin, "epoch_*.pt")), key=os.path.getmtime)
    for yol in yollar[:-son_n]:
        try:
            os.remove(yol)
        except OSError:
            pass


def _csv_logla(log_dizini: Optional[str], satir: Dict):
    if not log_dizini:
        return
    os.makedirs(log_dizini, exist_ok=True)
    yol = os.path.join(log_dizini, "talimat_metrics.csv")
    yeni = not os.path.exists(yol)
    alanlar = ["epoch", "train_loss", "val_loss", "perplexity", "lr", "grad_norm", "max_grad_katman", "max_grad_norm"]
    with open(yol, "a", encoding="utf-8") as f:
        if yeni:
            f.write(",".join(alanlar) + "\n")
        f.write(",".join(str(satir.get(a, "")) for a in alanlar) + "\n")
    with open(os.path.join(log_dizini, "talimat_metrics.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"run": "talimat", **satir}, ensure_ascii=False) + "\n")


def _resume_yukle(yol: str, model, opt, scheduler, strict: bool = True) -> int:
    ckpt = torch.load(yol, map_location=next(model.parameters()).device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=strict)
        if "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return int(ckpt.get("epoch", ckpt.get("cag", 0)))
    model.load_state_dict(ckpt, strict=strict)
    return 0


def main():
    _torch_gerekli()
    if model_olustur is None or agirlik_kaydet is None:
        raise ImportError("mimari/kuresel_model.py yüklenemedi; torch kurulumunu kontrol edin")
    ap = argparse.ArgumentParser(description="Hiper-Geometrik AI talimat fine-tuning")
    ap.add_argument("--config", default=None,
                    help="model/eğitim YAML config yolu (varsayılan: hga/config/model_config.yaml)")
    ap.add_argument("--cag", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--ogrenme-hizi", type=float, default=None)
    ap.add_argument("--label-smoothing", type=float, default=0.02)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--katman", type=int, default=None)
    ap.add_argument("--baglam", type=int, default=None)
    ap.add_argument("--seyrek-satir", type=int, default=None)
    ap.add_argument("--seyrek-boyut", type=int, default=None)
    ap.add_argument("--seyrek-yok", action="store_true")
    ap.add_argument("--amp", default=None, choices=["auto", "evet", "hayir"])
    ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument("--validation-split", type=float, default=None)
    ap.add_argument("--early-stopping-patience", type=int, default=None)
    ap.add_argument("--early-stopping-min-delta", type=float, default=None)
    ap.add_argument("--warmup-cag", type=int, default=None)
    ap.add_argument("--lr-min-factor", type=float, default=None)
    ap.add_argument("--log-dizini", default=None)
    ap.add_argument("--checkpoint-dizini", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--son-checkpoint", type=int, default=None)
    args = ap.parse_args()

    cfg = model_config_yukle(args.config)
    mcfg, tcfg = cfg["model"], cfg["training"]
    cag = args.cag if args.cag is not None else tcfg.toplam_cag
    batch = args.batch if args.batch is not None else min(16, tcfg.batch_size)
    lr = args.ogrenme_hizi if args.ogrenme_hizi is not None else tcfg.ogrenme_hizi
    n = args.n if args.n is not None else mcfg.n
    katman = args.katman if args.katman is not None else mcfg.katman_sayisi
    baglam = args.baglam if args.baglam is not None else mcfg.baglam_penceresi
    seyrek_satir = args.seyrek_satir if args.seyrek_satir is not None else mcfg.seyrek_tablo_boyutu
    seyrek_boyut = args.seyrek_boyut if args.seyrek_boyut is not None else mcfg.seyrek_boyut
    amp = args.amp if args.amp is not None else tcfg.amp
    grad_clip = args.grad_clip if args.grad_clip is not None else tcfg.grad_clip
    val_split = args.validation_split if args.validation_split is not None else tcfg.validation_split
    patience = (args.early_stopping_patience if args.early_stopping_patience is not None
                else tcfg.early_stopping_patience)
    min_delta = (args.early_stopping_min_delta if args.early_stopping_min_delta is not None
                 else tcfg.early_stopping_min_delta)
    warmup_cag = args.warmup_cag if args.warmup_cag is not None else tcfg.warmup_cag
    lr_min_factor = args.lr_min_factor if args.lr_min_factor is not None else tcfg.lr_min_factor
    log_dizini = args.log_dizini if args.log_dizini is not None else tcfg.log_dizini
    ckpt_dizini = args.checkpoint_dizini if args.checkpoint_dizini is not None else tcfg.checkpoint_dizini
    son_ckpt = args.son_checkpoint if args.son_checkpoint is not None else tcfg.son_checkpoint_sayisi

    print("\n🎯 Talimat (Instruction) Fine-Tuning — Kronecker zinciri sürümü")
    tt = TalimatToplayici(os.path.join(KOK, "talimat_verisi.json"))
    talimatlar = tt.hazirla_veya_yukle()

    talimat_metni = " ".join(f"{it['soru']} {it['cevap']}" for it in talimatlar)
    tok = bpe_tokenizer_hazirla(talimat_metni, baglam=baglam,
                                max_vocab_size=mcfg.sozluk_boyutu)
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
                          checkpoint_kullan=mcfg.checkpoint_kullan,
                          seyrek_tablo_boyutu=(0 if args.seyrek_yok else seyrek_satir),
                          seyrek_boyut=seyrek_boyut,
                          seyrek_tablo_sayisi=mcfg.seyrek_tablo_sayisi,
                          seyrek_erisim_izleme=mcfg.seyrek_erisim_izleme)

    aygit = next(model.parameters()).device
    amp_acik = (aygit.type == "cuda" and (amp == "evet" or
                (amp == "auto" and torch.cuda.is_available() and torch.cuda.is_bf16_supported())))

    temel_yol = os.path.join(KOK, f"hiper_model_{n}.pt")
    if os.path.exists(temel_yol):
        try:
            agirlik_yukle(model, temel_yol, strict=True)
            print("✅ Temel ağırlık strict=True ile yüklendi")
        except Exception as e:
            print(f"⚠️ Temel ağırlık yüklenemedi (mimari uyumsuz olabilir) — "
                  f"rastgele ağırlıkla devam ediliyor: {e}")

    X, Y = talimat_ornekleri_olustur(tok, talimatlar, baglam)
    if not X:
        print("❌ Eğitim örneği üretilemedi (sözlük çok küçük olabilir).")
        sys.exit(1)
    Xtr, Ytr, Xval, Yval = train_val_bol(X, Y, validation_split=val_split)
    print(f"✨ Eğitim örneği: train={len(Xtr):,}, val={0 if Xval is None else len(Xval):,} "
          f"(bağlam={baglam}, AMP={amp_acik}, grad_clip={grad_clip})")

    dataset = TensorDataset(torch.tensor(Xtr, dtype=torch.long), torch.tensor(Ytr, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = warmup_cosine_scheduler(
        opt, total_steps=max(1, cag), warmup_steps=warmup_cag,
        min_factor=lr_min_factor,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing), ignore_index=tok.PAD_ID)

    baslanan = 0
    if args.resume:
        baslanan = _resume_yukle(args.resume, model, opt, scheduler)
        print(f"↩️ Checkpoint yüklendi: {args.resume} (epoch={baslanan}; hedef={cag})")

    if baslanan >= cag:
        print("✅ Checkpoint zaten hedef epoch sayısına ulaşmış; eğitim atlandı.")
    model.train()
    en_iyi_val = float("inf")
    sabir = 0
    for ep in range(baslanan + 1, cag + 1):
        tl, adim, basla = 0.0, 0, time.time()
        son_grad_norm, max_ad, max_norm = 0.0, "", 0.0
        for bx, by in loader:
            bx, by = bx.to(aygit), by.to(aygit)
            opt.zero_grad(set_to_none=True)
            with _autocast(aygit, amp_acik):
                logits = model(bx)
                loss = loss_fn(logits, by)
            _loss_kontrol(loss, logits)
            loss.backward()
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip,
                                                            error_if_nonfinite=True)
            except TypeError:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            max_ad, max_norm, grad_sonlu = _grad_ozet(model)
            if not grad_sonlu:
                raise FloatingPointError("Talimat gradient NaN/Inf içeriyor")
            opt.step()
            tl += float(loss.item())
            adim += 1
            son_grad_norm = float(grad_norm)
        scheduler.step()

        train_loss = tl / max(1, adim)
        val_loss, ppl = _degerlendir(model, Xval, Yval, batch, loss_fn, aygit, amp_acik)
        lr_now = float(opt.param_groups[0]["lr"])
        _csv_logla(log_dizini, {
            "epoch": ep, "train_loss": train_loss, "val_loss": val_loss,
            "perplexity": ppl, "lr": lr_now, "grad_norm": son_grad_norm,
            "max_grad_katman": max_ad, "max_grad_norm": max_norm,
        })
        _checkpoint_kaydet(ckpt_dizini, "latest.pt", model, opt, scheduler,
                           ep, train_loss, val_loss)
        _checkpoint_kaydet(ckpt_dizini, f"epoch_{ep:04d}.pt", model, opt, scheduler,
                           ep, train_loss, val_loss)
        if val_loss is not None and val_loss + min_delta < en_iyi_val:
            en_iyi_val = val_loss
            sabir = 0
            _checkpoint_kaydet(ckpt_dizini, "best.pt", model, opt, scheduler,
                               ep, train_loss, val_loss)
        elif val_loss is not None and patience is not None:
            sabir += 1
        _eski_checkpoint_temizle(ckpt_dizini, son_ckpt)

        if ep % 20 == 0 or ep == cag or ep == 1:
            val_txt = "" if val_loss is None else f" val={val_loss:.4f} ppl={ppl:.2f}"
            print(f"  🌟 Epoch {ep:03d}/{cag} loss={train_loss:.4f}{val_txt} "
                  f"lr={lr_now:.2e} grad={son_grad_norm:.3f} "
                  f"max={max_ad}:{max_norm:.3f} süre={time.time() - basla:.1f}s")
        if val_loss is not None and patience is not None and sabir >= patience:
            print(f"  ⏹️ Early stopping: {patience} epoch boyunca validation iyileşmedi.")
            break

    talimat_yol = os.path.join(KOK, f"hiper_model_{n}_talimat.pt")
    agirlik_kaydet(model, talimat_yol)
    agirlik_kaydet(model, temel_yol)
    if model.seyrek_tablo is not None:
        print(f"🧠 {model.seyrek_doluluk_metni()}")
    print(f"💾 Kaydedildi: {talimat_yol}")


if __name__ == "__main__":
    main()
