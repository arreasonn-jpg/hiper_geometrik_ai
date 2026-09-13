# -*- coding: utf-8 -*-
"""Gerçek mini eğitim smoke raporu.

Amaç: büyük eğitim iddiası değil; küçük model + küçük Türkçe corpus ile eğitim
hattının (loss/grad/checkpoint/log/perplexity) uçtan uca çalıştığını ölçmek.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from typing import Dict, Optional

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [KOK, os.path.join(KOK, "mimari")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _cihaz():
    try:
        import torch
    except Exception as e:
        return {"torch_available": False, "cuda_available": False, "note": str(e)}
    d = {"torch_available": True,
         "torch_version": getattr(torch, "__version__", None),
         "cuda_available": bool(torch.cuda.is_available()),
         "device": "cuda" if torch.cuda.is_available() else "cpu"}
    if not torch.cuda.is_available():
        d["note"] = "CUDA yok; smoke CPU fallback ile koştu"
    return d


def mini_egitim_smoke_raporu(rapor_yol: Optional[str] = None,
                             markdown_yol: Optional[str] = None,
                             log_dizini: Optional[str] = None,
                             checkpoint_dizini: Optional[str] = None,
                             cag: int = 1,
                             batch: int = 16,
                             tekrar: int = 8,
                             sessiz: bool = True) -> Dict:
    """Mini eğitimi çalıştır ve JSON-serileştirilebilir rapor döndür/yaz.

    Torch yoksa exception fırlatmak yerine ``durum='skipped_dependency'`` döner;
    böylece sistem Python smoke testleri de güvenli kalır.
    """
    try:
        import torch
    except Exception as e:
        rapor = {
            "rapor_tipi": "hga-mini-training-smoke-v1",
            "durum": "skipped_dependency",
            "hata": str(e),
            "cihaz": _cihaz(),
        }
        if rapor_yol:
            _json_yaz(rapor_yol, rapor)
        if markdown_yol:
            _metin_yaz(markdown_yol, mini_egitim_smoke_markdown(rapor))
        return rapor

    from bpe_tokenizer import BPETokenizer
    from kuresel_model import HiperGeometrikAI

    from egitim.egitici import KureselEgitimMotoru
    from egitim.saglamlik import checkpoint_uyumluluk_raporu
    from hga.evaluation import mini_turkce_corpus, perplexity_benchmark

    torch.manual_seed(1234)
    log_dizini = log_dizini or os.path.join(KOK, "logs", "mini_training_smoke")
    checkpoint_dizini = checkpoint_dizini or os.path.join(KOK, "checkpoints", "mini_training_smoke")
    os.makedirs(log_dizini, exist_ok=True)
    os.makedirs(checkpoint_dizini, exist_ok=True)

    corpus = mini_turkce_corpus() * max(1, int(tekrar))
    metin = "\n".join(corpus)
    tok = BPETokenizer(baglam_penceresi=8, max_vocab_size=256, min_freq=1)
    tok.fit_on_text(metin, verbose=False)
    ids = tok.encode(metin)
    model = HiperGeometrikAI(n=8, katman_sayisi=1, baglam_penceresi=8,
                             emb_dim=8, num_heads=2, sozluk_boyutu=max(len(tok.sozluk), 64),
                             dropout=0.0, seyrek_tablo_boyutu=0, bilgilendir=False)
    motor = KureselEgitimMotoru(
        model, ogrenme_hizi=5e-4, toplam_cag=int(cag), karisik_hassasiyet="auto",
        grad_clip=1.0, warmup_cag=0, lr_min_factor=0.1,
        log_dizini=log_dizini, checkpoint_dizini=checkpoint_dizini,
        son_checkpoint_sayisi=2)

    if sessiz:
        import io
        with redirect_stdout(io.StringIO()):
            motor.ngram_egitim_dongusu(ids, cag_sayisi=int(cag), batch_size=int(batch),
                                       validation_split=0.2,
                                       early_stopping_patience=None)
    else:
        motor.ngram_egitim_dongusu(ids, cag_sayisi=int(cag), batch_size=int(batch),
                                   validation_split=0.2,
                                   early_stopping_patience=None)

    son = motor.cag_gecmisi[-1]
    grad = motor.gradient_gecmisi[-1]
    latest = os.path.join(checkpoint_dizini, "latest.pt")
    ckpt_rapor = checkpoint_uyumluluk_raporu(model, latest, strict=True) if os.path.exists(latest) else None
    ppl = perplexity_benchmark(model, tok, mini_turkce_corpus(), batch_size=32).to_dict()
    rapor = {
        "rapor_tipi": "hga-mini-training-smoke-v1",
        "durum": "ok",
        "olusturma_zamani_utc": datetime.now(timezone.utc).isoformat(),
        "sinirlar": "Küçük smoke koşusu; kalite/ölçek iddiası değildir.",
        "cihaz": _cihaz(),
        "veri": {
            "cumle_sayisi": len(corpus),
            "token_sayisi": len(ids),
            "tekrar": int(tekrar),
        },
        "model": {
            "n": 8, "K": 1, "baglam": 8, "emb_dim": 8, "heads": 2,
            "vocab": max(len(tok.sozluk), 64),
            "parametre": sum(p.numel() for p in model.parameters()),
        },
        "egitim": {
            "cag": son.cag,
            "train_loss": son.train_loss,
            "val_loss": son.val_loss,
            "perplexity": son.perplexity,
            "lr": son.lr,
            "grad_norm": son.grad_norm,
            "max_grad_katman": son.max_grad_katman,
            "max_grad_norm": son.max_grad_norm,
            "loss_sonlu": bool(math.isfinite(float(son.train_loss))),
            "grad_sonlu": bool(grad.sonlu),
        },
        "benchmark": ppl,
        "artefaktlar": {
            "metrics_csv": _gosterilebilir_yol(os.path.join(log_dizini, "metrics.csv")),
            "metrics_jsonl": _gosterilebilir_yol(os.path.join(log_dizini, "metrics.jsonl")),
            "latest_checkpoint": _gosterilebilir_yol(latest),
            "best_checkpoint": _gosterilebilir_yol(os.path.join(checkpoint_dizini, "best.pt")),
        },
        "checkpoint_uyumluluk": ckpt_rapor,
    }
    if rapor_yol:
        _json_yaz(rapor_yol, rapor)
    if markdown_yol:
        _metin_yaz(markdown_yol, mini_egitim_smoke_markdown(rapor))
    return rapor


def mini_egitim_smoke_markdown(rapor: Dict) -> str:
    eg = rapor.get("egitim", {})
    bm = rapor.get("benchmark", {})
    lines = [
        "# HGA Mini Eğitim Smoke Raporu",
        "",
        f"- Durum: `{rapor.get('durum')}`",
        f"- Rapor tipi: `{rapor.get('rapor_tipi')}`",
        f"- Sınır: {rapor.get('sinirlar', 'Torch yoksa smoke atlanır')}",
        f"- Cihaz: `{rapor.get('cihaz', {}).get('device', 'n/a')}`",
        "",
        "## Eğitim",
        f"- Çağ: `{eg.get('cag')}`",
        f"- Train loss: `{eg.get('train_loss')}`",
        f"- Val loss: `{eg.get('val_loss')}`",
        f"- Val perplexity: `{eg.get('perplexity')}`",
        f"- Grad norm: `{eg.get('grad_norm')}`",
        f"- Loss sonlu: `{eg.get('loss_sonlu')}`",
        f"- Grad sonlu: `{eg.get('grad_sonlu')}`",
        "",
        "## Mini Benchmark",
        f"- Perplexity: `{bm.get('perplexity')}`",
        f"- Tokenizer OK: `{bm.get('tokenizer_ok')}`",
    ]
    return "\n".join(lines) + "\n"


def _gosterilebilir_yol(yol: str) -> str:
    """Repo içi artefaktları raporda taşınabilir göreli yol olarak göster."""
    abs_yol = os.path.abspath(yol)
    try:
        if os.path.commonpath([KOK, abs_yol]) == KOK:
            return os.path.relpath(abs_yol, KOK)
    except ValueError:
        pass
    return yol


def _json_yaz(yol: str, veri: Dict):
    os.makedirs(os.path.dirname(os.path.abspath(yol)) or ".", exist_ok=True)
    with open(yol, "w", encoding="utf-8") as f:
        json.dump(veri, f, ensure_ascii=False, indent=2, sort_keys=True)


def _metin_yaz(yol: str, metin: str):
    os.makedirs(os.path.dirname(os.path.abspath(yol)) or ".", exist_ok=True)
    with open(yol, "w", encoding="utf-8") as f:
        f.write(metin)


def main(argv=None):
    ap = argparse.ArgumentParser(description="HGA gerçek mini eğitim smoke raporu")
    ap.add_argument("--out", default=os.path.join(KOK, "raporlar", "mini_training_report.json"))
    ap.add_argument("--markdown", default=os.path.join(KOK, "raporlar", "mini_training_report.md"))
    ap.add_argument("--log-dizini", default=None)
    ap.add_argument("--checkpoint-dizini", default=None)
    ap.add_argument("--cag", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tekrar", type=int, default=8)
    ap.add_argument("--sesli", action="store_true", help="eğitim stdout'unu bastırma")
    args = ap.parse_args(argv)
    rapor = mini_egitim_smoke_raporu(
        rapor_yol=args.out, markdown_yol=args.markdown,
        log_dizini=args.log_dizini, checkpoint_dizini=args.checkpoint_dizini,
        cag=args.cag, batch=args.batch, tekrar=args.tekrar, sessiz=not args.sesli)
    print(f"Mini eğitim smoke durumu: {rapor.get('durum')}")
    print(f"JSON: {args.out}")
    print(f"Markdown: {args.markdown}")


if __name__ == "__main__":
    main()
