# -*- coding: utf-8 -*-
"""Kompakt HGA benchmark/sağlık raporu üretimi.

Rapor bilinçli olarak küçük ve deterministiktir: CI'da hızlı koşar, GPU yoksa CPU
bilgisini açıkça yazar, checkpoint yoksa perplexity sonucunu "rastgele ağırlık"
notuyla etiketler.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .turkish_benchmark import mini_turkce_corpus, tokenizer_kapsami, perplexity_benchmark
from .hallucination import hallucination_metrics
from hga.experience import AritmetikOrtam, ExperienceEvaluator, ExperienceGenerator, aritmetik_etki_alani
from hga.memory import DeneyimSlotlari


def _device_report() -> Dict[str, Any]:
    try:
        import torch
    except Exception as e:
        return {"torch_available": False, "cuda_available": False, "note": str(e)}
    info: Dict[str, Any] = {
        "torch_available": True,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if torch.cuda.is_available():  # gerçek GPU ortamında bilgi zenginleşir
        try:
            idx = torch.cuda.current_device()
            info.update({
                "cuda_device_index": int(idx),
                "cuda_device_name": torch.cuda.get_device_name(idx),
                "cuda_memory_allocated_mb": round(torch.cuda.memory_allocated(idx) / (1024 ** 2), 3),
                "cuda_memory_reserved_mb": round(torch.cuda.memory_reserved(idx) / (1024 ** 2), 3),
            })
        except Exception as e:  # pragma: no cover - GPU ortamına bağlı
            info["cuda_probe_error"] = str(e)
    else:
        info["note"] = "CUDA yok; VRAM metrikleri CPU fallback olarak raporlandı"
    return info


def _mini_hallucination() -> Dict[str, Any]:
    store = aritmetik_etki_alani()
    gen = ExperienceGenerator(tip_filtresi=False)
    ev = ExperienceEvaluator()
    adaylar = gen.uret(store)
    for a in adaylar:
        ev.degerlendir(a, store)
    return hallucination_metrics(adaylar, store=store,
                                 validator=AritmetikOrtam().aday_dogrula).to_dict()


def _memory_report() -> Dict[str, Any]:
    slot = DeneyimSlotlari(slot_sayisi=16)
    for i in range(6):
        slot.yaz(f"K{i}", ("E1", "R", f"E{i}"))
    for k, anahtar in [("K0", ("E1", "R", "E0")), ("K3", ("E1", "R", "E3")),
                       ("K3", ("E1", "R", "E3")), ("YOK", ("E1", "R", "EY"))]:
        slot.icerir(k, anahtar)
    k = slot.kapasite()
    k["doluluk_orani"] = k["dolu_slot"] / k["toplam_slot"] if k["toplam_slot"] else 0.0
    return k


def _tiny_perplexity(checkpoint: Optional[str] = None,
                     tokenizer_yol: Optional[str] = None,
                     n: int = 8,
                     katman: int = 1,
                     baglam: int = 8,
                     vocab: int = 256) -> Dict[str, Any]:
    try:
        import torch  # noqa: F401
    except Exception as e:
        return {"ok": False, "skipped": "torch yok", "error": str(e)}

    from bpe_tokenizer import BPETokenizer
    from kuresel_model import HiperGeometrikAI, agirlik_yukle

    corpus = mini_turkce_corpus()
    train = corpus[:-2]
    heldout = corpus[-2:]
    tok = BPETokenizer(baglam_penceresi=baglam, max_vocab_size=vocab, min_freq=1)
    notlar: List[str] = []
    if tokenizer_yol and os.path.exists(tokenizer_yol):
        tok.yukle(tokenizer_yol)
        notlar.append(f"tokenizer={tokenizer_yol}")
    else:
        tok.fit_on_text("\n".join(train), verbose=False)
        notlar.append("tokenizer yalnız train split üzerinde fit edildi; held-out mini smoke")

    model = HiperGeometrikAI(n=n, katman_sayisi=katman, baglam_penceresi=baglam,
                             emb_dim=8, num_heads=2, sozluk_boyutu=max(len(tok.sozluk), 64),
                             dropout=0.0, seyrek_tablo_boyutu=0, bilgilendir=False)
    if checkpoint:
        agirlik_yukle(model, checkpoint, strict=True)
        notlar.append(f"checkpoint={checkpoint}")
    else:
        notlar.append("checkpoint yok: rastgele ağırlıklar; kalite iddiası değildir")
    r = perplexity_benchmark(model, tok, heldout, batch_size=32)
    d = r.to_dict()
    d["ok"] = bool(math.isfinite(float(d.get("perplexity", float("inf")))))
    d["heldout_cumle_sayisi"] = len(heldout)
    d["train_cumle_sayisi"] = len(train)
    d["notlar"] = list(d.get("notlar", [])) + notlar
    return d


def benchmark_raporu_olustur(checkpoint: Optional[str] = None,
                             tokenizer_yol: Optional[str] = None,
                             tiny: bool = True,
                             n: int = 8,
                             katman: int = 1,
                             baglam: int = 8,
                             vocab: int = 256) -> Dict[str, Any]:
    corpus = mini_turkce_corpus()
    from bpe_tokenizer import BPETokenizer
    tok = BPETokenizer(baglam_penceresi=baglam, max_vocab_size=vocab, min_freq=1)
    tok.fit_on_text("\n".join(corpus[:-2]), verbose=False)
    rapor = {
        "rapor_tipi": "hga-mini-benchmark-v1",
        "olusturma_zamani_utc": datetime.now(timezone.utc).isoformat(),
        "sinirlar": {
            "mini": bool(tiny),
            "not": "Rapor smoke/held-out sağlık ölçümüdür; eğitimli kalite iddiası değildir.",
        },
        "cihaz": _device_report(),
        "tokenizer": tokenizer_kapsami(tok, corpus),
        "perplexity": _tiny_perplexity(checkpoint=checkpoint, tokenizer_yol=tokenizer_yol,
                                        n=n, katman=katman, baglam=baglam, vocab=vocab),
        "halusinasyon": _mini_hallucination(),
        "seyrek_bellek": _memory_report(),
    }
    return rapor


def benchmark_raporu_markdown(rapor: Dict[str, Any]) -> str:
    lines = [
        "# HGA Mini Benchmark Raporu",
        "",
        f"- Rapor tipi: `{rapor.get('rapor_tipi')}`",
        f"- Zaman (UTC): `{rapor.get('olusturma_zamani_utc')}`",
        f"- Sınır: {rapor.get('sinirlar', {}).get('not')}",
        "",
        "## Cihaz",
    ]
    for k, v in rapor.get("cihaz", {}).items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## Ana Metrikler"]
    token = rapor.get("tokenizer", {})
    ppl = rapor.get("perplexity", {})
    hal = rapor.get("halusinasyon", {})
    mem = rapor.get("seyrek_bellek", {})
    lines += [
        f"- Tokenizer karakter kapsama: `{token.get('karakter_kapsama')}`",
        f"- Tokenizer bilinmeyen oranı: `{token.get('unk_orani')}`",
        f"- Held-out mini perplexity: `{ppl.get('perplexity')}`",
        f"- Hallucination oranı: `{hal.get('hallucination_rate')}`",
        f"- Bellek doluluk oranı: `{mem.get('doluluk_orani')}`",
        "",
        "## Notlar",
    ]
    for note in ppl.get("notlar", []):
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def benchmark_raporu_kaydet(rapor: Dict[str, Any], json_yol: Optional[str] = None,
                            markdown_yol: Optional[str] = None) -> Dict[str, str]:
    yollar: Dict[str, str] = {}
    if json_yol:
        os.makedirs(os.path.dirname(os.path.abspath(json_yol)) or ".", exist_ok=True)
        with open(json_yol, "w", encoding="utf-8") as f:
            json.dump(rapor, f, ensure_ascii=False, indent=2, sort_keys=True)
        yollar["json"] = json_yol
    if markdown_yol:
        os.makedirs(os.path.dirname(os.path.abspath(markdown_yol)) or ".", exist_ok=True)
        with open(markdown_yol, "w", encoding="utf-8") as f:
            f.write(benchmark_raporu_markdown(rapor))
        yollar["markdown"] = markdown_yol
    return yollar


__all__ = ["benchmark_raporu_olustur", "benchmark_raporu_markdown", "benchmark_raporu_kaydet"]
