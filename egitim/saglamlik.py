# -*- coding: utf-8 -*-
"""Eğitim sağlamlık smoke-test yardımcıları.

Bu modül CI/yerel ortamda hızlı doğrulama için tasarlanmıştır: AMP açıkken
loss/logit sonlu mu, checkpoint açık/kapalı ileri-geri gradyanları uyumlu mu?
"""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Union

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _torch_gerekli():
    if torch is None:
        raise ImportError("Sağlamlık smoke testleri için torch gerekli")


def amp_dogrula(model, x, y, loss_fn=None, dtype=None) -> Dict:
    """AMP açıkken tek ileri/geri adımda NaN/Inf oluşuyor mu kontrol et."""
    _torch_gerekli()
    device = next(model.parameters()).device
    x, y = x.to(device), y.to(device)
    loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
    amp_acik = device.type == "cuda"
    dtype = dtype or (torch.bfloat16 if amp_acik and torch.cuda.is_bf16_supported()
                      else torch.float16)
    model.train()
    model.zero_grad(set_to_none=True)
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if amp_acik else _nullcontext()
    with ctx:
        logits = model(x)
        loss = loss_fn(logits, y)
    ok = bool(torch.isfinite(logits).all().item() and torch.isfinite(loss).item())
    if ok:
        loss.backward()
        grad_ok = all(p.grad is None or torch.isfinite(p.grad).all().item()
                      for p in model.parameters())
    else:
        grad_ok = False
    return {
        "amp_acik": amp_acik,
        "dtype": str(dtype).replace("torch.", ""),
        "loss": float(loss.detach().cpu().item()) if torch.isfinite(loss) else float("nan"),
        "logit_sonlu": bool(torch.isfinite(logits).all().item()),
        "loss_sonlu": bool(torch.isfinite(loss).item()),
        "grad_sonlu": bool(grad_ok),
        "ok": bool(ok and grad_ok),
    }


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def checkpoint_state_manifest(state_dict: Mapping) -> Dict[str, Dict[str, object]]:
    """Bir state_dict için şekil/dtype manifesti üret."""
    manifest: Dict[str, Dict[str, object]] = {}
    for ad, tensor in state_dict.items():
        shape = tuple(getattr(tensor, "shape", ()))
        manifest[str(ad)] = {
            "shape": list(shape),
            "dtype": str(getattr(tensor, "dtype", "")),
            "numel": int(tensor.numel()) if hasattr(tensor, "numel") else 0,
        }
    return manifest


def checkpoint_uyumluluk_raporu(model, checkpoint: Union[str, Mapping],
                                strict: bool = True) -> Dict:
    """Checkpoint'i yüklemeden modelle şekil/anahtar uyumluluğunu raporla.

    Amaç: mimari değiştiğinde ``load_state_dict`` hatasını geç yakalamak yerine
    hangi katmanda neden uyumsuzluk olduğunu önce raporlamak. ``checkpoint`` yol
    ya da halihazırda yüklenmiş dict olabilir.
    """
    _torch_gerekli()
    if isinstance(checkpoint, str):
        ckpt = torch.load(checkpoint, map_location=next(model.parameters()).device)
    else:
        ckpt = checkpoint
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, Mapping) else ckpt
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    ckpt_keys = set(state.keys())
    ortak = sorted(model_keys & ckpt_keys)
    sekil_uyumsuz = {}
    for k in ortak:
        m_shape = tuple(model_state[k].shape)
        c_shape = tuple(state[k].shape)
        if m_shape != c_shape:
            sekil_uyumsuz[k] = {"model": list(m_shape), "checkpoint": list(c_shape)}
    eksik = sorted(model_keys - ckpt_keys)
    fazla = sorted(ckpt_keys - model_keys)
    strict_sorun = bool(strict and (eksik or fazla))
    ok = not sekil_uyumsuz and not strict_sorun
    return {
        "ok": bool(ok),
        "strict": bool(strict),
        "ortak_anahtar": len(ortak),
        "eksik": eksik,
        "fazla": fazla,
        "sekil_uyumsuz": sekil_uyumsuz,
        "checkpoint_version": ckpt.get("checkpoint_version") if isinstance(ckpt, Mapping) else None,
        "model_meta": ckpt.get("model_meta") if isinstance(ckpt, Mapping) else None,
    }


def checkpoint_gradyan_dogrula(factory: Callable[[bool], object], x,
                               atol: float = 1e-6) -> Dict:
    """Checkpoint açık/kapalı modüllerin ileri+geri eşdeğerliğini doğrula.

    ``factory(False)`` checkpoint kapalı, ``factory(True)`` açık modülü üretmeli;
    ikisinin ``state_dict`` anahtarları uyumlu olmalıdır.
    """
    _torch_gerekli()
    m1 = factory(False)
    m2 = factory(True)
    m2.load_state_dict(m1.state_dict())
    m1.train(); m2.train()
    x1 = x.detach().clone().requires_grad_(True)
    x2 = x.detach().clone().requires_grad_(True)
    y1 = m1(x1).sum()
    y2 = m2(x2).sum()
    y1.backward(); y2.backward()
    ileri_ok = bool(torch.allclose(y1.detach(), y2.detach(), atol=atol))
    girdi_grad_ok = bool(torch.allclose(x1.grad, x2.grad, atol=atol))
    param_grad_ok = True
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        if n1 != n2 or not torch.allclose(p1.grad, p2.grad, atol=atol):
            param_grad_ok = False
            break
    return {
        "ileri_ok": ileri_ok,
        "girdi_grad_ok": girdi_grad_ok,
        "param_grad_ok": bool(param_grad_ok),
        "ok": bool(ileri_ok and girdi_grad_ok and param_grad_ok),
    }


__all__ = [
    "amp_dogrula", "checkpoint_gradyan_dogrula",
    "checkpoint_state_manifest", "checkpoint_uyumluluk_raporu",
]
