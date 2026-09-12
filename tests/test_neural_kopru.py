# -*- coding: utf-8 -*-
"""
NeuralKopru testleri — doğrulanmış deneyim ↔ modelin seyrek belleği
====================================================================
Çalıştırma (torch gerekir):
    .venv/bin/python tests/test_neural_kopru.py
    .venv/bin/python -m pytest tests/test_neural_kopru.py -q

Torch yoksa: kurulumun dürüstçe ImportError verdiğini doğrular.
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [KOK, os.path.join(KOK, "mimari")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hga.memory import NeuralKopru, torch_var_mi  # noqa: E402

UCLU_A = ("E_001", "R_001", "E_004")
UCLU_B = ("E_002", "R_001", "E_005")


def _model():
    if not torch_var_mi():
        return None
    from kuresel_model import HiperGeometrikAI
    return HiperGeometrikAI(
        n=64, katman_sayisi=2, baglam_penceresi=8, emb_dim=32,
        num_heads=4, sozluk_boyutu=256, dropout=0.0,
        seyrek_tablo_boyutu=2048, seyrek_boyut=8, bilgilendir=False)


def test_kopru_torch_yoksa_guvenli_hata():
    if torch_var_mi():
        print("  (torch mevcut — gerçek testler aşağıda)")
        return
    hata = None
    try:
        NeuralKopru(object())
    except ImportError as e:
        hata = e
    assert hata is not None, "torch yokken NeuralKopru sessizce kurulmamalı"


def test_kopru_anahtar_deterministik():
    if not torch_var_mi():
        return
    k = NeuralKopru(_model())
    assert k.anahtar(UCLU_A)[0].item() == k.anahtar(UCLU_A)[0].item()
    assert k.anahtar(UCLU_A)[0].item() != k.anahtar(UCLU_B)[0].item()


def test_kopru_bos_tablo_olu_yol():
    """Boş tabloda gen=0 → gen_kopru gradyanı 0 → yol ÖLÜ (dürüst başlangıç)."""
    if not torch_var_mi():
        return
    k = NeuralKopru(_model())
    assert k.gen(UCLU_A).abs().sum().item() == 0.0
    assert k.kopru_canli_mi(UCLU_A) is False


def test_kopru_yaz_sonrasi_canli():
    """Doğrulanmış üçlüler gradyanla yazıldıktan sonra yol CANLANIR."""
    if not torch_var_mi():
        return
    k = NeuralKopru(_model())
    once, _ = k.doluluk()
    k.coklu_yaz([UCLU_A, UCLU_B])
    sonra, _ = k.doluluk()
    assert sonra > once                       # satırlar doldu
    assert k.gen(UCLU_A).abs().sum().item() > 0.0
    assert k.kopru_canli_mi(UCLU_A) is True   # gen_kopru artık gradyan alıyor


def test_kopru_model_egitimi_birlikte():
    """Deneyim satırları doluyken token eğitimi devam eder; loss düşer ve
    tabloya token pencereleri de eklenir (paylaşımlı bellek)."""
    if not torch_var_mi():
        return
    import torch
    model = _model()
    k = NeuralKopru(model)
    k.coklu_yaz([UCLU_A, UCLU_B])
    doluluk_deneyim, _ = k.doluluk()

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    desen = [10 + (i % 4) for i in range(120)]
    x = torch.tensor([desen[i:i + 8] for i in range(112)], dtype=torch.long)
    y = torch.tensor(desen[8:], dtype=torch.long)
    ilk = son = None
    model.train()
    for adim in range(60):
        i = (adim * 16) % (112 - 16)
        opt.zero_grad()
        loss = loss_fn(model(x[i:i + 16]), y[i:i + 16])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if adim == 0:
            ilk = loss.item()
        son = loss.item()
    assert son < ilk * 0.85, f"loss düşmedi: {ilk:.3f} → {son:.3f}"

    doluluk_son, _ = k.doluluk()
    assert doluluk_son > doluluk_deneyim   # token pencereleri de tabloya yazıldı


if __name__ == "__main__":
    testler = [(ad, fn) for ad, fn in sorted(globals().items())
               if ad.startswith("test_") and callable(fn)]
    basarisiz = 0
    for ad, fn in testler:
        try:
            fn()
            print(f"  ✅ {ad}")
        except Exception as e:  # noqa: BLE001
            basarisiz += 1
            print(f"  ❌ {ad}: {type(e).__name__}: {e}")
    print("\nTÜM TESTLER GEÇTİ" if basarisiz == 0 else f"{basarisiz} test başarısız")
    sys.exit(1 if basarisiz else 0)
