# -*- coding: utf-8 -*-
"""
Hiper-Geometrik AI — Mimari Duman (Smoke) Testi
================================================
Kullanım:
    python test_mimari.py          # tek dosya, bağımlılıksız çalışır
    pytest test_mimari.py -q       # pytest ile de çalışır

Kapsam (rapor maddeleriyle eşleşir):
  - bilinear A@X@B sandviçin doğruluğu (elle karşılaştırma)        (8.2)
  - zincir: şekil, gradyan akışı, checkpoint eşdeğerliği           (8.3/8.4.2)
  - encoder dış çarpımı (0→1 katmanı)                              (5.3)
  - decoder: çift odak merceği, softmax YOK, iki mercek DE gradyan alır (8.6.4)
  - nedensel dikkat (gelecek token geçmişi DEĞİŞTİREMEZ)           (8.4.5)
  - model: ileri/geri, ÖLÜ PARAMETRE YOK                           (8.1.2)
  - fabrika: n gerçekten modele geçer                              (8.1.3)
  - strict kaydet/yükle + uyumsuzluk YÜZEYE ÇIKAR                  (8.1.5)
  - BPE tokenizer API                                             (8.4.6)
  - kapasite raporu tutarlılığı                                   (8.6.7)
  - mini eğitim: loss gözle görülür şekilde düşer
"""
import os
import sys

KOK = os.path.abspath(os.path.dirname(__file__))
for _p in [KOK, os.path.join(KOK, "mimari")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from kuresel_model import (HiperGeometrikAI, model_olustur, agirlik_yukle)
from kuresel_bag import KureselBagKatmani, KureselZincir
from encoder import GeometrikVeriEncoder
from decoder import FraktalDecoder
from hiper_attention import HiperGeometrikAttention
from bpe_tokenizer import BPETokenizer

KUCUK = dict(n=64, katman_sayisi=2, baglam_penceresi=8, emb_dim=32,
             num_heads=4, sozluk_boyutu=256, dropout=0.0)


def test_bilinear_sandvic():
    """Tek katman: Y gerçekten A @ X @ B mi? (eski sürümde matris çarpımı YOKTU)"""
    kat = KureselBagKatmani(32)
    X = torch.randn(3, 32, 32)
    Y = kat(X)
    assert Y.shape == (3, 32, 32)
    beklenen = torch.einsum("ij,bjk,kl->bil", kat.mercek_A, X, kat.mercek_B)
    assert torch.allclose(Y, beklenen, atol=1e-5)


def test_zincir_sekil_ve_gradyan():
    """Zincir: şekil korunur, TÜM parametreler sağlıklı gradyan alır."""
    z = KureselZincir(32, katman_sayisi=3, dropout=0.0)
    X = torch.randn(2, 32, 32, requires_grad=True)
    Y = z(X)
    assert Y.shape == X.shape
    Y.sum().backward()
    for ad, p in z.named_parameters():
        assert p.grad is not None, f"{ad} gradyan almıyor (ölü parametre!)"
        assert torch.isfinite(p.grad).all(), f"{ad} gradyanı NaN/Inf"
    assert X.grad is not None and torch.isfinite(X.grad).all()


def test_zincir_checkpoint_esdegerligi():
    """Gradient checkpointing açık/kapalı aynı sonucu verir (rapor 8.4.2)."""
    torch.manual_seed(0)
    z1 = KureselZincir(16, katman_sayisi=2, dropout=0.0)
    z2 = KureselZincir(16, katman_sayisi=2, dropout=0.0, checkpoint_kullan=True)
    z2.load_state_dict(z1.state_dict())
    z1.train(); z2.train()
    X = torch.randn(2, 16, 16)
    assert torch.allclose(z1(X), z2(X), atol=1e-6)


def test_encoder_dis_carpim():
    """0→1 katmanı: (B, giris) → tanh(u ⊗ v) → (B, n, n)."""
    enc = GeometrikVeriEncoder(giris_boyutu=16, n=8)
    duz = torch.randn(4, 16)
    X = enc(duz)
    assert X.shape == (4, 8, 8)
    u, v = enc.proj_u(duz), enc.proj_v(duz)
    beklenen = torch.tanh(torch.einsum("bi,bj->bij", u, v))
    assert torch.allclose(X, beklenen, atol=1e-6)
    # rank-1 dış çarpım + eleman bazlı tanh → tam ranklı başlangıç (tip. rank > 1)
    assert torch.linalg.matrix_rank(X[0]).item() > 1


def test_decoder_cift_odak_ve_softmax_yok():
    """Decoder: iki odak merceği de eğitilir; çıktı softmax DEĞİLDİR."""
    dec = FraktalDecoder(n=16, sozluk_boyutu=100)
    X = torch.randn(5, 16, 16)
    logits = dec(X)
    assert logits.shape == (5, 100)
    # olasılık dağılımı DEĞİL (CrossEntropyLoss ile uyum — çifte softmax yok)
    assert not torch.allclose(logits.sum(dim=-1), torch.ones(5), atol=1e-2)
    logits.sum().backward()
    # ESKİ SÜRÜMDEKİ HATA BURADAYDI: mercek_B hiç eğitilmiyordu
    assert dec.mercek_1.grad is not None and dec.mercek_1.grad.abs().sum() > 0
    assert dec.mercek_2.grad is not None and dec.mercek_2.grad.abs().sum() > 0


def test_dikkat_nedensel():
    """Gelecekteki token, geçmiş pozisyonların çıktısını DEĞİŞTIREMEZ."""
    torch.manual_seed(1)
    att = HiperGeometrikAttention(emb_dim=16, num_heads=4, dropout=0.0,
                                  is_causal=True).eval()
    att.alpha.data.fill_(1.0)  # ReZero'yu aç → dikkatin katkısı gerçek olsun
    x1 = torch.randn(2, 6, 16)
    x2 = x1.clone()
    x2[:, 5, :] = torch.randn(2, 16)      # yalnızca SON (gelecek) tokenı değiştir
    y1, y2 = att(x1), att(x2)
    assert torch.allclose(y1[:, :5], y2[:, :5], atol=1e-5), "nedensellik ihlali!"
    assert not torch.allclose(y1[:, 5], y2[:, 5], atol=1e-5)


def test_model_ileri_geri_olu_parametre_yok():
    """Tam model: ileri + geri geçişte TEK bir ölü parametre bile olmamalı."""
    torch.manual_seed(0)
    model = HiperGeometrikAI(**KUCUK, bilgilendir=False)
    x = torch.randint(4, 256, (8, 8))
    logits = model(x)
    assert logits.shape == (8, 256)
    hedef = torch.randint(4, 256, (8,))
    loss = torch.nn.functional.cross_entropy(logits, hedef)
    loss.backward()
    oluler = [ad for ad, p in model.named_parameters() if p.grad is None]
    assert not oluler, f"Gradyan almayan (ölü) parametreler: {oluler}"
    assert torch.isfinite(loss)


def test_model_pencere_esnekligi():
    """Kısa girdi sola doldurulur, uzun girdi son pencereye kırpılır."""
    model = HiperGeometrikAI(**KUCUK, bilgilendir=False)
    assert model(torch.randint(4, 256, (2, 3))).shape == (2, 256)
    assert model(torch.randint(4, 256, (2, 20))).shape == (2, 256)


def test_fabrika_n_gercekten_gecer():
    """RAPOR 8.1.3 REGRESYON TESTİ: n artık modele gerçekten iletiliyor."""
    for n in (128, 256):
        model = model_olustur(sozluk_boyutu=512, n=n, katman_sayisi=2,
                              baglam_penceresi=8, emb_dim=32, bilgilendir=False)
        assert model.n == n
        assert model.kuresel_bag.katmanlar[0].n == n
        assert model.encoder.proj_u.out_features == n
        assert model.decoder.anlam_cozucu.in_features == 2 * n


def test_strict_kaydet_yukle():
    """strict=True: uyumlu yüklenir, uyumsuz SESSİZCE geçmez (rapor 8.1.5)."""
    kwargs = dict(sozluk_boyutu=128, n=32, katman_sayisi=2,
                  baglam_penceresi=8, emb_dim=16, bilgilendir=False)
    model = model_olustur(**kwargs)
    yol = os.path.join(KOK, "_test_model.pt")
    try:
        torch.save(model.state_dict(), yol)
        model2 = model_olustur(**kwargs)
        sonuc = agirlik_yukle(model2, yol, strict=True)
        assert sonuc.missing_keys == [] and sonuc.unexpected_keys == []
        # uyumsuz mimari (farklı sözlük boyutu) hata VERMELİ
        model3 = model_olustur(sozluk_boyutu=64, n=32, katman_sayisi=2,
                               baglam_penceresi=8, emb_dim=16, bilgilendir=False)
        hata = None
        try:
            agirlik_yukle(model3, yol, strict=True)
        except RuntimeError as e:
            hata = e
        assert hata is not None, "uyumsuz checkpoint strict=True ile sessizce geçti!"
    finally:
        if os.path.exists(yol):
            os.remove(yol)


def test_bpe_api():
    """BPE tokenizer: kurulum, encode/decode, kilitli sözlük, uyumluluk API'si."""
    tok = BPETokenizer(baglam_penceresi=8, max_vocab_size=500, min_freq=1)
    metin = ("hiper geometrik yapay zeka türkçe metin öğreniyor. "
             "küresel bağ katmanı matris çarpımı ile çalışır. "
             "kronecker zinciri katman katman büyür. ") * 20
    tok.fit_on_text(metin, verbose=False)
    ids = tok.encode("küresel bağ katmanı")
    assert isinstance(ids, list) and len(ids) > 0
    geri = tok.decode(ids)
    assert "küresel" in geri and "bağ" in geri, f"geri okuma bozuk: {geri!r}"
    # GeometrikTokenizer uyumluluk katmanı
    assert isinstance(tok.sozluk, dict) and len(tok.sozluk) == tok.sozluk_boyutu
    padded = tok.encode("merhaba", pad=True)
    assert len(padded) == 8                       # sağa hizalı, pencereye sabit
    assert hasattr(tok, "fit")                    # takma ad
    # kaydet/yükle round-trip
    yol = os.path.join(KOK, "_test_bpe.json")
    try:
        tok.kaydet(yol)
        tok2 = BPETokenizer(baglam_penceresi=8)
        tok2.yukle(yol)
        assert tok2.encode("küresel bağ katmanı") == ids
    finally:
        if os.path.exists(yol):
            os.remove(yol)


def test_kapasite_raporu():
    """Kapasite muhasebesi formüllerle tutarlı (n⁴, n^2K, gerçek param)."""
    cfg = dict(n=64, katman_sayisi=3, baglam_penceresi=8, emb_dim=32,
               num_heads=4, sozluk_boyutu=256, dropout=0.0)
    model = HiperGeometrikAI(**cfg, bilgilendir=False)
    r = model.kapasite_raporu()
    assert r["gercek_parametre"] == model.gercek_parametre_sayisi() > 0
    assert r["sanal_kose"] == 64 ** 2
    assert r["katman_basi_sanal_operator"] == 64 ** 4
    assert r["etkilesim_uzayi_ust_siniri"] == 64 ** (2 * 3)
    assert r["katman_basi_tam_matris_ram_gb"] == 64 ** 4 * 4 / 1024 ** 3


def test_mini_egitim_loss_duser():
    """Öğrenmesi kolay desen üzerinde loss gözle görülür şekilde düşmeli."""
    torch.manual_seed(0)
    model = HiperGeometrikAI(**KUCUK, bilgilendir=False)
    desen = [10 + (i % 4) for i in range(400)]        # 10,11,12,13,10,11,...
    x = torch.tensor([desen[i:i + 8] for i in range(392)], dtype=torch.long)
    y = torch.tensor(desen[8:], dtype=torch.long)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    ilk = son = None
    for adim in range(150):
        i = (adim * 16) % (392 - 16)
        opt.zero_grad()
        loss = loss_fn(model(x[i:i + 16]), y[i:i + 16])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if adim == 0:
            ilk = loss.item()
        son = loss.item()
    assert son < ilk * 0.5, f"loss yeterince düşmedi: {ilk:.3f} → {son:.3f}"


if __name__ == "__main__":
    print("Hiper-Geometrik AI — duman testi başlıyor...\n")
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
    print(f"\n{'🎉 TÜM TESTLER GEÇTİ' if basarisiz == 0 else f'⚠️ {basarisiz} test başarısız'}")
    sys.exit(1 if basarisiz else 0)
