# -*- coding: utf-8 -*-
"""Terminal ve Gradio arayüzlerinin ortak çalışma yardımcıları.

Amaç: ``calistir.py`` ve ``arayuz.py`` içinde tekrar eden tokenizer/model yükleme
ve BPE üretim mantığını tek yerde tutmak. Ağır bağımlılıklar (torch, gradio) bu
modülde top-level import edilmez; fonksiyon çağrıldığında açık hata verir.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

SORU_ETIKETI = "soru"
CEVAP_ETIKETI = "cevap"
DURDURMA_KELIMELERI = {"son", "soru", "cevap"}

Logger = Optional[Callable[[str], None]]


def _log(logger: Logger, mesaj: str) -> None:
    if logger:
        logger(mesaj)


def sys_path_hazirla(kok: str) -> None:
    """Eski script düzeniyle uyumlu import yollarını ekle."""
    kok = os.path.abspath(kok)
    for p in [kok, os.path.join(kok, "mimari"), os.path.join(kok, "egitim")]:
        if p not in sys.path:
            sys.path.insert(0, p)


def config_yukle(kok: str, config_yolu: Optional[str] = None):
    sys_path_hazirla(kok)
    from model_config import yukle as model_config_yukle
    return model_config_yukle(config_yolu)


def tokenizer_hazirla(kok: str, baglam: Optional[int] = None,
                      sozluk_boyutu: Optional[int] = None,
                      logger: Logger = None,
                      config_yolu: Optional[str] = None):
    """Kilitli BPE tokenizer yükle; yoksa korpustan kur.

    ``baglam`` ve ``sozluk_boyutu`` verilmezse ``hga/config/model_config.yaml``
    kullanılır. Böylece UI varsayılanları eğitimle aynı kaynaktan gelir.
    """
    sys_path_hazirla(kok)
    from bpe_tokenizer import BPETokenizer

    cfg = config_yukle(kok, config_yolu)
    mcfg = cfg["model"]
    baglam = int(baglam if baglam is not None else mcfg.baglam_penceresi)
    sozluk_boyutu = int(sozluk_boyutu if sozluk_boyutu is not None else mcfg.sozluk_boyutu)

    tok = BPETokenizer(baglam_penceresi=baglam, max_vocab_size=sozluk_boyutu)
    sozluk_yolu = os.path.join(kok, "bpe_sozluk.json")
    korpus_yolu = os.path.join(kok, "turkce_metin.txt")
    if os.path.exists(sozluk_yolu):
        tok.yukle(sozluk_yolu)
        _log(logger, f"🔒 Kilitli BPE sözlüğü yüklendi: {tok.sozluk_boyutu} parça")
    elif os.path.exists(korpus_yolu):
        with open(korpus_yolu, "r", encoding="utf-8") as f:
            tok.fit_on_text(f.read(800000))
        tok.kaydet(sozluk_yolu)
        _log(logger, f"🔧 BPE sözlüğü kuruldu ve kilitlendi: {tok.sozluk_boyutu} parça → bpe_sozluk.json")
    else:
        _log(logger, "⚠️ bpe_sozluk.json / turkce_metin.txt bulunamadı — sözlüksüz (demo) mod.")
    return tok


def bilgi_katmani_hazirla(kok: str, logger: Logger = None):
    """talimat_verisi.json varsa yükle; yoksa yerleşik talimat setine düş."""
    sys_path_hazirla(kok)
    from bilgi_katmani import BilgiKatmani
    from talimat_toplayici import ZENGIN

    talimatlar = []
    yol = os.path.join(kok, "talimat_verisi.json")
    if os.path.exists(yol):
        try:
            with open(yol, "r", encoding="utf-8") as f:
                talimatlar = json.load(f)
        except Exception as e:
            _log(logger, f"⚠️ talimat_verisi.json okunamadı ({e}) — yerleşik set kullanılacak.")
    if not talimatlar:
        talimatlar = ZENGIN
    _log(logger, f"Talimat: {len(talimatlar)} örnek")
    return BilgiKatmani(talimatlar), talimatlar


def model_vocab_boyutu(tokenizer, config_model) -> int:
    """Küçük/demo sözlükte checkpoint uyumu için config vocab'ına dön."""
    mevcut = len(getattr(tokenizer, "sozluk", {}) or {})
    return mevcut if mevcut >= 100 else int(config_model.sozluk_boyutu)


def model_ve_agirlik_yukle(kok: str, tokenizer, n: Optional[int] = None,
                           katman: Optional[int] = None,
                           baglam: Optional[int] = None,
                           logger: Logger = None,
                           config_yolu: Optional[str] = None):
    """Config uyumlu modeli oluştur ve varsa strict checkpoint yükle.

    Dönüş: ``(model, yuklenen_dosya_adi_veya_None, config)``.
    """
    sys_path_hazirla(kok)
    from kuresel_model import model_olustur, agirlik_yukle

    cfg = config_yukle(kok, config_yolu)
    mcfg = cfg["model"]
    n = int(n if n is not None else mcfg.n)
    katman = int(katman if katman is not None else mcfg.katman_sayisi)
    baglam = int(baglam if baglam is not None else mcfg.baglam_penceresi)
    vocab = model_vocab_boyutu(tokenizer, mcfg)

    model = model_olustur(
        vocab,
        n=n,
        baglam_penceresi=baglam,
        emb_dim=mcfg.emb_dim,
        num_heads=mcfg.num_heads,
        katman_sayisi=katman,
        dropout=mcfg.dropout,
        encoder_aktivasyon=mcfg.encoder_aktivasyon,
        zincir_aktivasyon=mcfg.zincir_aktivasyon,
        checkpoint_kullan=mcfg.checkpoint_kullan,
        seyrek_tablo_boyutu=mcfg.seyrek_tablo_boyutu,
        seyrek_boyut=mcfg.seyrek_boyut,
        seyrek_tablo_sayisi=mcfg.seyrek_tablo_sayisi,
        seyrek_erisim_izleme=mcfg.seyrek_erisim_izleme,
    )

    yuklenen = None
    for ad in [f"hiper_model_{n}_talimat.pt", f"hiper_model_{n}.pt"]:
        yol = os.path.join(kok, ad)
        if os.path.exists(yol):
            try:
                agirlik_yukle(model, yol, strict=True)
                _log(logger, f"✅ Model ağırlıkları yüklendi: {ad}")
                yuklenen = ad
                break
            except Exception as e:
                _log(logger, f"⚠️ {ad} yüklenemedi (mimari uyumsuz?): {e}")
    if yuklenen is None:
        _log(logger, "⚠️ Uyumlu ağırlık yok — model rastgele başlatıldı.")
    model.eval()
    return model, yuklenen, cfg


def to_token_ids(tokenizer, metin: str) -> List[int]:
    if hasattr(tokenizer, "encode"):
        res = tokenizer.encode(metin)
    elif hasattr(tokenizer, "text_to_ids"):
        res = tokenizer.text_to_ids(metin)
    else:
        res = [tokenizer.sozluk.get(w.lower(), 1) for w in metin.split()]
    try:
        import torch
        if isinstance(res, torch.Tensor):
            return res.detach().cpu().flatten().tolist()
    except Exception:
        pass
    return list(res)


def _ban_idleri(tokenizer, kelimeler: Optional[Iterable[str]]) -> List[int]:
    if not kelimeler:
        return []
    sozluk = getattr(tokenizer, "sozluk", {}) or {}
    return [int(sozluk[w]) for w in kelimeler if w in sozluk]


def metin_uret(model, tokenizer, prompt: str, baglam: int,
               izinli=None, max_token: int = 40,
               strateji: str = "sample", temperature: float = 0.7,
               special_token_siniri: int = 3,
               ban_kelimeler: Optional[Sequence[str]] = None,
               tekrar_cezasi: bool = False,
               max_kelime: Optional[int] = None,
               kv_cache: bool = True) -> str:
    """Ortak BPE üretim döngüsü.

    ``MODEL_GENERATED`` çıktı doğrulanmış sayılmaz; bu fonksiyon yalnız metin
    üretir. Çağıran UI, cevap etiketini/güven bilgisini açıkça göstermelidir.
    """
    import torch

    eos_id = getattr(tokenizer, "EOS_ID", 3)
    uretilen_ids = list(to_token_ids(tokenizer, prompt)) or [getattr(tokenizer, "BOS_ID", 2)]
    kelimeler: List[str] = []
    parca = ""
    ban = set(_ban_idleri(tokenizer, ban_kelimeler))
    baglam = int(baglam)
    max_token = int(max_token)
    cache = None
    cache_yolu = bool(kv_cache and hasattr(model, "forward_cacheli_pencere"))
    try:
        aygit = next(model.parameters()).device
    except Exception:
        aygit = torch.device("cpu")

    with torch.inference_mode():
        for step in range(max_token):
            pencere = uretilen_ids[-baglam:]
            if len(pencere) < baglam:
                pencere = [0] * (baglam - len(pencere)) + pencere
            girdi = torch.tensor([pencere], dtype=torch.long, device=aygit)
            if cache_yolu:
                logits, cache = model.forward_cacheli_pencere(girdi, cache=cache)
                logits = logits[0]
            else:
                logits = model(girdi)[0]
            logits[:special_token_siniri] = -float("inf")
            if izinli is not None:
                logits[~izinli] = -float("inf")
            if step >= 2:
                for b in ban:
                    logits[b] -= 4.0
            if tekrar_cezasi and uretilen_ids:
                logits[uretilen_ids[-1]] -= 6.0
            if not torch.isfinite(logits).any():
                break

            if strateji == "greedy":
                nxt = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)
                nxt = int(torch.multinomial(probs, num_samples=1).item())
            uretilen_ids.append(nxt)
            if nxt == eos_id:
                break

            ham = getattr(tokenizer, "id_to_kelime", {}).get(nxt, "")
            if not ham or ham in ("<pad>", "<unk>", "<bos>"):
                continue
            if ham.endswith("</w>"):
                kelime = parca + ham[:-4]
                parca = ""
                if kelime in DURDURMA_KELIMELERI:
                    break
                if kelime and not (kelimeler and kelimeler[-1] == kelime):
                    kelimeler.append(kelime)
            else:
                parca += ham
            if max_kelime is not None and len(kelimeler) >= max_kelime:
                break

    if parca and parca not in DURDURMA_KELIMELERI:
        kelimeler.append(parca)
    return " ".join(kelimeler).strip()


__all__ = [
    "SORU_ETIKETI", "CEVAP_ETIKETI", "DURDURMA_KELIMELERI",
    "sys_path_hazirla", "config_yukle", "tokenizer_hazirla",
    "bilgi_katmani_hazirla", "model_vocab_boyutu", "model_ve_agirlik_yukle",
    "to_token_ids", "metin_uret",
]
