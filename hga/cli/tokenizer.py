"""HGA CLI komutu (Faz 2 refactor)."""
from __future__ import annotations

import json
import os
from typing import Optional

from hga.cli.common import KOK


def _veri_canli_smoke(konular=None, cikis=None, out=None, kontrollu: bool = False):
    """Canlı Wikipedia → kalite → manifest smoke raporu."""
    from hga.data import canli_wikipedia_smoke
    konu_listesi = None if konular is None else [k.strip() for k in konular.split(",") if k.strip()]
    fetcher = None
    if kontrollu:
        def fetcher(konu):
            return (f"{konu} Türkçe veri hattı için kontrollü bir smoke metnidir. "
                    f"İstanbul, Iğdır, şeker, çay, öğüt ve üzüm örnekleri Türkçe karakterleri korur. "
                    f"Bu cümleler ağ bağımlılığı olmadan kalite filtresinden geçirilir.")
    rapor = canli_wikipedia_smoke(konu_listesi, proje_kok=KOK, cikis=cikis, fetcher=fetcher)
    if kontrollu:
        rapor.notlar.append("kontrollü fetcher kullanıldı; ağ bağımlılığı yok")
    d = rapor.to_dict()
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"Canlı veri smoke raporu yazıldı: {out}")
    else:
        print("Canlı veri smoke raporu:")
        print(json.dumps(d, ensure_ascii=False, indent=2, sort_keys=True))

def _manifest(yol):
    """Tek dosya için SHA-256 veri manifesti yazdır."""
    if not yol:
        raise SystemExit("manifest komutu için dosya yolu gerekli")
    from hga.data import dosya_hashle
    m = dosya_hashle(yol)
    print("Veri manifesti:")
    for k, v in m.to_dict().items():
        print(f"  {k:<8}: {v}")

def _veri_kalite(yol):
    """Dosya veya mini metin üzerinde veri kalite filtresi."""
    from hga.data import temizle_cumleler
    from hga.experience.corpus import cumlelere_bol
    if yol:
        with open(yol, "r", encoding="utf-8") as f:
            metin = f.read()
    else:
        metin = "Ali ataya bindi. Ali ataya bindi. bozuk � metin. 12345 !!! ???."
    temiz, rapor = temizle_cumleler(cumlelere_bol(metin))
    print("Veri kalite raporu:")
    for k, v in rapor.to_dict().items():
        print(f"  {k:<18}: {v}")
    print("Temiz örnekler:")
    for c in temiz[:5]:
        print(f"  - {c}")

def _checkpoint_rapor(yol, config_yol=None, strict: bool = True,
                      n: Optional[int] = None, katman: Optional[int] = None,
                      baglam: Optional[int] = None, vocab: Optional[int] = None,
                      emb: Optional[int] = None, heads: Optional[int] = None,
                      seyrek_satir: Optional[int] = None,
                      seyrek_boyut: Optional[int] = None):
    """Checkpoint'i yüklemeden model anahtar/şekil uyumluluğunu raporla."""
    if not yol:
        raise SystemExit("checkpoint-rapor komutu için checkpoint yolu gerekli")
    try:
        import torch  # noqa: F401
    except ImportError:
        print("Checkpoint raporu için torch gerekli. Örn: .venv/bin/python -m hga checkpoint-rapor ckpt.pt")
        return
    if not os.path.exists(yol):
        raise SystemExit(f"checkpoint bulunamadı: {yol}")

    from kuresel_model import model_olustur

    from egitim.saglamlik import checkpoint_uyumluluk_raporu
    from hga.config import model_config_yukle

    cfg = model_config_yukle(config_yol)["model"]
    n = int(n if n is not None else cfg.n)
    katman = int(katman if katman is not None else cfg.katman_sayisi)
    baglam = int(baglam if baglam is not None else cfg.baglam_penceresi)
    vocab = int(vocab if vocab is not None else cfg.sozluk_boyutu)
    emb = int(emb if emb is not None else cfg.emb_dim)
    heads = int(heads if heads is not None else cfg.num_heads)
    seyrek_satir = int(seyrek_satir if seyrek_satir is not None else cfg.seyrek_tablo_boyutu)
    seyrek_boyut = int(seyrek_boyut if seyrek_boyut is not None else cfg.seyrek_boyut)

    model = model_olustur(sozluk_boyutu=vocab, n=n, baglam_penceresi=baglam,
                          emb_dim=emb, num_heads=heads, katman_sayisi=katman,
                          dropout=cfg.dropout,
                          encoder_aktivasyon=cfg.encoder_aktivasyon,
                          zincir_aktivasyon=cfg.zincir_aktivasyon,
                          checkpoint_kullan=False, bilgilendir=False,
                          seyrek_tablo_boyutu=seyrek_satir,
                          seyrek_boyut=seyrek_boyut,
                          seyrek_tablo_sayisi=cfg.seyrek_tablo_sayisi,
                          seyrek_erisim_izleme=cfg.seyrek_erisim_izleme)
    rapor = checkpoint_uyumluluk_raporu(model, yol, strict=strict)
    print("Checkpoint uyumluluk raporu:")
    for k in ["ok", "strict", "checkpoint_version", "model_meta", "ortak_anahtar"]:
        print(f"  {k:<20}: {rapor.get(k)}")
    print(f"  eksik               : {len(rapor['eksik'])}")
    print(f"  fazla               : {len(rapor['fazla'])}")
    print(f"  sekil_uyumsuz       : {len(rapor['sekil_uyumsuz'])}")
    for ad, sekil in list(rapor["sekil_uyumsuz"].items())[:8]:
        print(f"    - {ad}: model={sekil['model']} checkpoint={sekil['checkpoint']}")
    if rapor["eksik"]:
        print("  ilk_eksik           :", rapor["eksik"][:8])
    if rapor["fazla"]:
        print("  ilk_fazla           :", rapor["fazla"][:8])

def _perplexity_benchmark(yol=None, checkpoint=None, tokenizer_yol=None,
                          tiny: bool = True, batch_size: int = 64,
                          n: Optional[int] = None, katman: Optional[int] = None,
                          baglam: Optional[int] = None, vocab: Optional[int] = None):
    """Mini Türkçe perplexity smoke'u; torch yoksa güvenli bilgi ver."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print("Perplexity benchmark için torch gerekli. Örn: .venv/bin/python -m hga perplexity --tiny")
        return

    from bpe_tokenizer import BPETokenizer
    from kuresel_model import HiperGeometrikAI, agirlik_yukle

    from hga.config import model_config_yukle
    from hga.evaluation import mini_turkce_corpus, perplexity_benchmark

    if yol:
        with open(yol, "r", encoding="utf-8") as f:
            metin = f.read()
        cumleler = [c.strip() for c in metin.splitlines() if c.strip()] or [metin]
    else:
        cumleler = mini_turkce_corpus()
    metin = "\n".join(cumleler)

    cfg = model_config_yukle()["model"]
    baglam = int(baglam if baglam is not None else (8 if tiny else cfg.baglam_penceresi))
    vocab = int(vocab if vocab is not None else (256 if tiny else cfg.sozluk_boyutu))
    n = int(n if n is not None else (8 if tiny else cfg.n))
    katman = int(katman if katman is not None else (1 if tiny else cfg.katman_sayisi))

    tok = BPETokenizer(baglam_penceresi=baglam, max_vocab_size=vocab, min_freq=1)
    notlar = []
    sozluk = tokenizer_yol or os.path.join(KOK, "bpe_sozluk.json")
    if os.path.exists(sozluk):
        tok.yukle(sozluk)
        notlar.append(f"tokenizer={sozluk}")
    else:
        tok.fit_on_text(metin, verbose=False)
        notlar.append("tokenizer aynı metinde fit edildi: sadece smoke, held-out ölçüm değil")

    model = HiperGeometrikAI(n=n, katman_sayisi=katman, baglam_penceresi=baglam,
                             emb_dim=(8 if tiny else cfg.emb_dim),
                             num_heads=(2 if tiny else cfg.num_heads),
                             sozluk_boyutu=max(len(tok.sozluk), 64),
                             dropout=0.0 if tiny else cfg.dropout,
                             seyrek_tablo_boyutu=0 if tiny else cfg.seyrek_tablo_boyutu,
                             seyrek_boyut=cfg.seyrek_boyut,
                             bilgilendir=False)
    if checkpoint:
        agirlik_yukle(model, checkpoint, strict=True)
        notlar.append(f"checkpoint={checkpoint}")
    else:
        notlar.append("checkpoint yok: rastgele ağırlıklarla smoke ölçümü")

    rapor = perplexity_benchmark(model, tok, cumleler, batch_size=batch_size)
    rapor.notlar.extend(notlar)
    print("Mini Türkçe perplexity benchmark:")
    for k, v in rapor.to_dict().items():
        print(f"  {k:<16}: {v}")

def _tokenizer_benchmark():
    """Mini Türkçe tokenizer benchmark'ı."""
    from bpe_tokenizer import BPETokenizer

    from hga.config import model_config_yukle
    from hga.evaluation import mini_turkce_corpus, tokenizer_kapsami
    corpus = mini_turkce_corpus()
    sozluk_boyutu = model_config_yukle()["model"].sozluk_boyutu
    tok = BPETokenizer(max_vocab_size=sozluk_boyutu, min_freq=1)
    tok.fit_on_text("\n".join(corpus), verbose=False)
    r = tokenizer_kapsami(tok, corpus)
    print("Mini Türkçe tokenizer benchmark:")
    for k, v in r.items():
        print(f"  {k:<24}: {v}")
