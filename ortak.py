# -*- coding: utf-8 -*-
"""Ortak yardımcılar: model fabrikası, ağırlık yükleme, sözlük hazırlama, loglama.

Bu modül, daha önce ``calistir.py``, ``talimat_egitici.py`` ve ``arayuz.py``
içinde birebir kopyalanan kodun tek adresidir.

Önemli düzeltme: Eski kopyalarda parametre adı listesi ``["n_gen", "gen_sayisi",
"boyut"]`` aranıyor ve ``HiperGeometrikAI``'ın gerçek parametre adı ``n`` hiçbir
zaman eşleşmediği için n değeri modele iletilmiyordu. Burada doğrudan gerçek
imza kullanılır; ``inspect`` ile isim tahmini yapılmaz.
"""
import logging
import os
import sys

KOK_DIZIN = os.path.abspath(os.path.dirname(__file__))
if KOK_DIZIN not in sys.path:
    sys.path.insert(0, KOK_DIZIN)

import torch  # noqa: E402  (sys.path ayarından sonra import edilmeli)

from mimari.kuresel_model import HiperGeometrikAI  # noqa: E402
from mimari.tokenizer import GeometrikTokenizer  # noqa: E402
from egitim.talimat_toplayici import ZENGIN  # noqa: E402

log = logging.getLogger("hiper")

LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"


def log_kur(seviye=logging.INFO):
    """Konsol için tek noktadan logging yapılandırması."""
    logging.basicConfig(format=LOG_FORMAT, level=seviye, datefmt="%H:%M:%S")
    return log


def model_olustur(sozluk_boyutu, n=1000, baglam_penceresi=8, emb_dim=64, num_heads=4):
    """``HiperGeometrikAI``'ı GERÇEK parametre adlarıyla oluşturur.

    Eski kopyalardaki ``inspect.signature`` tabanlı isim tahmininin aksine
    burada modelin gerçek imzası doğrudan kullanılır; böylece ``n`` değeri
    gerçekten modele iletilir (bkz. tests/test_ortak.py).
    """
    return HiperGeometrikAI(
        sozluk_boyutu=sozluk_boyutu,
        n=n,
        baglam_penceresi=baglam_penceresi,
        emb_dim=emb_dim,
        num_heads=num_heads,
    )


def model_yolu(n, talimat=False):
    """Model ağırlık dosyasının tam yolu (hiper_model_<n>.pt / ..._talimat.pt)."""
    ad = f"hiper_model_{n}_talimat.pt" if talimat else f"hiper_model_{n}.pt"
    return os.path.join(KOK_DIZIN, ad)


def agirlik_yukle(model, yol):
    """Ağırlık dosyasını yükler; uyuşmayan anahtarları RAPORLAR.

    Eski davranış (``strict=False`` + uyarı yok) anahtar/boyut uyuşmazlıklarını
    sessizce yutuyordu. Burada:
    - Eksik/beklenmeyen anahtar varsa uyarı loglanır (ilk birkaçı listelenir).
    - Hiçbir ağırlık eşleşmediyse ``RuntimeError`` fırlatılır (sessiz başlangıç yok).
    """
    durum = torch.load(yol, map_location="cpu", weights_only=True)
    bilgi = model.load_state_dict(durum, strict=False)
    if bilgi.missing_keys or bilgi.unexpected_keys:
        eslesen = len(durum) - len(bilgi.unexpected_keys)
        log.warning(
            "Ağırlık uyuşmazlığı (%s): %d anahtardan %d eşleşti | eksik: %s | beklenmeyen: %s",
            os.path.basename(yol),
            len(durum),
            eslesen,
            bilgi.missing_keys[:5],
            bilgi.unexpected_keys[:5],
        )
        if eslesen == 0:
            raise RuntimeError(
                f"{yol}: Hiçbir ağırlık bu mimariyle eşleşmedi. "
                "Model mimarisi değişmiş olabilir; ağırlığı yeniden eğitin."
            )
    return bilgi


def sozluk_yolu():
    return os.path.join(KOK_DIZIN, "sozluk.json")


def korpus_yolu():
    return os.path.join(KOK_DIZIN, "turkce_metin.txt")


def tokenizer_hazirla(max_vocab_size=8000):
    """Sözlüğü kararlı bir sırayla hazırlar ve tokenizer döndürür.

    Öncelik sırası:
    1. ``sozluk.json`` varsa YÜKLE (kilitli sözlük — eğitim sonrası kimlikler
       değişmemeli).
    2. Yoksa ``turkce_metin.txt`` korpusundan SÖZLÜK KUR ve ``sozluk.json``
       olarak kaydet.
    3. O da yoksa gömülü talimat setinden (ZENGIN) minik bir sözlük kur; model
       veri akışı çalıştırılmadan da hatasız açılsın (sinir ağı kısmı bu
       durumda anlamsız üretir; kural tabanlı yanıtlar yine çalışır).
    """
    tok = GeometrikTokenizer(max_vocab_size=max_vocab_size)
    s_yol, k_yol = sozluk_yolu(), korpus_yolu()

    if os.path.exists(s_yol):
        tok.yukle(s_yol)
        log.info("Kilitli sözlük yüklendi: %s (%d kelime)", s_yol, len(tok.sozluk))
    elif os.path.exists(k_yol):
        with open(k_yol, "r", encoding="utf-8") as f:
            tok.fit(f.read(800000))
        tok.kaydet(s_yol)
        log.info("Sözlük korpusdan kuruldu ve kaydedildi: %s (%d kelime)", s_yol, len(tok.sozluk))
    else:
        metin = " ".join(f"{it['soru']} {it['cevap']}" for it in ZENGIN)
        tok.fit(metin)
        log.warning(
            "sozluk.json ve turkce_metin.txt bulunamadı; gömülü talimat setinden "
            "minik sözlük kuruldu (%d kelime). Sinir ağı kısmı anlamlı üretim için "
            "eğitilmelidir (bkz. README 'Eğitim Akışı').",
            len(tok.sozluk),
        )
    return tok


# Üretim (generation) parametreleri — calistir.py ve arayuz.py ortak kullanır.
DURDURMA_KELIMELERI = {"son", "soru", "cevap", "<eos>"}


def metin_uret(model, tokenizer, giris_ids, baglam, max_token=40,
               sicaklik=0.7, top_k=40, tekrar_cezasi=1.5):
    """Next-token üretimi: top-k örnekleme + genel tekrar cezası.

    Not: Eski arayüzdeki gibi elle seçilmiş kelimeleri yasaklayan bir liste
    YOKTUR ('üç', 'bir', 've', ...). Bu, modelin en sık kelimeleri tekrar
    üretmesini gizlemek için semptomu maskeliyordu. Bunun yerine:
    - özel tokenlar (<PAD>/<UNK>/<BOS>/<EOS>) bloklanır,
    - son pencere içinde geçen tokenlara genel bir tekrar cezası uygulanır,
    - top-k örnekleme ile çeşitlilik sınırlı tutulur.
    """
    uretilen_ids = list(giris_ids)
    kelimeler = []

    with torch.inference_mode():
        for _ in range(max_token):
            pencere = uretilen_ids[-baglam:]
            if len(pencere) < baglam:
                pencere = [0] * (baglam - len(pencere)) + pencere

            logits = model(torch.tensor([pencere], dtype=torch.long))[0]

            # Özel tokenları blokla (PAD, UNK, BOS, EOS)
            logits[:4] = -float("inf")

            # Son penceredeki tokenlara tekrar cezası
            for t in set(pencere):
                if logits[t] > 0:
                    logits[t] /= tekrar_cezasi
                else:
                    logits[t] *= tekrar_cezasi

            # Top-k ile sınırlı, sıcaklıkla ölçeklenmiş örnekleme
            if top_k and top_k < logits.size(0):
                esik = torch.topk(logits, top_k).values[-1]
                logits[logits < esik] = -float("inf")
            probs = torch.softmax(logits / sicaklik, dim=-1)
            next_token = int(torch.multinomial(probs, num_samples=1).item())

            uretilen_ids.append(next_token)
            kelime = tokenizer.id_to_kelime.get(next_token, "")

            if kelime in DURDURMA_KELIMELERI:
                break
            if kelime and kelime not in ("<unk>", "<pad>", "<bos>"):
                if kelimeler and kelimeler[-1] == kelime:
                    continue
                kelimeler.append(kelime)

    return kelimeler
