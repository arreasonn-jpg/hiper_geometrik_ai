# -*- coding: utf-8 -*-
"""
Bilgi Katmanı — 3 Katmanlı "Önce Ara, Yoksa Sınırlı Genelle" Mekanizması
========================================================================
(rapor Bölüm 10.5)

Halüsinasyon–genelleme ödünleşimini (rapor 10.3: ezber ↔ genelleme) yönetmek
için üretim tarafında üç karar katmanı:

  1. KATMAN_TAM   — TAM EŞLEŞME: normalize edilmiş soru, kayıtlı bir
                    soru/kural ile birebir eşleşir → KAYITLI cevap
                    doğrudan verilir. Hiçbir şey "üretilmez", yalnızca
                    "hatırlanır" → halüsinasyon riski ~0 (rapor 10.5.1).
  2. KATMAN_KISMI — KISMİ EŞLEŞME (n-gram/Jaccard ≥ eşik): eşleşen kaydın
                    kelimeleriyle BEYAZ LİSTE kurulur; sinir ağı yalnız
                    bu BİLİNEN parçalarla üretim yapar — "aradaki
                    boşlukları serbestçe dolduramaz" (rapor 10.5.2).
  3. KATMAN_ACIK  — AÇIK GENELLEME: eşleşme yok; sinir ağı serbest üretir
                    ama çıktı açıkça [DOĞRULANMAMIŞ] etiketi ve güven
                    skoruyla işaretlenir — skor artık kullanıcıdan
                    GİZLENMEZ (rapor 10.5.3).

Bu, endüstrideki RAG (Retrieval-Augmented Generation) yaklaşımının
basitleştirilmiş hâlidir. Eski arayuz.py'deki intent_cevap() fonksiyonu bu
mekanizmanın ilkel bir örneğiydi (rapor 10.4) — artık tek kaynak burada;
hem terminal (calistir.py) hem Gradio (arayuz.py) aynı katmanı kullanır.
"""
import re

try:  # Bilgi arama saf Python çalışır; beyaz liste maskesi için torch gerekir.
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _torch_gerekli():
    if torch is None:
        raise ImportError("Beyaz liste maskesi oluşturmak için torch gerekli.")
    return torch


KATMAN_TAM = "tam"
KATMAN_KISMI = "kismi"
KATMAN_ACIK = "acik"

KATMAN_ETIKET = {
    KATMAN_TAM: "🔎 [KAYITLI BİLGİ ✓]",
    KATMAN_KISMI: "🤔 [KISMİ EŞLEŞME]",
    KATMAN_ACIK: "⚠️ [DOĞRULANMAMIŞ — sinir ağı üretimi]",
}


def norm(t):
    """Türkçe metni eşleştirme için sadeleştir.

    NOT: Kesme işareti SİLİNİR (boşlukla değiştirilmez): 'Türkiye'nin' →
    'turkiyenin' — kayıtlı 'türkiyenin ...' soruyla birebir eşleşsin.
    (Eski arayuz.py sürümü kesme işaretini boşluğa çevirdiği için bu tam
    eşleşmeler kaçırılıyordu.)
    """
    t = (t or "").lower().strip()
    t = t.replace("'", "").replace("\u2019", "").replace("\u2018", "")
    repl = {
        "\u0131": "i",  # ı
        "\u0130": "i",  # İ
        "\u015f": "s",  # ş
        "\u015e": "s",  # Ş
        "\u011f": "g",  # ğ
        "\u011e": "g",  # Ğ
        "\u00fc": "u",  # ü
        "\u00dc": "u",  # Ü
        "\u00f6": "o",  # ö
        "\u00d6": "o",  # Ö
        "\u00e7": "c",  # ç
        "\u00c7": "c",  # Ç
    }
    for a, b in repl.items():
        t = t.replace(a, b)
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t.replace("neresidir", "neresi")


# ── Standart el kuralları (eski arayuz.py intent_cevap'ından; sıra önemli) ──
STANDART_KURALLAR = [
    (lambda q: ("baskent" in q or "ankara" in q) and
               ("turkiye" in q or "neresi" in q or "nedir" in q or "baskent" in q),
     "Türkiye Cumhuriyeti'nin başkenti Ankara'dır."),
    (lambda q: "baskent" in q and "neresi" in q,
     "Türkiye Cumhuriyeti'nin başkenti Ankara'dır."),
    (lambda q: any(x in q for x in ("merhaba", "selam", "gunaydin", "nasilsin")),
     "Merhaba, iyiyim teşekkür ederim. Size nasıl yardımcı olabilirim?"),
    (lambda q: "nasil" in q and "sin" in q,
     "Merhaba, iyiyim teşekkür ederim. Size nasıl yardımcı olabilirim?"),
    (lambda q: "duygu" in q,
     "Duygu, insanın dış dünyadaki olaylara karşı hissettiği psikolojik tepkilerdir."),
    (lambda q: "yapay" in q and "zeka" in q,
     "Yapay zeka, insan zekasını taklit eden, öğrenen ve problem çözen bilgisayar sistemleridir."),
    (lambda q: "tesseract" in q or "hiperkup" in q,
     "Tesseract, üç boyutlu küpün dört boyutlu karşılığı olan hiper küptür."),
    (lambda q: "fraktal" in q,
     "Fraktal, her ölçekte kendine benzeyen karmaşık geometrik şekillerdir."),
    (lambda q: "kimsin" in q or "adin ne" in q,
     "Ben Hiper-Geometrik Fraktal AI. Türkçe sohbet için tasarlandım."),
    (lambda q: "ataturk" in q,
     "Mustafa Kemal Atatürk, Türkiye Cumhuriyeti'nin kurucusu ve ilk cumhurbaşkanıdır."),
    (lambda q: "tesekkur" in q,
     "Rica ederim, her zaman yardımcı olmaktan mutluluk duyarım."),
    (lambda q: "gorusuruz" in q or "hosca" in q,
     "Görüşmek üzere, kendinize iyi bakın."),
]


class BilgiKatmani:
    """Rapor 10.5'teki 3 katmanlı karar mekanizması."""

    def __init__(self, talimatlar=None, esik_kismi=0.40, kurallar="standart"):
        """
        talimatlar : [{"soru": ..., "cevap": ...}, ...] — kayıtlı bilgi
        kurallar   : "standart" (STANDART_KURALLAR) ya da [(kosul_fn, cevap), ...]
        """
        self.esik_kismi = float(esik_kismi)
        self.kurallar = STANDART_KURALLAR if kurallar == "standart" else list(kurallar or [])
        self.kayitlar = []
        for it in (talimatlar or []):
            self.ekle(it.get("soru", ""), it.get("cevap", ""))

    def ekle(self, soru, cevap):
        """Yeni kayıtlı bilgi ekle (çalışma anında da çağrılabilir)."""
        if not soru or not cevap:
            return
        s = norm(soru)
        self.kayitlar.append({"soru": soru, "cevap": cevap,
                              "soru_norm": s, "soru_kume": set(s.split())})

    def ara(self, metin):
        """→ (katman, cevap|None, skor, eslesme|None)

        katman  : KATMAN_TAM / KATMAN_KISMI / KATMAN_ACIK
        eslesme : KATMAN_KISMI'da en iyi kayıt (beyaz liste için)
        """
        q = norm(metin)

        # 1. katman: el kuralları (yüksek öncelik)
        for kosul, cevap in self.kurallar:
            if kosul(q):
                return KATMAN_TAM, cevap, 1.0, None

        # 1. katman: kayıtlı soruyla birebir eşleşme
        for k in self.kayitlar:
            if q and q == k["soru_norm"]:
                return KATMAN_TAM, k["cevap"], 1.0, k

        # 2. katman: kısmi eşleşme (Jaccard + alt dizgi bonusu)
        if q:
            q_set = set(q.split())
            en_iyi, en_iyi_skor = None, 0.0
            for k in self.kayitlar:
                s_set = k["soru_kume"]
                if not s_set:
                    continue
                skor = len(q_set & s_set) / (len(q_set | s_set) or 1)
                if k["soru_norm"] in q or q in k["soru_norm"]:
                    skor += 0.55
                if skor > en_iyi_skor:
                    en_iyi, en_iyi_skor = k, skor
            if en_iyi and en_iyi_skor >= self.esik_kismi:
                return KATMAN_KISMI, en_iyi["cevap"], en_iyi_skor, en_iyi

        # 3. katman: açık genelleme (sinir ağına düş)
        return KATMAN_ACIK, None, 0.0, None


def beyaz_liste_olustur(tokenizer, eslesme, vocab_boyutu):
    """KATMAN_KISMI için üretim beyaz listesi → bool maske (uzunluk=vocab).

    Yalnızca eşleşen kaydın soru+cevap kelimeleri ile 'son' durdurucusunun
    parçaları serbest bırakılır; model bilinmeyen kelimelerle üretim
    yapamaz (rapor 10.5.2: "sadece bildiğin parçaları kullan").
    """
    torch_mod = _torch_gerekli()
    maske = torch_mod.zeros(int(vocab_boyutu), dtype=torch_mod.bool)
    metin = f"{eslesme['soru']} {eslesme['cevap']} son"
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(metin)
    else:
        ids = [tokenizer.sozluk.get(w.lower(), 1) for w in metin.split()]
    for i in ids:
        i = int(i)
        if 4 <= i < int(vocab_boyutu):   # 0-3 (PAD/UNK/BOS/EOS) kapalı kalır
            maske[i] = True
    return maske
