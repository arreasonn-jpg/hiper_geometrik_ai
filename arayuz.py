# -*- coding: utf-8 -*-
"""Hiper-Geometrik AI — Gradio sohbet arayüzü.

Değişiklik geçmişi (önemli):
- N_GEN dropdown artık GERÇEKTEN çalışıyor: seçim değişince model o boyutta
  yeniden kurulur ve varsa ağırlığı yüklenir (eski sürümde dropdown değeri
  hiçbir yerde kullanılmıyordu).
- Sinir ağı üretiminden elle kelime yasağı listesi ('üç', 'bir', 've', ...)
  kaldırıldı; bunun yerine ortak.metin_uret top-k örnekleme + genel tekrar
  cezası kullanır.
- Talimat verisi dosyası yoksa gömülü ZENGIN setine düşülür.
"""
import json
import os
import re
import sys
import time

KOK = os.path.abspath(os.path.dirname(__file__))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

import gradio as gr

import ortak
from egitim.talimat_toplayici import ZENGIN
from mimari.kuresel_model import HiperGeometrikAI  # noqa: F401  (referans/tutarlılık)
from mimari.tokenizer import GeometrikTokenizer


def _gradio_major():
    """Gradio ana sürümü (css parametresinin yeri 6.0'da launch()'a taşındı)."""
    try:
        return int(gr.__version__.split(".")[0])
    except (AttributeError, ValueError):
        return 0


SISTEM = {"model": None, "tok": None, "n_gen": 1000, "talimatlar": [], "loglar": [], "modeller": {}}

# Jaccard benzerliğiyle talimat eşleştirmede kabul eşiği (elle ayarlanabilir).
BENZERLIK_ESIGI = 0.40


def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"
    SISTEM["loglar"].append(s)
    SISTEM["loglar"] = SISTEM["loglar"][-40:]
    return "\n".join(SISTEM["loglar"])


def norm(t):
    """Türkçe metni eşleştirme için sadeleştir — güvenli replace ile."""
    t = (t or "").lower().strip()
    t = t.replace("'", " ").replace("\u2019", " ").replace("\u2018", " ")
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
    t = t.replace("neresidir", "neresi")
    return t


def model_yukle(n):
    """Verilen boyut için model kurar (önbellekli), varsa ağırlığını yükler."""
    if n in SISTEM["modeller"]:
        SISTEM["model"] = SISTEM["modeller"][n]
        SISTEM["n_gen"] = n
        return SISTEM["model"]

    model = ortak.model_olustur(len(SISTEM["tok"].sozluk), n, 8)
    for yol in [ortak.model_yolu(n, talimat=True), ortak.model_yolu(n)]:
        if os.path.exists(yol):
            try:
                ortak.agirlik_yukle(model, yol)
                log(f"Model yüklendi: {os.path.basename(yol)}")
                break
            except Exception as e:
                log(f"Uyarı {os.path.basename(yol)}: {e}")
        else:
            log(f"Ağırlık yok: {os.path.basename(yol)}")
    model.eval()
    SISTEM["modeller"][n] = model
    SISTEM["model"] = model
    SISTEM["n_gen"] = n
    return model


def model_degistir(n):
    """N_GEN dropdown değişince çağrılır; modeli yeniden kurar/yükler."""
    model_yukle(int(n))
    log(f"Mimari değişti: n={n}")
    return durum()


def intent_cevap(mesaj):
    """Kural tabanlı + benzerlik: bilinen sorulara net cevap."""
    q = norm(mesaj)

    # Başkent
    if ("baskent" in q or "ankara" in q) and (
        "turkiye" in q or "neresi" in q or "nedir" in q or "baskent" in q
    ):
        return "Türkiye Cumhuriyeti'nin başkenti Ankara'dır.", 1.0
    if "baskent" in q and "neresi" in q:
        return "Türkiye Cumhuriyeti'nin başkenti Ankara'dır.", 1.0

    # Selamlaşma
    if any(x in q for x in ["merhaba", "selam", "gunaydin", "nasilsin"]):
        return "Merhaba, iyiyim teşekkür ederim. Size nasıl yardımcı olabilirim?", 1.0
    if "nasil" in q and "sin" in q:
        return "Merhaba, iyiyim teşekkür ederim. Size nasıl yardımcı olabilirim?", 1.0

    # Duygu
    if "duygu" in q:
        return "Duygu, insanın dış dünyadaki olaylara karşı hissettiği psikolojik tepkilerdir.", 1.0

    # Yapay zeka
    if "yapay" in q and "zeka" in q:
        return "Yapay zeka, insan zekasını taklit eden, öğrenen ve problem çözen bilgisayar sistemleridir.", 1.0

    # Tesseract / Fraktal
    if "tesseract" in q or "hiperkup" in q:
        return "Tesseract, üç boyutlu küpün dört boyutlu karşılığı olan hiper küptür.", 1.0
    if "fraktal" in q:
        return "Fraktal, her ölçekte kendine benzeyen karmaşık geometrik şekillerdir.", 1.0

    # Kimlik
    if "kimsin" in q or "adin ne" in q:
        return "Ben Hiper-Geometrik Fraktal AI. Türkçe sohbet için tasarlandım.", 1.0

    # Atatürk
    if "ataturk" in q:
        return "Mustafa Kemal Atatürk, Türkiye Cumhuriyeti'nin kurucusu ve ilk cumhurbaşkanıdır.", 1.0

    # Teşekkür / veda
    if "tesekkur" in q:
        return "Rica ederim, her zaman yardımcı olmaktan mutluluk duyarım.", 1.0
    if "gorusuruz" in q or "hosca" in q:
        return "Görüşmek üzere, kendinize iyi bakın.", 1.0

    # Veri seti benzerliği (talimat_verisi.json'dan; dosya yoksa ZENGIN)
    best, best_score = None, 0.0
    q_set = set(q.split())
    for it in SISTEM["talimatlar"]:
        s = norm(it.get("soru", ""))
        s_set = set(s.split())
        if not s_set:
            continue
        inter = len(q_set & s_set)
        union = len(q_set | s_set) or 1
        score = inter / union
        if s in q or q in s:
            score += 0.55
        if score > best_score:
            best_score, best = score, it
    if best and best_score >= BENZERLIK_ESIGI:
        cevap = (best.get("cevap") or "").strip()
        if not cevap:
            return None, best_score
        guzel = cevap[0].upper() + cevap[1:]
        if not guzel.endswith("."):
            guzel += "."
        return guzel, best_score
    return None, best_score


def baslat():
    ortak.log_kur()
    tok = GeometrikTokenizer(8000)

    sozluk = ortak.sozluk_yolu()
    korpus = ortak.korpus_yolu()
    if os.path.exists(sozluk):
        tok.yukle(sozluk)
        log(f"Kilitli sözlük yüklendi: {len(tok.sozluk)}")
    elif os.path.exists(korpus):
        with open(korpus, "r", encoding="utf-8") as f:
            tok.fit(f.read(800000))
        tok.kaydet(sozluk)
        log(f"Sözlük oluşturuldu: {len(tok.sozluk)}")
    else:
        tok.fit(" ".join(f"{it['soru']} {it['cevap']}" for it in ZENGIN))
        log(f"⚠️ Veri yok; gömülü talimat setinden minik sözlük: {len(tok.sozluk)}")
    SISTEM["tok"] = tok

    talimat_yol = os.path.join(KOK, "talimat_verisi.json")
    if os.path.exists(talimat_yol):
        with open(talimat_yol, "r", encoding="utf-8") as f:
            SISTEM["talimatlar"] = json.load(f)
    else:
        SISTEM["talimatlar"] = list(ZENGIN)  # dosya yoksa gömülü sete düş
    log(f"Talimat: {len(SISTEM['talimatlar'])} örnek")

    model_yukle(1000)


def sinir_agi_uret(prompt, max_token=24):
    """Sinir ağı üretimi — ortak.metin_uret üzerinden (kelime yasağı yok)."""
    tok = SISTEM["tok"]
    model = SISTEM["model"]
    ids = tok.encode(prompt) or [2]
    kelimeler = ortak.metin_uret(model, tok, ids, baglam=8, max_token=int(max_token),
                                 sicaklik=0.7, top_k=40)
    return " ".join(kelimeler).strip()


def chat(mesaj, history, max_token, talimat_modu):
    history = history or []
    if not mesaj or not mesaj.strip():
        yield history, ""
        return

    history = history + [
        {"role": "user", "content": mesaj},
        {"role": "assistant", "content": ""},
    ]

    if talimat_modu:
        cevap, skor = intent_cevap(mesaj)
        if cevap:
            history[-1]["content"] = cevap
            log(f"Intent Match skor={skor:.2f}: {mesaj[:40]}")
            yield history, ""
            return

    prompt = f"soru {mesaj.strip()} cevap" if talimat_modu else mesaj.strip()
    metin = sinir_agi_uret(prompt, int(max_token))
    if not metin:
        metin = "Anladım. Lütfen soruyu biraz daha açık yazar mısınız?"
    else:
        metin = metin[0].upper() + metin[1:]
        if not metin.endswith("."):
            metin += "."
    history[-1]["content"] = metin
    log(f"Neural: {mesaj[:40]}")
    yield history, ""


def durum():
    n = len(SISTEM["tok"].sozluk) if SISTEM["tok"] else 0
    model = SISTEM["model"]
    param = f"{sum(p.numel() for p in model.parameters()):,}" if model else "-"
    return (
        f"Mimari: {SISTEM['n_gen']}-Gen (~{param} parametre)\n"
        f"Sözlük: {n} kelime (kilitli)\n"
        f"Talimat: {len(SISTEM['talimatlar'])} örnek\n"
        f"Motor: Intent Match + Neural\n"
        f"Sürüm: v14.0.0"
    )


baslat()

CSS = """
body{background:#0a0a0f!important;color:#f3f4f6!important;font-family:Inter,sans-serif}
.panel-kart{background:rgba(20,20,32,.85);border:1px solid rgba(139,92,246,.25);border-radius:12px;padding:14px;margin-bottom:10px}
.accent-title{background:linear-gradient(135deg,#8b5cf6,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800}
"""


def _chatbot(**kwargs):
    """Gradio 4/5 ('messages' tipi gerekli) ve 6+ (tek format) uyumlu Chatbot."""
    try:
        return gr.Chatbot(type="messages", **kwargs)
    except TypeError:
        return gr.Chatbot(**kwargs)


# Gradio 6.0+ css'i launch()'a taşımıştır; 4/5'te yalnızca Blocks kurucusunda geçerlidir.
_CSS_IN_LAUNCH = _gradio_major() >= 6
_blocks_kwargs = dict(title="Hiper-Geometrik AI v14.0.0")
if not _CSS_IN_LAUNCH:
    _blocks_kwargs["css"] = CSS

with gr.Blocks(**_blocks_kwargs) as demo:
    gr.HTML(
        "<div style='text-align:center'>"
        "<h1 class='accent-title'>Hiper-Geometrik AI — v14.0.0</h1>"
        "<p style='color:#9ca3af'>Kilitli Sözlük • Intent Match • Net Asistan Cevapları</p>"
        "</div>"
    )
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["panel-kart"]):
                gr.Markdown("### Model")
                gen = gr.Dropdown(
                    [500, 1000, 2000], value=1000, label="N_GEN (model boyutu)",
                    info="Değiştirince model yeniden kurulur ve ağırlığı yüklenir (varsa).",
                )
                talimat = gr.Checkbox(True, label="Sohbet Asistanı Modu")
            with gr.Group(elem_classes=["panel-kart"]):
                gr.Markdown("### Durum")
                st = gr.Textbox(value=durum(), lines=6, interactive=False)
                gr.Button("Yenile", size="sm").click(durum, None, st)
            with gr.Group(elem_classes=["panel-kart"]):
                gr.Markdown("### Log")
                gr.Textbox(value=log("Hazır"), lines=7, interactive=False)
        with gr.Column(scale=3):
            chatbox = _chatbot(label="Canlı Sohbet", height=520)
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Örn: Türkiye'nin başkenti neresidir?",
                    scale=8,
                    show_label=False,
                    container=False,
                )
                btn = gr.Button("Gönder", scale=2, variant="primary")
            with gr.Row():
                gr.Button("Temizle", size="sm").click(lambda: ([], ""), None, [chatbox, msg])
                b1 = gr.Button("Merhaba, nasılsın?", size="sm")
                b2 = gr.Button("Türkiye'nin başkenti neresidir?", size="sm")
                b3 = gr.Button("Duygu nedir?", size="sm")
                b4 = gr.Button("Yapay zeka nedir?", size="sm")
            with gr.Accordion("Parametreler", open=False):
                max_tok = gr.Slider(8, 60, value=24, step=1, label="Max Kelime")
            b1.click(lambda: "Merhaba, nasılsın?", None, msg)
            b2.click(lambda: "Türkiye'nin başkenti neresidir?", None, msg)
            b3.click(lambda: "Duygu nedir?", None, msg)
            b4.click(lambda: "Yapay zeka nedir?", None, msg)
            # N_GEN dropdown artık gerçek: değişim → model_degistir
            gen.change(model_degistir, gen, st)
            btn.click(chat, [msg, chatbox, max_tok, talimat], [chatbox, msg])
            msg.submit(chat, [msg, chatbox, max_tok, talimat], [chatbox, msg])

if __name__ == "__main__":
    launch_kwargs = dict(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        inbrowser=True,
        show_error=True,
    )
    if _CSS_IN_LAUNCH:
        launch_kwargs["css"] = CSS
    demo.launch(**launch_kwargs)
