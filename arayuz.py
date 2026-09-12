# -*- coding: utf-8 -*-
import sys, os, re, json, time, torch

KOK = os.path.abspath(os.path.dirname(__file__))
for p in [KOK, os.path.join(KOK, "mimari"), os.path.join(KOK, "egitim")]:
    if p not in sys.path: sys.path.insert(0, p)

import gradio as gr
from kuresel_model import (model_olustur, agirlik_yukle,
                           VARSAYILAN_N, VARSAYILAN_KATMAN, VARSAYILAN_BAGLAM)
from bpe_tokenizer import BPETokenizer

# Tek doğruluk kaynağı: mimari/kuresel_model.py
# (eski yerel model_olustur kopyası ve inspect-tabanlı ad eşleştirme hilesi kaldırıldı)
SISTEM = {"model": None, "tok": None, "n_gen": VARSAYILAN_N,
          "katman": VARSAYILAN_KATMAN, "baglam": VARSAYILAN_BAGLAM,
          "talimatlar": [], "loglar": []}

def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"
    SISTEM["loglar"].append(s)
    SISTEM["loglar"] = SISTEM["loglar"][-40:]
    return "\n".join(SISTEM["loglar"])

def norm(t):
    """Türkçe metni eşleştirme için sadeleştir — güvenli replace ile."""
    t = (t or "").lower().strip()
    t = t.replace("'", " ").replace("\u2019", " ").replace("\u2018", " ")
    # Tek tek replace (maketrans bozulmasın diye)
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

def intent_cevap(mesaj):
    """Kural tabanlı + benzerlik: bilinen sorulara net cevap."""
    q_raw = (mesaj or "").lower().strip()
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

    # Veri seti benzerliği
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
    if best and best_score >= 0.40:
        cevap = (best.get("cevap") or "").strip()
        if not cevap:
            return None, best_score
        guzel = cevap[0].upper() + cevap[1:]
        if not guzel.endswith("."):
            guzel += "."
        return guzel, best_score
    return None, best_score

def baslat():
    # BPE tokenizer (rapor 8.4.6): kilitli sözlük varsa yükle, yoksa kur
    tok = BPETokenizer(baglam_penceresi=VARSAYILAN_BAGLAM, max_vocab_size=8000)
    sozluk = os.path.join(KOK, "bpe_sozluk.json")
    korpus = os.path.join(KOK, "turkce_metin.txt")
    if os.path.exists(sozluk):
        tok.yukle(sozluk)
        log(f"Kilitli BPE sozlugu yuklendi: {tok.sozluk_boyutu} parca")
    elif os.path.exists(korpus):
        with open(korpus, "r", encoding="utf-8") as f:
            tok.fit_on_text(f.read(800000))
        tok.kaydet(sozluk)
        log(f"BPE sozlugu kurulup kilitlendi: {tok.sozluk_boyutu} parca")
    SISTEM["tok"] = tok

    talimat_yol = os.path.join(KOK, "talimat_verisi.json")
    if os.path.exists(talimat_yol):
        with open(talimat_yol, "r", encoding="utf-8") as f:
            SISTEM["talimatlar"] = json.load(f)
    log(f"Talimat: {len(SISTEM['talimatlar'])} ornek")

    # Model: TEK doğruluk kaynağından (mimari/kuresel_model.py)
    n = SISTEM["n_gen"]
    model = model_olustur(max(len(tok.sozluk), 100), n=n,
                          baglam_penceresi=SISTEM["baglam"],
                          katman_sayisi=SISTEM["katman"])
    for ad in [f"hiper_model_{n}_talimat.pt", f"hiper_model_{n}.pt"]:
        yol = os.path.join(KOK, ad)
        if os.path.exists(yol):
            try:
                agirlik_yukle(model, yol, strict=True)  # rapor 8.1.5
                log(f"Model (strict=True): {ad}")
                break
            except Exception as e:
                log(f"UYARI {ad} yuklenemedi: {str(e)[:140]}")
    model.eval()
    SISTEM["model"] = model

def sinir_agi_uret(prompt, max_token=24):
    """BPE parçalarını kelimelere DOĞRU biçimde birleştirerek üretim yapar
    (kelime içi parçalar bitişik, kelime sonları boşluklu)."""
    tok = SISTEM["tok"]   # BPETokenizer
    model = SISTEM["model"]
    baglam = getattr(model, "baglam_penceresi", SISTEM["baglam"])
    out = list(tok.encode(prompt) or [2])
    kelimeler, parca = [], ""
    ban = set()
    for w in ["üç", "bir", "ve", "için", "ile", "bu", "da", "de", "en", "her", "çok", "tck", "yil", "yıl"]:
        if w in tok.sozluk:
            ban.add(tok.sozluk[w])

    with torch.inference_mode():
        for step in range(max_token):
            pencere = out[-baglam:]
            pencere = [0] * (baglam - len(pencere)) + pencere
            logits = model(torch.tensor([pencere], dtype=torch.long))[0]
            logits[:4] = -1e9
            if step >= 2:
                for b in ban:
                    logits[b] -= 4.0
            if len(out) >= 1:
                logits[out[-1]] -= 6.0
            nxt = int(torch.argmax(logits).item())
            out.append(nxt)
            if nxt == tok.EOS_ID:
                break
            ham = tok.id_to_kelime.get(nxt, "")
            if not ham or ham in ("<pad>", "<unk>", "<bos>"):
                continue
            if ham.endswith("</w>"):
                parca += ham[:-4]
                if parca in ("son", "soru", "cevap"):
                    break
                if parca and not (kelimeler and kelimeler[-1] == parca):
                    kelimeler.append(parca)
                parca = ""
            else:
                parca += ham
            if len(kelimeler) >= 16:
                break
    return " ".join(kelimeler).strip()

def chat(mesaj, history, n_gen, max_token, talimat_modu):
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
    return (
        f"Mimari: n={SISTEM['n_gen']}, K={SISTEM['katman']} bilinear Kronecker zinciri\n"
        f"Baglam: {SISTEM['baglam']} token (BPE, alt-kelime)\n"
        f"Sozluk: {n} parca (KILITLI)\n"
        f"Talimat: {len(SISTEM['talimatlar'])} ornek\n"
        f"Motor: Intent Match + Neural\n"
        f"Surum: v14.0"
    )

baslat()

CSS = """
body{background:#0a0a0f!important;color:#f3f4f6!important;font-family:Inter,sans-serif}
.panel-kart{background:rgba(20,20,32,.85);border:1px solid rgba(139,92,246,.25);border-radius:12px;padding:14px;margin-bottom:10px}
.accent-title{background:linear-gradient(135deg,#8b5cf6,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800}
"""

with gr.Blocks(title="Hiper-Geometrik AI v14.0") as demo:
    gr.HTML(
        "<div style='text-align:center'>"
        "<h1 class='accent-title'>Hiper-Geometrik AI — v14.0</h1>"
        "<p style='color:#9ca3af'>Kilitli BPE Sozlugu • Bilinear Kronecker Zinciri • Intent Match</p>"
        "</div>"
    )
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["panel-kart"]):
                gr.Markdown("### Model")
                gen = gr.Dropdown([128, 256, 512], value=VARSAYILAN_N, label="n (kuresel bag boyutu)")
                talimat = gr.Checkbox(True, label="Sohbet Asistani Modu")
            with gr.Group(elem_classes=["panel-kart"]):
                gr.Markdown("### Durum")
                st = gr.Textbox(value=durum(), lines=6, interactive=False)
                gr.Button("Yenile", size="sm").click(durum, None, st)
            with gr.Group(elem_classes=["panel-kart"]):
                gr.Markdown("### Log")
                gr.Textbox(value=log("Hazir"), lines=7, interactive=False)
        with gr.Column(scale=3):
            chatbox = gr.Chatbot(label="Canli Sohbet", height=520)
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Orn: Turkiye'nin baskenti neresidir?",
                    scale=8,
                    show_label=False,
                    container=False,
                )
                btn = gr.Button("Gonder", scale=2, variant="primary")
            with gr.Row():
                gr.Button("Temizle", size="sm").click(lambda: ([], ""), None, [chatbox, msg])
                b1 = gr.Button("Merhaba, nasilsin?", size="sm")
                b2 = gr.Button("Turkiye'nin baskenti neresidir?", size="sm")
                b3 = gr.Button("Duygu nedir?", size="sm")
                b4 = gr.Button("Yapay zeka nedir?", size="sm")
            with gr.Accordion("Parametreler", open=False):
                max_tok = gr.Slider(8, 60, value=24, step=1, label="Max Kelime")
            b1.click(lambda: "Merhaba, nasilsin?", None, msg)
            b2.click(lambda: "Turkiye'nin baskenti neresidir?", None, msg)
            b3.click(lambda: "Duygu nedir?", None, msg)
            b4.click(lambda: "Yapay zeka nedir?", None, msg)
            btn.click(chat, [msg, chatbox, gen, max_tok, talimat], [chatbox, msg])
            msg.submit(chat, [msg, chatbox, gen, max_tok, talimat], [chatbox, msg])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, css=CSS)
