# -*- coding: utf-8 -*-
import sys, os, re, json, time, torch

KOK = os.path.abspath(os.path.dirname(__file__))
for p in [KOK, os.path.join(KOK, "mimari"), os.path.join(KOK, "egitim")]:
    if p not in sys.path: sys.path.insert(0, p)

import gradio as gr
from kuresel_model import (model_olustur, agirlik_yukle,
                           VARSAYILAN_N, VARSAYILAN_KATMAN, VARSAYILAN_BAGLAM)
from bpe_tokenizer import BPETokenizer
from bilgi_katmani import (BilgiKatmani, KATMAN_TAM, KATMAN_KISMI, KATMAN_ACIK,
                           beyaz_liste_olustur)

# Tek doğruluk kaynağı: mimari/kuresel_model.py + bilgi_katmani.py
# (eski yerel model_olustur kopyası ve intent_cevap/norm buraya indirgendi —
#  artık terminal (calistir.py) ile AYNI 3 katmanlı karar mekanizması)
SISTEM = {"model": None, "tok": None, "bilgi": None,
          "n_gen": VARSAYILAN_N, "katman": VARSAYILAN_KATMAN,
          "baglam": VARSAYILAN_BAGLAM, "talimatlar": [], "loglar": []}

def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"
    SISTEM["loglar"].append(s)
    SISTEM["loglar"] = SISTEM["loglar"][-40:]
    return "\n".join(SISTEM["loglar"])

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

    # 3 katmanlı halüsinasyon kontrolü (rapor 10.5) — calistir.py ile ortak
    SISTEM["bilgi"] = BilgiKatmani(SISTEM["talimatlar"])

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

def sinir_agi_uret(prompt, max_token=24, izinli=None):
    """BPE parçalarını kelimelere DOĞRU biçimde birleştirerek üretim yapar.

    izinli (bool maske) verilirse yalnız beyaz listedeki token'lar seçilebilir
    (KATMAN_KISMI kısıtlı üretimi, rapor 10.5.2).
    """
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
            if izinli is not None:
                logits[~izinli] = -1e9            # beyaz liste dışını engelle
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

    izinli = None
    if talimat_modu:
        # 3 katmanlı karar mekanizması (rapor 10.5) — skor artık GİZLİ DEĞİL
        katman, cevap, skor, eslesme = SISTEM["bilgi"].ara(mesaj)

        if katman == KATMAN_TAM:
            # 1. katman: kayıtlı cevap doğrudan — halüsinasyon ~0
            history[-1]["content"] = f"🔎 (kayıtlı bilgi) {cevap}"
            log(f"Katman 1 (tam): {mesaj[:40]}")
            yield history, ""
            return

        if katman == KATMAN_KISMI:
            # 2. katman: eşleşen kaydın kelimeleriyle kısıtlı üretim
            izinli = beyaz_liste_olustur(SISTEM["tok"], eslesme,
                                         SISTEM["model"].sozluk_boyutu)
            onek = f"🤔 (kısmi eşleşme, güven {skor:.2f}) "
        else:
            # 3. katman: açık genelleme — açıkça işaretli
            onek = "⚠️ (doğrulanmamış yapay zeka üretimi) "
    else:
        onek = ""

    prompt = f"soru {mesaj.strip()} cevap" if talimat_modu else mesaj.strip()
    metin = sinir_agi_uret(prompt, int(max_token), izinli=izinli)
    if not metin:
        metin = "Anladım. Lütfen soruyu biraz daha açık yazar mısınız?"
    else:
        metin = metin[0].upper() + metin[1:]
        if not metin.endswith("."):
            metin += "."
    history[-1]["content"] = onek + metin
    log(f"Neural (katman={'2' if izinli is not None else '3'}): {mesaj[:40]}")
    yield history, ""

def durum():
    n = len(SISTEM["tok"].sozluk) if SISTEM["tok"] else 0
    model = SISTEM["model"]
    seyrek = ""
    if model is not None and hasattr(model, "seyrek_doluluk_metni"):
        seyrek = f"Seyrek bellegimiz: {model.seyrek_doluluk_metni()}\n"
    return (
        f"Mimari: n={SISTEM['n_gen']}, K={SISTEM['katman']} bilinear Kronecker zinciri\n"
        f"Baglam: {SISTEM['baglam']} token (BPE, alt-kelime)\n"
        f"Sozluk: {n} parca (KILITLI)\n"
        f"Talimat: {len(SISTEM['talimatlar'])} ornek\n"
        f"{seyrek}"
        f"Motor: 3 Katmanli Bilgi (kayitli/kismi/acik) + Neural\n"
        f"Surum: v15.0"
    )

baslat()

CSS = """
body{background:#0a0a0f!important;color:#f3f4f6!important;font-family:Inter,sans-serif}
.panel-kart{background:rgba(20,20,32,.85);border:1px solid rgba(139,92,246,.25);border-radius:12px;padding:14px;margin-bottom:10px}
.accent-title{background:linear-gradient(135deg,#8b5cf6,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800}
"""

with gr.Blocks(title="Hiper-Geometrik AI v15.0") as demo:
    gr.HTML(
        "<div style='text-align:center'>"
        "<h1 class='accent-title'>Hiper-Geometrik AI — v15.0</h1>"
        "<p style='color:#9ca3af'>Kilitli BPE • Bilinear Kronecker Zinciri • Seyrek Bellek • 3 Katmanli Bilgi Kontrolu</p>"
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
                st = gr.Textbox(value=durum(), lines=7, interactive=False)
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
