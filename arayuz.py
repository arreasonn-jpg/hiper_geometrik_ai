# -*- coding: utf-8 -*-
import sys, os, time

KOK = os.path.abspath(os.path.dirname(__file__))
for p in [KOK, os.path.join(KOK, "mimari"), os.path.join(KOK, "egitim")]:
    if p not in sys.path: sys.path.insert(0, p)

import gradio as gr
from model_config import VARSAYILAN_MODEL_CONFIG
from bilgi_katmani import KATMAN_TAM, KATMAN_KISMI, KATMAN_ACIK, beyaz_liste_olustur
from hga.ui_runtime import (
    SORU_ETIKETI, CEVAP_ETIKETI,
    config_yukle,
    tokenizer_hazirla as ortak_tokenizer_hazirla,
    bilgi_katmani_hazirla as ortak_bilgi_katmani_hazirla,
    model_ve_agirlik_yukle,
    metin_uret,
)

# Tek doğruluk kaynağı: model_config.yaml + hga.ui_runtime + bilgi_katmani.py.
# Terminal (calistir.py) ile aynı tokenizer/model yükleme ve üretim döngüsü kullanılır.
def n_secenekleri():
    return sorted({128, 256, 512, SISTEM["n_gen"]})


SISTEM = {"model": None, "tok": None, "bilgi": None,
          "n_gen": VARSAYILAN_MODEL_CONFIG.n,
          "katman": VARSAYILAN_MODEL_CONFIG.katman_sayisi,
          "baglam": VARSAYILAN_MODEL_CONFIG.baglam_penceresi,
          "talimatlar": [], "loglar": []}

def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"
    SISTEM["loglar"].append(s)
    SISTEM["loglar"] = SISTEM["loglar"][-40:]
    return "\n".join(SISTEM["loglar"])

def modeli_yukle(n_gen=None):
    """Seçili n için config uyumlu modeli ortak runtime üzerinden yükle."""
    n = int(n_gen if n_gen is not None else SISTEM["n_gen"])
    model, _, _ = model_ve_agirlik_yukle(
        KOK, SISTEM["tok"], n=n, katman=SISTEM["katman"],
        baglam=SISTEM["baglam"], logger=log,
    )
    SISTEM["model"] = model
    SISTEM["n_gen"] = n


def modeli_gerekirse_yenile(n_gen):
    try:
        n = int(n_gen)
    except Exception:
        n = SISTEM["n_gen"]
    if n != SISTEM["n_gen"] or SISTEM["model"] is None:
        modeli_yukle(n)


def baslat():
    cfg = config_yukle(KOK)
    mcfg = cfg["model"]
    SISTEM["n_gen"] = mcfg.n
    SISTEM["katman"] = mcfg.katman_sayisi
    SISTEM["baglam"] = mcfg.baglam_penceresi

    # BPE tokenizer (rapor 8.4.6): config'ten gelen varsayılanlarla hazırlanır.
    tok = ortak_tokenizer_hazirla(
        KOK, baglam=SISTEM["baglam"], sozluk_boyutu=mcfg.sozluk_boyutu,
        logger=log,
    )
    SISTEM["tok"] = tok

    # 3 katmanlı halüsinasyon kontrolü (rapor 10.5) — calistir.py ile ortak.
    bilgi, talimatlar = ortak_bilgi_katmani_hazirla(KOK, logger=log)
    SISTEM["bilgi"] = bilgi
    SISTEM["talimatlar"] = talimatlar

    modeli_yukle(SISTEM["n_gen"])

def sinir_agi_uret(prompt, max_token=24, izinli=None):
    """Gradio üretimi; gerçek BPE döngüsü hga.ui_runtime.metin_uret'tedir."""
    tok = SISTEM["tok"]
    model = SISTEM["model"]
    baglam = getattr(model, "baglam_penceresi", SISTEM["baglam"])
    ban = ["üç", "bir", "ve", "için", "ile", "bu", "da", "de", "en", "her", "çok", "tck", "yil", "yıl"]
    return metin_uret(
        model, tok, prompt, baglam, izinli=izinli, max_token=max_token,
        strateji="greedy", special_token_siniri=4, ban_kelimeler=ban,
        tekrar_cezasi=True, max_kelime=16,
    )

def chat(mesaj, history, n_gen, max_token, talimat_modu):
    modeli_gerekirse_yenile(n_gen)
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

    prompt = f"{SORU_ETIKETI} {mesaj.strip()} {CEVAP_ETIKETI}" if talimat_modu else mesaj.strip()
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
                gen = gr.Dropdown(n_secenekleri(), value=SISTEM["n_gen"], label="n (kuresel bag boyutu)")
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
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=False, css=CSS)
