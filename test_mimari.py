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
  - seyrek 'boş küme' tablosu: sıfır başlangıç, yalnız dokunulan
    satırlar dolar, anahtar deterministik, Bloom çift hash          (9.3/9.4)
  - modelde seyrek bellek yolu: iki adımda gen_kopru ÖĞRENİR       (9.5)
  - bilgi katmanı: 3 katmanlı karar mekanizması + beyaz liste       (10.5)
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
from seyrek_tablo import HashlenmisKureselTablo
from bilgi_katmani import (BilgiKatmani, beyaz_liste_olustur,
                           KATMAN_TAM, KATMAN_KISMI, KATMAN_ACIK)

KUCUK = dict(n=64, katman_sayisi=2, baglam_penceresi=8, emb_dim=32,
             num_heads=4, sozluk_boyutu=256, dropout=0.0,
             seyrek_tablo_boyutu=2048, seyrek_boyut=8)


# ══════════════════════════════════════════════════════════════════════
# Mimari çekirdek (rapor 8)
# ══════════════════════════════════════════════════════════════════════
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
                              baglam_penceresi=8, emb_dim=32, bilgilendir=False,
                              seyrek_tablo_boyutu=1024, seyrek_boyut=8)
        assert model.n == n
        assert model.kuresel_bag.katmanlar[0].n == n
        assert model.encoder.proj_u.out_features == n
        assert model.decoder.anlam_cozucu.in_features == 2 * n


def test_strict_kaydet_yukle():
    """strict=True: uyumlu yüklenir, uyumsuz SESSİZCE geçmez (rapor 8.1.5)."""
    kwargs = dict(sozluk_boyutu=128, n=32, katman_sayisi=2,
                  baglam_penceresi=8, emb_dim=16, bilgilendir=False,
                  seyrek_tablo_boyutu=256, seyrek_boyut=8)
    model = model_olustur(**kwargs)
    yol = os.path.join(KOK, "_test_model.pt")
    try:
        torch.save(model.state_dict(), yol)
        model2 = model_olustur(**kwargs)
        sonuc = agirlik_yukle(model2, yol, strict=True)
        assert sonuc.missing_keys == [] and sonuc.unexpected_keys == []
        # uyumsuz mimari (farklı sözlük boyutu) hata VERMELİ
        model3 = model_olustur(sozluk_boyutu=64, n=32, katman_sayisi=2,
                               baglam_penceresi=8, emb_dim=16, bilgilendir=False,
                               seyrek_tablo_boyutu=256, seyrek_boyut=8)
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
    """Kapasite muhasebesi formüllerle tutarlı (n⁴, n^2K, gerçek param, seyrek)."""
    cfg = dict(n=64, katman_sayisi=3, baglam_penceresi=8, emb_dim=32,
               num_heads=4, sozluk_boyutu=256, dropout=0.0,
               seyrek_tablo_boyutu=2048, seyrek_boyut=8)
    model = HiperGeometrikAI(**cfg, bilgilendir=False)
    r = model.kapasite_raporu()
    assert r["gercek_parametre"] == model.gercek_parametre_sayisi() > 0
    assert r["sanal_kose"] == 64 ** 2
    assert r["katman_basi_sanal_operator"] == 64 ** 4
    assert r["etkilesim_uzayi_ust_siniri"] == 64 ** (2 * 3)
    assert r["katman_basi_tam_matris_ram_gb"] == 64 ** 4 * 4 / 1024 ** 3
    # seyrek muhasebe (rapor 9)
    assert r["yogun_gercek_parametre"] < r["gercek_parametre"]
    assert r["seyrek_fiziksel_satir"] == 2048
    assert r["seyrek_fiziksel_parametre"] == 2048 * 8
    assert r["kavramsal_anahtar_uzayi"] == 256 ** 8      # sözlük^pencere
    assert r["dolu_satir"] == 0                           # başlangıçta tamamı boş


# ══════════════════════════════════════════════════════════════════════
# Seyrek 'boş küme' belleği (rapor 9)
# ══════════════════════════════════════════════════════════════════════
def test_seyrek_tablo_bos_baslangic_ve_dolusma():
    """Başlangıçta HER küme boş; optimizasyon sonrası YALNIZ dokunulan
    satırlar dolar (rapor 9.3: boş küme = sıfır vektör, dokunulmayan
    satır asla değişmez)."""
    torch.manual_seed(0)
    t = HashlenmisKureselTablo(tablo_boyutu=1024, boyut=8)
    dolu, toplam = t.doluluk_orani()
    assert dolu == 0 and toplam == 1024

    x = torch.randint(4, 100, (16, 4))
    cikti = t(x)
    assert cikti.shape == (16, 8)
    assert cikti.abs().sum().item() == 0.0        # boş küme → nötr sıfır vektör

    cikti.sum().backward()                        # dokunulan satırlara gradyan akar
    opt = torch.optim.AdamW(t.parameters(), lr=1e-2)
    opt.step()
    dolu2, _ = t.doluluk_orani()
    assert 1 <= dolu2 <= 16, f"yalnız dokunulanlar dolmalı: {dolu2}"


def test_seyrek_tablo_katismayan_dokunulmaz():
    """İki kez lookup: birinci pencerenin satırları dolar, ikinci (farklı)
    pencerenin satırları ilk adımda HÂLÂ boş kalır (bellek izolasyonu)."""
    t = HashlenmisKureselTablo(tablo_boyutu=4096, boyut=4)
    x1 = torch.tensor([[10, 20, 30, 40]])
    x2 = torch.tensor([[99, 88, 77, 66]])
    t(x1).sum().backward()
    opt = torch.optim.SGD(t.parameters(), lr=0.1)
    opt.step()
    v1, v2 = t(x1), t(x2)
    assert v1.abs().sum() > 0                     # görülen pencere doldu
    assert v2.abs().sum() == 0                    # görülmemiş pencere boş kaldı


def test_seyrek_anahtar_deterministik_ve_ayristirici():
    """Aynı pencere → aynı anahtar/adres; farklı pencere → farklı anahtar."""
    t = HashlenmisKureselTablo(tablo_boyutu=1024, boyut=8)
    x = torch.tensor([[5, 9, 2, 7], [5, 9, 2, 7], [5, 9, 2, 8]])
    k = t.anahtar(x)
    assert k[0] == k[1], "aynı pencere farklı anahtar üretti!"
    assert k[0] != k[2], "farklı pencere aynı anahtar üretti!"
    a = t.adres(k, 0)
    assert (a >= 0).all() and (a < t.tablo_boyutu).all()


def test_seyrek_cift_hash_bloom():
    """tablo_sayisi=2 (Bloom tarzı çift hash) çalışır ve her iki tablo da
    adreslenir (rapor 9.4.1)."""
    torch.manual_seed(2)
    t = HashlenmisKureselTablo(tablo_boyutu=512, boyut=8, tablo_sayisi=2)
    cikti = t(torch.randint(4, 100, (6, 4)))
    assert cikti.shape == (6, 8)
    cikti.sum().backward()
    opt = torch.optim.AdamW(t.parameters(), lr=1e-2)
    opt.step()
    dolu, toplam = t.doluluk_orani()
    assert toplam == 1024                          # iki tablo birden
    assert dolu >= 2                               # her iki tabloda da doluş oldu


def test_model_seyrek_yol_ogreniyor():
    """RAPOR 9.5 entegrasyonu: seyrek yol iki adımda canlanır.

    Adım 1'de tablo sıfır → gen=0 → gen_kopru gradyanı sıfır (tasarım gereği:
    köprü sıfırla başlatılsaydı bu yol ÖLÜ kalırdı). Tablo dolduktan sonra
    (adım 2) köprü de gradyan almaya başlar — ölü parametre yok.
    """
    torch.manual_seed(0)
    model = HiperGeometrikAI(**KUCUK, bilgilendir=False)
    # köprü sıfır DEĞİL (ölü yol tuzağı bilinçli olarak önlenmiş)
    assert model.gen_kopru.weight.abs().sum() > 0
    assert model.seyrek_doluluk_metni().startswith("dolu küme: 0/")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    x = torch.randint(4, KUCUK["sozluk_boyutu"], (8, KUCUK["baglam_penceresi"]))
    y = torch.randint(4, KUCUK["sozluk_boyutu"], (8,))

    for _ in range(2):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

    dolu, _ = model.seyrek_tablo.doluluk_orani()
    assert dolu >= 1, "eğitim sonrası hiç küme dolmadı"
    # ikinci adımdan sonra köprü de gradyan alıyor (gen ≠ 0 oldu)
    assert model.gen_kopru.weight.grad is not None
    assert model.gen_kopru.weight.grad.abs().sum() > 0


# ══════════════════════════════════════════════════════════════════════
# Bilgi katmanı: 3 katmanlı karar mekanizması (rapor 10)
# ══════════════════════════════════════════════════════════════════════
def test_bilgi_katmani_tam_eslesme():
    """1. katman: normalize edilmiş soru kayıtla birebir eşleşir.
    (Kesme işareti silinerek 'Türkiye'nin' → 'turkiyenin' eşleşmesi sağlanır.)"""
    bk = BilgiKatmani([{"soru": "türkiyenin başkenti neresidir",
                        "cevap": "türkiye cumhuriyetinin başkenti ankaradır"}])
    katman, cevap, skor, _ = bk.ara("Türkiye'nin başkenti neresidir?")
    assert katman == KATMAN_TAM and skor == 1.0
    assert "ankara" in cevap.lower()


def test_bilgi_katmani_kismi_eslesme():
    """2. katman: kelimeler kısmen örtüşür → beyaz liste için eşleşme döner."""
    bk = BilgiKatmani([{"soru": "su kaç derecede kaynar",
                        "cevap": "su deniz seviyesinde yüz derecede kaynar"}],
                      kurallar=[])            # el kuralları kapalı (izole test)
    katman, cevap, skor, eslesme = bk.ara("su kaç derecede kayar")
    assert katman == KATMAN_KISMI
    assert 0.0 < skor < 1.0
    assert eslesme is not None and "kaynar" in eslesme["cevap"]


def test_bilgi_katmani_acik_genelleme():
    """3. katman: hiçbir eşleşme yok → sinir ağına düş (skor 0)."""
    bk = BilgiKatmani([{"soru": "su kaç derecede kaynar",
                        "cevap": "su deniz seviyesinde yüz derecede kaynar"}],
                      kurallar=[])
    katman, cevap, skor, eslesme = bk.ara("kuantum dolanıklığı nedir")
    assert katman == KATMAN_ACIK and cevap is None and skor == 0.0


def test_bilgi_katmani_kurallar():
    """Standart el kuralları (eski arayuz intent_cevap davranışı) korunur."""
    bk = BilgiKatmani()                              # kurallar="standart"
    for soru in ["merhaba nasılsın", "türkiyenin başkenti neresidir",
                 "fraktal nedir", "sen kimsin"]:
        katman, cevap, skor, _ = bk.ara(soru)
        assert katman == KATMAN_TAM and cevap, f"kural tetiklenmedi: {soru}"


def test_beyaz_liste_maskeleme():
    """KATMAN_KISMI beyaz listesi: yalnız eşleşen kaydın kelimeleri + 'son'
    serbest; bilinmeyen token'lar maskelenir (rapor 10.5.2)."""
    tok = BPETokenizer(baglam_penceresi=8, max_vocab_size=500, min_freq=1)
    metin = ("su deniz seviyesinde yüz derecede kaynar. "
             "kronecker zinciri katman katman büyür. ") * 15
    tok.fit_on_text(metin, verbose=False)
    eslesme = {"soru": "su kaç derecede kaynar",
               "cevap": "su deniz seviyesinde yüz derecede kaynar"}
    vocab = max(len(tok.sozluk), 64)
    maske = beyaz_liste_olustur(tok, eslesme, vocab)
    assert maske.dtype == torch.bool and maske.shape[0] == vocab
    izinli_parcalar = {tok.id_to_kelime[i] for i in maske.nonzero().flatten().tolist()}
    # kayıttaki kelimeler serbest
    for kelime in ["su", "yüz", "derecede", "kaynar"]:
        parcalar = tok._fast_parcala(kelime)
        assert set(parcalar) <= izinli_parcalar, f"{kelime} beyaz listede değil!"
    # kayıtta OLMAYAN bir kelime serbest olmamalı
    yabanci = set(tok._fast_parcala("kronecker")) - izinli_parcalar
    assert yabanci, "'kronecker' da beyaz listede olmamalıydı (kayıt dışı)"


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
