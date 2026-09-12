# -*- coding: utf-8 -*-
"""
Hiper-Geometrik AI — Bütünleşik Model Gövdesi (Kronecker Zinciri Sürümü)
=======================================================================

Mimari akış (her örnek için):
    token id'leri (B, S)
      → Embedding + pozisyon kodlaması                      (B, S, emb)
      → Nedensel çok kafalı dikkat (Pre-LN + ReZero + SDPA)  (B, S, emb)
      → düzleştir                                            (B, S·emb)
      → [0→1 Katman]  GeometrikVeriEncoder: tanh(u ⊗ v)       (B, n, n)
        ("n² sanal köşe" — README'deki 1. Katman)
      → [1→2+ Katman] KureselZincir: K adet A @ X @ B sandviçi (B, n, n)
        (katman başına n⁴ sanal operatör girdisi = Kronecker İllüzyonu)
      → LayerNorm + FraktalDecoder (bilinear odak mercekleri)
      → ham logitler (B, sozluk_boyutu)   [softmax YOK — loss hallediyor]

Tek doğruluk kaynağı (rapor 8.1):
    model_olustur(...) → eski 3 dosyaya kopyalanmış model_olustur buraya
                         indirgendi; 'n' artık SESSİZCE kaybolmuyor,
                         gerçekten modele iletiliyor.
    agirlik_yukle(...) → strict=True varsayılan: checkpoint uyumsuzluğu
                         sessizce yutulmaz, açıkça raporlanır.
    kapasite_raporu()  → gerçek parametre / sanal kapasite muhasebesi.
"""
import torch
import torch.nn as nn

from kuresel_bag import KureselZincir
from encoder import GeometrikVeriEncoder
from decoder import FraktalDecoder
from hiper_attention import HiperGeometrikAttention

# ── Tek doğruluk kaynağı: varsayılan mimari ayarları (rapor 8.3) ──────────
VARSAYILAN_N = 256         # küresel bağ boyutu (öneri: 128–256)
VARSAYILAN_KATMAN = 4      # bilinear katman sayısı K (öneri: 4–8)
VARSAYILAN_BAGLAM = 16     # bağlam penceresi (token)
VARSAYILAN_EMB = 128       # gömme boyutu
VARSAYILAN_KAFA = 4        # dikkat kafası sayısı
VARSAYILAN_SOZLUK = 8000   # BPE sözlük üst sınırı


class HiperGeometrikAI(nn.Module):
    def __init__(self, n=VARSAYILAN_N, baglam_penceresi=VARSAYILAN_BAGLAM,
                 sozluk_boyutu=VARSAYILAN_SOZLUK, emb_dim=VARSAYILAN_EMB,
                 num_heads=VARSAYILAN_KAFA, katman_sayisi=VARSAYILAN_KATMAN,
                 dropout=0.1, checkpoint_kullan=False, bilgilendir=True):
        super().__init__()
        self.n = n
        self.baglam_penceresi = baglam_penceresi
        self.katman_sayisi = katman_sayisi
        self.sozluk_boyutu = sozluk_boyutu

        self.kelime_gomme = nn.Embedding(sozluk_boyutu, emb_dim, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.randn(1, baglam_penceresi, emb_dim) * 0.02)

        # Nedensel dikkat — hiper_attention.py artık ölü kod DEĞİL (rapor 8.4.5)
        self.dikkat = HiperGeometrikAttention(emb_dim, num_heads, dropout, is_causal=True)

        # 0→1 katmanı: dış çarpım köprüsü — encoder.py artık modelin İÇİNDE (rapor 5.3)
        self.encoder = GeometrikVeriEncoder(baglam_penceresi * emb_dim, n)

        # 1→2+ katmanları: K adet bilinear (A@X@B) sandviç zinciri (rapor 8.2/8.3)
        self.kuresel_bag = KureselZincir(n, katman_sayisi, dropout, checkpoint_kullan)

        self.norm_cikis = nn.LayerNorm(n)
        self.decoder = FraktalDecoder(n, sozluk_boyutu)

        if bilgilendir:
            r = self.kapasite_raporu()
            print(f"[MİMARİ] n={n}, K={katman_sayisi} küresel bağ zinciri kuruldu: "
                  f"{r['gercek_parametre']:,} gerçek parametre | "
                  f"katman başına ~{r['katman_basi_sanal_operator']:.2e} sanal operatör "
                  f"(Kronecker) | zincir etkileşim üst sınırı ~{r['etkilesim_uzayi_ust_siniri']:.2e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S) token id'leri → (B, sozluk_boyutu) ham logit
        if x.dim() != 2:
            raise ValueError(f"Beklenen girdi (B, S), gelen: {tuple(x.shape)}")
        x = self._pencereye_sigdir(x)
        seq_len = x.size(1)

        emb = self.kelime_gomme(x) + self.pos_encoder[:, :seq_len, :]
        emb = self.dikkat(emb)                    # nedensel dikkat
        duz = emb.reshape(emb.size(0), -1)        # (B, S·emb)

        matris = self.encoder(duz)                # (B, n, n) — 0→1: n² sanal köşe
        matris = self.kuresel_bag(matris)         # (B, n, n) — K bilinear katman
        matris = self.norm_cikis(matris)
        return self.decoder(matris)               # (B, sozluk) — softmax YOK

    def _pencereye_sigdir(self, x: torch.Tensor) -> torch.Tensor:
        """Girdiyi tam bağlam penceresine getir: uzunsa SON pencereyi al,
        kısaysa BAŞINA PAD (id=0) ekle (sağa hizalı bağlam)."""
        S = self.baglam_penceresi
        if x.size(1) > S:
            return x[:, -S:]
        if x.size(1) < S:
            pad = torch.zeros(x.size(0), S - x.size(1), dtype=x.dtype, device=x.device)
            return torch.cat([pad, x], dim=1)
        return x

    # ── Kapasite muhasebesi (dürüst sayılar) ──────────────────────────
    def gercek_parametre_sayisi(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def kapasite_raporu(self) -> dict:
        """Gerçek parametre ve sanal kapasite sayıları.

        DİKKAT: 'sanal' alanlar gerçek eğitilebilir parametre DEĞİLDİR;
        Kronecker yapılı operatörün TEMSİL ettiği boyutlardır. Öğrenebilir
        serbestlik her zaman gerçek parametre sayısıyla sınırlıdır.
        """
        zincir = self.kuresel_bag.kapasite()
        return {
            "n": self.n,
            "katman_sayisi": self.katman_sayisi,
            "baglam_penceresi": self.baglam_penceresi,
            "gercek_parametre": self.gercek_parametre_sayisi(),
            "sanal_kose": self.n ** 2,                                    # 0→1 dış çarpım
            "katman_basi_sanal_operator": zincir["katman_basi_sanal_operator"],   # n⁴
            "etkilesim_uzayi_ust_siniri": zincir["etkilesim_uzayi_ust_siniri"],   # n^(2K)
            "katman_basi_tam_matris_ram_gb": zincir["katman_basi_sanal_operator"] * 4 / 1024 ** 3,
            "zincirin_gercek_ram_mb": zincir["gercek_parametre"] * 4 / 1024 ** 2,
        }


# ══════════════════════════════════════════════════════════════════════════
# TEK DOĞRULUK KAYNAĞI: Model fabrikası (rapor 8.1.3)
# ══════════════════════════════════════════════════════════════════════════
def model_olustur(sozluk_boyutu=VARSAYILAN_SOZLUK, n=VARSAYILAN_N,
                  baglam_penceresi=VARSAYILAN_BAGLAM, emb_dim=VARSAYILAN_EMB,
                  num_heads=VARSAYILAN_KAFA, katman_sayisi=VARSAYILAN_KATMAN,
                  dropout=0.1, checkpoint_kullan=False, aygit="cpu",
                  bilgilendir=True):
    """HiperGeometrikAI'ı TEK YERDEN kurar.

    Önceki sürümde bu fonksiyon calistir.py, arayuz.py ve
    egitim/talimat_egitici.py içinde ÜÇ AYRI KOPYA hâlindeydi ve 'n'
    parametresi, anahtar adları eşleşmediği için ('n_gen'/'gen_sayisi'/
    'boyut' ararken model 'n' bekliyordu) HİÇBİR ZAMAN modele iletilmiyordu:
    n'i büyütmeye çalışan her değişiklik sessizce yok sayılıyordu.
    """
    if aygit == "cuda" and not torch.cuda.is_available():
        print("[MİMARİ] ⚠️ CUDA bulunamadı — CPU'ya dönülüyor.")
        aygit = "cpu"
    model = HiperGeometrikAI(
        n=n, baglam_penceresi=baglam_penceresi, sozluk_boyutu=sozluk_boyutu,
        emb_dim=emb_dim, num_heads=num_heads, katman_sayisi=katman_sayisi,
        dropout=dropout, checkpoint_kullan=checkpoint_kullan,
        bilgilendir=bilgilendir,
    )
    return model.to(aygit)


def agirlik_yukle(model: nn.Module, yol: str, strict: bool = True):
    """Model ağırlığı yükler — strict=True VARSAYILAN (rapor 8.1.5).

    Eski sürümdeki strict=False, mimari değişikliklerinden sonra checkpoint
    uyumsuzluklarını sessizce yutuyordu ("model büyüdü sanılıyor ama
    ağırlıkların bir kısmı hiç yüklenmiyor"). Artık uyumsuzluk hata olarak
    yüzeye çıkar; çağıran taraf açık bir uyarıyla raporlayıp rastgele
    ağırlıkla devam etmeyi SEÇEBİLİR ama bu artık sessiz değildir.
    """
    durum = torch.load(yol, map_location="cpu", weights_only=True)
    return model.load_state_dict(durum, strict=strict)


def agirlik_kaydet(model: nn.Module, yol: str) -> str:
    torch.save(model.state_dict(), yol)
    return yol


def kapasite_metni(model: "HiperGeometrikAI | None" = None, **kwargs) -> str:
    """İnsan tarafından okunabilir dürüst kapasite tablosu."""
    if model is None:
        model = HiperGeometrikAI(bilgilendir=False, **kwargs)
    r = model.kapasite_raporu()
    return (
        "═" * 64 + "\n"
        " HİPER-GEOMETRİK AI — KAPASİTE RAPORU (dürüst muhasebe)\n" +
        "═" * 64 + "\n"
        f"  Mimari                        : n={r['n']}, K={r['katman_sayisi']} katman, "
        f"bağlam={r['baglam_penceresi']}\n"
        f"  GERÇEK eğitilebilir parametre : {r['gercek_parametre']:,}\n"
        f"  0→1 katmanı sanal köşe (n²)    : {r['sanal_kose']:,}\n"
        f"  Katman başına sanal operatör   : {r['katman_basi_sanal_operator']:,}  (n⁴, Kronecker Bᵀ⊗A)\n"
        f"  Zincir etkileşim üst sınırı   : {r['etkilesim_uzayi_ust_siniri']:.3e}  (n^2K)\n"
        f"  Katmanı TAM matris tutmak RAM  : {r['katman_basi_tam_matris_ram_gb']:.1f} GB\n"
        f"  Zincirin GERÇEK RAM tüketimi  : {r['zincirin_gercek_ram_mb']:.2f} MB\n" +
        "─" * 64 + "\n"
        "  NOT: 'sanal' sayılar gerçek parametre DEĞİLDİR; Kronecker yapılı\n"
        "  operatörün temsil ettiği boyutlardır. Öğrenebilir serbestlik her\n"
        "  zaman gerçek parametre sayısıyla sınırlıdır (düşük-rank kısıtı).\n" +
        "═" * 64
    )


if __name__ == "__main__":
    # python mimari/kuresel_model.py  →  varsayılan mimarinin kapasite raporu
    print(kapasite_metni())
