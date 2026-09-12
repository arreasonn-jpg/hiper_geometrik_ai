# -*- coding: utf-8 -*-
"""
Hiper-Geometrik AI — Bütünleşik Model Gövdesi
=============================================
(Kronecker zinciri + seyrek 'boş küme' bellek)

Mimari akış (her örnek için):
    token id'leri (B, S)
      → Embedding + pozisyon kodlaması                       (B, S, emb)
      → Nedensel çok kafalı dikkat (Pre-LN + ReZero + SDPA)   (B, S, emb)
      → düzleştir                                             (B, S·emb)
      → [Seyrek bellek] pencere anahtarı → hash tablosu       (B, seyrek_boyut)
        "gen" vektörü (boş küme → sıfır; rapor 9) — gen_kopru ile
        bağlam vektörüne eklenir. Seyrek tablo HANGİ kümelerin dolu
        olduğunu taşır; Kronecker zinciri etkileşimi işler (rapor 9.5).
      → [0→1 Katman]  GeometrikVeriEncoder: tanh(u ⊗ v)         (B, n, n)
        ("n² sanal köşe" — README'deki 1. Katman)
      → [1→2+ Katman] KureselZincir: K adet A @ X @ B sandviçi  (B, n, n)
        (katman başına n⁴ sanal operatör girdisi = Kronecker İllüzyonu)
      → LayerNorm + FraktalDecoder (bilinear odak mercekleri)
      → ham logitler (B, sozluk_boyutu)   [softmax YOK — loss hallediyor]

Tek doğruluk kaynağı (rapor 8.1):
    model_olustur(...) → model kurulumu TEK yerden; 'n' gerçekten iletilir.
    agirlik_yukle(...) → strict=True varsayılan: uyumsuzluk sessizce yutulmaz.
    kapasite_raporu()  → gerçek parametre / sanal kapasite / seyrek doluluk
                         muhasebesi (dürüst sayılar).
"""
import torch
import torch.nn as nn

from kuresel_bag import KureselZincir
from encoder import GeometrikVeriEncoder
from decoder import FraktalDecoder
from hiper_attention import HiperGeometrikAttention
from seyrek_tablo import HashlenmisKureselTablo

# ── Tek doğruluk kaynağı: varsayılan mimari ayarları (rapor 8.3 / 9.6.4) ──
VARSAYILAN_N = 256              # küresel bağ boyutu (öneri: 128–256)
VARSAYILAN_KATMAN = 4           # bilinear katman sayısı K (öneri: 4–8)
VARSAYILAN_BAGLAM = 16          # bağlam penceresi (token)
VARSAYILAN_EMB = 128            # gömme boyutu
VARSAYILAN_KAFA = 4             # dikkat kafası sayısı
VARSAYILAN_SOZLUK = 8000        # BPE sözlük üst sınırı
VARSAYILAN_SEYREK_SATIR = 1_048_576   # seyrek tablo satırı (128 MB @ 32 boyut)
VARSAYILAN_SEYREK_BOYUT = 32          # her "küme"nin vektör boyutu


class HiperGeometrikAI(nn.Module):
    def __init__(self, n=VARSAYILAN_N, baglam_penceresi=VARSAYILAN_BAGLAM,
                 sozluk_boyutu=VARSAYILAN_SOZLUK, emb_dim=VARSAYILAN_EMB,
                 num_heads=VARSAYILAN_KAFA, katman_sayisi=VARSAYILAN_KATMAN,
                 dropout=0.1, checkpoint_kullan=False, bilgilendir=True,
                 seyrek_tablo_boyutu=VARSAYILAN_SEYREK_SATIR,
                 seyrek_boyut=VARSAYILAN_SEYREK_BOYUT, seyrek_tablo_sayisi=1):
        """
        seyrek_tablo_boyutu : 0 → seyrek bellek kapalı; >0 → fiziksel satır sayısı
        seyrek_tablo_sayisi : 1 (tek hash) veya 2 (Bloom tarzı çift hash, rapor 9.4.1)
        """
        super().__init__()
        self.n = n
        self.baglam_penceresi = baglam_penceresi
        self.katman_sayisi = katman_sayisi
        self.sozluk_boyutu = sozluk_boyutu

        self.kelime_gomme = nn.Embedding(sozluk_boyutu, emb_dim, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.randn(1, baglam_penceresi, emb_dim) * 0.02)

        # Nedensel dikkat — hiper_attention.py artık ölü kod DEĞİL (rapor 8.4.5)
        self.dikkat = HiperGeometrikAttention(emb_dim, num_heads, dropout, is_causal=True)

        # [Seyrek bellek — rapor 9] Kavramsal uzayı katrilyonların üzerinde
        # (sözlük^pencere) olan hash'lenmiş 'boş küme' tablosu. Başlangıçta
        # tamamı sıfır; yalnız veride görülen pencerelerin satırları dolar.
        self.seyrek_tablo = None
        if seyrek_tablo_boyutu and seyrek_tablo_boyutu > 0:
            self.seyrek_tablo = HashlenmisKureselTablo(
                tablo_boyutu=seyrek_tablo_boyutu, boyut=seyrek_boyut,
                tablo_sayisi=seyrek_tablo_sayisi)
            # Köprü: gen vektörünü bağlam vektörüne enjekte eder.
            # DİKKAT (ölü-yol tuzağı): köprüyü SIFIR başlatmak yanlış olurdu —
            # tablo da sıfırken gradyan hiçbir tarafa akamazdı (0×0). Köprü
            # normal başlar; tablo sıfırken gen=0 → katkı başlangıçta nötr,
            # tablo doldukça köprü onu kullanmayı ÖĞRENİR.
            self.gen_kopru = nn.Linear(seyrek_boyut, baglam_penceresi * emb_dim)
            nn.init.zeros_(self.gen_kopru.bias)

        # 0→1 katmanı: dış çarpım köprüsü (rapor 5.3)
        self.encoder = GeometrikVeriEncoder(baglam_penceresi * emb_dim, n)

        # 1→2+ katmanları: K adet bilinear (A@X@B) sandviç zinciri (rapor 8.2/8.3)
        self.kuresel_bag = KureselZincir(n, katman_sayisi, dropout, checkpoint_kullan)

        self.norm_cikis = nn.LayerNorm(n)
        self.decoder = FraktalDecoder(n, sozluk_boyutu)

        if bilgilendir:
            r = self.kapasite_raporu()
            seyrek_bilgi = (f", seyrek bellek {r['seyrek_fiziksel_satir']:,} satır "
                            f"(kavramsal uzay ~{float(r['kavramsal_anahtar_uzayi']):.2e})"
                            if self.seyrek_tablo is not None else ", seyrek bellek kapalı")
            print(f"[MİMARİ] n={n}, K={katman_sayisi} küresel bağ zinciri kuruldu: "
                  f"{r['gercek_parametre']:,} gerçek parametre "
                  f"({r['yogun_gercek_parametre']:,} yoğun){seyrek_bilgi} | "
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

        # Seyrek bellek enjeksiyonu: pencerenin 'gen' vektörü (boş küme → 0)
        if self.seyrek_tablo is not None:
            gen = self.seyrek_tablo(x)            # (B, seyrek_boyut)
            duz = duz + self.gen_kopru(gen)

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
        """TOPLAM gerçek parametre (yoğun + seyrek fiziksel depo)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def yogun_parametre_sayisi(self) -> int:
        """Yalnız yoğun (dense) bileşenlerin parametresi — seyrek tablo hariç."""
        return sum(p.numel() for ad, p in self.named_parameters()
                   if p.requires_grad and not ad.startswith("seyrek_tablo."))

    def kapasite_raporu(self) -> dict:
        """Gerçek parametre, sanal kapasite ve seyrek bellek muhasebesi.

        DİKKAT: 'sanal' ve 'kavramsal' alanlar gerçek eğitilebilir parametre
        DEĞİLDİR. Seyrek tabloda 'dolu_satır' çalışma-zamanı istatistiğidir:
        fiziksel depo baştan ayrılır, boş satırlar eğitimde asla değişmez.
        """
        zincir = self.kuresel_bag.kapasite()
        r = {
            "n": self.n,
            "katman_sayisi": self.katman_sayisi,
            "baglam_penceresi": self.baglam_penceresi,
            "gercek_parametre": self.gercek_parametre_sayisi(),
            "yogun_gercek_parametre": self.yogun_parametre_sayisi(),
            "sanal_kose": self.n ** 2,                                    # 0→1 dış çarpım
            "katman_basi_sanal_operator": zincir["katman_basi_sanal_operator"],   # n⁴
            "etkilesim_uzayi_ust_siniri": zincir["etkilesim_uzayi_ust_siniri"],   # n^(2K)
            "katman_basi_tam_matris_ram_gb": zincir["katman_basi_sanal_operator"] * 4 / 1024 ** 3,
            "zincirin_gercek_ram_mb": zincir["gercek_parametre"] * 4 / 1024 ** 2,
        }
        if self.seyrek_tablo is not None:
            t = self.seyrek_tablo.kapasite()
            dolu, toplam = self.seyrek_tablo.doluluk_orani()
            r.update({
                "seyrek_fiziksel_satir": t["fiziksel_satir"],
                "seyrek_boyut": t["boyut"],
                "seyrek_fiziksel_parametre": t["fiziksel_parametre"],
                "seyrek_ram_mb": t["ram_mb"],
                "dolu_satir": dolu,
                # Kavramsal anahtar uzayı: sözlük^pencere (Python büyük tamsayısı)
                "kavramsal_anahtar_uzayi": self.sozluk_boyutu ** self.baglam_penceresi,
            })
        return r

    def seyrek_doluluk_metni(self) -> str:
        """'1 katrilyonluk kapasitenin şu an kaç kümesi dolu' istatistiği (rapor 9.4.4)."""
        if self.seyrek_tablo is None:
            return "seyrek bellek kapalı"
        dolu, toplam = self.seyrek_tablo.doluluk_orani()
        uzay = self.sozluk_boyutu ** self.baglam_penceresi
        return (f"dolu küme: {dolu:,}/{toplam:,} fiziksel satır "
                f"(%{100 * dolu / max(1, toplam):.3f}) | "
                f"kavramsal uzay ~{float(uzay):.2e}")


# ══════════════════════════════════════════════════════════════════════════
# TEK DOĞRULUK KAYNAĞI: Model fabrikası (rapor 8.1.3)
# ══════════════════════════════════════════════════════════════════════════
def model_olustur(sozluk_boyutu=VARSAYILAN_SOZLUK, n=VARSAYILAN_N,
                  baglam_penceresi=VARSAYILAN_BAGLAM, emb_dim=VARSAYILAN_EMB,
                  num_heads=VARSAYILAN_KAFA, katman_sayisi=VARSAYILAN_KATMAN,
                  dropout=0.1, checkpoint_kullan=False, aygit="cpu",
                  bilgilendir=True,
                  seyrek_tablo_boyutu=VARSAYILAN_SEYREK_SATIR,
                  seyrek_boyut=VARSAYILAN_SEYREK_BOYUT,
                  seyrek_tablo_sayisi=1):
    """HiperGeometrikAI'ı TEK YERDEN kurar.

    Önceki sürümde bu fonksiyon calistir.py, arayuz.py ve
    egitim/talimat_egitici.py içinde ÜÇ AYRI KOPYA hâlindeydi ve 'n'
    parametresi, anahtar adları eşleşmediği için ('n_gen'/'gen_sayisi'/
    'boyut' ararken model 'n' bekliyordu) HİÇBİR ZAMAN modele iletilmiyordu.
    seyrek_tablo_boyutu=0 → seyrek bellek olmadan kur (eski davranış).
    """
    if aygit == "cuda" and not torch.cuda.is_available():
        print("[MİMARİ] ⚠️ CUDA bulunamadı — CPU'ya dönülüyor.")
        aygit = "cpu"
    model = HiperGeometrikAI(
        n=n, baglam_penceresi=baglam_penceresi, sozluk_boyutu=sozluk_boyutu,
        emb_dim=emb_dim, num_heads=num_heads, katman_sayisi=katman_sayisi,
        dropout=dropout, checkpoint_kullan=checkpoint_kullan,
        bilgilendir=bilgilendir,
        seyrek_tablo_boyutu=seyrek_tablo_boyutu, seyrek_boyut=seyrek_boyut,
        seyrek_tablo_sayisi=seyrek_tablo_sayisi,
    )
    return model.to(aygit)


def agirlik_yukle(model: nn.Module, yol: str, strict: bool = True):
    """Model ağırlığı yükler — strict=True VARSAYILAN (rapor 8.1.5).

    Eski sürümdeki strict=False, mimari değişikliklerinden sonra checkpoint
    uyumsuzluklarını sessizce yutuyordu. Artık uyumsuzluk hata olarak yüzeye
    çıkar; çağıran taraf açık bir uyarıyla raporlayıp rastgele ağırlıkla
    devam etmeyi SEÇEBİLİR ama bu artık sessiz değildir.
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
    satirlar = [
        "═" * 70,
        " HİPER-GEOMETRİK AI — KAPASİTE RAPORU (dürüst muhasebe)",
        "═" * 70,
        f"  Mimari                        : n={r['n']}, K={r['katman_sayisi']} katman, "
        f"bağlam={r['baglam_penceresi']}",
        f"  GERÇEK eğitilebilir parametre : {r['gercek_parametre']:,}  (toplam)",
        f"    ├─ yoğun (dense)            : {r['yogun_gercek_parametre']:,}",
        f"  0→1 katmanı sanal köşe (n²)    : {r['sanal_kose']:,}",
        f"  Katman başına sanal operatör   : {r['katman_basi_sanal_operator']:,}  (n⁴, Kronecker Bᵀ⊗A)",
        f"  Zincir etkileşim üst sınırı   : {r['etkilesim_uzayi_ust_siniri']:.3e}  (n^2K)",
        f"  Katmanı TAM matris tutmak RAM  : {r['katman_basi_tam_matris_ram_gb']:.1f} GB",
        f"  Zincirin GERÇEK RAM tüketimi  : {r['zincirin_gercek_ram_mb']:.2f} MB",
    ]
    if "seyrek_fiziksel_satir" in r:
        satirlar += [
            "─" * 70,
            " SEYREK 'BOŞ KÜME' BELLEĞİ (hashing trick, rapor 9)",
            f"  Kavramsal anahtar uzayı       : {float(r['kavramsal_anahtar_uzayi']):.3e}  (sözlük^pencere)",
            f"  Fiziksel depo                 : {r['seyrek_fiziksel_satir']:,} satır × "
            f"{r['seyrek_boyut']} boyut = {r['seyrek_fiziksel_parametre']:,} parametre "
            f"({r['seyrek_ram_mb']:.0f} MB)",
            f"  Dolu küme (şu an)             : {r['dolu_satir']:,} satır — eğitim "
            "ile büyür,",
            "                                   dokunulmayan satırlar ASLA değişmez (boş küme)",
        ]
    satirlar += [
        "─" * 70,
        "  NOT: 'sanal'/'kavramsal' sayılar gerçek parametre DEĞİLDİR;",
        "  temsil edilen boyutlardır. Öğrenebilir serbestlik her zaman",
        "  gerçek parametre sayısıyla sınırlıdır (düşük-rank kısıtı).",
        "═" * 70,
    ]
    return "\n".join(satirlar)


if __name__ == "__main__":
    # python mimari/kuresel_model.py  →  varsayılan mimarinin kapasite raporu
    print(kapasite_metni())
