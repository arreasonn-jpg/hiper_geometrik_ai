# -*- coding: utf-8 -*-
"""
Hash'lenmiş Seyrek "Boş Küme" Tablosu (Hashing Trick / Bloom Embedding)
=======================================================================
(rapor Bölüm 9)

Kavramsal olarak KATRİLYONLARCA "boş küme" (slot) adreslenebilen, fiziksel
olarak sabit boyutlu bir hash'lenmiş gömme tablosu:

  - KAVRAMSAL anahtar uzayı: bağlam penceresinin n-gram uzayı —
    sözluk^pencere. Örn. 8000 parçalık sözlükte 16'lık pencere ≈ 1,8×10⁶¹;
    4'lük pencerede bile 8000⁴ ≈ 4×10¹⁵, yani katrilyon üzeri (rapor 9.3).
  - FİZİKSEL tablo: `tablo_boyutu` satır × `boyut` boyut. RAM =
    satır × boyut × 4 bayt. Fiziksel tablonun TAMAMI RAM'de ayrılır;
    "boş küme" kavramı şudur (rapor 9.4.2): satırın değeri SIFIR vektördür
    ve optimize edici dokunulmayan satıra hiç gradyan/step uygulamaz
    (AdamW'da gradyanı 0 olan sıfır satır aynen sıfır kalır).
  - Eğitim ilerledikçe yalnızca veride gerçekten görülen pencerelere denk
    gelen satırlar dolar; `doluluk_orani()` bu sayıyı ölçer (rapor 9.4.4).

Anahtar tasarımı (rapor 9.6.3 — buradaki seçim): anahtar = TAM BAĞLAM
PENCERESİ (token id dizisi). Pencere → 31-bit karma (taşmasız, vektörel):
polinomsal taban-65537 hash + splitmix tarzı sonlandırıcı; ardından satır
adresi = (anahtar × tuz + kayma) % tablo_boyutu.

Çakışma (rapor 9.4.1): iki farklı pencere aynı satıra düşebilir (modulo
nedeniyle — büyük endüstriyel sistemlerde de olur). `tablo_sayisi=2` ile
Bloom tarzı ÇİFT hash açılır: iki farklı tuzlu tablo, çıktılar toplanır;
çakışma etkisi pratikte azalır.

RAM örnekleri (float32):
      262.144 satır × 32 boyut ≈  32 MB
    1.048.576 satır × 32 boyut ≈ 128 MB   (varsayılan)
   20.000.000  satır × 32 boyut ≈ 2,4 GB  (rapor 9.2'deki örnek)

Dürüst not: nn.Embedding FİZİKSEL tabloyu baştan tahsis eder; bellek kullanımı
dolu satır sayısına değil, tablo_boyutu'na bağlıdır. "Sadece dokunulanlar
bellekte" davranışı (rapor 9.2'nin dikte/LMDB senaryosu, 9.4.3) bu sınıfla
DEĞİL, dinamik anahtar-değer deposuyla sağlanır — bu projede gerek yok.
"""
from typing import List

import torch
import torch.nn as nn


class HashlenmisKureselTablo(nn.Module):
    """Kavramsal olarak devasa, fiziksel olarak sabit boyutlu seyrek bellek.

    Bütün satırlar sıfırla başlar (her küme "boş"). Bir pencere ilk kez
    görüldüğünde adreslenen satır gradyan alır ve eğitimle dolar; hiç
    görülmeyen kümeler sonsuza dek sıfır (nötr) kalır.
    """

    MASK31 = 0x7FFFFFFF   # 2^31 - 1  → tüm çarpımlar 2^63'ü aşmaz (taşma yok)
    TABAN = 65537         # 2^16 + 1 (asal); vocab ≤ 65536 iken konumsal hash

    # `register_buffer` ile kaydedilen tensörler: torch'un `nn.Module.__getattr__`
    # imzası bunları `Tensor | Module` olarak döndürür. Gerçekte hepsi Tensor'dur;
    # sınıf düzeyinde bildirerek statik analize doğru tipi veriyoruz.
    tuzlar: torch.Tensor
    _adim: torch.Tensor
    son_erisim: torch.Tensor
    erisim_sayisi: torch.Tensor

    def __init__(self, tablo_boyutu=1_048_576, boyut=32,
                 hash_tuzu=0x2545F491, tablo_sayisi=1,
                 erisim_izleme: bool = False):
        super().__init__()
        if tablo_sayisi not in (1, 2):
            raise ValueError("tablo_sayisi yalnızca 1 (tek hash) veya 2 (Bloom) olabilir")
        self.tablo_boyutu = int(tablo_boyutu)
        self.boyut = int(boyut)
        self.tablo_sayisi = int(tablo_sayisi)
        self.erisim_izleme = bool(erisim_izleme)

        # Farklı tuzlar → aynı anahtar, farklı tablolarda farklı satıra düşer
        tuzler = [int(hash_tuzu), (int(hash_tuzu) * 2 + 1) & self.MASK31]
        self.register_buffer("tuzlar",
                             torch.tensor(tuzler[:tablo_sayisi], dtype=torch.long))

        # Fiziksel tablo(lar): BAŞLANGIÇTA TAMAMI SIFIR = tüm kümeler "boş"
        self.tablolar = nn.ModuleList(
            [nn.Embedding(self.tablo_boyutu, self.boyut)
             for _ in range(tablo_sayisi)]
        )
        for t in self._tablolar():
            nn.init.zeros_(t.weight)

        # Opsiyonel erişim istatistikleri: varsayılan kapalı tutulur; açılırsa
        # LRU/decay temizliği için son erişim ve erişim sayısı tutulur.
        if self.erisim_izleme:
            self.register_buffer("_adim", torch.zeros((), dtype=torch.long))
            self.register_buffer("son_erisim", torch.full(
                (self.tablo_sayisi, self.tablo_boyutu), -1, dtype=torch.long))
            self.register_buffer("erisim_sayisi", torch.zeros(
                (self.tablo_sayisi, self.tablo_boyutu), dtype=torch.long))

    # ── Anahtar hesaplama (taşmasız int64 aritmetiği, vektörel) ────────
    def anahtar(self, token_idleri: torch.Tensor) -> torch.Tensor:
        """(B, S) token penceresi → (B,) int64 kavramsal anahtar (31-bit).

        acc = (acc·TABAN + token) mod 2^31 — her adımda maskelenir, hiçbir
        ara çarpım 2^63'ü aşmaz (acc < 2^31, TABAN < 2^16 → çarpım < 2^47).
        Ardından splitmix tarzı xor-kaydırma sonlandırıcı (çarpımlar < 2^62).
        """
        x = token_idleri.to(torch.long)
        acc = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for j in range(x.size(1)):
            acc = (acc * self.TABAN + x[:, j]) & self.MASK31
        acc = ((acc ^ (acc >> 16)) * 0x7FEB352D) & self.MASK31
        acc = ((acc ^ (acc >> 15)) * 0x846CA68B) & self.MASK31
        return acc ^ (acc >> 16)

    def adres(self, anahtar: torch.Tensor, tablo_indeksi: int = 0) -> torch.Tensor:
        """Kavramsal anahtar → fiziksel satır numarası (tuz bazlı karıştırma)."""
        tuz = self.tuzlar[tablo_indeksi].to(anahtar.device)
        return (anahtar * tuz + (tablo_indeksi + 1) * 0x9E37) % self.tablo_boyutu

    def _tablo(self, indeks: int) -> "nn.Embedding":
        """ModuleList elemanını gerçek tipiyle döndür.

        `nn.ModuleList.__getitem__` statik olarak `Module` döndürür; bu tablo
        yalnız `nn.Embedding` içerir. Runtime davranışı değişmez.
        """
        katman = self.tablolar[indeks]
        assert isinstance(katman, nn.Embedding)
        return katman

    def _tablolar(self) -> List["nn.Embedding"]:
        """Tüm gömme tablolarını tipli liste olarak ver."""
        return [self._tablo(i) for i in range(len(self.tablolar))]

    # ── Arayüzler ───────────────────────────────────────────────────────
    @torch.no_grad()
    def _erisim_guncelle(self, adresler):
        if not self.erisim_izleme:
            return
        self._adim += 1
        for i, adr in enumerate(adresler):
            uniq = torch.unique(adr.detach().to(self.son_erisim.device))
            self.son_erisim[i, uniq] = self._adim
            self.erisim_sayisi[i, uniq] += 1

    def getir_anahtarla(self, anahtar: torch.Tensor) -> torch.Tensor:
        """Hazır kavramsal anahtarlarla arama (rapor 9.3'teki API)."""
        adresler = [self.adres(anahtar, i) for i in range(self.tablo_sayisi)]
        self._erisim_guncelle(adresler)
        cikti: torch.Tensor = self._tablo(0)(adresler[0])
        for i in range(1, self.tablo_sayisi):
            cikti = cikti + self._tablo(i)(adresler[i])
        return cikti

    def forward(self, token_idleri: torch.Tensor) -> torch.Tensor:
        """(B, S) token penceresi → (B, boyut) 'gen' vektörü.

        Boş kümeler SIFIR vektör döndürür (yukarı katkısı nötr). Yalnız
        adreslenen satırlar gradyan alır — dokunulmayan satırlar eğitim
        boyunca değişmez.
        """
        return self.getir_anahtarla(self.anahtar(token_idleri))

    # ── İzleme (rapor 9.4.4) ────────────────────────────────────────────
    @torch.no_grad()
    def doluluk_orani(self):
        """(dolu_satır, toplam_satır) — sıfırdan farklı satır sayısı.

        DİKKAT: tablonun tamamını tarar (O(satır×boyut)); çağ başı gibi
        seyrek çağırın, eğitim adımı başına değil.
        """
        dolu = 0
        for t in self.tablolar:
            dolu += int((t.weight.abs().sum(dim=1) > 1e-8).sum().item())
        return dolu, self.tablo_boyutu * self.tablo_sayisi

    @torch.no_grad()
    def adres_imzalari(self, token_idleri: torch.Tensor) -> torch.Tensor:
        """Her benzersiz pencerenin fiziksel adres imzası.

        Çift hash/Bloom modunda imza ``(adres_tablo_0, adres_tablo_1)`` olur;
        pratik çakışma ancak imzanın tamamı çakışırsa sayılır. Tek hash'te
        imza tek sütundur. Dönüş şekli: ``(benzersiz_pencere, tablo_sayisi)``.
        """
        x = token_idleri.to(torch.long)
        if x.dim() != 2:
            raise ValueError(f"Beklenen token penceresi (B, S), gelen: {tuple(x.shape)}")
        # Aynı pencerenin tekrarını collision saymamak için önce benzersizleştir.
        uniq = torch.unique(x, dim=0)
        anahtar = self.anahtar(uniq)
        adresler = [self.adres(anahtar, i) for i in range(self.tablo_sayisi)]
        return torch.stack(adresler, dim=1)

    @torch.no_grad()
    def carpisma_istatistigi(self, token_idleri: torch.Tensor,
                             esik: float = 0.05) -> dict:
        """Hash collision oranını ölç.

        ``collision_rate = 1 - benzersiz_adres_imzası / benzersiz_pencere``.
        Tekrarlanan aynı pencere collision kabul edilmez. ``esik`` varsayılan
        %5'tir; oran bunu aşarsa ``uyari=True`` döner.
        """
        imzalar = self.adres_imzalari(token_idleri).cpu()
        benzersiz_pencere = int(imzalar.size(0))
        if benzersiz_pencere == 0:
            return {
                "benzersiz_pencere": 0,
                "benzersiz_adres_imzasi": 0,
                "carpisma": 0,
                "collision_rate": 0.0,
                "uyari": False,
                "esik": float(esik),
            }
        benzersiz_imza = int(torch.unique(imzalar, dim=0).size(0))
        carpisma = benzersiz_pencere - benzersiz_imza
        oran = carpisma / max(1, benzersiz_pencere)
        return {
            "benzersiz_pencere": benzersiz_pencere,
            "benzersiz_adres_imzasi": benzersiz_imza,
            "carpisma": carpisma,
            "collision_rate": oran,
            "uyari": oran > float(esik),
            "esik": float(esik),
            "tablo_sayisi": self.tablo_sayisi,
            "tablo_boyutu": self.tablo_boyutu,
        }

    # Türkçe API takma adı — rapor diliyle uyumlu.
    hash_carpisma_orani = carpisma_istatistigi

    def bellek_kullanimi(self) -> dict:
        """Fiziksel seyrek tablonun yaklaşık bellek muhasebesi."""
        dtype = self._tablo(0).weight.dtype if len(self.tablolar) else torch.float32
        bayt = torch.tensor([], dtype=dtype).element_size()
        toplam_bayt = self.tablo_boyutu * self.tablo_sayisi * self.boyut * bayt
        if self.erisim_izleme:
            toplam_bayt += self.son_erisim.numel() * self.son_erisim.element_size()
            toplam_bayt += self.erisim_sayisi.numel() * self.erisim_sayisi.element_size()
        return {
            "dtype": str(dtype).replace("torch.", ""),
            "parametre_bayt": bayt,
            "toplam_bayt": toplam_bayt,
            "ram_mb": toplam_bayt / 1024 ** 2,
            "erisim_izleme": self.erisim_izleme,
        }

    @torch.no_grad()
    def decay_uygula(self, carpan: float = 0.99, min_erisim: int = 0) -> int:
        """Bellek eskime mekanizması: seçili satırları çarpanla küçült.

        ``erisim_izleme`` kapalıysa tüm dolu satırlara uygulanır. Açıksa
        ``erisim_sayisi >= min_erisim`` filtresi kullanılabilir. Dönüş:
        etkilenen satır sayısı.
        """
        carpan = float(carpan)
        etkilenen = 0
        for i, t in enumerate(self._tablolar()):
            dolu = t.weight.abs().sum(dim=1) > 1e-8
            if self.erisim_izleme and min_erisim > 0:
                dolu = dolu & (self.erisim_sayisi[i].to(dolu.device) >= int(min_erisim))
            if dolu.any():
                t.weight[dolu] *= carpan
                etkilenen += int(dolu.sum().item())
        return etkilenen

    @torch.no_grad()
    def lru_temizle(self, max_yas: int) -> int:
        """LRU temizliği: ``max_yas`` adımdan eski erişilen satırları sıfırla.

        ``erisim_izleme=True`` gerektirir. Dönüş: sıfırlanan satır sayısı.
        """
        if not self.erisim_izleme:
            raise RuntimeError("LRU temizliği için erisim_izleme=True ile kurun")
        temizlenen = 0
        esik = self._adim - int(max_yas)
        for i, t in enumerate(self._tablolar()):
            eski = (self.son_erisim[i].to(t.weight.device) >= 0) & (self.son_erisim[i].to(t.weight.device) < esik)
            if eski.any():
                t.weight[eski] = 0
                temizlenen += int(eski.sum().item())
                self.son_erisim[i, eski.to(self.son_erisim.device)] = -1
                self.erisim_sayisi[i, eski.to(self.erisim_sayisi.device)] = 0
        return temizlenen

    def adres_uzayi_raporu(self, sozluk_boyutu: int, baglam_penceresi: int) -> dict:
        """Kavramsal adres uzayı / fiziksel depo / doluluk raporu."""
        dolu, toplam = self.doluluk_orani()
        kavramsal = int(sozluk_boyutu) ** int(baglam_penceresi)
        return {
            "kavramsal_anahtar_uzayi": kavramsal,
            "fiziksel_satir": toplam,
            "dolu_satir": dolu,
            "fiziksel_doluluk_orani": dolu / max(1, toplam),
            "adres_sikistirma_orani": toplam / kavramsal if kavramsal else 0.0,
            "ram_mb": self.bellek_kullanimi()["ram_mb"],
        }

    def kapasite(self) -> dict:
        bellek = self.bellek_kullanimi()
        return {
            "fiziksel_satir": self.tablo_boyutu * self.tablo_sayisi,
            "boyut": self.boyut,
            "tablo_sayisi": self.tablo_sayisi,
            "fiziksel_parametre": self.tablo_boyutu * self.tablo_sayisi * self.boyut,
            "parametre_bayt": bellek["parametre_bayt"],
            "ram_mb": bellek["ram_mb"],
        }
