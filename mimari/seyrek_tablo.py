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

    def __init__(self, tablo_boyutu=1_048_576, boyut=32,
                 hash_tuzu=0x2545F491, tablo_sayisi=1):
        super().__init__()
        if tablo_sayisi not in (1, 2):
            raise ValueError("tablo_sayisi yalnızca 1 (tek hash) veya 2 (Bloom) olabilir")
        self.tablo_boyutu = int(tablo_boyutu)
        self.boyut = int(boyut)
        self.tablo_sayisi = int(tablo_sayisi)

        # Farklı tuzlar → aynı anahtar, farklı tablolarda farklı satıra düşer
        tuzler = [int(hash_tuzu), (int(hash_tuzu) * 2 + 1) & self.MASK31]
        self.register_buffer("tuzlar",
                             torch.tensor(tuzler[:tablo_sayisi], dtype=torch.long))

        # Fiziksel tablo(lar): BAŞLANGIÇTA TAMAMI SIFIR = tüm kümeler "boş"
        self.tablolar = nn.ModuleList(
            [nn.Embedding(self.tablo_boyutu, self.boyut)
             for _ in range(tablo_sayisi)]
        )
        for t in self.tablolar:
            nn.init.zeros_(t.weight)

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

    # ── Arayüzler ───────────────────────────────────────────────────────
    def getir_anahtarla(self, anahtar: torch.Tensor) -> torch.Tensor:
        """Hazır kavramsal anahtarlarla arama (rapor 9.3'teki API)."""
        cikti = self.tablolar[0](self.adres(anahtar, 0))
        for i in range(1, self.tablo_sayisi):
            cikti = cikti + self.tablolar[i](self.adres(anahtar, i))
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

    def kapasite(self) -> dict:
        return {
            "fiziksel_satir": self.tablo_boyutu * self.tablo_sayisi,
            "boyut": self.boyut,
            "tablo_sayisi": self.tablo_sayisi,
            "fiziksel_parametre": self.tablo_boyutu * self.tablo_sayisi * self.boyut,
            "ram_mb": self.tablo_boyutu * self.tablo_sayisi * self.boyut * 4 / 1024 ** 2,
        }
