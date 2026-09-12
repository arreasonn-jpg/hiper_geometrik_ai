# -*- coding: utf-8 -*-
"""
Ablasyon Deneyi — "Belleğe yazılan doğrulanmış bilgi, öğrenmeyi etkiliyor mu?"
===============================================================================
(v1.0+ — rapor §17, §20; "katrilyon" tezinin ölçülebilir kapanışı)

Soru: Doğrulanmış deneyim bilgisini modelin seyrek belleğine yazmak, aşağı-akış
bir okuyucunun işini GERÇEKTEN kolaylaştırıyor mu? Bu modül aynı veri üzerinde
iki koşu yapar ve tek sayıyla yanıt verir:

    KONTROL (boş bellek) : tablo tamamen sıfırken salt-okuma kafası eğitilir.
                           Girdi her üçlü için SIFIR vektördür → kafa yalnızca
                           sınıf oranını ezberleyebilir → dengeli veride ~%50.
    DENEY  (bilgi yazılı): önce doğrulanmış(+)/çürütülmüş(−) üçlülerin satırları
                           yalnızca tablo eğitilerek ±prototip vektörlerine yazılır
                           (Faz W), sonra tablo dondurulup salt-okuma kafası eğitilir
                           (Faz R) → kafa sinyali okur → ~%100.

Neden dürüst: Ölçüm, modelin `gen_kopru` gibi KARMAŞIK çıktı yolundan değil,
belleğin HAM içeriğinden (satır vektörü) yapılır — "sinyal gerçekten bellekte mi?"
sorusuna en temiz yanıttır. `NeuralKopru`'nun gösterdiği "ölü-yol → canlı-yol"
geçişinin İÇERİK boyutunu tamamlar: yol yalnızca canlı değil, aynı zamanda
BİLGİ TAŞIYOR.

Protokol:
    * dengeli veri: N doğru + N yanlış üçlü (etiket 1/0) — şans = %50
    * Faz W (yaz):  MSE(gen(üçlü), (2·etiket−1)·prototip) — yalnızca tablo eğitilir
    * Faz R (oku):  BCE(okuma(gen), etiket) — yalnızca okuma kafası eğitilir
    * çıktı: (kontrol_doğruluk, deney_doğruluk, yazılan_satır)

torch yalnızca kurulumda içe aktarılır; hga paketinin geri kalanı torch'suz
çalışmaya devam eder.
"""
from typing import List, Tuple


def _torch():
    try:
        import torch
        return torch
    except ImportError as e:  # torch yok → dürüstçe yüzeye çıkar
        raise ImportError(
            "AblasyonDeneyi için torch gerekli (pip install -r gereksinimler.txt)."
        ) from e


# Üçlü + etiket: etiket 1.0 = doğru (doğrulanmış), 0.0 = yanlış (çürütülmüş)
Ornek = Tuple[Tuple[str, str, str], float]


class AblasyonDeneyi:
    """Belleğe yazılan bilginin aşağı-akış etkisini iki koşuyla ölçer."""

    def __init__(self, model, tohum: int = 0):
        from .neural_kopru import NeuralKopru
        self.torch = _torch()
        self.kopru = NeuralKopru(model)
        self.tablo = self.kopru.tablo
        self.boyut = self.tablo.boyut
        self.torch.manual_seed(tohum)

    # ── Yardımcılar ──────────────────────────────────────────────────────
    def _prototip(self):
        """Norm=1, tümü pozitif prototip vektörü (hedef yön)."""
        return self.torch.ones(1, self.boyut) / (self.boyut ** 0.5)

    def _sifirla(self):
        """Tabloyu yeniden boşalt (bağımsız koşular için)."""
        for t in self.tablo.tablolar:
            self.torch.nn.init.zeros_(t.weight)
            t.weight.requires_grad_(True)

    def _vektorler(self, ornekler: List[Ornek]):
        X = self.torch.cat([self.kopru.gen(u) for u, _ in ornekler])
        y = self.torch.tensor([[e] for _, e in ornekler], dtype=self.torch.float32)
        return X, y

    # ── Faz W: doğrulanmış bilgiyi belleğe yaz ───────────────────────────
    def bilgi_yaz(self, ornekler: List[Ornek], adim: int = 500,
                  lr: float = 0.1) -> float:
        """Yalnızca tablo satırlarını eğit: doğru→+prototip, yanlış→−prototip.

        Dönen değer, son yazma kaybıdır (yakınsama göstergesi).
        """
        proto = self._prototip()
        opt = self.torch.optim.AdamW(self.tablo.parameters(), lr=lr)
        mse = self.torch.nn.MSELoss()
        son_kayip = 0.0
        for _ in range(int(adim)):
            opt.zero_grad()
            kayip = sum(mse(self.kopru.gen(u),
                            (2.0 * e - 1.0) * proto)
                        for u, e in ornekler)
            kayip.backward()
            opt.step()
            son_kayip = kayip.item()
        return son_kayip

    # ── Faz R: tabloyu dondur, salt-okuma kafası eğit ────────────────────
    def okuma_egit(self, ornekler: List[Ornek], adim: int = 500,
                   lr: float = 0.05):
        """Tabloyu dondurur; Linear(1) kafası yalnızca ham bellek vektöründen
        okumayı öğrenir. (kafa, doğruluk) döner."""
        for p in self.tablo.parameters():
            p.requires_grad_(False)
        okuma = self.torch.nn.Linear(self.boyut, 1)
        opt = self.torch.optim.AdamW(okuma.parameters(), lr=lr)
        kayip_fn = self.torch.nn.BCEWithLogitsLoss()
        X, y = self._vektorler(ornekler)
        for _ in range(int(adim)):
            opt.zero_grad()
            kayip = kayip_fn(okuma(X), y)
            kayip.backward()
            opt.step()
        with self.torch.no_grad():
            dogru = int(((okuma(X) > 0).float().view(-1) == y.view(-1)).sum().item())
        return okuma, dogru / len(ornekler)

    # ── Ana deney ────────────────────────────────────────────────────────
    def kos(self, ornekler: List[Ornek]) -> dict:
        """İki bağımsız koşu; {'kontrol': ..., 'deney': ..., 'yazilan_satir': ...}."""
        # KONTROL: boş bellek → salt okuma (sinyal yok)
        self._sifirla()
        _, dogruluk_bos = self.okuma_egit(ornekler)

        # DENEY: önce bilgi yaz, sonra oku (sinyal var)
        self._sifirla()
        self.bilgi_yaz(ornekler)
        dolu, toplam = self.kopru.doluluk()
        _, dogruluk_yazili = self.okuma_egit(ornekler)

        return {
            "kontrol_dogruluk": round(dogruluk_bos, 4),
            "deney_dogruluk": round(dogruluk_yazili, 4),
            "yazilan_satir": dolu,
            "toplam_satir": toplam,
            "ornek_sayisi": len(ornekler),
        }
