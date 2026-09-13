# -*- coding: utf-8 -*-
"""
NeuralKopru — Doğrulanmış Deneyimleri MODELİN Seyrek Belleğine Bağla
=====================================================================
(v0.6 tamamlayıcısı — rapor §19 v0.6, EK-B, "ölü-yol tuzağı" notu)

`TorchKoprusu` KENDİ bağımsız seyrek tablosunu kurar; `NeuralKopru` ise bunu
bir adım ileri götürür: doğrulanmış deneyim üçlülerini, modelin EĞİTİMDE
kullandığı `model.seyrek_tablo`'ya (aynı `HashlenmisKureselTablo`) bağlar ve
`model.gen_kopru` köprüsünden geçerek geometrik latent uzaya enjekte eder.

    doğrulanmış üçlü → 3 bileşenli token id → tablo.anahtar() → satır
                     → "gen" vektörü → gen_kopru → (S·emb) latent katkı

Neden önemli (model gövdesindeki "ölü-yol tuzağı" notuyla birebir):
    * Tablo BOŞKEN (satır = sıfır) `gen_kopru(0)` katkısı sıfırdır ve ağırlık
      gradyanı 0'dır → yol ÖLÜDÜR.
    * Doğrulanmış deneyim satırları gradyanla DOLDURULDUKTAN sonra `gen_kopru`
      deneyim vektörünü taşırken gradyan alır → yol CANLANIR.
    `kopru_canli_mi()` bu geçişi ölçer: yazmadan önce False, yazdıktan sonra True.

Böylece Knowledge/Experience katmanının doğrulanmış bilgisi, geometrik çekirdeğin
öğrenilebilir yollarına SOMUT ve ölçülebilir biçimde bağlanmış olur.

NOT: torch yalnızca kurulumda içe aktarılır; hga paketinin geri kalanı torch'suz
çalışmaya devam eder. Model seyrek bellek kapalıysa (`seyrek_tablo=None`) veya
`gen_kopru` yoksa kurulum AÇIKÇA hata verir (sessizce boş köprü döndürmez).
"""
from typing import List, Tuple

from .kopru import bilesen_token, torch_var_mi


class NeuralKopru:
    """Doğrulanmış deneyimleri modelin kendi seyrek belleğine yazar/okur."""

    def __init__(self, model):
        if not torch_var_mi():
            raise ImportError(
                "NeuralKopru için torch gerekli (pip install -e .).")
        import torch
        self.torch = torch
        self.model = model
        self.tablo = getattr(model, "seyrek_tablo", None)
        if self.tablo is None:
            raise ValueError("model seyrek bellek KAPALI — NeuralKopru kurulamaz "
                             "(seyrek_tablo_boyutu > 0 olmalı)")
        self.kopru = getattr(model, "gen_kopru", None)
        if self.kopru is None:
            raise ValueError("modelde gen_kopru köprüsü yok — NeuralKopru kurulamaz")

    # ── Üçlü → tensor / anahtar / vektör ─────────────────────────────────
    def uclu_tensoru(self, uclusu: Tuple[str, str, str]):
        """(özne_id, ilişki_id, nesne_id) → (1, 3) int64 tensor."""
        ids = [bilesen_token(b) for b in uclusu]
        return self.torch.tensor([ids], dtype=self.torch.long)

    def anahtar(self, uclusu: Tuple[str, str, str]):
        """Üçlünün kavramsal anahtarı (modelin kendi tablo şemasıyla)."""
        return self.tablo.anahtar(self.uclu_tensoru(uclusu))

    def gen(self, uclusu: Tuple[str, str, str]):
        """Üçlünün 'gen' vektörü (boş kümeye düşerse sıfır)."""
        return self.tablo(self.uclu_tensoru(uclusu))

    def enjekte(self, uclusu: Tuple[str, str, str]):
        """Üçlünün latent katkısı: gen_kopru(gen) → (1, S·emb)."""
        return self.kopru(self.gen(uclusu))

    # ── Yazma: satırları gradyanla doldur (eğitimde olan budur) ──────────
    def coklu_yaz(self, ucluler: List[Tuple[str, str, str]],
                  lr: float = 1e-2) -> int:
        """Doğrulanmış üçlülerin satırlarını TEK gradyan adımıyla doldur.

        Boş satır (sıfır) gradyanla dolana kadar yol ölüdür; bu adım onu canlandırır.
        Dokunulmayan satırlar AdamW'da sıfır kalır (boş küme davranışı korunur).
        """
        if not ucluler:
            return 0
        genler = [self.gen(u) for u in ucluler]
        toplam = self.torch.cat(genler).sum()
        self.model.zero_grad(set_to_none=True)
        toplam.backward()
        self.torch.optim.AdamW(self.tablo.parameters(), lr=lr).step()
        return len(ucluler)

    # ── Ölçüm: köprü canlı mı? ───────────────────────────────────────────
    def kopru_canli_mi(self, uclusu: Tuple[str, str, str]) -> bool:
        """gen_kopru, bu üçlünün vektörünü taşırken gradyan alıyor mu?

        Satır boşken (gen=0) ağırlık gradyanı 0'dır → False (yol ölü).
        Satır dolduktan sonra → True (yol canlı). Bu, modelin belgelediği
        "ölü-yol tuzağı"nın deneyim tarafındaki ölçümüdür.
        """
        self.model.zero_grad(set_to_none=True)
        katki = self.enjekte(uclusu)
        katki.sum().backward()
        w = self.kopru.weight
        return w.grad is not None and w.grad.abs().sum().item() > 0

    # ── Durum ────────────────────────────────────────────────────────────
    def doluluk(self) -> Tuple[int, int]:
        """(dolu_satır, toplam_satır) — modelin seyrek tablosunun doluluğu."""
        dolu, toplam = self.tablo.doluluk_orani()
        return int(dolu), int(toplam)

    def rapor(self) -> dict:
        dolu, toplam = self.doluluk()
        return {"dolu_satir": dolu, "toplam_satir": toplam}
