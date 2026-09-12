# -*- coding: utf-8 -*-
"""
Genelleme Ablasyonu — "Bellekteki bilgi, görülmeyen olgulara genelleştiriyor mu?"
==================================================================================
(v1.0+ — rapor §17/§20; `GorevAblasyonu`'nun genelleme boyutu)

`AblasyonDeneyi` sinyalin bellekte olduğunu, `GorevAblasyonu` belleğin modelin
KENDİ tamamlama görevini çözdüğünü gösterdi. `GenellemeAblasyonu` ise üçüncü ve
en güçlü soruyu sorar:

    Belleğe yazılan doğrulanmış bilgi, okuyucunun EĞİTİMDE HİÇ GÖRMEDİĞİ
    sorguları yanıtlamasını sağlıyor mu?

Deney kurgusu:
    * M olgu: (özne, ilişki, nesne). Nesneler AZ sayıda kategoridendir (her
      kategori birden çok öznede görünür) → aynı nesne kodu paylaşılır.
    * Faz W (bilgi yaz): HER olgunun [özne, ilişki] penceresine, nesnenin
      KANONİK kodu (one-hot vektör) yazılır (yalnız seyrek_tablo eğitilir).
      Böylece bilgi, özneden bağımsız, paylaşılan bir biçimde saklanır.
    * Faz R (oku): bellek dondurulur; okuyucu = modelin KENDİ `gen_kopru`
      köprüsü + küçük sınıflandırma kafası. Okuyucu yalnızca EĞİTİM alt
      kümesinde (her nesne en az bir kez görünür) eğitilir.
    * Ölçüm: HELD-OUT sorgularda (eğitimde görülmeyen [özne, ilişki]
      pencereleri) tamamlama doğruluğu.

    KONTROL (bellek boş): gen=0 → okuyucu sabit girdi görür → eğitimde bile
        yalnızca sınıf dağılımını ezberler → ~şans; held-out ~şans.
    DENEY (bilgi yazılı): okuyucu KANONİK kodu çözmeyi öğrenir; kod nesneler
        arasında paylaşıldığı için held-out sorgularda da doğru nesneyi okur
        → ~%100.

NEDEN OKUYUCU `gen_kopru` + KÜÇÜK KAFA (tam yoğun gövde değil):
    Yoğun gövde eğitilebilir bırakılırsa model eğitim sorgularını EZBERLER
    (train ~%100) ama held-out ~şans kalır — ölçüldü ve bu modülün dürüstlük
    notunda belgeleniyor. Belleğin genelleme katkısını YALITMAK için gövde
    donuk tutulur; tek öğrenen kanal köprü + kafa olur. Böylece "genelleme,
    bellekten mi geldi?" sorusu temiz yanıtlanır.

NOT: torch yalnızca kurulumda içe aktarılır; hga paketinin geri kalanı torch'suz
çalışmaya devam eder.
"""
from typing import Dict, List, Tuple

from .neural_kopru import NeuralKopru


def _torch():
    try:
        import torch
        return torch
    except ImportError as e:  # torch yok → dürüstçe yüzeye çıkar
        raise ImportError(
            "GenellemeAblasyonu için torch gerekli "
            "(pip install -r gereksinimler.txt).") from e


Uclu = Tuple[str, str, str]


class GenellemeAblasyonu:
    """Bellekteki kanonik bilginin held-out sorgulara genellemesini ölçer."""

    def __init__(self, model, tohum: int = 0):
        self.torch = _torch()
        self.model = model
        self.kopru_nesne = NeuralKopru(model)   # tablo + gen_kopru doğrular/erişir
        self.tablo = self.kopru_nesne.tablo
        self.boyut = self.tablo.boyut
        # gen_kopru çıktı boyutu = bağlam penceresi × gömme boyutu
        self.kopru_cikti = (self.model.baglam_penceresi *
                            self.model.kelime_gomme.embedding_dim)
        self.torch.manual_seed(tohum)
        self.sozluk_id: Dict[str, int] = {}
        self.nesne_indeks: Dict[str, int] = {}
        self.nesne_tokenleri: List[str] = []

    # ── Sözlük / pencere ─────────────────────────────────────────────────
    def sozluk_kur(self, ucluler: List[Uclu]) -> Dict[str, int]:
        """Üçlülerdeki benzersiz token'lardan 1..N id eşlemesi (0=PAD)."""
        tokenler = sorted({t for u in ucluler for t in u})
        self.sozluk_id = {t: i + 1 for i, t in enumerate(tokenler)}
        return dict(self.sozluk_id)

    def _pencere(self, ozne: str, iliski: str):
        S = self.model.baglam_penceresi
        ids = [0] * (S - 2) + [self.sozluk_id[ozne], self.sozluk_id[iliski]]
        return self.torch.tensor([ids], dtype=self.torch.long)

    def _gen(self, ozne: str, iliski: str):
        return self.tablo(self._pencere(ozne, iliski))

    # ── Nesne kategorileri + kanonik kod ─────────────────────────────────
    def nesne_kodlari(self, ucluler: List[Uclu]) -> int:
        """Benzersiz nesneleri sıralayıp indeksler; K (kategori sayısı) döner."""
        self.nesne_tokenleri = sorted({u[2] for u in ucluler})
        self.nesne_indeks = {n: i for i, n in enumerate(self.nesne_tokenleri)}
        K = len(self.nesne_tokenleri)
        if K > self.boyut:
            raise ValueError(
                f"nesne kategorisi ({K}) bellek boyutunu ({self.boyut}) aşıyor — "
                f"daha büyük seyrek_boyut gerekir.")
        return K

    def _kod(self, indeks: int):
        """Nesnenin kanonik kodu: boyut uzunluklu one-hot vektör."""
        v = self.torch.zeros(1, self.boyut)
        v[0, indeks] = 1.0
        return v

    # ── Yardımcılar ──────────────────────────────────────────────────────
    def _sifirla(self):
        for t in self.tablo.tablolar:
            self.torch.nn.init.zeros_(t.weight)
            t.weight.requires_grad_(True)

    def _bellek_yolu(self, aktif: bool):
        for ad, p in self.model.named_parameters():
            p.requires_grad_(aktif and ad.startswith("seyrek_tablo."))

    def _kopru_yolu(self, aktif: bool):
        for ad, p in self.model.named_parameters():
            p.requires_grad_(aktif and ad.startswith("gen_kopru."))

    # ── Faz W: bilgiyi KANONİK kod olarak yaz ────────────────────────────
    def bilgi_yaz(self, ucluler: List[Uclu], adim: int = 400,
                  lr: float = 0.1) -> float:
        """Her olgunun penceresine nesnenin one-hot kodu yazılır (yalnız tablo)."""
        self._bellek_yolu(True)
        opt = self.torch.optim.AdamW(
            [p for _, p in self.model.named_parameters() if p.requires_grad],
            lr=lr)
        mse = self.torch.nn.MSELoss()
        son = 0.0
        for _ in range(int(adim)):
            opt.zero_grad()
            kayip = sum(mse(self._gen(o, i), self._kod(self.nesne_indeks[n]))
                        for o, i, n in ucluler)
            kayip.backward()
            opt.step()
            son = kayip.item()
        return son

    # ── Faz R: okuyucu (gen_kopru + kafa) eğitimi ────────────────────────
    def okuma_egit(self, ucluler: List[Uclu], egitim_indisleri: List[int],
                   adim: int = 600, lr: float = 0.01):
        """Bellek donuk; `gen_kopru` + Linear(kopru_çıktı, K) kafası eğitilir.

        Okuyucu, yalnızca egitim_indisleri'ndeki olgular üzerinde eğitilir.
        """
        K = len(self.nesne_tokenleri)
        self._bellek_yolu(False)
        self._kopru_yolu(True)
        kafa = self.torch.nn.Linear(self.kopru_cikti, K)
        opt = self.torch.optim.AdamW(
            [p for _, p in self.model.named_parameters() if p.requires_grad]
            + list(kafa.parameters()), lr=lr)
        ce = self.torch.nn.CrossEntropyLoss()
        egitim = [ucluler[i] for i in egitim_indisleri]
        for _ in range(int(adim)):
            opt.zero_grad()
            logits = self.torch.cat(
                [kafa(self.model.gen_kopru(self._gen(o, i))) for o, i, _ in egitim])
            hedef = self.torch.tensor([self.nesne_indeks[n] for _, _, n in egitim])
            kayip = ce(logits, hedef)
            kayip.backward()
            opt.step()
        return kafa

    def _dogruluk(self, kafa, ucluler: List[Uclu], indisler: List[int]) -> float:
        with self.torch.no_grad():
            dogru = 0
            for i in indisler:
                o, il, n = ucluler[i]
                logit = kafa(self.model.gen_kopru(self._gen(o, il)))
                if logit.argmax(1).item() == self.nesne_indeks[n]:
                    dogru += 1
        return dogru / max(1, len(indisler))

    # ── Veri bölme: her nesne eğitimde en az bir kez ─────────────────────
    def bol(self, ucluler: List[Uclu]):
        """(egitim_indisleri, heldout_indisleri) — nesne başına ilk olgu eğitime."""
        gruplar: Dict[str, List[int]] = {}
        for i, (_, _, n) in enumerate(ucluler):
            gruplar.setdefault(n, []).append(i)
        egitim, heldout = [], []
        for n in sorted(gruplar):
            sira = sorted(gruplar[n], key=lambda i: ucluler[i][0])
            egitim.append(sira[0])
            heldout.extend(sira[1:])
        return egitim, heldout

    # ── Ana deney ────────────────────────────────────────────────────────
    def kos(self, ucluler: List[Uclu]) -> dict:
        """Kontrol (boş bellek) vs deney (bilgi yazılı) held-out genellemesi."""
        if not ucluler:
            raise ValueError("en az bir üçlü gerekli.")
        self.sozluk_kur(ucluler)
        K = self.nesne_kodlari(ucluler)
        egitim, heldout = self.bol(ucluler)
        if not heldout:
            raise ValueError("genelleme için nesne başına en az 2 olgu gerekli "
                             "(her nesne eğitimde ve held-out'ta görünmeli).")

        # KONTROL: boş bellek → okuyucu sabit girdi görür → şans
        self._sifirla()
        kafa_kontrol = self.okuma_egit(ucluler, egitim)
        kontrol_egitim = self._dogruluk(kafa_kontrol, ucluler, egitim)
        kontrol_heldout = self._dogruluk(kafa_kontrol, ucluler, heldout)

        # DENEY: önce bilgi yaz, sonra oku
        self._sifirla()
        self.bilgi_yaz(ucluler)
        dolu, toplam = self.tablo.doluluk_orani()
        kafa_deney = self.okuma_egit(ucluler, egitim)
        deney_egitim = self._dogruluk(kafa_deney, ucluler, egitim)
        deney_heldout = self._dogruluk(kafa_deney, ucluler, heldout)

        return {
            "kontrol_egitim_dogruluk": round(kontrol_egitim, 4),
            "kontrol_heldout_dogruluk": round(kontrol_heldout, 4),
            "deney_egitim_dogruluk": round(deney_egitim, 4),
            "deney_heldout_dogruluk": round(deney_heldout, 4),
            "nesne_sayisi": K,
            "egitim_olgu": len(egitim),
            "heldout_olgu": len(heldout),
            "yazilan_satir": dolu,
            "toplam_satir": toplam,
        }
