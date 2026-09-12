# -*- coding: utf-8 -*-
"""
Görev Ablasyonu — "Yazılan bilgi, modelin KENDİ görevini etkiliyor mu?"
========================================================================
(v1.0+ — rapor §17/§20 kapanışının bir üst basamağı; §19 v0.6 + EK-B)

`AblasyonDeneyi` belleğin HAM içeriğini ayrı bir okuma kafasıyla ölçer
(soru: "sinyal bellekte mi?"). `GorevAblasyonu` ise ölçümü modelin KENDİ
ileri geçiş yoluna taşır (soru: "bellek, modelin GERÇEK bir görevi çözmesini
sağlıyor mu?").

Görev: DİZİSEL TAMAMLAMA (modelin yerli next-token görevi).
    girdi  = [özne, ilişki]  penceresi        (özne ve ilişki token id'leri)
    hedef  = nesne token'ı                    (tamamlanacak üçüncü öğe)

Deneyin kritik kurgusu — YOĞUN GÖVDE DONDURULUR:
    embedding, dikkat, encoder, küresel bağ, decoder VE gen_kopru hariç
    her şey eğitilemez. Geriye TEK eğitilebilir kanal kalır: seyrek bellek
    (seyrek_tablo) + onu bağlam vektörüne enjekte eden köprü (gen_kopru).
    Böylece "model bu görevi belleği olmadan yapabilir mi?" sorusu yalıtılır:

    KONTROL — bellek yolu BOŞ ve DONUK: gen(0) → katkı sabit (bias). Dense gövde
              donuk-rastgele olduğu için doğru nesneyi üretemez → ~şans (1/k).
              (Eğitim de yapılamaz: eğitilebilir parametre yoktur.)
    DENEY   — bellek yolu EĞİTİLİR: doğrulanmış (özne, ilişki, nesne) üçlülerinin
              [özne, ilişki] penceresine denk gelen satırları, çapraz-entropi
              sinyaliyle nesneyi işaret edecek biçimde doldurulur → ~%100.

Ters yönlü iki kanıt (bilginin GERÇEKTEN tabloda yaşadığını gösterir):
    TABLO-SIFIR (`tablo_sifir`) — DENEY ~%100 iken tablo sıfırlanıp köprü
              EĞİTİLMİŞ ağırlıklarıyla donuk bırakılırsa tamamlama ~şansa
              düşer. Kazanım geri ÇEKİLİNCE kaybolur → bilgi köprüde değil
              tablo SATIRLARINDADIR.
    HELD-OUT SINIRI (`heldout_siniri`) — model yalnızca eğitim üçlülerinin
              satırlarını doldurur; hiç yazılmamış [özne, ilişki] pencereleri
              (yeni olgular) ~şans kalır. Dürüst sınır: bellekte satırı
              OLMAYAN olgular yoktan var olmaz (genellemez).

Bu, "ölü-yol → canlı-yol" geçişinin GÖREV boyutudur: `yol_canli_mi()` köprü
ağırlığının gradyanını ölçer; yazmadan önce (satır boş → gen=0) gradyan 0'dır
(False), yazdıktan sonra gradyan akar (True). Yol yalnızca canlı değil, aynı
zamanda bir görevi ÇÖZECEK bilgiyi taşır.

Dürüst sınır: dense gövde donuk olduğu için ağ, argmax'ı doğru nesneye
taşıyabilir ama softmax'ı aşırı keskinleştiremez (kayıp zemini donuk okuyucuyla
sınırlıdır). Metrik bu yüzden kayıp değil, TAMAMLAMA DOĞRULUĞU'dur.

NOT: torch yalnızca kurulumda içe aktarılır; hga paketinin geri kalanı torch'suz
çalışmaya devam eder. Model seyrek bellek kapalıysa (`seyrek_tablo=None`) veya
`gen_kopru` yoksa kurulum AÇIKÇA hata verir.
"""
from typing import Dict, List, Tuple


def _torch():
    try:
        import torch
        return torch
    except ImportError as e:  # torch yok → dürüstçe yüzeye çıkar
        raise ImportError(
            "GorevAblasyonu için torch gerekli (pip install -r gereksinimler.txt)."
        ) from e


# Bir üçlü: (özne, ilişki, nesne) — tamamlama görevinin girdisi/hededfi
Uclu = Tuple[str, str, str]


class GorevAblasyonu:
    """Doğrulanmış bilgiyi modelin kendi tamamlama görevi üzerinden ölçer."""

    def __init__(self, model, tohum: int = 0):
        self.torch = _torch()
        self.model = model
        self.tablo = getattr(model, "seyrek_tablo", None)
        if self.tablo is None:
            raise ValueError("model seyrek bellek KAPALI — GorevAblasyonu kurulamaz "
                             "(seyrek_tablo_boyutu > 0 olmalı)")
        self.kopru = getattr(model, "gen_kopru", None)
        if self.kopru is None:
            raise ValueError("modelde gen_kopru köprüsü yok — GorevAblasyonu kurulamaz")
        self.torch.manual_seed(tohum)
        self.sozluk_id: Dict[str, int] = {}

    # ── Küçük sözlük: token → tamsayı (0 = PAD) ─────────────────────────
    def sozluk_kur(self, ucluler: List[Uclu]) -> Dict[str, int]:
        """Üçlülerdeki benzersiz token'lardan 1..N id eşlemesi kurar (0=PAD)."""
        tokenler = sorted({t for u in ucluler for t in u})
        self.sozluk_id = {t: i + 1 for i, t in enumerate(tokenler)}
        return dict(self.sozluk_id)

    # ── Tamamlama pencereleri ───────────────────────────────────────────
    def pencereler(self, ucluler: List[Uclu]):
        """(özne, ilişki, nesne) → (girdi (B,S), hedef (B,)) tamamlama örnekleri.

        Girdi penceresi [PAD…PAD, özne_id, ilişki_id]; hedef nesne_id.
        """
        if not self.sozluk_id:
            raise ValueError("önce sozluk_kur(ucluler) çağrılmalı (veya kos()).")
        S = self.model.baglam_penceresi
        if S < 2:
            raise ValueError("baglam_penceresi en az 2 olmalı (özne + ilişki).")
        en_buyuk = max(self.sozluk_id.values())
        if en_buyuk >= self.model.sozluk_boyutu:
            raise ValueError(
                f"sözlük {en_buyuk} token gerektiriyor ama modelin "
                f"sozluk_boyutu={self.model.sozluk_boyutu}.")
        satirlar, hedefler = [], []
        for ozne, iliski, nesne in ucluler:
            ids = [0] * (S - 2) + [self.sozluk_id[ozne], self.sozluk_id[iliski]]
            satirlar.append(ids)
            hedefler.append(self.sozluk_id[nesne])
        return (self.torch.tensor(satirlar, dtype=self.torch.long),
                self.torch.tensor(hedefler, dtype=self.torch.long))

    # ── Bellek yolunu aç/kapat (yoğun gövde her zaman donuk) ────────────
    def _bellek_yolu(self, aktif: bool):
        for ad, p in self.model.named_parameters():
            p.requires_grad_(aktif and (ad.startswith("seyrek_tablo.")
                                        or ad.startswith("gen_kopru.")))

    def _sifirla(self):
        """Tabloyu yeniden boşalt (bağımsız koşular için)."""
        for t in self.tablo.tablolar:
            self.torch.nn.init.zeros_(t.weight)
            t.weight.requires_grad_(True)

    # ── Ölçüm ───────────────────────────────────────────────────────────
    def tamamlama_dogrulugu(self, ucluler: List[Uclu]) -> float:
        """Donuk/açık modelin (özne, ilişki) → nesne tamamlama doğruluğu."""
        X, y = self.pencereler(ucluler)
        with self.torch.no_grad():
            dogru = int(((self.model(X).argmax(1) == y).sum().item()))
        return dogru / len(ucluler)

    def kontrol(self, ucluler: List[Uclu]) -> float:
        """Bellek yolu boş + donuk → modelin ham tamamlama doğruluğu (~şans)."""
        self._sifirla()
        self._bellek_yolu(False)
        return self.tamamlama_dogrulugu(ucluler)

    # ── Faz W: bilgiyi belleğe yaz (görev sinyaliyle) ───────────────────
    def bilgi_yaz(self, ucluler: List[Uclu], adim: int = 800,
                  lr: float = 3e-2) -> float:
        """Yalnızca bellek yolunu (seyrek_tablo + gen_kopru) tamamlama göreviyle
        eğitir; yoğun gövde donuk kalır. Dönen değer son çapraz-entropidir.

        Dürüst not: kayıp zemini donuk okuyucuyla sınırlıdır; asıl metrik
        `tamamlama_dogrulugu()`'dur (kayıp düşmese de argmax doğruya oturur).
        """
        self._bellek_yolu(True)
        X, y = self.pencereler(ucluler)
        opt = self.torch.optim.AdamW(
            [p for _, p in self.model.named_parameters() if p.requires_grad],
            lr=lr)
        ce = self.torch.nn.CrossEntropyLoss()
        onceki_mod = self.model.training
        self.model.train()
        son_kayip = 0.0
        try:
            for _ in range(int(adim)):
                opt.zero_grad()
                kayip = ce(self.model(X), y)
                kayip.backward()
                opt.step()
                son_kayip = kayip.item()
        finally:
            self.model.train(onceki_mod)
        return son_kayip

    # ── Ölü-yol → canlı-yol (görev penceresi üzerinden) ─────────────────
    def yol_canli_mi(self, ozne: str, iliski: str, nesne: str) -> bool:
        """Tamamlama sorgusu [özne, ilişki] gen_kopru'ya gradyan taşıyor mu?

        Satır boşken (gen=0) köprü AĞIRLIĞININ gradyanı 0'dır → False (ölü-yol).
        Satır dolduktan sonra gradyan akar → True (canlı-yol). Bu, modelin
        belgelediği "ölü-yol tuzağı"nın GÖREV tarafındaki ölçümüdür.
        """
        self.model.zero_grad(set_to_none=True)
        X, y = self.pencereler([(ozne, iliski, nesne)])
        logits = self.model(X)
        if not logits.requires_grad:
            # bellek yolu tamamen donuk → gradyan akamaz (ölü-yol)
            return False
        self.torch.nn.CrossEntropyLoss()(logits, y).backward()
        w = self.kopru.weight
        return w.grad is not None and w.grad.abs().sum().item() > 0

    # ── Ana deney ───────────────────────────────────────────────────────
    def kos(self, ucluler: List[Uclu]) -> dict:
        """İki koşu: kontrol (boş+donuk) vs deney (bilgi yazılı) tamamlama."""
        if not ucluler:
            raise ValueError("en az bir üçlü gerekli.")
        self.sozluk_kur(ucluler)
        ozne, iliski, nesne = ucluler[0]

        # 1) Ölü-yol ölçümü: boş tablo + AKTİF bellek yolu → gen=0 → gradyan 0
        self._sifirla()
        self._bellek_yolu(True)
        canli_once = self.yol_canli_mi(ozne, iliski, nesne)

        # 2) KONTROL: boş tablo + DONUK bellek yolu → yoğun gövde donuk-rastgele
        self._sifirla()
        self._bellek_yolu(False)
        kontrol_dog = self.tamamlama_dogrulugu(ucluler)

        # 3) DENEY: bilgi yaz → tamamlama ~%100; yol canlanır
        son_kayip = self.bilgi_yaz(ucluler)
        deney_dog = self.tamamlama_dogrulugu(ucluler)
        canli_sonra = self.yol_canli_mi(ozne, iliski, nesne)

        dolu, toplam = self.tablo.doluluk_orani()
        return {
            "kontrol_dogruluk": round(kontrol_dog, 4),
            "deney_dogruluk": round(deney_dog, 4),
            "son_kayip": round(son_kayip, 4),
            "yazilan_satir": dolu,
            "toplam_satir": toplam,
            "ornek_sayisi": len(ucluler),
            "yol_canlandi": bool(canli_sonra) and not bool(canli_once),
        }

    # ── TABLO-SIFIR: bilgi köprüde değil, TABLODA yaşar ─────────────────
    def tablo_sifir(self, ucluler: List[Uclu], adim: int = 800,
                    lr: float = 3e-2) -> dict:
        """Bilgi yazılıyken tablo sıfırlanırsa görev çöker mi? (~şansa düşer)

        Kurgu: DENEY (bellek eğitildi) ~%100 → tablo SIFIRLANIR ama köprü
        (`gen_kopru`) EĞİTİLMİŞ ağırlıklarıyla DONUK bırakılır → tamamlama
        ~şansa düşer. Bu, görevi çözen bilginin köprüde DEĞİL, tablo
        SATIRLARINDA saklandığını kanıtlar: köprü yalnızca tablodan gelen
        vektörü bağlam vektörüne taşır; tablo boşalınca (gen=0) taşıyacak
        sinyal kalmaz (geriye yalnızca bias kalır).

        Bu, "kontrol → deney" kanıtının zıt yönüdür: bilgiyi geri ÇEKİNCE
        kazanım da geri gider.
        """
        if not ucluler:
            raise ValueError("en az bir üçlü gerekli.")
        self.sozluk_kur(ucluler)
        self.bilgi_yaz(ucluler, adim=adim, lr=lr)
        deney_dog = self.tamamlama_dogrulugu(ucluler)
        self._sifirla()            # tablo sıfır; köprü ağırlıkları EĞİTİLMİŞ kalır
        self._bellek_yolu(False)   # yalnızca okuma (donuk)
        sifir_dog = self.tamamlama_dogrulugu(ucluler)
        return {
            "deney_dogruluk": round(deney_dog, 4),
            "tablo_sifir_dogruluk": round(sifir_dog, 4),
            "etki": round(deney_dog - sifir_dog, 4),
        }

    # ── Dürüst sınır: yazılmamış olgular genellemez ─────────────────────
    def heldout_siniri(self, tren_ucluler: List[Uclu], test_ucluler: List[Uclu],
                       adim: int = 800, lr: float = 3e-2) -> dict:
        """Model yalnızca EĞİTİM üçlülerinin satırlarını doldurur; yeni pencereler
        (held-out) genellemez.

        Yoğun gövde donuk-rastgele olduğu için öğrenme TAMAMEN tablo
        satırlarına yazılır. `test_ucluler`'deki [özne, ilişki] pencereleri
        eğitimde hiç görülmediyse, denk gelen satırlar BOŞ kalır ve model bu
        yeni olguları tamamlayamaz → ~şans. Bu, yolun dürüst sınırıdır:
        bellekte satırı yazılmamış olgular yoktan var olmaz.

        DİKKAT: held-out pencereleri eğitim pencerelerinden FARKLI olmalıdır
        (örn. yeni özneler) — aynı [özne, ilişki] penceresi aynı satıra denk
        gelir ve zaten dolmuş olur.
        """
        if not tren_ucluler or not test_ucluler:
            raise ValueError("hem eğitim hem test üçlüsü gerekli.")
        self.sozluk_kur(list(tren_ucluler) + list(test_ucluler))
        self.bilgi_yaz(tren_ucluler, adim=adim, lr=lr)
        tren_dog = self.tamamlama_dogrulugu(tren_ucluler)
        test_dog = self.tamamlama_dogrulugu(test_ucluler)
        return {
            "egitim_dogruluk": round(tren_dog, 4),
            "heldout_dogruluk": round(test_dog, 4),
            "etki": round(tren_dog - test_dog, 4),
        }
