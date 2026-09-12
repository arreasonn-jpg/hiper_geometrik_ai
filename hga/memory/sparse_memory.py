# -*- coding: utf-8 -*-
"""
Sparse Memory — Deneyim Slotları (mevcut seyrek belleğe köprü)
===============================================================
(v0.1 — rapor §13, §22 commit 6; v0.6 için hazırlık)

Mevcut geometrik çekirdekteki `mimari/seyrek_tablo.py` (torch tabanlı
`HashlenmisKureselTablo`), token PENcerelerini hash'leyip sabit boyutlu bir
tabloya yazar. Bu modül, AYNI fikri deneyim üçlüleri için SAF PYTHON ile
uygular: deneyim parmak izi → 31-bit splitmix hash → sabit sayıda slot.

Neden saf Python: v0.1'de knowledge/experience katmanı torch'a bağımlı
olmadan, tek başına test edilebilir ve doğrulanabilir olmalıdır. Torch'taki
`HashlenmisKureselTablo` ile gerçek entegrasyon (deneyim vektörlerini o
tabloya yazmak) yol haritasının v0.6 "bridge layer" adımıdır; bu sınıf o
köprünün protokolünü (hash + slot adresleme + doluluk/çakışma ölçümü) şimdiden
sabitler.

Çakışma ölçümü bilinçli olarak buradadır (rapor §21: "Hash collisions:
collision ölçümü ve gerektiğinde collision-aware storage").
"""
import hashlib
from typing import Dict, List, Optional, Tuple

# mevcut seyrek_tablo.py ile aynı sabitler (splitmix sonlandırıcılı polinomsal hash)
_MASK31 = 0x7FFFFFFF
_TABAN = 65537


def _kararli_bilesen_hash(b) -> int:
    """Python'un süreç-bağımlı ``hash`` fonksiyonu yerine kararlı 31-bit hash."""
    veri = str(b).encode("utf-8", errors="surrogatepass")
    h = hashlib.blake2b(veri, digest_size=8, person=b"hga-slot").digest()
    return int.from_bytes(h, "little") & _MASK31


def parmak_izi(anahtar_bilesenleri) -> int:
    """Bir deneyimi 31-bit deterministik parmak izine indirger.

    Girdi, hash'lenecek bileşenlerin (ör. (özne_id, ilişki_id, nesne_id))
    bir iterable'ıdır. Aynı üçlü → her süreçte ve her seed'de aynı parmak izi.
    """
    acc = 0
    for b in anahtar_bilesenleri:
        acc = (acc * _TABAN + _kararli_bilesen_hash(b)) & _MASK31
    acc = ((acc ^ (acc >> 16)) * 0x7FEB352D) & _MASK31
    acc = ((acc ^ (acc >> 15)) * 0x846CA68B) & _MASK31
    return acc ^ (acc >> 16)


class DeneyimSlotlari:
    """Sabit boyutlu, seyrek deneyim slot deposu (collision-aware).

    * Başlangıçta TÜM slotlar boştur (None).
    * `yaz()` bir deneyimi adresler; aynı adrese farklı bir deneyim düşerse
      ÇAKIŞMA sayılır ve kaydedilir.
    * `doluluk_orani()` ve `cakisma_orani()` deney metrikleridir (§21).
    """

    def __init__(self, slot_sayisi: int = 4096, tablo_sayisi: int = 1):
        if slot_sayisi < 1:
            raise ValueError("slot_sayisi >= 1 olmalı")
        if tablo_sayisi not in (1, 2):
            raise ValueError("tablo_sayisi 1 veya 2 (Bloom tarzı) olabilir")
        self.slot_sayisi = int(slot_sayisi)
        self.tablo_sayisi = int(tablo_sayisi)
        # her tablo: slot → experience_id (None = boş)
        self._tablolar: List[Dict[int, Optional[str]]] = [
            {} for _ in range(tablo_sayisi)
        ]
        self._son_erisim: List[Dict[int, int]] = [{} for _ in range(tablo_sayisi)]
        self._erisim_sayisi: List[Dict[int, int]] = [{} for _ in range(tablo_sayisi)]
        self._tik = 0
        self.okuma_sayisi = 0
        self.yazma_sayisi = 0
        self.cakismalar: List[Dict] = []

    def _adres(self, iz: int, tuz: int) -> int:
        return (iz * tuz) % self.slot_sayisi

    def yaz(self, experience_id: str, anahtar_bilesenleri) -> int:
        """Deneyimi slotlara yaz; birincil slot numarasını döner.

        Aynı deneyim tekrar yazılırsa idempotenttir (aynı slot, çakışma
        sayılmaz). Farklı deneyim aynı slot(lar)a düşerse çakışma kaydedilir.
        """
        self._tik += 1
        self.yazma_sayisi += 1
        iz = parmak_izi(anahtar_bilesenleri)
        adresler = [self._adres(iz, tuz + 1) for tuz in range(self.tablo_sayisi)]
        for t, adres in enumerate(adresler):
            mevcut = self._tablolar[t].get(adres)
            if mevcut is None:
                self._tablolar[t][adres] = experience_id
            elif mevcut != experience_id:
                self.cakismalar.append({
                    "tablo": t, "slot": adres,
                    "onceki": mevcut, "yeni": experience_id,
                })
            self._son_erisim[t][adres] = self._tik
            self._erisim_sayisi[t][adres] = self._erisim_sayisi[t].get(adres, 0) + 1
        return adresler[0]

    def icerir(self, experience_id: str, anahtar_bilesenleri) -> bool:
        self._tik += 1
        self.okuma_sayisi += 1
        iz = parmak_izi(anahtar_bilesenleri)
        adresler = [self._adres(iz, tuz + 1) for tuz in range(self.tablo_sayisi)]
        var = all(self._tablolar[t].get(a) == experience_id
                  for t, a in enumerate(adresler))
        if var:
            for t, a in enumerate(adresler):
                self._son_erisim[t][a] = self._tik
                self._erisim_sayisi[t][a] = self._erisim_sayisi[t].get(a, 0) + 1
        return var

    def doluluk_orani(self) -> Tuple[int, int]:
        dolu = sum(len(t) for t in self._tablolar)
        return dolu, self.slot_sayisi * self.tablo_sayisi

    def cakisma_orani(self) -> float:
        toplam = sum(len(t) for t in self._tablolar)
        return round(len(self.cakismalar) / toplam, 4) if toplam else 0.0

    def lru_temizle(self, max_yas: int) -> int:
        """Son erişimi ``max_yas`` adımdan eski olan slotları temizle."""
        max_yas = int(max_yas)
        temizlenen = 0
        esik = self._tik - max_yas
        for t in range(self.tablo_sayisi):
            eski = [slot for slot, son in self._son_erisim[t].items() if son < esik]
            for slot in eski:
                self._tablolar[t].pop(slot, None)
                self._son_erisim[t].pop(slot, None)
                self._erisim_sayisi[t].pop(slot, None)
                temizlenen += 1
        return temizlenen

    def okuma_yazma_raporu(self) -> Dict:
        """Okuma/yazma asimetrisi metriği."""
        toplam = self.okuma_sayisi + self.yazma_sayisi
        return {
            "okuma": self.okuma_sayisi,
            "yazma": self.yazma_sayisi,
            "okuma_orani": self.okuma_sayisi / toplam if toplam else 0.0,
            "yazma_orani": self.yazma_sayisi / toplam if toplam else 0.0,
        }

    def kapasite(self) -> Dict:
        dolu, toplam = self.doluluk_orani()
        return {
            "slot_sayisi": self.slot_sayisi,
            "tablo_sayisi": self.tablo_sayisi,
            "dolu_slot": dolu,
            "toplam_slot": toplam,
            "cakisma": len(self.cakismalar),
            "cakisma_orani": self.cakisma_orani(),
            "okuma_yazma": self.okuma_yazma_raporu(),
        }
