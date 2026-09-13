# -*- coding: utf-8 -*-
"""
Korpus Borusu — veri toplayıcı çıktısı → sözlük büyütme → REAL_DATA
=====================================================================
(v1.0+ — rapor §8 "dış korpus ölçeği"; §12)

`egitim/veri_toplayici.py` (`OtomatikVeriToplayici.metni_kaydet`) `turkce_metin.txt`
dosyasına bir metin havuzu yazar. Bu modül o çıktıyı Experience Engine'in bilgi
tabanına GERÇEK VERİ olarak akıtan borudur:

    metin → cümlelere böl → sözlüğü büyüt → üçlü ayıkla → REAL_DATA olarak yaz

Ağ erişimi, `requests` veya `pyarrow` GEREKTİRMEZ: boru yalnızca DOSYA/ metin okur.
Böylece veri toplayıcı (ağ) ile bilgi tabanı (bilgi) birbirinden bağımsız kalır —
toplayıcı dosyayı üretir, boru dosyayı tüketir.

    * `korpus_borusu(store, metin, ...)`  — metinden uçtan uca akıtır
    * `korpus_dosyasindan(store, yol, ...)` — dosyadan akıtır
    * `veri_toplayici_ciktisindan(store, kok, ...)` — veri toplayıcının
      varsayılan `turkce_metin.txt` dosyasını akıtır
"""
import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional

from ..data.quality import VeriKaliteRaporu, temizle_cumleler
from .corpus import cumlelere_bol
from .cumle_ayiklayici import VARSAYILAN_SOZLUK, CumleAyiklayici, cumlelerden_bilgi_aktar
from .sozluk_buyutme import SozlukBuyutmeRaporu, sozlugu_buyut

VARSAYILAN_KORPUS_DOSYASI = "turkce_metin.txt"


@dataclass
class KorpusRaporu:
    cumle_sayisi: int = 0
    buyutme: Optional[SozlukBuyutmeRaporu] = None
    kalite: Optional[VeriKaliteRaporu] = None
    aktarilan_uclu: int = 0
    son_varlik: int = 0
    son_kanit: int = 0

    def to_dict(self) -> Dict:
        d = asdict(self)
        if self.buyutme is not None:
            d["buyutme"] = self.buyutme.to_dict()
        if self.kalite is not None:
            d["kalite"] = self.kalite.to_dict()
        return d


def korpus_borusu(store, metin: str, sozluk: Optional[Dict] = None,
                  iliski_kisitlari: Optional[Dict] = None,
                  kalite_filtresi: bool = False) -> KorpusRaporu:
    """Metni sözlük büyütme + üçlü ayıklama + REAL_DATA aktarımından geçirir.

    Sözlük verilmezse `VARSAYILAN_SOZLUK`'tan başlanır (derin kopya alınır;
    global sözlük asla değişmez). `iliski_kisitlari`, `cumlelerden_bilgi_aktar`'a
    aktarılır (örn. binmek→öznesi insan + nesnesi binilebilir). ``kalite_filtresi``
    açılırsa duplicate/spam/bozuk encoding cümleleri atılır ve raporlanır.
    """
    cumleler = cumlelere_bol(metin)
    kalite_raporu = None
    if kalite_filtresi:
        cumleler, kalite_raporu = temizle_cumleler(cumleler)
    buyumus, buyutme_raporu, _ = sozlugu_buyut(sozluk or VARSAYILAN_SOZLUK,
                                               cumleler)
    ayiklayici = CumleAyiklayici(buyumus)
    aktarilan = cumlelerden_bilgi_aktar(store, cumleler, ayiklayici=ayiklayici,
                                        iliski_kisitlari=iliski_kisitlari)
    ozet = store.ozet()
    return KorpusRaporu(
        cumle_sayisi=len(cumleler),
        buyutme=buyutme_raporu,
        kalite=kalite_raporu,
        aktarilan_uclu=len(aktarilan),
        son_varlik=ozet.get("varlik", 0),
        son_kanit=ozet.get("kanit", 0),
    )


def korpus_dosyasindan(store, yol: str, sozluk: Optional[Dict] = None,
                       iliski_kisitlari: Optional[Dict] = None,
                       kalite_filtresi: bool = False) -> KorpusRaporu:
    """Bir metin dosyasını okuyup REAL_DATA olarak akıtır."""
    with open(yol, "r", encoding="utf-8") as f:
        return korpus_borusu(store, f.read(), sozluk=sozluk,
                             iliski_kisitlari=iliski_kisitlari,
                             kalite_filtresi=kalite_filtresi)


def veri_toplayici_ciktisindan(store, proje_kok: str,
                               dosya_adi: str = VARSAYILAN_KORPUS_DOSYASI,
                               sozluk: Optional[Dict] = None,
                               iliski_kisitlari: Optional[Dict] = None,
                               kalite_filtresi: bool = False
                               ) -> KorpusRaporu:
    """`OtomatikVeriToplayici.metni_kaydet` çıktısı `turkce_metin.txt`'i akıtır.

    Veri toplayıcıyı İÇE AKTARMAZ (requests/pyarrow gerekmez); yalnızca onun
    yazdığı dosyayı okur. Dosya yoksa dürüst bir hata döner (sessizce boş
    rapor üretmez).
    """
    yol = os.path.join(proje_kok, dosya_adi)
    if not os.path.exists(yol):
        raise FileNotFoundError(
            f"veri toplayıcı çıktısı bulunamadı: {yol} — önce "
            f"OtomatikVeriToplayici(proje_kok).metni_kaydet(metin) çalıştırın.")
    return korpus_dosyasindan(store, yol, sozluk=sozluk,
                              iliski_kisitlari=iliski_kisitlari,
                              kalite_filtresi=kalite_filtresi)
