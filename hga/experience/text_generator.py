# -*- coding: utf-8 -*-
"""
TextGenerator — Deneyimden Metin/Olay Üretimi (v0.2)
=====================================================
(v0.2 — rapor §19)

v0.1 Generator'ı KONTROLLÜ KAVRAMSAL KOMBİNASYON üretir (§7); v0.2 onu
GERÇEK metin/olay üretimine genişletir. İlk adım, tam bir doğal dil üretici
kurmak yerine İLİŞKİYE ÖZEL, değiştirilebilir üreteçler sağlar: böylece cümle
üretimi kavramsal üçlüden beslenir ve ileride gerçek bir dil modeline veya
morfolojik üretece değiştirilebilir.

Üreteç sözleşmesi (geriye dönük uyumlu):

    * 2 parametreli:  ureteci(ozne, nesne) -> str
    * 3 parametreli:  ureteci(ozne, nesne, baglam) -> str
        baglam = {"subject": Entity, "object": Entity, "relation": Relation}

Yerleşik "binmek" üreteci, `turkce.py`'deki ek uyumu yardımcılarını kullanır:
cins isim nesnede "Ali ataya bindi.", özel isim nesnede "Ali, Ata'ya bindi."
"""
import inspect
from typing import Callable, Dict, Optional

from ..knowledge.schemas import ExperienceCandidate
from .turkce import ayrilma_eki, belirtme_eki, bulunma_eki, kucult, yonelme_eki


def _norm(token: str) -> str:
    return (token or "").strip().lower()


def _ozel_mi(baglam) -> bool:
    return bool(baglam and getattr(baglam.get("object"), "is_ozel", False))


def _binmek_ureteci(ozne: str, nesne: str, baglam: Optional[Dict] = None) -> str:
    """"binmek" için ek uyumlu Türkçe cümle (yönelme)."""
    ozel = _ozel_mi(baglam)
    if ozel:
        cekilmis = yonelme_eki(nesne, ozel_isim=True)
        return f"{ozne}, {cekilmis} bindi."
    cekilmis = yonelme_eki(kucult(nesne))
    return f"{ozne} {cekilmis} bindi."


def _sevmek_ureteci(ozne: str, nesne: str, baglam: Optional[Dict] = None) -> str:
    """"sevmek" için ek uyumlu Türkçe cümle (belirtme)."""
    ozel = _ozel_mi(baglam)
    if ozel:
        cekilmis = belirtme_eki(nesne, ozel_isim=True)
        return f"{ozne}, {cekilmis} seviyor."
    cekilmis = belirtme_eki(kucult(nesne))
    return f"{ozne} {cekilmis} seviyor."


def _durmak_ureteci(ozne: str, nesne: str, baglam: Optional[Dict] = None) -> str:
    """"durmak" için ek uyumlu Türkçe cümle (bulunma)."""
    ozel = _ozel_mi(baglam)
    cekilmis = bulunma_eki(nesne, ozel_isim=ozel) if ozel else bulunma_eki(kucult(nesne))
    return f"{ozne} {cekilmis} duruyor."


def _gelmek_ureteci(ozne: str, nesne: str, baglam: Optional[Dict] = None) -> str:
    """"gelmek" için ek uyumlu Türkçe cümle (ayrılma)."""
    ozel = _ozel_mi(baglam)
    cekilmis = ayrilma_eki(nesne, ozel_isim=ozel) if ozel else ayrilma_eki(kucult(nesne))
    return f"{ozne} {cekilmis} geliyor."


def _varsayilan_uretecler() -> Dict[str, Callable]:
    """İlişki adına göre Türkçe şablon üreteçleri."""
    return {
        "binmek": _binmek_ureteci,
        "sevmek": _sevmek_ureteci,
        "durmak": _durmak_ureteci,
        "gelmek": _gelmek_ureteci,
    }


def _cagir(ureteci: Callable, ozne: str, nesne: str, baglam: Dict) -> str:
    """Üreteci 2 veya 3 parametreyle çağır (imza denetimiyle)."""
    try:
        imza = inspect.signature(ureteci)
        konumsal = [p for p in imza.parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        zorunlu = [p for p in konumsal if p.default is p.empty]
        if len(konumsal) >= 3 or len(zorunlu) >= 3:
            return str(ureteci(ozne, nesne, baglam))
    except (TypeError, ValueError):
        pass
    return str(ureteci(ozne, nesne))


class TextGenerator:
    """Kavramsal üçlüden metin/olay üretir (ilişkiye özel, pluggable)."""

    def __init__(self, uretecler: Optional[Dict[str, Callable]] = None):
        self.uretecler: Dict[str, Callable] = _varsayilan_uretecler()
        if uretecler:
            for k, v in uretecler.items():
                self.kayit_ekle(k, v)

    def kayit_ekle(self, iliski_token: str, ureteci: Callable) -> None:
        """Bir ilişki için metin üreteci kaydet (değiştirilebilir)."""
        self.uretecler[_norm(iliski_token)] = ureteci

    def cumle(self, store, aday: ExperienceCandidate) -> str:
        """Değerlendirilmiş bir deneyim adayını Türkçe cümleye çevirir."""
        subject = store.entities.getir(aday.subject_id)
        object_ = store.entities.getir(aday.object_id)
        iliski = store.relations.iliski_al(aday.relation_id)
        ureteci = self.uretecler.get(_norm(iliski.token))
        baglam = {"subject": subject, "object": object_, "relation": iliski}
        if ureteci is None:
            # kayıtlı üreteç yoksa yapılandırılmış nötr çıktı (üçlü korunur)
            return f"{subject.token} --{iliski.token}--> {object_.token}"
        return _cagir(ureteci, subject.token, object_.token, baglam)

    def olay(self, store, aday: ExperienceCandidate) -> Dict:
        """Yapılandırılmış olay kaydı: üçlü + cümle + durum."""
        return {
            "experience_id": aday.experience_id,
            "ozne": store.entities.getir(aday.subject_id).token,
            "iliski": store.relations.iliski_al(aday.relation_id).token,
            "nesne": store.entities.getir(aday.object_id).token,
            "cumle": self.cumle(store, aday),
            "durum": aday.state.value,
        }
