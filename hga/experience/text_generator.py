# -*- coding: utf-8 -*-
"""
TextGenerator — Deneyimden Metin/Olay Üretimi (v0.2)
=====================================================
(v0.2 — rapor §19)

v0.1 Generator'ı KONTROLLÜ KAVRAMSAL KOMBİNASYON üretir (§7); v0.2 onu
GERÇEK metin/olay üretimine genişletir. İlk adım, tam bir doğal dil üretici
kurmak yerine İLİŞKİYE ÖZEL, değiştirilebilir üreteçler (template) sağlar:
böylece cümle üretimi kavramsal üçlüden beslenir ve ileride gerçek bir dil
modeline veya morfolojik üretece değiştirilebilir.

Tasarım:
  * `kayit_ekle(iliski_token, ureteci)` — her ilişki için bir metin üreteci
    kaydedilir; `ureteci(özne, nesne) -> str`.
  * `cumle()` — değerlendirilmiş bir deneyim adayını metne çevirir.
  * `olay()` — üçlü + cümle + durum içeren yapılandırılmış olay kaydı üretir.

NOT: Yerleşik üreteçler BİLİNÇLİ OLARAK morfolojik olarak güvenli (nötr)
kalıplar kullanır; Türkçe ek uyumu (yönelme -a/-e, araç "ile") tam bir
morfoloji modülü gerektirir ve sonraki aşamadadır.
"""
from typing import Callable, Dict, Optional

from ..knowledge.schemas import ExperienceCandidate


def _norm(token: str) -> str:
    return (token or "").strip().lower()


def _varsayilan_uretecler() -> Dict[str, Callable[[str, str], str]]:
    """İlişki adına göre nötr, güvenli Türkçe şablon üreteçleri."""
    return {
        # "{s}, {o} ile bindi." — morfolojik olarak güvenli araç bağlacı
        "binmek": lambda s, o: f"{s}, {o} ile bindi.",
    }


class TextGenerator:
    """Kavramsal üçlüden metin/olay üretir (ilişkiye özel, pluggable)."""

    def __init__(self, uretecler: Optional[Dict[str, Callable[[str, str], str]]] = None):
        self.uretecler: Dict[str, Callable[[str, str], str]] = _varsayilan_uretecler()
        if uretecler:
            for k, v in uretecler.items():
                self.kayit_ekle(k, v)

    def kayit_ekle(self, iliski_token: str,
                   ureteci: Callable[[str, str], str]) -> None:
        """Bir ilişki için metin üreteci kaydet (değiştirilebilir)."""
        self.uretecler[_norm(iliski_token)] = ureteci

    def cumle(self, store, aday: ExperienceCandidate) -> str:
        """Değerlendirilmiş bir deneyim adayını Türkçe cümleye çevirir."""
        ozne = store.entities.getir(aday.subject_id).token
        nesne = store.entities.getir(aday.object_id).token
        iliski = store.relations.iliski_al(aday.relation_id)
        ureteci = self.uretecler.get(_norm(iliski.token))
        if ureteci is None:
            # kayıtlı üreteç yoksa yapılandırılmış nötr çıktı (üçlü korunur)
            return f"{ozne} --{iliski.token}--> {nesne}"
        return ureteci(ozne, nesne)

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
