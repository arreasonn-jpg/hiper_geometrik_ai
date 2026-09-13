# -*- coding: utf-8 -*-
"""
Mini-Environment — Deterministik Doğrulayıcı Ortam (v0.5)
===========================================================
(rapor §18, §19 v0.5)

Genel dilde sonucun objektif kurallarla ölçülebildiği bir environment yoktur
(rapor §18 — AlphaGo benzetmesinin doğru kullanımı). Bu yüzden HGA'ya
domain-specific doğrulayıcılar gerekir. v0.5'te bunun ilk örneği:

    * `AritmetikOrtam` — toplama/çıkarma/çarpma/bölme ifadelerini GÜVENLİ
      (eval KULLANMADAN, `ast` ile) değerlendirir; "2+3=5" → True,
      "2+3=6" → False, bozuk ifade → None (belirsiz).

Ortam, ConflictResolver'ın beklediği `(store, aday) -> Optional[bool]`
imzasıyla doğrudan deterministik test olarak takılabilir (rapor §11 döngüsü:
"Kanıt veya deterministik test ara").
"""
import ast
import operator
import re
from typing import Optional

from ..knowledge.schemas import ExperienceCandidate

# Yalnızca izin verilen AST düğümleri → güvenli aritmetik (eval yok)
_ISLEM = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class AritmetikOrtam:
    """Deterministik aritmetik doğrulayıcı (mini-environment)."""

    def deger(self, ifade: str) -> Optional[float]:
        """Bir aritmetik ifadeyi güvenli değerlendir; bozuk/belirsiz → None."""
        try:
            dugum = ast.parse(str(ifade), mode="eval").body
            return float(self._hesapla(dugum))
        except Exception:
            return None

    def _hesapla(self, dugum):
        if isinstance(dugum, ast.Constant) and isinstance(dugum.value, (int, float)):
            return dugum.value
        if isinstance(dugum, ast.BinOp) and type(dugum.op) in _ISLEM:
            sol = self._hesapla(dugum.left)
            sag = self._hesapla(dugum.right)
            if sag == 0 and isinstance(dugum.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                raise ZeroDivisionError
            return _ISLEM[type(dugum.op)](sol, sag)
        if isinstance(dugum, ast.UnaryOp) and type(dugum.op) in _ISLEM:
            return _ISLEM[type(dugum.op)](self._hesapla(dugum.operand))
        raise ValueError("izin verilmeyen ifade")

    def iddia_dogrula(self, iddia: str) -> Optional[bool]:
        """'sol=sag' biçimindeki aritmetik iddiayı doğrula.

        True = doğru, False = yanlış, None = ayrıştırılamadı (belirsiz).
        """
        if "=" not in iddia:
            return None
        sol, sag = iddia.split("=", 1)
        v_sol, v_sag = self.deger(sol), self.deger(sag)
        if v_sol is None or v_sag is None:
            return None
        return abs(v_sol - v_sag) < 1e-9

    # ── ExperienceCandidate adaptörü (ConflictResolver ile takılabilir) ──
    def aday_dogrula(self, store, aday: ExperienceCandidate) -> Optional[bool]:
        """İlişki "eşittir" ise özne/nesne ifadelerini deterministik sınar.

        Örn. (özne="2+3", ilişki="eşittir", nesne="5") → True.
        İlişki uyuşmuyorsa veya ifadeler bozuksa None (belirsiz).
        """
        try:
            iliski = store.relations.iliski_al(aday.relation_id)
            if iliski.token.strip().lower() not in ("esittir", "eşittir", "esit"):
                return None
            ozne = store.entities.getir(aday.subject_id).token
            nesne = store.entities.getir(aday.object_id).token
            return self.iddia_dogrula(f"{ozne}={nesne}")
        except KeyError:
            return None


class MantikOrtam:
    """Temel mantık doğrulayıcı (modus ponens).

    Desteklenen metin biçimleri:
      * ``A -> B, A |- B``
      * ``A → B, A ⊢ B``

    True = çıkarım geçerli, False = aynı öncüllerden verilen sonuç çıkmaz,
    None = ifade ayrıştırılamadı.
    """

    _AYIR = re.compile(r"\s*(?:->|→|=>|⇒)\s*")
    _TURNSTILE = re.compile(r"(?:\|-|⊢)")

    @staticmethod
    def _norm_ifade(ifade: str) -> str:
        return re.sub(r"\s+", " ", str(ifade).strip().lower())

    def modus_ponens(self, kosul: str, olgu: str, sonuc: str) -> Optional[bool]:
        parca = self._AYIR.split(str(kosul), maxsplit=1)
        if len(parca) != 2:
            return None
        a, b = map(self._norm_ifade, parca)
        olgu_n = self._norm_ifade(olgu)
        sonuc_n = self._norm_ifade(sonuc)
        if not a or not b or not olgu_n or not sonuc_n:
            return None
        if olgu_n == a:
            return sonuc_n == b
        # Bu mini ortam yalnız modus ponens ölçer; başka çıkarım belirsizdir.
        return None

    def iddia_dogrula(self, iddia: str) -> Optional[bool]:
        if not iddia:
            return None
        bol = self._TURNSTILE.split(iddia, maxsplit=1)
        if len(bol) != 2:
            return None
        oncul_metin, sonuc = bol[0], bol[1]
        oncul = [p.strip() for p in oncul_metin.split(",") if p.strip()]
        if len(oncul) < 2:
            return None
        # Her koşul + her olgu ikilisini dene.
        for kosul in oncul:
            if len(self._AYIR.split(kosul, maxsplit=1)) != 2:
                continue
            for olgu in oncul:
                if olgu == kosul:
                    continue
                r = self.modus_ponens(kosul, olgu, sonuc)
                if r is not None:
                    return r
        return None

    def aday_dogrula(self, store, aday: ExperienceCandidate) -> Optional[bool]:
        """İlişki token'ı 'çıkarır/entails/gerektirir' ise metinleri sınar.

        Özne koşul (``A -> B``), nesne ise sonuç olarak yorumlanır; özne
        tarafındaki virgülden sonraki parça olgu kabul edilir: ``A -> B, A``.
        """
        try:
            iliski = store.relations.iliski_al(aday.relation_id)
            token = iliski.token.strip().lower()
            if token not in ("çıkarır", "cikarir", "entails", "gerektirir", "implies"):
                return None
            ozne = store.entities.getir(aday.subject_id).token
            nesne = store.entities.getir(aday.object_id).token
            return self.iddia_dogrula(f"{ozne} |- {nesne}")
        except KeyError:
            return None


class TutarlilikOrtam:
    """Bilgi deposu tutarlılık doğrulayıcı.

    İlişkinin ``requires_subject_props`` ve ``requires_object_props`` kısıtlarını
    mevcut PropertyIndex'e karşı deterministik kontrol eder. Bilinen değerler
    hedefle çelişirse False, bütün gereksinimler biliniyor ve uyumluysa True,
    eksik/az güvenilir bilgi varsa None döner.
    """

    def __init__(self, guven_esigi: float = 0.5):
        self.guven_esigi = float(guven_esigi)

    @staticmethod
    def _uyumlu(deger: float, hedef: float) -> bool:
        return (deger >= 0.5) == (hedef >= 0.5)

    def ozellik_celiskisi(self, store, entity_id: str, ad: str, yeni_deger: float,
                          guven_esigi: Optional[float] = None) -> Optional[bool]:
        """Yeni özellik mevcut kayıtla çelişiyor mu?

        True = çelişki var, False = çelişki yok, None = mevcut kayıt yok veya
        güven eşiğinin altında.
        """
        pv = store.properties.al(entity_id, ad)
        if pv is None:
            return None
        esik = self.guven_esigi if guven_esigi is None else float(guven_esigi)
        if pv.confidence < esik:
            return None
        return bool((pv.deger >= 0.5) != (float(yeni_deger) >= 0.5))

    def aday_dogrula(self, store, aday: ExperienceCandidate) -> Optional[bool]:
        try:
            r = store.relations.iliski_al(aday.relation_id)
            store.entities.getir(aday.subject_id)
            store.entities.getir(aday.object_id)
        except KeyError:
            return None

        eksik = False
        for taraf, entity_id, gereksinimler in (
            ("özne", aday.subject_id, r.requires_subject_props),
            ("nesne", aday.object_id, r.requires_object_props),
        ):
            for ad, hedef in gereksinimler.items():
                pv = store.properties.al(entity_id, ad)
                if pv is None or pv.confidence < self.guven_esigi:
                    eksik = True
                    continue
                if not self._uyumlu(pv.deger, hedef):
                    aday.evidence.append(
                        f"tutarlılık ihlali: {taraf} {entity_id} için {ad}={pv.deger}, beklenen {hedef}")
                    return False
        return None if eksik else True


__all__ = ["AritmetikOrtam", "MantikOrtam", "TutarlilikOrtam"]
