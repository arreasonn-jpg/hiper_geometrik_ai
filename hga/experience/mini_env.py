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
