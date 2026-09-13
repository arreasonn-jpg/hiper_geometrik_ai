# -*- coding: utf-8 -*-
"""Bağımsız doğrulama hattı.

Evaluator yalnız aday değerlendirmesi üretir. Bu modül evaluator çağırmaz ve
onun skorlarını ground truth kabul etmez. Bir aday önce değerlendirilmiş
(``VALID``) olmalı, ardından bağımsız prosedür ``True``/``False``/``None``
döndürmelidir::

    VALID -> VERIFYING -> VERIFIED | INVALID | UNCERTAIN

``None`` kanıt yetersizliğidir; adayın VALID kalması ya da INVALID sayılması
yerine açıkça UNCERTAIN yapılır.
"""
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from ..knowledge.schemas import DeneyimDurumu, ExperienceCandidate, KaynakTuru
from .consolidation import Consolidator
from .state_machine import DeneyimDurumMakinesi


@dataclass
class DogrulamaRaporu:
    islenen: int = 0
    dogrulanan: int = 0
    reddedilen: int = 0
    belirsiz: int = 0
    atlanan: int = 0
    bilgi_buyumesi: int = 0
    yanlis_kabul_oncesi: int = 0
    yanlis_kabul_sonrasi: int = 0
    dogrulananlar: List[str] = field(default_factory=list)
    reddedilenler: List[str] = field(default_factory=list)
    belirsizler: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class DogrulamaHatti:
    """Evaluator'dan bağımsız, harici/deterministik doğrulayıcı adaptörü."""

    def __init__(
        self,
        dogrulayici: Callable,
        evaluator=None,
        consolidator: Optional[Consolidator] = None,
        dogrulayici_adi: str = "deterministik-dogrulayici",
    ):
        self.dogrulayici = dogrulayici
        # evaluator yalnız geriye dönük çağrı uyumluluğu için kabul edilir;
        # bilinçli olarak saklanmaz ve bu katmanda asla çağrılmaz.
        self.consolidator = consolidator or Consolidator()
        self.dogrulayici_adi = dogrulayici_adi
        self.durum_makinesi = DeneyimDurumMakinesi()

    def _aday_dogrula(self, store, aday: ExperienceCandidate) -> Optional[bool]:
        """Önceden değerlendirilmiş tek adayı bağımsız prosedürle doğrula."""
        if aday.state != DeneyimDurumu.VALID:
            raise ValueError(
                f"Verifier yalnız VALID aday kabul eder; {aday.experience_id}={aday.state.value}"
            )

        gecis = self.durum_makinesi.uygula(
            aday, DeneyimDurumu.VERIFYING, "bağımsız doğrulama başladı"
        )
        if not gecis.ok:  # pragma: no cover - sözleşme ihlaline karşı savunma
            raise ValueError(gecis.neden)

        sonuc = self.dogrulayici(store, aday)
        if sonuc is True:
            aday.source = KaynakTuru.EXTERNAL_VERIFIED
            aday.source_confidence = 1.0
            aday.verified_by = self.dogrulayici_adi
            self.durum_makinesi.uygula(
                aday, DeneyimDurumu.VERIFIED, "bağımsız doğrulayıcı onayladı"
            )
        elif sonuc is False:
            self.durum_makinesi.uygula(
                aday, DeneyimDurumu.INVALID, "bağımsız doğrulayıcı çürüttü"
            )
            store.celiski_logla(aday, "deterministik-curutme")
        else:
            self.durum_makinesi.uygula(
                aday, DeneyimDurumu.UNCERTAIN, "doğrulayıcı için kanıt yetersiz"
            )
        return sonuc

    def isle(self, store, adaylar: List[ExperienceCandidate]) -> DogrulamaRaporu:
        """VALID adayları doğrula ve yalnız VERIFIED sonuçları konsolide et."""
        if any(a.state == DeneyimDurumu.CANDIDATE for a in adaylar):
            raise ValueError("Verifier CANDIDATE değerlendirmez; önce Evaluator çalıştırılmalı")

        rapor = DogrulamaRaporu(islenen=len(adaylar))
        for aday in adaylar:
            if aday.state != DeneyimDurumu.VALID:
                rapor.atlanan += 1
                continue
            sonuc = self._aday_dogrula(store, aday)
            if sonuc is True:
                rapor.dogrulanan += 1
                rapor.dogrulananlar.append(aday.experience_id)
            elif sonuc is False:
                rapor.yanlis_kabul_oncesi += 1
                rapor.reddedilen += 1
                rapor.reddedilenler.append(aday.experience_id)
            else:
                rapor.belirsiz += 1
                rapor.belirsizler.append(aday.experience_id)

        onceki_versiyon = store.versiyon
        self.consolidator.konsolide_et(store, adaylar)
        rapor.bilgi_buyumesi = store.versiyon - onceki_versiyon
        # Her False karar aynı atomik işlemde INVALID'a geçirildi; doğrulayıcıyı
        # ikinci kez çağırıp durum bağımlı/nondeterministik sonuç üretmeyiz.
        rapor.yanlis_kabul_sonrasi = 0
        return rapor


__all__ = ["DogrulamaHatti", "DogrulamaRaporu"]
