# -*- coding: utf-8 -*-
"""
TorchKoprusu — Deneyim ↔ Geometrik Seyrek Bellek Köprüsü (v0.6)
=================================================================
(rapor §19 v0.6, §22 commit 6, EK-B)

Mevcut çekirdekteki `mimari/seyrek_tablo.py` (`HashlenmisKureselTablo`, torch
tabanlı) token pencerelerini hash'leyip sabit boyutlu bir tabloya yazar. Bu
modül, deneyim üçlülerini (özne, ilişki, nesne) aynı tabloya bağlar:

    üçlü → 3 bileşenli token id vektörü → tablo.anahtar() → satır(lar) → "gen" vektörü

Böylece Knowledge/Experience katmanının kavramsal deneyimleri, geometrik
çekirdeğin seyrek belleğine yazılabilir ve bilinear zincirin etkileşim uzayına
enjekte edilebilir (EK-B: "Sparse memory → Uzun süreli Experience/Knowledge
Store"). v0.6 köprüsünün protokolü `hga/memory/sparse_memory.py`'de saf Python
ile de aynıdır (aynı parmak izi mantığı).

NOT: torch yalnızca köprü kurulurken içe aktarılır; bu paketin geri kalanı
torch'suz çalışmaya devam eder. torch yoksa `TorchKoprusu()` AÇIKÇA ImportError
fırlatır (sessizce boş köprü döndürmez) — `torch_var_mi()` ile önceden
sorgulanabilir.
"""
from typing import Optional, Tuple

from .sparse_memory import parmak_izi


def torch_var_mi() -> bool:
    """Torch bu ortamda kurulu mu?"""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def bilesen_token(bilesen: str) -> int:
    """Bir deneyim bileşenini (entity_id/relation_id) token id'sine indirger.

    entity_id'ler tokenizer token ID'lerinden AYRI olduğu için (rapor §4) burada
    deterministik bir kavramsal kimlik → tamsayı eşlemesi kullanılır.
    """
    return parmak_izi([bilesen]) & 0xFFFF   # 0..65535


# Geriye dönük uyumluluk takma adı
_bilesen_token = bilesen_token


class TorchKoprusu:
    """Deneyim üçlülerini torch `HashlenmisKureselTablo`'ya bağlar."""

    def __init__(self, tablo_boyutu: int = 1_048_576, boyut: int = 32):
        try:
            import torch
            from mimari.seyrek_tablo import HashlenmisKureselTablo
        except ImportError as e:  # torch yok → dürüstçe yüzeye çıkar
            raise ImportError(
                "TorchKoprusu için torch + mimari/seyrek_tablo.py gerekli "
                "(pip install -r gereksinimler.txt). v0.6 köprüsü yalnızca "
                "torch kurulu ortamlarda kurulabilir.") from e
        self.torch = torch
        self.tablo = HashlenmisKureselTablo(tablo_boyutu=tablo_boyutu, boyut=boyut)

    # ── Üçlü → tensor ────────────────────────────────────────────────────
    def uclu_tensoru(self, uclusu: Tuple[str, str, str]):
        """(özne_id, ilişki_id, nesne_id) → (1, 3) int64 tensor."""
        ids = [bilesen_token(b) for b in uclusu]
        return self.torch.tensor([ids], dtype=self.torch.long)

    def anahtar(self, uclusu: Tuple[str, str, str]):
        """Üçlünün kavramsal anahtarı (tablo.anahtar ile aynı şema)."""
        return self.tablo.anahtar(self.uclu_tensoru(uclusu))

    def vektor(self, uclusu: Tuple[str, str, str]):
        """Üçlünün 'gen' vektörü (boş kümeye düşerse sıfır vektör)."""
        return self.tablo(self.uclu_tensoru(uclusu))

    def dokun(self, uclusu: Tuple[str, str, str]):
        """Üçlünün adreslediği satırlara dokunur (eğitimde dolar)."""
        return self.vektor(uclusu)

    def doluluk(self) -> Tuple[int, int]:
        """(dolu_satır, toplam_satır) — torch tablosunun doluluğu."""
        return self.tablo.doluluk_orani()

    def kapasite(self) -> dict:
        return self.tablo.kapasite()
