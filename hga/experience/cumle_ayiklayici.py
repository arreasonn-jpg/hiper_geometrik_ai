# -*- coding: utf-8 -*-
"""
Cümle Ayıklayıcı — "Gerçek veri → Temsil" giriş noktası
=========================================================
(rapor §2, §12, §19 v1.0: döngü "Gerçek veri → Temsil" ile BAŞLAR)

Tam bir ayrıştırıcı kurmak yerine, bilinçli olarak DAR ve LEXICON TABANLI bir
ayıklayıcı sağlar: Türkçe cümleleri (özne, ilişki, nesne) üçlüsüne çevirir ve
bunları `REAL_DATA` kaynağıyla KnowledgeStore'a aktarır. Böylece sinir ağı
çekirdeği dışındaki deneyim döngüsü GERÇEK veriyle beslenir.

Lexicon girişi (yüzey biçim → kavramsal kayıt):

    "ata": {"token": "Ata", "tip": "hayvan", "ozellikler": {"binilebilir": 1}}

`cumlelerden_bilgi_aktar()`:
    1. her cümle için üçlü ayıklar,
    2. özne/nesne varlıklarını (varsa özellikleriyle) REAL_DATA olarak kurar,
    3. ilişkiyi (yoksa) tanımlar ve üçlüye ait REAL_DATA kanıtı yazar.

DÜRÜSTLÜK NOTU: Bu, gerçek bir sözdizimsel ayrıştırıcı DEĞİLDİR; sınırlı bir
sözlük üzerinden pattern eşleme yapar. Kapsam dışı cümleler `None` döner ve
sessizce atlanır — sistem asla uydurmaz.
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..knowledge.schemas import KaynakTuru

# ── Yardımcılar ──────────────────────────────────────────────────────────
_KUCULT = {"İ": "i", "I": "i", "Ş": "s", "Ğ": "g", "Ü": "u", "Ö": "o", "Ç": "c"}
_KUCUK = {"ı": "i", "ş": "s", "ğ": "g", "ü": "u", "ö": "o", "ç": "c",
          "â": "a", "î": "i", "û": "u"}


def ascii_norm(metin: str) -> str:
    """Türkçe metni eşleştirme için ASCII'ye indirger (İ→i sorunu çözülür)."""
    t = metin or ""
    for a, b in _KUCULT.items():
        t = t.replace(a, b)
    t = t.lower()
    for a, b in _KUCUK.items():
        t = t.replace(a, b)
    return re.sub(r"[^a-z0-9\s]", " ", t).strip()


@dataclass
class AyiklananUclu:
    ozne: str
    iliski: str
    nesne: str

    def uclusu(self):
        return (self.ozne, self.iliski, self.nesne)


# ── Varsayılan sözlük (sınırlı ama çalışır) ────────────────────────────────
# 'nesne_durumu' (isteğe bağlı) ilişkinin beklediği nesne durumudur; yalnız
# sözlük büyütme (`sozluk_buyutme.py`) yeni nesne çıkarırken kullanır.
VARSAYILAN_SOZLUK: Dict[str, Dict] = {
    "binmek": {
        "iliski": "Binmek",
        "nesne_durumu": "yonelme",
        "yuklemler": ["bindi", "biniyor", "biner", "binmis", "binecek"],
        "ozneler": {
            "ali": {"token": "Ali", "tip": "insan", "ozellikler": {"canli": 1}},
            "ayse": {"token": "Ayşe", "tip": "insan", "ozellikler": {"canli": 1}},
        },
        "nesneler": {
            "ata": {"token": "Ata", "tip": "hayvan", "ozellikler": {"binilebilir": 1}},
            "ataya": {"token": "Ata", "tip": "hayvan", "ozellikler": {"binilebilir": 1}},
            "araba": {"token": "Araba", "tip": "tasit", "ozellikler": {"binilebilir": 1}},
            "arabaya": {"token": "Araba", "tip": "tasit", "ozellikler": {"binilebilir": 1}},
            "gokyuzu": {"token": "Gökyüzü", "tip": "mekan", "ozellikler": {"binilebilir": 0}},
            "gokyuzune": {"token": "Gökyüzü", "tip": "mekan", "ozellikler": {"binilebilir": 0}},
        },
    },
    "bakmak": {
        "iliski": "Bakmak",
        "nesne_durumu": "yonelme",
        "yuklemler": ["bakti", "bakiyor", "bakar", "bakmis"],
        "ozneler": {
            "ali": {"token": "Ali", "tip": "insan", "ozellikler": {"canli": 1}},
            "ayse": {"token": "Ayşe", "tip": "insan", "ozellikler": {"canli": 1}},
        },
        "nesneler": {
            "gokyuzu": {"token": "Gökyüzü", "tip": "mekan", "ozellikler": {}},
            "gokyuzune": {"token": "Gökyüzü", "tip": "mekan", "ozellikler": {}},
            "yildiz": {"token": "Yıldız", "tip": "mekan", "ozellikler": {}},
        },
    },
    # ── Genişletilmiş sözlük (v1.0+): elle küratörlü yeni ilişkiler ──────
    # İlişkiler ASLA çalışma anında uydurulmaz; bunlar BİLİNEN, elle doğrulanmış
    # fiil desenleridir. Yüklem biçimleri `turkce.py`'nin testli `fiil_cekimi`
    # çıktısıyla birebir uyumludur (3. tekil, dört zaman).
    "gitmek": {
        "iliski": "Gitmek",
        "nesne_durumu": "yonelme",
        "yuklemler": ["gitti", "gidiyor", "gidecek", "gider"],
        "ozneler": {
            "ali": {"token": "Ali", "tip": "insan", "ozellikler": {"canli": 1}},
            "ayse": {"token": "Ayşe", "tip": "insan", "ozellikler": {"canli": 1}},
        },
        "nesneler": {
            "okula": {"token": "Okul", "tip": "mekan", "ozellikler": {}},
            "eve": {"token": "Ev", "tip": "mekan", "ozellikler": {}},
            "sinemaya": {"token": "Sinema", "tip": "mekan", "ozellikler": {}},
            "parka": {"token": "Park", "tip": "mekan", "ozellikler": {}},
        },
    },
    "gelmek": {
        "iliski": "Gelmek",
        "nesne_durumu": "yonelme",
        "yuklemler": ["geldi", "geliyor", "gelecek", "gelir"],
        "ozneler": {
            "ali": {"token": "Ali", "tip": "insan", "ozellikler": {"canli": 1}},
            "ayse": {"token": "Ayşe", "tip": "insan", "ozellikler": {"canli": 1}},
        },
        "nesneler": {
            "okula": {"token": "Okul", "tip": "mekan", "ozellikler": {}},
            "eve": {"token": "Ev", "tip": "mekan", "ozellikler": {}},
        },
    },
    "okumak": {
        "iliski": "Okumak",
        "nesne_durumu": "belirtme",
        "yuklemler": ["okudu", "okuyor", "okuyacak", "okur"],
        "ozneler": {
            "ali": {"token": "Ali", "tip": "insan", "ozellikler": {"canli": 1}},
            "ayse": {"token": "Ayşe", "tip": "insan", "ozellikler": {"canli": 1}},
        },
        "nesneler": {
            "kitabi": {"token": "Kitap", "tip": "esya", "ozellikler": {}},
            "gazeteyi": {"token": "Gazete", "tip": "esya", "ozellikler": {}},
            "mektubu": {"token": "Mektup", "tip": "esya", "ozellikler": {}},
        },
    },
    "yazmak": {
        "iliski": "Yazmak",
        "nesne_durumu": "belirtme",
        "yuklemler": ["yazdi", "yaziyor", "yazacak", "yazar"],
        "ozneler": {
            "ali": {"token": "Ali", "tip": "insan", "ozellikler": {"canli": 1}},
            "ayse": {"token": "Ayşe", "tip": "insan", "ozellikler": {"canli": 1}},
        },
        "nesneler": {
            "mektubu": {"token": "Mektup", "tip": "esya", "ozellikler": {}},
            "siiri": {"token": "Şiir", "tip": "esya", "ozellikler": {}},
        },
    },
    "sevmek": {
        "iliski": "Sevmek",
        "nesne_durumu": "belirtme",
        "yuklemler": ["sevdi", "seviyor", "sevecek", "sever"],
        "ozneler": {
            "ali": {"token": "Ali", "tip": "insan", "ozellikler": {"canli": 1}},
            "ayse": {"token": "Ayşe", "tip": "insan", "ozellikler": {"canli": 1}},
        },
        "nesneler": {
            "kediyi": {"token": "Kedi", "tip": "hayvan", "ozellikler": {"canli": 1}},
            "denizi": {"token": "Deniz", "tip": "mekan", "ozellikler": {}},
        },
    },
}


class CumleAyiklayici:
    """Sınırlı lexicon tabanlı (özne, ilişki, nesne) ayıklayıcı."""

    def __init__(self, sozluk: Optional[Dict[str, Dict]] = None):
        self.sozluk = dict(sozluk or VARSAYILAN_SOZLUK)

    def ayikla(self, cumle: str) -> Optional[AyiklananUclu]:
        """Cümle → (özne, ilişki, nesne); eşleşmezse None."""
        kelimeler = ascii_norm(cumle).split()
        if not kelimeler:
            return None
        for _, sablon in self.sozluk.items():
            yuklem = next((k for k in kelimeler if k in sablon["yuklemler"]), None)
            if yuklem is None:
                continue
            ozne = next((k for k in kelimeler if k in sablon["ozneler"]), None)
            if ozne is None:
                continue
            nesne = next((k for k in kelimeler if k in sablon["nesneler"]), None)
            if nesne is None:
                continue
            return AyiklananUclu(
                ozne=sablon["ozneler"][ozne]["token"],
                iliski=sablon["iliski"],
                nesne=sablon["nesneler"][nesne]["token"],
            )
        return None

    def detay(self, cumle: str) -> Optional[Dict]:
        """Üçlü + kayıt bilgileri (tip/özellik) — aktarım için."""
        kelimeler = ascii_norm(cumle).split()
        if not kelimeler:
            return None
        for _, sablon in self.sozluk.items():
            yuklem = next((k for k in kelimeler if k in sablon["yuklemler"]), None)
            if yuklem is None:
                continue
            ozne_k = next((k for k in kelimeler if k in sablon["ozneler"]), None)
            if ozne_k is None:
                continue
            nesne_k = next((k for k in kelimeler if k in sablon["nesneler"]), None)
            if nesne_k is None:
                continue
            return {
                "ozne": sablon["ozneler"][ozne_k],
                "iliski": sablon["iliski"],
                "nesne": sablon["nesneler"][nesne_k],
            }
        return None


def cumlelerden_bilgi_aktar(store, cumleler: List[str],
                            ayiklayici: Optional[CumleAyiklayici] = None,
                            iliski_kisitlari: Optional[Dict[str, Dict]] = None
                            ) -> List[AyiklananUclu]:
    """Cümleleri ayıkla ve REAL_DATA olarak KnowledgeStore'a aktar.

    * özne/nesne varlıkları (özellikleriyle) kurulur/pekiştirilir,
    * ilişki (yoksa) tanımlanır; `iliski_kisitlari` ile kısıt verilebilir,
    * her üçlüye score=1.0 REAL_DATA kanıtı yazılır.

    Dönen liste, aktarılan üçlülerdir (kayıp cümleler atlanır).
    """
    ayiklayici = ayiklayici or CumleAyiklayici()
    kisitlar = iliski_kisitlari or {}
    aktarilan: List[AyiklananUclu] = []

    # mevcut ilişki token → id eşlemesi
    token_2_id = {r.token.strip().lower(): r.relation_id
                  for r in store.relations.iliskiler()}

    for c in cumleler:
        d = ayiklayici.detay(c)
        if d is None:
            continue

        ozne_rec = d["ozne"]
        nesne_rec = d["nesne"]

        s = store.varlik_ekle(
            ozne_rec["token"], entity_type=ozne_rec["tip"],
            properties=ozne_rec.get("ozellikler"),
            ozel_isim=(ozne_rec["tip"] == "insan"),
            source=KaynakTuru.REAL_DATA)
        o = store.varlik_ekle(
            nesne_rec["token"], entity_type=nesne_rec["tip"],
            properties=nesne_rec.get("ozellikler"),
            ozel_isim=(nesne_rec["tip"] == "insan"),
            source=KaynakTuru.REAL_DATA)

        iliski_token = d["iliski"]
        rid = token_2_id.get(iliski_token.strip().lower())
        if rid is None:
            k = kisitlar.get(iliski_token, {})
            r = store.iliski_tanimla(
                iliski_token,
                subject_types=k.get("subject_types"),
                object_types=k.get("object_types"),
                requires_object_props=k.get("requires_object_props"),
                requires_subject_props=k.get("requires_subject_props"),
                source=KaynakTuru.VERIFIED_RULE)
            rid = r.relation_id
            token_2_id[iliski_token.strip().lower()] = rid

        store.olgu_kaydet(s.entity_id, rid, o.entity_id, 1.0,
                          source=KaynakTuru.REAL_DATA, confidence=1.0)
        aktarilan.append(AyiklananUclu(ozne_rec["token"], iliski_token,
                                       nesne_rec["token"]))
    return aktarilan
