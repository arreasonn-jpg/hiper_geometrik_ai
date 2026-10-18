# -*- coding: utf-8 -*-
"""P1 — Çoklu ortam + cross-domain verifier izolasyonu.

Mevcut self-learning hattı tek bir ortamda (aritmetik) çalışıyordu. Tek ortam
iki soruyu cevapsız bırakır:

1. **Genellenebilirlik.** Hat yalnız aritmetiğe mi özel? Fizik, nedensellik,
   uzamsal, zamansal ve dilsel ortamlarda da aynı davranıyor mu?
2. **Kontaminasyon.** Bir ortamın doğrulayıcısı başka bir ortamın iddiasını
   onaylıyor mu? Onaylıyorsa "doğrulandı" etiketi anlamsızdır: doğrulayıcı
   alan bilgisi değil, yüzeysel bir örüntü kullanıyor demektir.

Bu modül beş ortam tanımlar ve **5×5 çapraz doğrulama matrisi** kurar. Her
ortamın olguları her ortamın doğrulayıcısına sunulur. Beklenen sonuç:

* Köşegen (kendi doğrulayıcısı) → doğru olguları kabul, yanlışları ret.
* Köşegen dışı → **ABSTAIN** (yetkisiz alan, karar verme).

Bir doğrulayıcının alanı dışında ``True``/``False`` döndürmesi kontaminasyondur
ve ayrı sayılır. Burada kritik ayrım şudur: *yanlış cevap vermek* ile
*yetkisiz olduğunu bilmek* farklı şeylerdir. İkincisi ölçülmeden "verifier
izolasyonu var" denemez.

Ortamlar kasıtla **birbirinden bağımsız sembol uzayları** kullanır; ortak
sembol olsaydı izolasyon testi sembol çakışmasını ölçerdi, alan bilgisini
değil.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Tuple

PROTOCOL = "multi_environment_verifier_isolation_v1"
SCHEMA_VERSION = 1

#: Doğrulayıcı kararları. ABSTAIN "bu benim alanım değil" demektir.
ACCEPT = "ACCEPT"
REJECT = "REJECT"
ABSTAIN = "ABSTAIN"

ENVIRONMENTS: Tuple[str, ...] = (
    "physics", "causal", "spatial", "temporal", "language",
)


@dataclass
class Claim:
    """Bir ortamda üretilmiş, doğru/yanlış değeri bilinen tek iddia."""

    environment: str
    subject: str
    relation: str
    object: str
    truth: bool

    def as_triple(self) -> Tuple[str, str, str]:
        return (self.subject, self.relation, self.object)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Ortamlar ────────────────────────────────────────────────────────────────
# Her ortam: (iddia üreteci, doğrulayıcı). Doğrulayıcı yalnız KENDİ alanının
# sembollerini tanır; tanımadığında ABSTAIN döner.

_PHYSICS_REL = "dusme_suresi_s"
_CAUSAL_REL = "neden_olur"
_SPATIAL_REL = "kuzeyinde"
_TEMPORAL_REL = "once_gelir"
_LANGUAGE_REL = "cogulu"


def _physics_claims(rng: random.Random, n: int) -> List[Claim]:
    """Serbest düşüş: h = ½gt² → t = √(2h/g). g=10 alınır (tam sayı zemini)."""
    iddialar = []
    for _ in range(n):
        t = rng.randint(1, 9)
        h = 5 * t * t                     # ½ · 10 · t²
        dogru = rng.random() < 0.5
        deger = t if dogru else t + rng.choice([-1, 1, 2])
        if deger < 1:
            deger = t + 1
            dogru = False
        iddialar.append(Claim("physics", f"PHYS_H{h}", _PHYSICS_REL,
                              f"PHYS_T{deger}", deger == t))
    return iddialar


def _physics_verify(claim: Claim) -> str:
    if claim.relation != _PHYSICS_REL:
        return ABSTAIN
    if not (claim.subject.startswith("PHYS_H")
            and claim.object.startswith("PHYS_T")):
        return ABSTAIN
    try:
        h = int(claim.subject[6:])
        t = int(claim.object[6:])
    except ValueError:
        return ABSTAIN
    return ACCEPT if h == 5 * t * t else REJECT


#: Nedensellik ortamının YER GERÇEĞİ grafiği (yalnız doğrulayıcı bilir).
_CAUSAL_EDGES: Dict[str, str] = {
    "CAUS_yagmur": "CAUS_islak_zemin",
    "CAUS_islak_zemin": "CAUS_kayma",
    "CAUS_kivilcim": "CAUS_yangin",
    "CAUS_yangin": "CAUS_duman",
    "CAUS_virus": "CAUS_ates",
}


def _causal_claims(rng: random.Random, n: int) -> List[Claim]:
    dugumler = sorted(set(_CAUSAL_EDGES) | set(_CAUSAL_EDGES.values()))
    kenarlar = sorted(_CAUSAL_EDGES.items())
    iddialar = []
    for _ in range(n):
        if rng.random() < 0.5:
            a, b = rng.choice(kenarlar)
            iddialar.append(Claim("causal", a, _CAUSAL_REL, b, True))
        else:
            a, b = rng.sample(dugumler, 2)
            iddialar.append(Claim("causal", a, _CAUSAL_REL, b,
                                  _CAUSAL_EDGES.get(a) == b))
    return iddialar


def _causal_verify(claim: Claim) -> str:
    if claim.relation != _CAUSAL_REL:
        return ABSTAIN
    if not (claim.subject.startswith("CAUS_")
            and claim.object.startswith("CAUS_")):
        return ABSTAIN
    return ACCEPT if _CAUSAL_EDGES.get(claim.subject) == claim.object else REJECT


#: Uzamsal ızgara: her şehir (x, y). Kuzey = daha büyük y.
_SPATIAL_GRID: Dict[str, Tuple[int, int]] = {
    "GEO_ankara": (3, 5), "GEO_izmir": (1, 3), "GEO_antalya": (2, 1),
    "GEO_samsun": (4, 8), "GEO_van": (8, 4), "GEO_edirne": (0, 6),
}


def _spatial_claims(rng: random.Random, n: int) -> List[Claim]:
    sehirler = sorted(_SPATIAL_GRID)
    iddialar = []
    for _ in range(n):
        a, b = rng.sample(sehirler, 2)
        iddialar.append(Claim(
            "spatial", a, _SPATIAL_REL, b,
            _SPATIAL_GRID[a][1] > _SPATIAL_GRID[b][1]))
    return iddialar


def _spatial_verify(claim: Claim) -> str:
    if claim.relation != _SPATIAL_REL:
        return ABSTAIN
    if claim.subject not in _SPATIAL_GRID or claim.object not in _SPATIAL_GRID:
        return ABSTAIN
    return (ACCEPT if _SPATIAL_GRID[claim.subject][1]
            > _SPATIAL_GRID[claim.object][1] else REJECT)


#: Zamansal sıra: olay → yıl.
_TEMPORAL_YEARS: Dict[str, int] = {
    "EVT_matbaa": 1450, "EVT_istanbul_fethi": 1453, "EVT_amerika": 1492,
    "EVT_fransiz_devrimi": 1789, "EVT_cumhuriyet": 1923, "EVT_ay": 1969,
}


def _temporal_claims(rng: random.Random, n: int) -> List[Claim]:
    olaylar = sorted(_TEMPORAL_YEARS)
    iddialar = []
    for _ in range(n):
        a, b = rng.sample(olaylar, 2)
        iddialar.append(Claim(
            "temporal", a, _TEMPORAL_REL, b,
            _TEMPORAL_YEARS[a] < _TEMPORAL_YEARS[b]))
    return iddialar


def _temporal_verify(claim: Claim) -> str:
    if claim.relation != _TEMPORAL_REL:
        return ABSTAIN
    if (claim.subject not in _TEMPORAL_YEARS
            or claim.object not in _TEMPORAL_YEARS):
        return ABSTAIN
    return (ACCEPT if _TEMPORAL_YEARS[claim.subject]
            < _TEMPORAL_YEARS[claim.object] else REJECT)


#: Dilsel ortam: Türkçe çoğul eki (ünlü uyumu).
_LANGUAGE_WORDS: Tuple[Tuple[str, str], ...] = (
    ("kitap", "kitaplar"), ("ev", "evler"), ("araba", "arabalar"),
    ("göz", "gözler"), ("okul", "okullar"), ("köy", "köyler"),
)


def _language_claims(rng: random.Random, n: int) -> List[Claim]:
    iddialar = []
    for _ in range(n):
        tekil, cogul = rng.choice(_LANGUAGE_WORDS)
        if rng.random() < 0.5:
            iddialar.append(Claim("language", f"LEX_{tekil}", _LANGUAGE_REL,
                                  f"LEX_{cogul}", True))
        else:
            yanlis = tekil + ("ler" if cogul.endswith("lar") else "lar")
            iddialar.append(Claim("language", f"LEX_{tekil}", _LANGUAGE_REL,
                                  f"LEX_{yanlis}", False))
    return iddialar


def _language_verify(claim: Claim) -> str:
    if claim.relation != _LANGUAGE_REL:
        return ABSTAIN
    if not (claim.subject.startswith("LEX_")
            and claim.object.startswith("LEX_")):
        return ABSTAIN
    tekil = claim.subject[4:]
    dogru = dict(_LANGUAGE_WORDS).get(tekil)
    if dogru is None:
        return ABSTAIN
    return ACCEPT if claim.object[4:] == dogru else REJECT


GENERATORS: Dict[str, Callable[[random.Random, int], List[Claim]]] = {
    "physics": _physics_claims,
    "causal": _causal_claims,
    "spatial": _spatial_claims,
    "temporal": _temporal_claims,
    "language": _language_claims,
}

VERIFIERS: Dict[str, Callable[[Claim], str]] = {
    "physics": _physics_verify,
    "causal": _causal_verify,
    "spatial": _spatial_verify,
    "temporal": _temporal_verify,
    "language": _language_verify,
}



# ── Çekişmeli sondalar (izolasyonun ZOR sınavı) ─────────────────────────────
# Ayrık sembol uzaylarıyla izolasyon kolaydır: ön ek eşlemesi yeter. Gerçek
# soru şudur — doğrulayıcı KENDİ sembollerini gördüğünde, ilişki başka bir
# alana aitse yine de konuşur mu? Aşağıdaki sondalar tam olarak bunu zorlar
# ve üç tuzak içerir:
#
#   1. ``own_symbols_foreign_relation``: doğrulayıcının tanıdığı semboller,
#      başka alanın ilişkisiyle. (Sembole bakıp karar veren bir doğrulayıcı
#      burada konuşur ve kirlenir.)
#   2. ``foreign_symbols_own_relation``: doğrulayıcının ilişkisi, başka
#      alanın sembolleriyle. (İlişkiye bakıp karar veren burada kirlenir.)
#   3. ``shared_surface``: iki alanda da geçerli GÖRÜNEN yüzey; yalnız
#      ikisinden birinin yetkisi vardır.
#
# Bu sondalarda DOĞRU davranış her zaman ABSTAIN'dir. Bir doğrulayıcının
# burada ACCEPT/REJECT demesi, alan bilgisi yerine yüzeysel bir ipucu
# kullandığını kanıtlar.

def build_adversarial_probes(environments: Sequence[str]) -> List[Dict[str, Any]]:
    """Her doğrulayıcı için yetki sınırını zorlayan sondalar üret."""
    ortamlar = list(environments)
    ornek_sembol = {
        "physics": ("PHYS_H80", "PHYS_T4"),
        "causal": ("CAUS_yagmur", "CAUS_islak_zemin"),
        "spatial": ("GEO_ankara", "GEO_izmir"),
        "temporal": ("EVT_matbaa", "EVT_amerika"),
        "language": ("LEX_kitap", "LEX_kitaplar"),
    }
    ortam_iliskisi = {
        "physics": _PHYSICS_REL, "causal": _CAUSAL_REL,
        "spatial": _SPATIAL_REL, "temporal": _TEMPORAL_REL,
        "language": _LANGUAGE_REL,
    }
    sondalar: List[Dict[str, Any]] = []
    for sahip in ortamlar:
        ozne, nesne = ornek_sembol[sahip]
        for yabanci in ortamlar:
            if yabanci == sahip:
                continue
            # 1) Kendi sembolleri + yabancı ilişki
            sondalar.append({
                "kind": "own_symbols_foreign_relation",
                "verifier": sahip,
                "claim": Claim(yabanci, ozne, ortam_iliskisi[yabanci],
                               nesne, False),
            })
            # 2) Yabancı semboller + kendi ilişkisi
            y_ozne, y_nesne = ornek_sembol[yabanci]
            sondalar.append({
                "kind": "foreign_symbols_own_relation",
                "verifier": sahip,
                "claim": Claim(yabanci, y_ozne, ortam_iliskisi[sahip],
                               y_nesne, False),
            })
            # 3) Karışık yüzey: bir öznenin kendi alanından, nesnenin
            #    yabancı alandan geldiği melez iddia.
            sondalar.append({
                "kind": "shared_surface",
                "verifier": sahip,
                "claim": Claim(yabanci, ozne, ortam_iliskisi[sahip],
                               y_nesne, False),
            })
    return sondalar


def _run_adversarial(environments: Sequence[str]) -> Dict[str, Any]:
    """Çekişmeli sondaları koş; DOĞRU cevap her zaman ABSTAIN'dir."""
    sondalar = build_adversarial_probes(environments)
    tur_bazinda: Dict[str, Dict[str, int]] = {}
    ihlaller: List[Dict[str, Any]] = []
    for sonda in sondalar:
        karar = VERIFIERS[sonda["verifier"]](sonda["claim"])
        tur = sonda["kind"]
        satir = tur_bazinda.setdefault(tur, {"probes": 0, "abstain": 0,
                                             "spoke": 0})
        satir["probes"] += 1
        if karar == ABSTAIN:
            satir["abstain"] += 1
        else:
            satir["spoke"] += 1
            if len(ihlaller) < 20:
                ihlaller.append({
                    "kind": tur,
                    "verifier": sonda["verifier"],
                    "claim": sonda["claim"].to_dict(),
                    "verdict": karar,
                })
    toplam = sum(v["probes"] for v in tur_bazinda.values())
    konusan = sum(v["spoke"] for v in tur_bazinda.values())
    for satir in tur_bazinda.values():
        satir["abstain_rate"] = round(  # type: ignore[assignment]
            satir["abstain"] / satir["probes"], 6) if satir["probes"] else 0.0
    return {
        "probes": toplam,
        "spoke": konusan,
        "abstain_rate": round((toplam - konusan) / toplam, 6) if toplam else 1.0,
        "by_kind": tur_bazinda,
        "violations": ihlaller,
        "note": ("Bu sondaların TAMAMINDA doğru davranış ABSTAIN'dir. "
                 "Konuşan bir doğrulayıcı, alan bilgisi yerine yüzeysel "
                 "bir ipucu (yalnız sembol ön eki ya da yalnız ilişki adı) "
                 "kullanıyor demektir."),
    }


# ── Rapor ───────────────────────────────────────────────────────────────────
@dataclass
class MultiEnvironmentReport:
    """Çoklu ortam ve çapraz doğrulayıcı izolasyon raporu."""

    protocol: str
    schema_version: int
    seeds: List[int]
    environments: List[str]
    claims_per_environment: int
    own_domain: Dict[str, Dict[str, Any]]
    cross_matrix: Dict[str, Dict[str, Dict[str, Any]]]
    contamination: Dict[str, Any]
    adversarial: Dict[str, Any]
    symbol_overlap: Dict[str, Any]
    checks: Dict[str, bool]
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_multi_environment_benchmark(
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    claims_per_environment: int = 200,
    environments: Sequence[str] = ENVIRONMENTS,
) -> MultiEnvironmentReport:
    """Beş ortamda üretim + 5×5 çapraz doğrulama matrisi.

    Args:
        seeds: Tohumlar. Her tohum bağımsız iddia kümesi üretir.
        claims_per_environment: Ortam ve tohum başına iddia sayısı.
        environments: Kullanılacak ortam alt kümesi.

    Raises:
        ValueError: boş/geçersiz girdi ya da bilinmeyen ortam.
    """
    tohumlar = [int(s) for s in seeds]
    if not tohumlar:
        raise ValueError("en az bir tohum gerekir")
    if int(claims_per_environment) < 1:
        raise ValueError("claims_per_environment >= 1 olmalı")
    bilinmeyen = [o for o in environments if o not in ENVIRONMENTS]
    if bilinmeyen:
        raise ValueError(f"bilinmeyen ortam(lar): {bilinmeyen}")
    ortamlar = list(environments)
    if not ortamlar:
        raise ValueError("en az bir ortam gerekir")

    # ── iddia üretimi ──────────────────────────────────────────────────────
    iddialar: Dict[str, List[Claim]] = {o: [] for o in ortamlar}
    for tohum in tohumlar:
        for ortam in ortamlar:
            rng = random.Random(tohum * 1000 + hash(ortam) % 997)
            iddialar[ortam].extend(
                GENERATORS[ortam](rng, int(claims_per_environment)))

    # ── kendi alanı: doğruluk ──────────────────────────────────────────────
    kendi: Dict[str, Dict[str, Any]] = {}
    for ortam in ortamlar:
        dogrulayici = VERIFIERS[ortam]
        tp = tn = fp = fn = cekimser = 0
        for iddia in iddialar[ortam]:
            karar = dogrulayici(iddia)
            if karar == ABSTAIN:
                cekimser += 1
            elif karar == ACCEPT:
                if iddia.truth:
                    tp += 1
                else:
                    fp += 1
            else:
                if iddia.truth:
                    fn += 1
                else:
                    tn += 1
        toplam = len(iddialar[ortam])
        karar_verilen = tp + tn + fp + fn
        kendi[ortam] = {
            "claims": toplam,
            "true_positive": tp, "true_negative": tn,
            "false_positive": fp, "false_negative": fn,
            "abstain": cekimser,
            "accuracy": round((tp + tn) / karar_verilen, 6)
            if karar_verilen else 0.0,
            "coverage": round(karar_verilen / toplam, 6) if toplam else 0.0,
            "false_acceptance_rate": round(fp / max(1, fp + tn), 6),
        }

    # ── çapraz matris ──────────────────────────────────────────────────────
    capraz: Dict[str, Dict[str, Dict[str, Any]]] = {}
    kontaminasyon_ornekleri: List[Dict[str, Any]] = []
    toplam_capraz = 0
    toplam_kontaminasyon = 0

    for kaynak in ortamlar:
        capraz[kaynak] = {}
        for dogrulayici_adi in ortamlar:
            dogrulayici = VERIFIERS[dogrulayici_adi]
            kabul = ret = cekimser = 0
            for iddia in iddialar[kaynak]:
                karar = dogrulayici(iddia)
                if karar == ACCEPT:
                    kabul += 1
                elif karar == REJECT:
                    ret += 1
                else:
                    cekimser += 1
            toplam = len(iddialar[kaynak])
            konusma = kabul + ret
            satir = {
                "claims": toplam,
                "accept": kabul, "reject": ret, "abstain": cekimser,
                "abstain_rate": round(cekimser / toplam, 6) if toplam else 0.0,
                "spoke_rate": round(konusma / toplam, 6) if toplam else 0.0,
                "is_own_domain": kaynak == dogrulayici_adi,
            }
            if kaynak != dogrulayici_adi:
                toplam_capraz += toplam
                toplam_kontaminasyon += konusma
                satir["contaminated"] = konusma > 0
                if konusma > 0 and len(kontaminasyon_ornekleri) < 20:
                    ornek = next(
                        i for i in iddialar[kaynak]
                        if VERIFIERS[dogrulayici_adi](i) != ABSTAIN)
                    kontaminasyon_ornekleri.append({
                        "claim_environment": kaynak,
                        "verifier": dogrulayici_adi,
                        "claim": ornek.to_dict(),
                        "verdict": VERIFIERS[dogrulayici_adi](ornek),
                    })
            capraz[kaynak][dogrulayici_adi] = satir

    kontaminasyon = {
        "cross_domain_claims": toplam_capraz,
        "cross_domain_decisions": toplam_kontaminasyon,
        "contamination_rate": round(
            toplam_kontaminasyon / toplam_capraz, 8) if toplam_capraz else 0.0,
        "isolation_rate": round(
            1 - toplam_kontaminasyon / toplam_capraz, 8)
        if toplam_capraz else 1.0,
        "contaminated_pairs": sorted(
            f"{k}→{d}" for k in ortamlar for d in ortamlar
            if k != d and capraz[k][d].get("contaminated")),
        "examples": kontaminasyon_ornekleri,
    }

    # ── çekişmeli sondalar ────────────────────────────────────────────────
    cekismeli = _run_adversarial(ortamlar)

    # ── sembol örtüşmesi ───────────────────────────────────────────────────
    # İzolasyon, ortamlar ortak sembol kullanmadığı için "bedava" gelmiş
    # olabilir. Bu bölüm o şüpheyi açıkça ele alır.
    semboller: Dict[str, set] = {}
    iliskiler: Dict[str, set] = {}
    for ortam in ortamlar:
        s: set = set()
        r: set = set()
        for iddia in iddialar[ortam]:
            s.update((iddia.subject, iddia.object))
            r.add(iddia.relation)
        semboller[ortam] = s
        iliskiler[ortam] = r
    ortusen_ciftler = []
    for i, a in enumerate(ortamlar):
        for b in ortamlar[i + 1:]:
            ortak_s = semboller[a] & semboller[b]
            ortak_r = iliskiler[a] & iliskiler[b]
            if ortak_s or ortak_r:
                ortusen_ciftler.append({
                    "pair": f"{a}|{b}",
                    "shared_symbols": sorted(ortak_s)[:10],
                    "shared_relations": sorted(ortak_r),
                })
    sembol_ortusmesi = {
        "symbols_per_environment": {o: len(s) for o, s in semboller.items()},
        "relations_per_environment": {o: sorted(r)
                                      for o, r in iliskiler.items()},
        "overlapping_pairs": ortusen_ciftler,
        "any_overlap": bool(ortusen_ciftler),
        "note": ("Ortamlar ayrık sembol uzayları kullanır. Bu, izolasyon "
                 "testini KOLAYLAŞTIRIR ve sonuç bu sınırla okunmalıdır: "
                 "ölçülen şey 'alan bilgisi' değil, 'yetki sınırına saygı'dır."),
    }

    # ── kapılar ────────────────────────────────────────────────────────────
    kapilar: Dict[str, bool] = {
        "all_environments_generated_claims": all(
            len(iddialar[o]) > 0 for o in ortamlar),
        "every_verifier_accurate_on_own_domain": all(
            kendi[o]["accuracy"] >= 0.99 for o in ortamlar),
        "every_verifier_full_coverage_on_own_domain": all(
            kendi[o]["coverage"] >= 0.99 for o in ortamlar),
        "no_cross_domain_contamination": toplam_kontaminasyon == 0,
        "every_environment_has_both_labels": all(
            any(i.truth for i in iddialar[o])
            and any(not i.truth for i in iddialar[o]) for o in ortamlar),
        "no_symbol_overlap_between_environments": not ortusen_ciftler,
        "verifiers_abstain_on_adversarial_probes": cekismeli["spoke"] == 0,
        "multi_seed": len(tohumlar) >= 2,
    }

    # ── bulgular ───────────────────────────────────────────────────────────
    bulgular: List[str] = [
        f"{len(ortamlar)} ortam × {len(tohumlar)} tohum × "
        f"{claims_per_environment} iddia = ortam başına "
        f"{len(iddialar[ortamlar[0]])} iddia.",
    ]
    dogruluk_ozet = ", ".join(
        f"{o} {kendi[o]['accuracy']:.4f}" for o in ortamlar)
    bulgular.append(f"Kendi alanında doğruluk: {dogruluk_ozet}.")

    if toplam_kontaminasyon == 0:
        bulgular.append(
            f"Çapraz doğrulayıcı kontaminasyonu YOK: {toplam_capraz:,} "
            "alan-dışı iddianın tamamında doğrulayıcılar ÇEKİMSER kaldı. "
            "Yani 'doğrulandı' etiketi alan bilgisine dayanıyor, "
            "yüzeysel bir örüntüye değil.")
    else:
        bulgular.append(
            f"KONTAMİNASYON: {toplam_kontaminasyon:,}/{toplam_capraz:,} "
            f"alan-dışı iddiada doğrulayıcı karar verdi "
            f"(oran {kontaminasyon['contamination_rate']:.6f}). "
            f"Kirlenen çiftler: {kontaminasyon['contaminated_pairs']}.")

    bulgular.append(
        "Bu sonucun sınırı: ortamlar ayrık sembol uzayları kullanıyor, "
        "bu yüzden izolasyon kısmen tasarımdan gelir. Ölçülen şey "
        "doğrulayıcının yetki sınırına saygı göstermesidir — daha zor olan "
        "'aynı sembollerle farklı alan' durumu bu sürümde test EDİLMEMİŞTİR.")

    if cekismeli["spoke"] == 0:
        bulgular.append(
            f"Çekişmeli sondalar: {cekismeli['probes']} tuzağın tamamında "
            "doğrulayıcılar çekimser kaldı. Yetki kontrolü hem sembolü hem "
            "ilişkiyi birlikte gerektiriyor; tek ipucuyla kandırılamıyor.")
    else:
        ihlal_turleri = sorted(
            t for t, v in cekismeli["by_kind"].items() if v["spoke"] > 0)
        bulgular.append(
            f"ÇEKİŞMELİ SONDA İHLALİ: {cekismeli['spoke']}/"
            f"{cekismeli['probes']} tuzakta doğrulayıcı konuştu "
            f"(ihlal türleri: {ihlal_turleri}). Yetki kontrolü yüzeysel bir "
            "ipucuna dayanıyor ve izolasyon iddiası bu ölçüde zayıflar.")

    zayif = [o for o in ortamlar if kendi[o]["accuracy"] < 0.99]
    if zayif:
        bulgular.append(
            f"Kendi alanında %99 altına düşen ortamlar: {zayif}.")

    imza = hashlib.sha256(json.dumps(
        {"protocol": PROTOCOL, "seeds": tohumlar, "environments": ortamlar,
         "n": int(claims_per_environment)},
        sort_keys=True).encode("utf-8")).hexdigest()[:12]
    kontaminasyon["signature"] = imza

    return MultiEnvironmentReport(
        protocol=PROTOCOL,
        schema_version=SCHEMA_VERSION,
        seeds=tohumlar,
        environments=ortamlar,
        claims_per_environment=int(claims_per_environment),
        own_domain=kendi,
        cross_matrix=capraz,
        contamination=kontaminasyon,
        adversarial=cekismeli,
        symbol_overlap=sembol_ortusmesi,
        checks=kapilar,
        findings=bulgular,
        limitations=[
            "Ortamlar sentetiktir ve kapalı-form kurallara dayanır; gerçek "
            "dünya fiziği/nedenselliği DEĞİLDİR.",
            "Doğrulayıcılar sembol ön ekiyle alan tanır. Bu, izolasyonu "
            "kolaylaştıran güçlü bir ipucudur ve gerçek bir sistemde "
            "bulunmayabilir.",
            "Ölçülen şey doğrulayıcı izolasyonudur; ÖĞRENME başarısı ya da "
            "bu ortamlarda model performansı değildir.",
            "Beş ortam da tek adımlı ilişkiler kullanır; çok adımlı "
            "alanlar arası çıkarım kapsam dışıdır.",
        ],
    )


def multi_environment_markdown(report: MultiEnvironmentReport) -> str:
    """Raporu Markdown'a çevir."""
    s = report
    satirlar = [
        "# Çoklu Ortam ve Verifier İzolasyonu (P1)",
        "",
        f"- Protokol: `{s.protocol}` v{s.schema_version} "
        f"(imza `{s.contamination['signature']}`)",
        f"- Ortamlar: `{s.environments}`",
        f"- Tohumlar: `{s.seeds}` × `{s.claims_per_environment}` iddia",
        "",
        "## Kendi alanında doğrulayıcı performansı",
        "",
        "| Ortam | İddia | Doğruluk | Kapsama | FAR | Çekimser |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for ortam in s.environments:
        m = s.own_domain[ortam]
        satirlar.append(
            f"| {ortam} | {m['claims']} | {m['accuracy']:.4f} | "
            f"{m['coverage']:.4f} | {m['false_acceptance_rate']:.4f} | "
            f"{m['abstain']} |")

    satirlar.extend([
        "",
        "## Çapraz doğrulama matrisi — çekimserlik oranı",
        "",
        "Satır = iddianın geldiği ortam, sütun = doğrulayıcı. "
        "Köşegen dışında **1.0000 beklenir** (yetkisiz alanda karar yok).",
        "",
        "| İddia ↓ / Verifier → | " + " | ".join(s.environments) + " |",
        "|---" * (len(s.environments) + 1) + "|",
    ])
    for kaynak in s.environments:
        hucreler = []
        for dv in s.environments:
            oran = s.cross_matrix[kaynak][dv]["abstain_rate"]
            if kaynak == dv:
                hucreler.append(f"_{oran:.4f}_")
            elif oran < 1.0:
                hucreler.append(f"**{oran:.4f}** ⚠")
            else:
                hucreler.append(f"{oran:.4f}")
        satirlar.append(f"| **{kaynak}** | " + " | ".join(hucreler) + " |")

    k = s.contamination
    satirlar.extend([
        "",
        "## Kontaminasyon",
        "",
        f"- Alan-dışı iddia: `{k['cross_domain_claims']:,}`",
        f"- Alan-dışı karar (kontaminasyon): `{k['cross_domain_decisions']:,}`",
        f"- **İzolasyon oranı: {k['isolation_rate']:.6f}**",
        f"- Kirlenen çiftler: `{k['contaminated_pairs'] or 'yok'}`",
        "",
        "## Çekişmeli sondalar (zor sınav)",
        "",
        f"- Sonda: `{s.adversarial['probes']}`, konuşan: "
        f"`{s.adversarial['spoke']}`, çekimserlik: "
        f"**{s.adversarial['abstain_rate']:.6f}**",
        "",
        "| Tuzak türü | Sonda | Çekimser | Oran |",
        "|---|---:|---:|---:|",
    ] + [
        f"| {tur} | {v['probes']} | {v['abstain']} | {v['abstain_rate']:.4f} |"
        for tur, v in sorted(s.adversarial["by_kind"].items())
    ] + [
        "",
        f"> {s.adversarial['note']}",
        "",
        "## Sembol örtüşmesi",
        "",
        f"- Ortamlar arası örtüşen sembol/ilişki: "
        f"`{'VAR' if s.symbol_overlap['any_overlap'] else 'yok'}`",
        "",
        f"> {s.symbol_overlap['note']}",
        "",
        "## Kabul kapıları",
        "",
        "| Kapı | Sonuç |",
        "|---|---|",
    ])
    satirlar.extend(f"| {ad} | {'GEÇTİ' if v else 'KALDI'} |"
                    for ad, v in s.checks.items())
    satirlar.extend(["", "## Bulgular", ""])
    satirlar.extend(f"- {b}" for b in s.findings)
    satirlar.extend(["", "## Sınırlar", ""])
    satirlar.extend(f"- {b}" for b in s.limitations)
    return "\n".join(satirlar) + "\n"


__all__ = [
    "ABSTAIN",
    "ACCEPT",
    "ENVIRONMENTS",
    "GENERATORS",
    "PROTOCOL",
    "REJECT",
    "SCHEMA_VERSION",
    "VERIFIERS",
    "Claim",
    "MultiEnvironmentReport",
    "build_adversarial_probes",
    "multi_environment_markdown",
    "run_multi_environment_benchmark",
]
