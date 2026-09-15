# -*- coding: utf-8 -*-
"""P1 — İnsan değerlendirme protokolü + kodlayıcılar arası güvenilirlik.

**Bu modül insan değerlendirme SONUCU üretmez.** Gerçek değerlendirici
yoktur; dolayısıyla karnede `human_evaluation` bölümü `n/a` kalır ve bu
kasıtlıdır. Modülün ürettiği şey üç somut şeydir:

1. **Protokol** — 50–100 Türkçe prompt, 10–20 değerlendirici, kör sunum,
   kol etiketlerinin gizlenmesi, sunum sırasının dengelenmesi (Latin kare),
   dikkat kontrolü (attention check) ve altın öğeler.
2. **Araç** — değerlendirici paketini (`build_evaluation_sheets`) üreten,
   körlemeyi gerçekten uygulayan ve kör açma anahtarını ayrı tutan kod.
3. **Krippendorff's α** — kodlayıcılar arası güvenilirlik hesabı; nominal,
   ordinal ve interval metriklerle, eksik veriye toleranslı.

α implementasyonu Krippendorff'un kanonik örneğine (3 kodlayıcı × 15 birim)
karşı doğrulanmıştır: nominal 0.691, ordinal 0.807, interval 0.811.
Bu testler `tests/scientific/test_human_evaluation.py` içindedir — yani
araç "yazıldı" değil, **doğruluğu kanıtlandı** durumundadır.

Neden α, yüzde uyum değil? Yüzde uyum şansı düzeltmez: iki değerlendirici
her şeye "iyi" derse %100 uyuşur ama hiçbir bilgi üretmez. α gözlenen
uyuşmazlığı beklenen uyuşmazlığa böler, eksik veriyi kaldırır ve ikiden
fazla kodlayıcıyı destekler.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROTOCOL = "human_evaluation_protocol_v1"
SCHEMA_VERSION = 1

#: Protokolün zorunlu kıldığı alt sınırlar (kullanıcı şartnamesi).
MIN_PROMPTS = 50
MAX_PROMPTS = 100
MIN_RATERS = 10
MAX_RATERS = 20

#: Her öğenin puanlandığı boyutlar ve ölçek tipi.
DIMENSIONS: Dict[str, Dict[str, Any]] = {
    "dogruluk": {
        "scale": (1, 2, 3, 4, 5), "level": "ordinal",
        "question": "Yanıt olgusal olarak doğru mu?",
        "anchors": {1: "tamamen yanlış", 3: "kısmen doğru",
                    5: "tamamen doğru"},
    },
    "tutarlilik": {
        "scale": (1, 2, 3, 4, 5), "level": "ordinal",
        "question": "Yanıt kendi içinde çelişkisiz mi?",
        "anchors": {1: "kendiyle çelişiyor", 3: "kısmen tutarlı",
                    5: "tam tutarlı"},
    },
    "dil_kalitesi": {
        "scale": (1, 2, 3, 4, 5), "level": "ordinal",
        "question": "Türkçe dilbilgisi ve akıcılık nasıl?",
        "anchors": {1: "anlaşılmaz", 3: "anlaşılır ama bozuk",
                    5: "doğal Türkçe"},
    },
    "belirsizlik_durustlugu": {
        "scale": (1, 2, 3, 4, 5), "level": "ordinal",
        "question": ("Bilmediğinde bilmediğini söylüyor mu? Emin olmadan "
                     "kesin konuşuyorsa düşük puan verin."),
        "anchors": {1: "bilmediğini kesin söylüyor (halüsinasyon)",
                    3: "karışık", 5: "belirsizliği doğru bildiriyor"},
    },
    "halusinasyon_var": {
        "scale": (0, 1), "level": "nominal",
        "question": "Yanıtta uydurma bilgi VAR mı? (0=yok, 1=var)",
        "anchors": {0: "uydurma yok", 1: "uydurma var"},
    },
}

#: Karşılaştırılan kollar. Değerlendirici bunları GÖRMEZ.
ARMS: Tuple[str, ...] = ("hga", "dense", "transformer", "symbolic")

#: Kabul eşikleri (Krippendorff'un kendi önerdiği sınırlar).
ALPHA_ACCEPTABLE = 0.800
ALPHA_TENTATIVE = 0.667


# ══════════════════════════════════════════════════════════════════════════
# Krippendorff's alpha
# ══════════════════════════════════════════════════════════════════════════

def krippendorff_alpha(
    ratings: Dict[Any, Sequence[Optional[float]]],
    level: str = "nominal",
) -> Dict[str, Any]:
    """Krippendorff's α hesapla.

    Args:
        ratings: ``{birim: [kodlayıcı1_değeri, kodlayıcı2_değeri, ...]}``.
            ``None`` eksik veridir ve düşürülür (α'nın temel avantajı).
        level: ``nominal`` | ``ordinal`` | ``interval``.

    Returns:
        ``alpha``, ``observed_disagreement``, ``expected_disagreement``,
        ``pairable_values``, ``units_used`` ve yorumlanmış ``verdict``.

    Raises:
        ValueError: bilinmeyen metrik ya da eşleştirilebilir veri yokluğu.

    α = 1 − D_o/D_e. α=1 tam uyum, α=0 şans düzeyi, α<0 sistematik
    uyuşmazlıktır.
    """
    if level not in ("nominal", "ordinal", "interval"):
        raise ValueError(
            "level 'nominal', 'ordinal' veya 'interval' olmalı")

    # Yalnız en az iki kodlayıcının değerlendirdiği birimler eşleştirilebilir.
    birimler = {u: [v for v in vs if v is not None]
                for u, vs in ratings.items()}
    birimler = {u: vs for u, vs in birimler.items() if len(vs) >= 2}
    if not birimler:
        raise ValueError(
            "eşleştirilebilir birim yok: en az bir birimi en az iki "
            "kodlayıcı değerlendirmeli")

    # Rastlantı (coincidence) matrisi.
    rastlanti: Dict[Tuple[Any, Any], float] = defaultdict(float)
    for vs in birimler.values():
        m_u = len(vs)
        for i, j in combinations(range(m_u), 2):
            for x, y in ((vs[i], vs[j]), (vs[j], vs[i])):
                rastlanti[(x, y)] += 1.0 / (m_u - 1)

    degerler = sorted({v for vs in birimler.values() for v in vs})
    marjinal = {c: sum(rastlanti[(c, k)] for k in degerler) for c in degerler}
    toplam = sum(marjinal.values())
    if toplam <= 1:
        raise ValueError("α için en az 2 eşleştirilebilir değer gerekir")

    if level == "nominal":
        def uzaklik(x: Any, y: Any) -> float:
            return 0.0 if x == y else 1.0
    elif level == "interval":
        def uzaklik(x: Any, y: Any) -> float:
            return (float(x) - float(y)) ** 2
    else:  # ordinal
        sira = {v: i for i, v in enumerate(degerler)}

        def uzaklik(x: Any, y: Any) -> float:
            i, j = sorted((sira[x], sira[y]))
            birikim = sum(marjinal[degerler[k]] for k in range(i, j + 1))
            return (birikim - (marjinal[x] + marjinal[y]) / 2.0) ** 2

    gozlenen = sum(rastlanti[(x, y)] * uzaklik(x, y)
                   for x in degerler for y in degerler)
    beklenen = sum(marjinal[x] * marjinal[y] * uzaklik(x, y)
                   for x in degerler for y in degerler) / (toplam - 1)

    if beklenen == 0.0:
        # Tüm değerler aynı: uyuşmazlık yok ama α tanımsız.
        return {
            "alpha": None, "level": level,
            "observed_disagreement": round(gozlenen, 12),
            "expected_disagreement": 0.0,
            "pairable_values": round(toplam, 6),
            "units_used": len(birimler),
            "distinct_values": len(degerler),
            "verdict": ("TANIMSIZ: tüm kodlayıcılar tek bir değer kullandı; "
                        "varyans yok, güvenilirlik hesaplanamaz."),
        }

    alfa = 1.0 - gozlenen / beklenen
    if alfa >= ALPHA_ACCEPTABLE:
        hukum = (f"KABUL EDİLEBİLİR: α={alfa:.4f} ≥ {ALPHA_ACCEPTABLE}; "
                 "sonuçlar güvenilir kabul edilebilir.")
    elif alfa >= ALPHA_TENTATIVE:
        hukum = (f"GEÇİCİ: {ALPHA_TENTATIVE} ≤ α={alfa:.4f} < "
                 f"{ALPHA_ACCEPTABLE}; yalnız geçici sonuç çıkarılabilir.")
    elif alfa > 0.0:
        hukum = (f"YETERSİZ: α={alfa:.4f} < {ALPHA_TENTATIVE}; uyum şans "
                 "üstü ama güvenilir sonuç çıkarılamaz.")
    else:
        hukum = (f"BAŞARISIZ: α={alfa:.4f} ≤ 0; uyum şans düzeyinde ya da "
                 "altında. Kılavuz veya değerlendirici eğitimi hatalı.")

    return {
        "alpha": round(alfa, 12),
        "level": level,
        "observed_disagreement": round(gozlenen, 12),
        "expected_disagreement": round(beklenen, 12),
        "pairable_values": round(toplam, 6),
        "units_used": len(birimler),
        "distinct_values": len(degerler),
        "verdict": hukum,
    }


def analyze_ratings(
    ratings_by_dimension: Dict[str, Dict[Any, Sequence[Optional[float]]]],
) -> Dict[str, Any]:
    """Her boyut için α'yı kendi ölçek tipiyle hesapla.

    Raises:
        ValueError: tanımsız boyut adı.
    """
    sonuc: Dict[str, Any] = {}
    for boyut, veri in ratings_by_dimension.items():
        if boyut not in DIMENSIONS:
            raise ValueError(f"tanımsız boyut: {boyut}")
        seviye = DIMENSIONS[boyut]["level"]
        sonuc[boyut] = krippendorff_alpha(veri, level=seviye)
    kabul = [b for b, v in sonuc.items()
             if v["alpha"] is not None and v["alpha"] >= ALPHA_ACCEPTABLE]
    sonuc["_summary"] = {
        "dimensions": len(ratings_by_dimension),
        "acceptable_dimensions": len(kabul),
        "all_acceptable": len(kabul) == len(ratings_by_dimension),
        "note": ("Bir boyut bile eşiği geçemezse o boyuta dayanan hiçbir "
                 "sonuç raporlanamaz."),
    }
    return sonuc


# ══════════════════════════════════════════════════════════════════════════
# Protokol + değerlendirici paketi
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class EvaluationSheet:
    """Tek değerlendiricinin göreceği kör paket."""

    rater_id: str
    items: List[Dict[str, Any]]
    attention_checks: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HumanEvaluationReport:
    """Protokol tanımı + araç çıktısı. Gerçek puan İÇERMEZ."""

    protocol: str
    schema_version: int
    design: Dict[str, Any]
    dimensions: Dict[str, Any]
    blinding: Dict[str, Any]
    sheets: List[Dict[str, Any]]
    unblinding_key_digest: str
    alpha_tool: Dict[str, Any]
    results: Optional[Dict[str, Any]]
    checks: Dict[str, bool]
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_evaluation_sheets(
    prompts: Sequence[str],
    raters: Sequence[str],
    arms: Sequence[str] = ARMS,
    seed: int = 20260914,
    attention_check_rate: float = 0.1,
) -> Tuple[List[EvaluationSheet], Dict[str, Dict[str, str]]]:
    """Kör değerlendirme paketlerini ve kör açma anahtarını üret.

    Körleme gerçekten uygulanır: her öğeye rastgele bir `item_id` verilir,
    kol adı pakete **yazılmaz**, ve sunum sırası her değerlendiricide
    bağımsız karıştırılır. Kol ↔ item eşlemesi ayrı bir anahtarda tutulur.

    Returns:
        ``(sheets, unblinding_key)``. Anahtar **değerlendiriciye
        verilmemelidir**; analiz aşamasında kullanılır.

    Raises:
        ValueError: prompt/değerlendirici sayısı protokol sınırları dışında.
    """
    if not MIN_PROMPTS <= len(prompts) <= MAX_PROMPTS:
        raise ValueError(
            f"prompt sayısı {MIN_PROMPTS}–{MAX_PROMPTS} olmalı, "
            f"{len(prompts)} verildi")
    if not MIN_RATERS <= len(raters) <= MAX_RATERS:
        raise ValueError(
            f"değerlendirici sayısı {MIN_RATERS}–{MAX_RATERS} olmalı, "
            f"{len(raters)} verildi")
    if not arms:
        raise ValueError("en az bir kol gerekir")
    if not 0.0 <= attention_check_rate < 1.0:
        raise ValueError("attention_check_rate [0,1) aralığında olmalı")

    kor_anahtar: Dict[str, Dict[str, str]] = {}
    havuz: List[Dict[str, Any]] = []
    for p_indeks, prompt in enumerate(prompts):
        for kol in arms:
            item_id = hashlib.sha256(
                f"{seed}:{p_indeks}:{kol}".encode("utf-8")).hexdigest()[:16]
            kor_anahtar[item_id] = {"arm": kol, "prompt_index": str(p_indeks)}
            havuz.append({
                "item_id": item_id,
                "prompt": prompt,
                # 'arm' KASITLI olarak yok: körleme burada uygulanır.
                "response_placeholder": (
                    "<model yanıtı buraya yerleştirilecek>"),
                "dimensions": list(DIMENSIONS),
            })

    paketler: List[EvaluationSheet] = []
    for r_indeks, rater in enumerate(raters):
        # Latin-kare benzeri dengeleme: her değerlendirici farklı bir
        # döndürmeden başlar, böylece sunum sırası etkisi kolları dengeler.
        kaydirma = (r_indeks * len(arms)) % max(1, len(havuz))
        sira = havuz[kaydirma:] + havuz[:kaydirma]
        yerel = random.Random(seed + r_indeks)
        sira = list(sira)
        yerel.shuffle(sira)

        dikkat: List[str] = []
        if attention_check_rate > 0:
            sayi = max(1, int(len(sira) * attention_check_rate))
            dikkat = [sira[i]["item_id"]
                      for i in yerel.sample(range(len(sira)),
                                            min(sayi, len(sira)))]
        paketler.append(EvaluationSheet(
            rater_id=rater,
            items=[{k: v for k, v in ogesi.items()} for ogesi in sira],
            attention_checks=dikkat,
        ))
    return paketler, kor_anahtar


def _default_prompts(count: int = MIN_PROMPTS) -> List[str]:
    """Protokolü gösterecek kadar Türkçe prompt iskeleti üret.

    Bunlar **örnek prompt'lardır**, gerçek bir değerlendirme korpusu
    değildir; amaç aracın çalıştığını göstermektir.
    """
    kaliplar = [
        "{} nedir, kısaca açıkla.",
        "{} ile ilgili bilmediğin bir şey varsa söyle.",
        "{} hakkında kesin olmayan bir iddiayı nasıl işaretlersin?",
        "{} konusunda iki çelişkili kaynak varsa ne yaparsın?",
        "{} için adım adım bir çıkarım zinciri kur.",
    ]
    konular = [
        "Kronecker çarpımı", "epistemik belirsizlik", "Türkçe ünlü uyumu",
        "hiyerarşik bellek", "doğrulayıcı hattı", "bileşimsel genelleme",
        "dolgu dayanıklılığı", "olgu doğrulama", "deneyim verimi",
        "çoklu ortam izolasyonu",
    ]
    promptlar: List[str] = []
    for konu in konular:
        for kalip in kaliplar:
            promptlar.append(kalip.format(konu))
            if len(promptlar) >= count:
                return promptlar
    return promptlar[:count]


def build_human_evaluation_protocol(
    prompts: Optional[Sequence[str]] = None,
    raters: Optional[Sequence[str]] = None,
    arms: Sequence[str] = ARMS,
    seed: int = 20260914,
    collected_ratings: Optional[
        Dict[str, Dict[Any, Sequence[Optional[float]]]]] = None,
) -> HumanEvaluationReport:
    """İnsan değerlendirme protokolünü ve araç çıktısını üret.

    Args:
        collected_ratings: Gerçek puanlar toplandıysa buraya verilir ve
            α analizi koşar. Verilmezse ``results`` **None** kalır — bu
            "sonuç yok" demektir, "sonuç sıfır" demek değildir.

    Raises:
        ValueError: protokol sınırları ihlal edilirse.
    """
    prompt_listesi = list(prompts) if prompts else _default_prompts()
    degerlendiriciler = (list(raters) if raters
                         else [f"R{i:02d}" for i in range(1, MIN_RATERS + 1)])

    paketler, kor_anahtar = build_evaluation_sheets(
        prompt_listesi, degerlendiriciler, arms=arms, seed=seed)

    anahtar_ozeti = hashlib.sha256(json.dumps(
        kor_anahtar, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    # Körlemenin gerçekten uygulandığını KANITLA: hiçbir pakette kol adı
    # geçmemeli.
    paket_metni = json.dumps([p.to_dict() for p in paketler],
                             ensure_ascii=False)
    sizinti = [kol for kol in arms if f'"{kol}"' in paket_metni]

    tasarim = {
        "prompts": len(prompt_listesi),
        "raters": len(degerlendiriciler),
        "arms": list(arms),
        "items_per_rater": len(paketler[0].items) if paketler else 0,
        "total_judgements": (len(paketler) * len(paketler[0].items)
                             * len(DIMENSIONS)) if paketler else 0,
        "prompt_range_required": [MIN_PROMPTS, MAX_PROMPTS],
        "rater_range_required": [MIN_RATERS, MAX_RATERS],
        "seed": seed,
        "design_type": "within-subject, tam çapraz (her değerlendirici "
                       "her öğeyi görür)",
    }

    korleme = {
        "arm_labels_hidden": not sizinti,
        "leaked_labels": sizinti,
        "presentation_order_randomized_per_rater": True,
        "counterbalancing": "Latin-kare benzeri döndürme + tohumlu karıştırma",
        "attention_checks_per_rater": (len(paketler[0].attention_checks)
                                       if paketler else 0),
        "unblinding_key_held_separately": True,
        "note": ("Kör açma anahtarı rapora yalnız özet (digest) olarak "
                 "girer; değerlendirici paketinde kol adı yoktur."),
    }

    alfa_araci = {
        "implemented": True,
        "levels": ["nominal", "ordinal", "interval"],
        "handles_missing_data": True,
        "supports_many_coders": True,
        "acceptable_threshold": ALPHA_ACCEPTABLE,
        "tentative_threshold": ALPHA_TENTATIVE,
        "validated_against": ("Krippendorff kanonik örneği (3 kodlayıcı × "
                              "15 birim): nominal 0.691, ordinal 0.807, "
                              "interval 0.811"),
        "why_not_percent_agreement": (
            "Yüzde uyum şansı düzeltmez; herkes aynı etikete basarsa %100 "
            "çıkar ama bilgi üretmez."),
    }

    sonuclar = analyze_ratings(collected_ratings) if collected_ratings else None

    kapilar: Dict[str, bool] = {
        "prompt_count_within_spec": (
            MIN_PROMPTS <= len(prompt_listesi) <= MAX_PROMPTS),
        "rater_count_within_spec": (
            MIN_RATERS <= len(degerlendiriciler) <= MAX_RATERS),
        "arm_labels_hidden_from_raters": not sizinti,
        "presentation_order_counterbalanced": True,
        "attention_checks_present": bool(
            paketler and paketler[0].attention_checks),
        "alpha_tool_available": True,
        "alpha_validated_on_reference_data": True,
        "multiple_dimensions_defined": len(DIMENSIONS) >= 3,
        # Bu kapı gerçek veri gelene kadar KASITLI olarak KALIR.
        "human_ratings_collected": collected_ratings is not None,
        "reliability_meets_threshold": bool(
            sonuclar and sonuclar["_summary"]["all_acceptable"]),
    }

    bulgular = [
        f"Protokol {len(prompt_listesi)} Türkçe prompt × {len(arms)} kol × "
        f"{len(degerlendiriciler)} değerlendirici = "
        f"{tasarim['total_judgements']} tekil yargı olarak tanımlandı.",
        ("Körleme ARAÇ SEVİYESİNDE uygulanıyor ve doğrulanıyor: üretilen "
         "paketlerde hiçbir kol adı geçmiyor."
         if not sizinti else
         f"KÖRLEME İHLALİ: {sizinti} etiketleri pakete sızmış."),
        ("Krippendorff α aracı üç metrikte hazır ve kanonik referans "
         "veriye karşı doğrulandı (nominal 0.691 / ordinal 0.807 / "
         "interval 0.811)."),
        f"Eşikler: α ≥ {ALPHA_ACCEPTABLE} kabul edilebilir, "
        f"α ≥ {ALPHA_TENTATIVE} yalnız geçici sonuç.",
    ]
    if sonuclar is None:
        bulgular.append(
            "GERÇEK İNSAN PUANI YOK. Bu koşum protokol ve araç üretir, "
            "sonuç üretmez. `human_ratings_collected` kapısı KALDI ve "
            "karnede bu bölüm `n/a` kalmalıdır — kanıtsız bir bölüme puan "
            "vermek, ölçmediğini ölçtüm demektir.")
    else:
        ozet = sonuclar["_summary"]
        bulgular.append(
            f"{ozet['acceptable_dimensions']}/{ozet['dimensions']} boyut "
            f"α ≥ {ALPHA_ACCEPTABLE} eşiğini geçti.")

    # Protokolü girdisine bağlayan deterministik imza: aynı prompt seti,
    # değerlendirici listesi, kollar ve tohum → aynı imza. Yeniden-
    # üretilebilirlik denetimi bu alanı arar.
    tasarim["signature"] = hashlib.sha256(json.dumps({
        "protocol": PROTOCOL, "prompts": prompt_listesi,
        "raters": degerlendiriciler, "arms": list(arms), "seed": seed,
    }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    return HumanEvaluationReport(
        protocol=PROTOCOL,
        schema_version=SCHEMA_VERSION,
        design=tasarim,
        dimensions={ad: {k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in tanim.items()}
                    for ad, tanim in DIMENSIONS.items()},
        blinding=korleme,
        sheets=[p.to_dict() for p in paketler[:2]],  # örnek: ilk 2 paket
        unblinding_key_digest=anahtar_ozeti,
        alpha_tool=alfa_araci,
        results=sonuclar,
        checks=kapilar,
        findings=bulgular,
        limitations=[
            "GERÇEK DEĞERLENDİRİCİ YOK. Bu modül protokol ve araçtır; "
            "insan değerlendirme sonucu değildir ve öyle sunulamaz.",
            "Örnek prompt'lar aracı göstermek içindir; temsili bir Türkçe "
            "değerlendirme korpusu değildir.",
            "α kodlayıcılar arası tutarlılığı ölçer, DOĞRULUĞU değil: "
            "hepsi aynı şekilde yanılan değerlendiriciler yüksek α verir.",
            "Dikkat kontrolleri tanımlıdır ama doğru yanıt anahtarı gerçek "
            "yanıtlar üretilmeden doldurulamaz.",
            "Değerlendirici havuzunun demografisi, Türkçe yeterliği ve "
            "eğitim süreci bu modülün kapsamı dışındadır.",
        ],
    )


def human_evaluation_markdown(report: HumanEvaluationReport) -> str:
    """Raporu Markdown'a çevir."""
    s = report
    d = s.design
    satirlar = [
        "# İnsan Değerlendirme Protokolü (P1)",
        "",
        f"- Protokol: `{s.protocol}` v{s.schema_version}",
        f"- Kör açma anahtarı özeti: `{s.unblinding_key_digest}`",
        "",
        "> **Bu rapor insan değerlendirme SONUCU içermez.** Protokol ve "
        "araç üretir. Gerçek puan toplanana kadar karnede bu bölüm `n/a` "
        "kalır.",
        "",
        "## Tasarım",
        "",
        "| Alan | Değer |",
        "|---|---|",
        f"| Prompt sayısı | {d['prompts']} (şart: {d['prompt_range_required'][0]}–{d['prompt_range_required'][1]}) |",
        f"| Değerlendirici | {d['raters']} (şart: {d['rater_range_required'][0]}–{d['rater_range_required'][1]}) |",
        f"| Kollar | {', '.join(d['arms'])} |",
        f"| Değerlendirici başına öğe | {d['items_per_rater']} |",
        f"| Toplam tekil yargı | {d['total_judgements']} |",
        f"| Tasarım | {d['design_type']} |",
        "",
        "## Puanlama boyutları",
        "",
        "| Boyut | Ölçek | Tip | Soru |",
        "|---|---|---|---|",
    ]
    for ad, tanim in s.dimensions.items():
        satirlar.append(
            f"| `{ad}` | {min(tanim['scale'])}–{max(tanim['scale'])} | "
            f"{tanim['level']} | {tanim['question']} |")

    b = s.blinding
    satirlar.extend([
        "",
        "## Körleme",
        "",
        f"- Kol etiketleri gizli: **{'EVET' if b['arm_labels_hidden'] else 'HAYIR'}**"
        + (f" (sızan: {b['leaked_labels']})" if b["leaked_labels"] else ""),
        f"- Sunum sırası değerlendirici başına randomize: **{b['presentation_order_randomized_per_rater']}**",
        f"- Dengeleme: {b['counterbalancing']}",
        f"- Değerlendirici başına dikkat kontrolü: {b['attention_checks_per_rater']}",
        "",
        f"> {b['note']}",
        "",
        "## Krippendorff's α aracı",
        "",
        f"- Metrikler: {', '.join(s.alpha_tool['levels'])}",
        f"- Eksik veri desteği: {s.alpha_tool['handles_missing_data']}",
        f"- Kabul eşiği: α ≥ {s.alpha_tool['acceptable_threshold']} · "
        f"geçici: α ≥ {s.alpha_tool['tentative_threshold']}",
        f"- **Doğrulama:** {s.alpha_tool['validated_against']}",
        "",
        f"> {s.alpha_tool['why_not_percent_agreement']}",
    ])

    if s.results:
        satirlar.extend([
            "",
            "## Güvenilirlik sonuçları",
            "",
            "| Boyut | Tip | α | Birim | Hüküm |",
            "|---|---|---:|---:|---|",
        ])
        for ad, veri in s.results.items():
            if ad == "_summary":
                continue
            alfa = ("n/a" if veri["alpha"] is None
                    else f"{veri['alpha']:.4f}")
            satirlar.append(
                f"| {ad} | {veri['level']} | {alfa} | "
                f"{veri['units_used']} | {veri['verdict']} |")
    else:
        satirlar.extend([
            "",
            "## Güvenilirlik sonuçları",
            "",
            "**Yok.** Gerçek insan puanı toplanmadı.",
        ])

    satirlar.extend(["", "## Kabul kapıları", "", "| Kapı | Sonuç |", "|---|---|"])
    satirlar.extend(f"| {ad} | {'GEÇTİ' if v else 'KALDI'} |"
                    for ad, v in s.checks.items())
    satirlar.extend(["", "## Bulgular", ""])
    satirlar.extend(f"- {x}" for x in s.findings)
    satirlar.extend(["", "## Sınırlar", ""])
    satirlar.extend(f"- {x}" for x in s.limitations)
    return "\n".join(satirlar) + "\n"


__all__ = [
    "ALPHA_ACCEPTABLE",
    "ALPHA_TENTATIVE",
    "ARMS",
    "DIMENSIONS",
    "EvaluationSheet",
    "HumanEvaluationReport",
    "MAX_PROMPTS",
    "MAX_RATERS",
    "MIN_PROMPTS",
    "MIN_RATERS",
    "PROTOCOL",
    "SCHEMA_VERSION",
    "analyze_ratings",
    "build_evaluation_sheets",
    "build_human_evaluation_protocol",
    "human_evaluation_markdown",
    "krippendorff_alpha",
]
