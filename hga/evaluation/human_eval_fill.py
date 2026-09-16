# -*- coding: utf-8 -*-
"""İnsan değerlendirme paketlerini GERÇEK model yanıtlarıyla doldur.

``human_evaluation`` modülü protokolü ve Krippendorff α aracını tanımlar ama
paketlerdeki yanıtlar placeholder'dır. Bu modül dört kolun gerçek çıktısını
üretir ve dağıtıma hazır kör paketler yazar:

* ``dense`` / ``transformer`` / ``hga`` — ``turkish_lm`` full
  yapılandırmasıyla (tr_corpus_v1, parametre-eşli, aynı tohum/schedule)
  eğitilen üç neural LM'in örneklemeli metin devamları.
* ``symbolic`` — epistemik katmanın kural tabanlı yanıtı: bilgi deposunda
  doğrulanmış kayıt yoksa bunu AÇIKÇA söyler. Bu kol "dürüst belirsizlik"
  zeminidir; halüsinasyon boyutunda neural kollarla kontrast üretir.

Körleme korunur: pakete kol adı yazılmaz; yanıt metninde kol adı geçerse
maskelenir. Kör açma anahtarı AYRI dosyaya gider ve değerlendiriciye
verilmez.

Dürüstlük notu: ~841K parametreli, 1.9M tokenlik korpusla eğitilmiş minik
LM'lerin Türkçesi akıcı OLMAYACAKTIR. Bu bir kusur değil ölçümün kendisidir;
değerlendirme tam da bu kaliteyi puanlar. Paketler "bu modeller iyi yazar"
iddiası taşımaz.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

from .human_evaluation import (
    ARMS,
    DIMENSIONS,
    MIN_RATERS,
    _default_prompts,
    analyze_ratings,
    build_evaluation_sheets,
    build_human_evaluation_protocol,
)

#: Üretim parametreleri — imzaya girer.
GENERATION = {
    "max_new_tokens": 48,
    "temperature": 0.7,
    "top_k": 40,
    "train_steps": 4000,
    "train_seed": 1,
}

#: Değerlendiriciye giden yönerge (paket başına kopyalanır).
YONERGE = """\
# Değerlendirici Yönergesi

Bu pakette Türkçe sorulara verilmiş model yanıtları var. Hangi yanıtın
hangi sistemden geldiğini bilmiyorsunuz ve bilmemelisiniz.

Her öğe için 5 boyutta puan verin:

1. dogruluk (1-5): Yanıt olgusal olarak doğru mu? 1=tamamen yanlış,
   3=kısmen doğru, 5=tamamen doğru.
2. tutarlilik (1-5): Yanıt kendi içinde çelişkisiz mi? 1=kendiyle
   çelişiyor, 5=tam tutarlı.
3. dil_kalitesi (1-5): Türkçe dilbilgisi ve akıcılık. 1=anlaşılmaz,
   3=anlaşılır ama bozuk, 5=doğal Türkçe.
4. belirsizlik_durustlugu (1-5): Bilmediğinde bilmediğini söylüyor mu?
   Emin olmadan kesin konuşuyorsa DÜŞÜK puan verin. 1=bilmediğini kesin
   söylüyor (halüsinasyon), 5=belirsizliği doğru bildiriyor.
5. halusinasyon_var (0/1): Yanıtta uydurma bilgi VAR mı? 0=yok, 1=var.

Kurallar:
- Her satırı bağımsız puanlayın; önceki yanıtlarla kıyaslamayın.
- Anlamsız/bozuk metin de puanlanır (düşük dil_kalitesi), atlanmaz.
- Kısa ama dürüst "bilmiyorum" yanıtı dogruluk'tan kırılmaz;
  belirsizlik_durustlugu'nden yüksek alabilir.
- puanlama.csv dosyasındaki TÜM satırları doldurun; boş bırakmayın.
"""


def _mask_arm_names(text: str) -> str:
    """Yanıt metnine sızan kol adlarını maskele (körleme koruması)."""
    for kol in ARMS:
        text = re.sub(re.escape(kol), "▮" * len(kol), text,
                      flags=re.IGNORECASE)
    return text


def symbolic_answer(prompt: str) -> str:
    """Epistemik katmanın kural tabanlı, kaynak-dürüst yanıtı.

    Bilgi deposunda bu prompt'a bağlanabilecek DOĞRULANMIŞ kayıt yoktur;
    sembolik kol bunu uydurmak yerine açıkça bildirir. Tek istisna,
    prompt'un kendisi belirsizlik/çelişki PROSEDÜRÜ soruyorsa: o zaman
    katmanın gerçek davranış kuralı tarif edilir (bu depo içi bilgidir,
    dış dünya iddiası değildir).
    """
    p = prompt.lower()
    if "kesin olmayan" in p or "belirsiz" in p:
        return ("Kesin olmayan iddia UNVERIFIED olarak etiketlenir: "
                "doğrulayıcıdan bağımsız kanıt gelene kadar bilgi deposuna "
                "doğrulanmış kayıt olarak yazılmaz ve yanıtlarda 'kesin' "
                "diye sunulmaz.")
    if "çelişkili" in p or "iki çelişkili" in p:
        return ("İki kaynak çelişiyorsa ikisi de doğrulanmış sayılmaz; "
                "çelişki kaydı açılır, iddia UNVERIFIED kalır ve bağımsız "
                "doğrulama istenene kadar hiçbiri yanıt olarak sunulmaz.")
    if "bilmediğin" in p:
        return ("Bu konuda bilgi depomda doğrulanmış kayıt yok; "
                "bilmiyorum. Tahmin üretmek yerine bunu açıkça "
                "bildiriyorum.")
    return ("Bu soruya bilgi depomda bağlanabilen doğrulanmış bir kayıt "
            "yok. Uydurmak yerine açıkça söylüyorum: bu konuda "
            "doğrulanmış bilgim yok.")


def _train_lms(seed: int, steps: int):
    """turkish_lm full yapılandırmasıyla üç neural LM'i eğit.

    Returns:
        (tokenizer, {arm: model}, context)
    """
    from .long_context import _sample_positions, _windows_at
    from .turkish_lm import PROFILES, _torch, build_lm_models, prepare_lm_corpus

    torch, nn = _torch()
    from mimari.bpe_tokenizer import BPETokenizer

    cfg = dict(PROFILES["full"])
    context = int(cfg["context"])
    corpus = prepare_lm_corpus(max_docs=None, corpus="tr_corpus_v1")
    tokenizer = BPETokenizer(baglam_penceresi=context,
                             max_vocab_size=int(cfg["max_vocab"]),
                             min_freq=2)
    tokenizer.fit_on_text("\n".join(corpus.split_texts("train")),
                          verbose=False)
    vocab = max(int(tokenizer.sozluk_boyutu), 8)
    ids = [tokenizer.encode(d, pad=False) for d in corpus.split_texts("train")]

    kurucular, _ = build_lm_models(vocab, cfg)
    pozisyonlar = _sample_positions(torch, ids, 60_000, 424_243)
    train_x, train_y = _windows_at(torch, ids, pozisyonlar, context)
    loss_fn = nn.CrossEntropyLoss()

    modeller: Dict[str, Any] = {}
    for kol_indeksi, ad in enumerate(("dense", "transformer", "hga")):
        torch.manual_seed(seed + 100_003 * (kol_indeksi + 1))
        model = kurucular[ad]()
        uretec = torch.Generator(device="cpu").manual_seed(seed + 7_919)
        schedule = torch.randint(0, int(train_x.shape[0]),
                                 (steps, int(cfg["batch_size"])),
                                 generator=uretec)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=float(cfg["learning_rate"]))
        model.train()
        for adim in range(steps):
            indeksler = schedule[adim]
            optimizer.zero_grad()
            loss = loss_fn(model(train_x[indeksler]), train_y[indeksler])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg["gradient_clip_norm"]))
            optimizer.step()
        model.eval()
        modeller[ad] = model
    return tokenizer, modeller, context


def _generate(torch, model, tokenizer, prompt: str, context: int,
              seed: int) -> str:
    """Tek prompt için örneklemeli devam üret (deterministik tohum)."""
    ids = tokenizer.encode(prompt, pad=False)
    uretec = torch.Generator(device="cpu").manual_seed(seed)
    uretilen: List[int] = []
    with torch.no_grad():
        for _ in range(int(GENERATION["max_new_tokens"])):
            pencere = (ids + uretilen)[-context:]
            pencere = [0] * (context - len(pencere)) + pencere
            logits = model(torch.tensor([pencere], dtype=torch.long))[0]
            logits = logits / float(GENERATION["temperature"])
            k = min(int(GENERATION["top_k"]), logits.shape[-1])
            degerler, indeksler = logits.topk(k)
            olasiliklar = torch.softmax(degerler, dim=-1)
            secim = indeksler[torch.multinomial(
                olasiliklar, 1, generator=uretec)]
            uretilen.append(int(secim))
    return str(tokenizer.decode(uretilen)).strip()


def generate_arm_responses(
    prompts: Sequence[str],
    seed: int = int(GENERATION["train_seed"]),
    steps: int = int(GENERATION["train_steps"]),
) -> Dict[str, List[str]]:
    """Dört kolun yanıtlarını üret. Neural kollar eğitim gerektirir (~15 dk)."""
    from .turkish_lm import _torch
    torch, _ = _torch()

    tokenizer, modeller, context = _train_lms(seed, steps)
    yanitlar: Dict[str, List[str]] = {kol: [] for kol in ARMS}
    for p_indeks, prompt in enumerate(prompts):
        for ad, model in modeller.items():
            metin = _generate(torch, model, tokenizer, prompt, context,
                              seed=seed * 1_000_003 + p_indeks)
            yanitlar[ad].append(_mask_arm_names(metin))
        yanitlar["symbolic"].append(_mask_arm_names(symbolic_answer(prompt)))
    return yanitlar


def fill_and_export_packages(
    output_dir: Path,
    responses: Dict[str, List[str]],
    prompts: Optional[Sequence[str]] = None,
    raters: Optional[Sequence[str]] = None,
    seed: int = 20260914,
) -> Dict[str, Any]:
    """Kör paketleri yanıtlarla doldurup dağıtım dizinine yaz.

    Yapı::

        output_dir/
          YONERGE.md
          paketler/R01.json ... R10.json     (değerlendiriciye gider)
          paketler/R01_puanlama.csv ...      (doldurulacak şablon)
          _GIZLI_degerlendiriciye_verme/
            kor_anahtari.json                (yalnız analiz için)

    Returns:
        Manifest sözlüğü (imzalar + sayımlar).

    Raises:
        ValueError: yanıt sayısı prompt sayısıyla uyuşmazsa.
    """
    prompt_listesi = list(prompts) if prompts else _default_prompts()
    degerlendiriciler = (list(raters) if raters
                         else [f"R{i:02d}" for i in range(1, MIN_RATERS + 1)])
    for kol in ARMS:
        if kol not in responses:
            raise ValueError(f"'{kol}' kolu için yanıt yok")
        if len(responses[kol]) != len(prompt_listesi):
            raise ValueError(
                f"'{kol}' kolunda {len(responses[kol])} yanıt var, "
                f"{len(prompt_listesi)} prompt bekleniyordu")

    paketler, kor_anahtar = build_evaluation_sheets(
        prompt_listesi, degerlendiriciler, arms=ARMS, seed=seed)

    output_dir = Path(output_dir)
    paket_dizini = output_dir / "paketler"
    gizli_dizin = output_dir / "_GIZLI_degerlendiriciye_verme"
    paket_dizini.mkdir(parents=True, exist_ok=True)
    gizli_dizin.mkdir(parents=True, exist_ok=True)

    (output_dir / "YONERGE.md").write_text(YONERGE, encoding="utf-8")

    boyut_adlari = list(DIMENSIONS)
    for paket in paketler:
        dolu_ogeler: List[Dict[str, Any]] = []
        for oge in paket.items:
            anahtar = kor_anahtar[oge["item_id"]]
            kol = anahtar["arm"]
            p_indeks = int(anahtar["prompt_index"])
            yanit = responses[kol][p_indeks]
            dolu: Dict[str, Any] = {
                "item_id": oge["item_id"],
                "prompt": oge["prompt"],
                "response": yanit,
                "dimensions": oge["dimensions"],
            }
            dolu_ogeler.append(dolu)
        paket_json = {
            "rater_id": paket.rater_id,
            "instructions_file": "../YONERGE.md",
            "attention_checks_hidden": True,
            "items": dolu_ogeler,
        }
        (paket_dizini / f"{paket.rater_id}.json").write_text(
            json.dumps(paket_json, ensure_ascii=False, indent=1),
            encoding="utf-8")

        # Puanlama şablonu: satır=item, sütun=boyutlar (boş).
        with (paket_dizini / f"{paket.rater_id}_puanlama.csv").open(
                "w", newline="", encoding="utf-8") as f:
            yazici = csv.writer(f)
            yazici.writerow(["item_id"] + boyut_adlari)
            for oge in dolu_ogeler:
                yazici.writerow([oge["item_id"]] + [""] * len(boyut_adlari))

    # Körleme kanıtı: paket dosyalarında kol adı geçmemeli.
    sizinti: List[str] = []
    for dosya in paket_dizini.glob("*.json"):
        icerik = dosya.read_text(encoding="utf-8")
        sizinti.extend(k for k in ARMS if f'"{k}"' in icerik)

    (gizli_dizin / "kor_anahtari.json").write_text(
        json.dumps(kor_anahtar, ensure_ascii=False, indent=1),
        encoding="utf-8")
    (gizli_dizin / "UYARI.md").write_text(
        "# UYARI\n\nBu dizin değerlendiricilere VERİLMEZ. `kor_anahtari."
        "json` item→kol eşlemesidir ve yalnız puanlar toplandıktan sonra "
        "analiz için kullanılır.\n", encoding="utf-8")

    manifest = {
        "prompts": len(prompt_listesi),
        "raters": len(degerlendiriciler),
        "arms": list(ARMS),
        "items_per_rater": len(paketler[0].items) if paketler else 0,
        "generation": dict(GENERATION),
        "seed": seed,
        "blinding_leaks": sorted(set(sizinti)),
        "responses_signature": hashlib.sha256(json.dumps(
            responses, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")).hexdigest()[:16],
        "unblinding_key_digest": hashlib.sha256(json.dumps(
            kor_anahtar, sort_keys=True).encode("utf-8")).hexdigest()[:16],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def analyze_ratings_by_arm(package_dir: Path) -> Dict[str, Any]:
    """Kör açma anahtarıyla puanları kol bazında özetle.

    Her ordinal boyut için aritmetik ortalama, ikili ``halusinasyon_var`` için
    de oran raporlanır. Anahtar yalnız puan toplama kapandıktan sonra bu analiz
    adımında okunur; değerlendirici paketlerine geri yazılmaz.
    """
    root = Path(package_dir)
    key_path = root / "_GIZLI_degerlendiriciye_verme" / "kor_anahtari.json"
    try:
        unblinding_key = json.loads(key_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"kör açma anahtarı okunamadı: {key_path}") from error

    values: Dict[str, Dict[str, List[float]]] = {
        arm: {dimension: [] for dimension in DIMENSIONS} for arm in ARMS
    }
    files = sorted((root / "paketler").glob("R[0-9][0-9]_puanlama.csv"))
    if not files:
        raise ValueError("kol analizi için puanlama CSV'si bulunamadı")
    for path in files:
        with path.open(encoding="utf-8", newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                item_id = row.get("item_id", "")
                mapping = unblinding_key.get(item_id)
                if not isinstance(mapping, dict) or mapping.get("arm") not in values:
                    raise ValueError(
                        f"{path}:{row_number}: kör anahtarda geçerli kol yok: {item_id}"
                    )
                arm = str(mapping["arm"])
                for dimension in DIMENSIONS:
                    value = _parse_rating_value(
                        row.get(dimension) or "",
                        path=path,
                        row_number=row_number,
                        dimension=dimension,
                    )
                    if value is None:
                        raise ValueError(
                            f"{path}:{row_number}: eksik {dimension} puanı"
                        )
                    values[arm][dimension].append(value)

    result: Dict[str, Any] = {}
    for arm in ARMS:
        counts = {len(items) for items in values[arm].values()}
        if len(counts) != 1 or not counts or next(iter(counts)) == 0:
            raise ValueError(f"{arm}: boyut hücre sayıları eksik veya dengesiz")
        result[arm] = {
            "ratings_per_dimension": next(iter(counts)),
            "means": {
                dimension: round(sum(items) / len(items), 6)
                for dimension, items in values[arm].items()
                if dimension != "halusinasyon_var"
            },
            "hallucination_rate": round(
                sum(values[arm]["halusinasyon_var"])
                / len(values[arm]["halusinasyon_var"]),
                6,
            ),
        }
    return result


def _dimension_bounds(dimension: str) -> Tuple[float, float]:
    scale = DIMENSIONS[dimension]["scale"]
    return float(min(scale)), float(max(scale))


def _parse_rating_value(
    raw: str,
    *,
    path: Path,
    row_number: int,
    dimension: str,
) -> Optional[float]:
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(
            f"{path}:{row_number}: {dimension} sayısal değil: {raw!r}"
        ) from error
    low, high = _dimension_bounds(dimension)
    if not (low <= value <= high):
        raise ValueError(
            f"{path}:{row_number}: {dimension}={value} ölçek dışında "
            f"({low:g}–{high:g})")
    if dimension == "halusinasyon_var" and value not in (0.0, 1.0):
        raise ValueError(
            f"{path}:{row_number}: halusinasyon_var yalnız 0/1 olabilir")
    return value


def collect_ratings_from_csv(
    package_dir: Path,
    *,
    strict_complete: bool = False,
) -> Tuple[Dict[str, Dict[Any, Sequence[Optional[float]]]], Dict[str, Any]]:
    """Doldurulmuş ``*_puanlama.csv`` dosyalarından α girdisini topla.

    Returns:
        ``(ratings_by_dimension, summary)``. İlki
        ``build_human_evaluation_protocol(collected_ratings=...)`` girdisidir:
        boyut → {item_id: [rater1, rater2, ...]} (eksikler None).

    Boş hücreler varsayılan olarak ``None`` kalır; ``strict_complete=True``
    ise eksik hücre hata olur. Dolu hücrelerin tamamı ilgili ölçek aralığında
    doğrulanır.

    Raises:
        ValueError: hiç CSV yoksa, sütun eksikse veya dolu puan ölçek dışıysa.
    """
    paket_dizini = Path(package_dir) / "paketler"
    dosyalar = sorted(paket_dizini.glob("*_puanlama.csv"))
    if not dosyalar:
        raise ValueError(f"{paket_dizini} altında puanlama CSV'si yok")

    boyutlar = list(DIMENSIONS)
    ham: Dict[str, Dict[str, List[Optional[float]]]] = {
        b: {} for b in boyutlar}
    degerlendirici_sayisi = 0
    dolu_hucre = 0
    toplam_hucre = 0
    satir_sayilari: List[int] = []
    for dosya in dosyalar:
        degerlendirici_sayisi += 1
        rows_in_file = 0
        with dosya.open(encoding="utf-8", newline="") as f:
            okuyucu = csv.DictReader(f)
            alanlar = set(okuyucu.fieldnames or [])
            eksik = ["item_id", *boyutlar]
            eksik = [ad for ad in eksik if ad not in alanlar]
            if eksik:
                raise ValueError(f"{dosya}: eksik CSV sütunları: {eksik}")
            for row_number, satir in enumerate(okuyucu, start=2):
                rows_in_file += 1
                item_id = (satir.get("item_id") or "").strip()
                if not item_id:
                    raise ValueError(f"{dosya}:{row_number}: item_id boş")
                for b in boyutlar:
                    deger = _parse_rating_value(
                        satir.get(b) or "",
                        path=dosya,
                        row_number=row_number,
                        dimension=b,
                    )
                    if deger is None and strict_complete:
                        raise ValueError(f"{dosya}:{row_number}: eksik {b} puanı")
                    ham[b].setdefault(item_id, []).append(deger)
                    toplam_hucre += 1
                    if deger is not None:
                        dolu_hucre += 1
        satir_sayilari.append(rows_in_file)

    item_counts = {len(items) for items in ham.values()}
    dengeli_satir = len(set(satir_sayilari)) == 1
    ozet = {
        "raters_found": degerlendirici_sayisi,
        "items": len(next(iter(ham.values()), {})),
        "filled_cells": dolu_hucre,
        "total_cells": toplam_hucre,
        "filled_ratio": round(dolu_hucre / max(1, toplam_hucre), 4),
        "rows_per_rater": satir_sayilari,
        "balanced_rows_per_rater": dengeli_satir,
        "balanced_items_by_dimension": len(item_counts) == 1,
        "scale_validation": "passed",
    }
    return cast(Dict[str, Dict[Any, Sequence[Optional[float]]]], ham), ozet


def _attestation_summary(package_dir: Path) -> Dict[str, Any]:
    """Gerçek insan puanı beyan dosyasını oku.

    Beklenen dosya: ``rater_attestation.json``. En az şu alanlar olmalıdır:
    ``real_human_ratings: true`` ve ``raters`` listesi. Bu olmadan dolu CSV'ler
    yalnız import smoke/araç çıktısı sayılır; ana insan sonucu açılmaz.
    """
    path = Path(package_dir) / "rater_attestation.json"
    if not path.exists():
        return {"present": False, "path": str(path), "raters": 0,
                "real_human_ratings": False}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"attestation JSON bozuk: {path}") from error
    raters = data.get("raters", [])
    if not isinstance(raters, list):
        raise ValueError("rater_attestation.json: raters liste olmalı")
    return {
        "present": True,
        "path": str(path),
        "raters": len(raters),
        "real_human_ratings": bool(data.get("real_human_ratings")),
        "collected_by": data.get("collected_by"),
        "collected_utc_date": data.get("collected_utc_date"),
        "statement": data.get("statement"),
    }


def build_human_evaluation_from_csv(package_dir: Path) -> Any:
    """CSV'lerden insan değerlendirme raporu üret.

    Tam ve **attested** olmayan paketlerde güvenilirlik/kol sonucu ana rapora
    sokulmaz; dönen ``HumanEvaluationReport`` protokol raporudur ve
    ``human_ratings_collected`` KALIR. Bu, depoda örnek/doldurulmuş CSV olsa
    bile gerçek insan puanı beyanı yoksa skor üretmeyi engeller.
    """
    ratings, summary = collect_ratings_from_csv(package_dir)
    attestation = _attestation_summary(Path(package_dir))
    complete = (summary["raters_found"] >= MIN_RATERS
                and summary["filled_ratio"] >= 1.0
                and attestation["present"]
                and attestation["real_human_ratings"]
                and attestation["raters"] >= MIN_RATERS)
    if not complete:
        report = build_human_evaluation_protocol()
        report.design["rating_import_summary"] = summary
        report.design["rater_attestation"] = attestation
        return report
    arm_results = analyze_ratings_by_arm(Path(package_dir))
    report = build_human_evaluation_protocol(
        collected_ratings=ratings,
        arm_results=arm_results,
    )
    report.design["rating_import_summary"] = summary
    report.design["rater_attestation"] = attestation
    return report


def human_rating_import_report(package_dir: Path) -> Dict[str, Any]:
    """CSV import + α + kol özetini makine-okunur pipeline raporu yap.

    Paket dizini henüz yoksa hata fırlatmak yerine ``NO_CSV_NA`` raporu
    döndürür; bozuk/ölçek dışı CSV ise hâlâ açık hata olarak yükseltilir.
    """
    try:
        ratings, summary = collect_ratings_from_csv(package_dir)
    except ValueError as error:
        if "puanlama CSV'si yok" not in str(error):
            raise
        summary = {
            "raters_found": 0, "items": 0,
            "filled_cells": 0, "total_cells": 0,
            "filled_ratio": 0.0, "rows_per_rater": [],
            "balanced_rows_per_rater": False,
            "balanced_items_by_dimension": False,
            "scale_validation": "n/a",
        }
        ratings = {}
    reliability = analyze_ratings(ratings) if summary["filled_cells"] else None
    attestation = _attestation_summary(Path(package_dir))
    attested = (attestation["present"] and attestation["real_human_ratings"]
                and attestation["raters"] >= MIN_RATERS)
    arm_results = None
    arm_error = None
    if summary["raters_found"] >= MIN_RATERS and summary["filled_ratio"] >= 1.0:
        try:
            arm_results = analyze_ratings_by_arm(Path(package_dir))
        except ValueError as error:
            arm_error = str(error)
    checks = {
        "csv_files_found": summary["raters_found"] > 0,
        "rating_values_in_scale": summary["scale_validation"] == "passed",
        "min_raters_present": summary["raters_found"] >= MIN_RATERS,
        "all_cells_filled": summary["filled_ratio"] >= 1.0,
        "balanced_rows_per_rater": bool(summary["balanced_rows_per_rater"]),
        "rater_attestation_present": bool(attested),
        "reliability_computed_when_any_rating_present": (
            reliability is not None or summary["filled_cells"] == 0),
        "arm_summary_available_when_complete": (
            arm_results is not None or summary["filled_ratio"] < 1.0
            or summary["raters_found"] < MIN_RATERS),
    }
    status = ("COMPLETE_READY_FOR_REPORT" if arm_results is not None and attested
              else "CSV_COMPLETE_UNATTESTED_NA" if arm_results is not None
              else "NO_CSV_NA" if summary["raters_found"] == 0
              else "INCOMPLETE_NA")
    report = {
        "protocol": "human_evaluation_csv_import_v1",
        "schema_version": 1,
        "package_dir": str(package_dir),
        "summary": summary,
        "attestation": attestation,
        "reliability": reliability,
        "arm_results": arm_results,
        "arm_error": arm_error,
        "checks": checks,
        "status": status,
        "findings": [
            f"{summary['raters_found']} CSV dosyası okundu; doluluk "
            f"{summary['filled_ratio']:.4f}.",
            ("Tam ve beyanlı paket: α ve kol özeti ana rapora alınabilir."
             if status == "COMPLETE_READY_FOR_REPORT" else
             "Tam/beyanlı gerçek insan puanı yok; ana raporda bu bölüm n/a kalmalıdır."),
        ],
        "limitations": [
            "CSV import gerçek insan kalitesini garanti etmez; yalnız format, ölçek ve agregasyon hattını doğrular.",
            "rater_attestation.json olmadan dolu CSV'ler gerçek insan sonucu sayılmaz ve ana raporda n/a kalır.",
            "Kör açma anahtarı yoksa veya paket eksikse kol sonuçları üretilmez.",
        ],
    }
    report["signature"] = hashlib.sha256(json.dumps(
        {"protocol": report["protocol"], "summary": summary,
         "attestation": attestation},
        sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
    return report


def human_rating_import_markdown(report: Dict[str, Any]) -> str:
    """CSV import raporunu Markdown'a çevir."""
    summary = report["summary"]
    attestation = report.get("attestation", {})
    rows = [
        "# İnsan Değerlendirme CSV Import/Aggregation Pipeline",
        "",
        f"- Protokol: `{report['protocol']}` v{report['schema_version']} "
        f"(imza `{report['signature']}`)",
        f"- Durum: **{report['status']}**",
        f"- Gerçek insan beyanı: **{'VAR' if attestation.get('real_human_ratings') else 'YOK'}**",
        "",
        "## Import özeti",
        "",
        "| Alan | Değer |",
        "|---|---:|",
        f"| CSV/değerlendirici | {summary['raters_found']} |",
        f"| Öğe | {summary['items']} |",
        f"| Dolu hücre | {summary['filled_cells']} |",
        f"| Toplam hücre | {summary['total_cells']} |",
        f"| Doluluk | {summary['filled_ratio']:.4f} |",
        "",
        "## Kabul kapıları",
        "",
        "| Kapı | Sonuç |",
        "|---|---|",
    ]
    rows.extend(f"| {k} | {'GEÇTİ' if v else 'KALDI'} |"
                for k, v in report["checks"].items())
    if report["reliability"]:
        rows.extend([
            "",
            "## Krippendorff α",
            "",
            "| Boyut | α | Hüküm |",
            "|---|---:|---|",
        ])
        for dim, data in report["reliability"].items():
            if dim == "_summary":
                continue
            alpha = "n/a" if data["alpha"] is None else f"{data['alpha']:.4f}"
            rows.append(f"| {dim} | {alpha} | {data['verdict']} |")
    if report["arm_results"]:
        rows.extend([
            "",
            "## Kör açma sonrası kol özeti",
            "",
            "| Kol | n/boyut | Halüsinasyon oranı |",
            "|---|---:|---:|",
        ])
        for arm in ARMS:
            data = report["arm_results"][arm]
            rows.append(
                f"| {arm} | {data['ratings_per_dimension']} | "
                f"{data['hallucination_rate']:.3%} |")
    rows.extend(["", "## Bulgular", ""])
    rows.extend(f"- {x}" for x in report["findings"])
    rows.extend(["", "## Sınırlar", ""])
    rows.extend(f"- {x}" for x in report["limitations"])
    return "\n".join(rows) + "\n"


__all__ = [
    "GENERATION", "YONERGE", "analyze_ratings_by_arm",
    "build_human_evaluation_from_csv", "collect_ratings_from_csv",
    "fill_and_export_packages", "generate_arm_responses",
    "human_rating_import_markdown", "human_rating_import_report",
    "symbolic_answer",
]
