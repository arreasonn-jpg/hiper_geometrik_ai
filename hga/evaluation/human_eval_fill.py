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
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .human_evaluation import (
    ARMS,
    DIMENSIONS,
    MIN_RATERS,
    _default_prompts,
    build_evaluation_sheets,
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
                    raw = (row.get(dimension) or "").strip()
                    if not raw:
                        raise ValueError(
                            f"{path}:{row_number}: eksik {dimension} puanı"
                        )
                    values[arm][dimension].append(float(raw))

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


def collect_ratings_from_csv(
    package_dir: Path,
) -> Tuple[Dict[str, Dict[Any, List[Optional[float]]]], Dict[str, Any]]:
    """Doldurulmuş ``*_puanlama.csv`` dosyalarından α girdisini topla.

    Returns:
        ``(ratings_by_dimension, summary)``. İlki
        ``build_human_evaluation_protocol(collected_ratings=...)`` girdisidir:
        boyut → {item_id: [rater1, rater2, ...]} (eksikler None).

    Raises:
        ValueError: hiç doldurulmuş CSV yoksa.
    """
    paket_dizini = Path(package_dir) / "paketler"
    dosyalar = sorted(paket_dizini.glob("*_puanlama.csv"))
    if not dosyalar:
        raise ValueError(f"{paket_dizini} altında puanlama CSV'si yok")

    boyutlar = list(DIMENSIONS)
    # item_id → rater sırasına göre puan listeleri
    ham: Dict[str, Dict[str, List[Optional[float]]]] = {
        b: {} for b in boyutlar}
    degerlendirici_sayisi = 0
    dolu_hucre = 0
    toplam_hucre = 0
    for dosya in dosyalar:
        degerlendirici_sayisi += 1
        with dosya.open(encoding="utf-8") as f:
            okuyucu = csv.DictReader(f)
            for satir in okuyucu:
                item_id = satir["item_id"]
                for b in boyutlar:
                    deger_metni = (satir.get(b) or "").strip()
                    deger: Optional[float] = (
                        float(deger_metni) if deger_metni else None)
                    ham[b].setdefault(item_id, []).append(deger)
                    toplam_hucre += 1
                    if deger is not None:
                        dolu_hucre += 1

    ozet = {
        "raters_found": degerlendirici_sayisi,
        "items": len(next(iter(ham.values()), {})),
        "filled_ratio": round(dolu_hucre / max(1, toplam_hucre), 4),
    }
    return ham, ozet


__all__ = [
    "GENERATION", "YONERGE", "analyze_ratings_by_arm",
    "collect_ratings_from_csv", "fill_and_export_packages",
    "generate_arm_responses", "symbolic_answer",
]
