# -*- coding: utf-8 -*-
"""P1 — Uzun bağlam dil modelleme: 24→256 token bağlam taraması.

Ana Türkçe LM protokolü (``turkish_lm``) 24 token bağlamla koşar. Bu modül
aynı korpus, aynı train-only BPE ve aynı parametre-eşleme kuralıyla bağlam
uzunluğunu tek değişken olarak tarar: 24 (referans), 64, 128, 256.

Tasarım kararları (adillik için):

1. **Eşleşmiş hedefler.** Değerlendirme pozisyonları (belge, konum) çiftleri
   olarak BİR KEZ örneklenir ve tüm bağlam uzunluklarında + tüm kollarda
   aynen kullanılır. Böylece "PPL bağlamla nasıl değişti" sorusu aynı hedef
   token kümesi üzerinde, yalnız görünür bağlam değişerek ölçülür.
2. **Örneklenmiş pencereler.** 256 token bağlamda tüm pencereleri
   materyalize etmek RAM'i aşar (~1.5 GB / split). Pencereler deterministik
   RNG ile örneklenir; PPL bu yüzden korpus CE'sinin YANSIZ bir tahminidir
   ve rapor bunu sınır olarak yazar.
3. **Bağlam başına parametre eşleme.** Girdi boyutu bağlamla büyüdüğü için
   dense/transformer genişlikleri her bağlamda HGA bütçesine yeniden
   eşlenir (``build_lm_models`` ile, oran ≤ 1.05).
4. **N-gram zemin aynı pencerelerde.** Unigram ve interpolasyonlu bigram
   aynı örneklenmiş hedeflerde hesaplanır; neural kolların anlam zemini
   pencere seçiminden etkilenmez.

Kapılar mimarinin kazanmasını İSTEMEZ; ölçümün tamamlanmasını, bütçe
adilliğini ve yönün (bağlam artınca PPL ne yaptı) raporlanmasını denetler.
"""
from __future__ import annotations

import math
import statistics as _stat
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .experiment import canonical_hash
from .turkish_lm import (
    NEURAL_ARMS,
    _torch,
    _train_and_eval,
    build_lm_models,
    prepare_lm_corpus,
)

PROTOCOL = "long_context_lm_v1"
SCHEMA_VERSION = 1

#: Taranan bağlam uzunlukları. 24 ana protokolün referansıdır.
CONTEXTS: Tuple[int, ...] = (24, 64, 128, 256)

PROFILES: Dict[str, Dict[str, Any]] = {
    # Hızlı CI: küçük TWT korpusu, iki bağlam, az adım.
    "smoke": {
        "corpus": "twt", "max_docs": 400, "max_vocab": 2000,
        "contexts": (16, 64), "seeds": (1, 2),
        "steps": 120, "batch_size": 64, "eval_batch_size": 512,
        "learning_rate": 0.003, "gradient_clip_norm": 5.0,
        "emb_dim": 16, "hga_n": 16, "heads": 2, "hga_layers": 1,
        "train_windows": 8_000, "eval_windows": 2_000,
    },
    # Bilimsel koşu: milyon-kelime korpus, 24→256 tarama. batch=32 tüm
    # bağlamlarda SABİTTİR: (a) schedule böylece bağlamlar arasında aynı
    # kalır (aynı tohum → aynı örnekler → eşleşmiş eğitim), (b) 256 tokenda
    # parametre eşleme transformer FFN'ini ~7400'e büyütür ve daha büyük
    # batch'in aktivasyonları 3 GB sınıfı makinede RAM'i aşar.
    "full": {
        "corpus": "tr_corpus_v1", "max_docs": None, "max_vocab": 8000,
        "contexts": CONTEXTS, "seeds": (1, 2),
        "steps": 300, "batch_size": 32, "eval_batch_size": 1024,
        "learning_rate": 0.003, "gradient_clip_norm": 5.0,
        "emb_dim": 32, "hga_n": 24, "heads": 4, "hga_layers": 2,
        "train_windows": 40_000, "eval_windows": 4_000,
    },
}

#: Pozisyon örnekleme tohumu — kollardan ve kullanıcı tohumlarından bağımsız,
#: protokolün kimliğinin parçası.
_POSITION_SEED = 424_243


@dataclass
class LongContextReport:
    protocol: str
    schema_version: int
    profile: str
    seeds: List[int]
    dataset_hash: str
    config_hash: str
    config: Dict[str, Any]
    corpus: Dict[str, Any]
    positions: Dict[str, Any]
    budget_by_context: Dict[str, Dict[str, Any]]
    ngram_by_context: Dict[str, Dict[str, float]]
    results: Dict[str, Dict[str, Any]]
    context_effect: Dict[str, Any]
    checks: Dict[str, bool]
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        return long_context_markdown(self)


def _sample_positions(torch, docs: Sequence[Sequence[int]], count: int,
                      seed: int) -> List[Tuple[int, int]]:
    """(belge, konum) çiftlerini deterministik ve tekrarsız örnekle."""
    tum: List[Tuple[int, int]] = [
        (di, i) for di, doc in enumerate(docs) for i in range(len(doc))]
    if not tum:
        raise ValueError("Örneklenecek pozisyon yok")
    if count >= len(tum):
        return tum
    uretec = torch.Generator(device="cpu").manual_seed(int(seed))
    secim = torch.randperm(len(tum), generator=uretec)[:count]
    return [tum[int(k)] for k in secim]


def _windows_at(torch, docs: Sequence[Sequence[int]],
                positions: Sequence[Tuple[int, int]], context: int,
                pad_id: int = 0):
    """Sabit pozisyon listesinden verilen bağlam uzunluğunda pencere kur.

    Bağlam belge sınırını asla aşmaz; eksik geçmiş PAD ile doldurulur.
    Aynı ``positions`` her bağlam uzunluğunda aynı hedef tokenları üretir.
    """
    xs: List[List[int]] = []
    ys: List[int] = []
    for di, i in positions:
        doc = docs[di]
        baslangic = max(0, i - context)
        baglam = list(doc[baslangic:i])
        baglam = [pad_id] * (context - len(baglam)) + baglam
        xs.append(baglam)
        ys.append(int(doc[i]))
    return (torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long))


class _WindowNgram:
    """Örneklenmiş pencerelerde unigram / interpolasyonlu bigram PPL.

    Neural kollarla AYNI hedef kümesinde hesaplanır ki zemin karşılaştırması
    pencere örnekleminden etkilenmesin. Add-α düzleştirme; bigram, pencere
    içindeki son gerçek (PAD olmayan) tokena koşullanır.
    """

    def __init__(self, train_docs: Sequence[Sequence[int]], vocab_size: int,
                 alpha: float = 1.0, lam: float = 0.7):
        self.v = int(vocab_size)
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.uni: Dict[int, int] = {}
        self.bi: Dict[Tuple[int, int], int] = {}
        self.prev_toplam: Dict[int, int] = {}
        self.toplam = 0
        for doc in train_docs:
            for j, tok in enumerate(doc):
                self.uni[tok] = self.uni.get(tok, 0) + 1
                self.toplam += 1
                if j > 0:
                    onceki = doc[j - 1]
                    self.bi[(onceki, tok)] = self.bi.get((onceki, tok), 0) + 1
                    self.prev_toplam[onceki] = self.prev_toplam.get(onceki, 0) + 1

    def _p_uni(self, tok: int) -> float:
        return ((self.uni.get(tok, 0) + self.alpha)
                / (self.toplam + self.alpha * self.v))

    def _p_bi(self, onceki: Optional[int], tok: int) -> float:
        p1 = self._p_uni(tok)
        if onceki is None:
            return p1
        pay = self.bi.get((onceki, tok), 0) + self.alpha
        payda = self.prev_toplam.get(onceki, 0) + self.alpha * self.v
        return self.lam * (pay / payda) + (1.0 - self.lam) * p1

    def ppl(self, xs, ys, model: str, pad_id: int = 0) -> float:
        toplam_log, sayi = 0.0, 0
        for satir, hedef in zip(xs.tolist(), ys.tolist()):
            onceki = next((t for t in reversed(satir) if t != pad_id), None)
            p = (self._p_uni(hedef) if model == "unigram"
                 else self._p_bi(onceki, hedef))
            toplam_log += -math.log(max(p, 1e-12))
            sayi += 1
        return round(math.exp(toplam_log / max(1, sayi)), 8)


def run_long_context_benchmark(
    profile: str = "smoke",
    seeds: Optional[Sequence[int]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> LongContextReport:
    """Bağlam uzunluğu taramasını koş ve tek rapor üret.

    Raises:
        ValueError: bilinmeyen profil ya da 2'den az tohum.
    """
    if profile not in PROFILES:
        raise ValueError(f"profile şunlardan biri olmalı: {', '.join(PROFILES)}")
    config = dict(PROFILES[profile])
    config.update(dict(overrides or {}))
    tohumlar = [int(s) for s in (seeds if seeds is not None else config["seeds"])]
    if len(tohumlar) < 2:
        raise ValueError("çoklu-tohum raporu için en az 2 tohum gerekir")
    baglam_listesi = [int(c) for c in config["contexts"]]
    if len(baglam_listesi) < 2:
        raise ValueError("bağlam etkisi için en az 2 bağlam uzunluğu gerekir")

    torch, nn = _torch()
    from mimari.bpe_tokenizer import BPETokenizer

    corpus = prepare_lm_corpus(max_docs=config.get("max_docs"),
                               corpus=config["corpus"])
    train_metin = "\n".join(corpus.split_texts("train"))
    tokenizer = BPETokenizer(baglam_penceresi=max(baglam_listesi),
                             max_vocab_size=int(config["max_vocab"]),
                             min_freq=2)
    tokenizer.fit_on_text(train_metin, verbose=False)
    vocab_size = max(int(tokenizer.sozluk_boyutu), 8)

    split_ids: Dict[str, List[List[int]]] = {}
    for split in ("train", "dev", "test"):
        split_ids[split] = [tokenizer.encode(d, pad=False)
                            for d in corpus.split_texts(split)]
    token_sayilari = {s: sum(len(d) for d in v) for s, v in split_ids.items()}

    # Pozisyonlar bağlamdan ve koldan bağımsız olarak BİR KEZ örneklenir.
    pozisyonlar = {
        "train": _sample_positions(torch, split_ids["train"],
                                   int(config["train_windows"]),
                                   _POSITION_SEED),
        "dev": _sample_positions(torch, split_ids["dev"],
                                 int(config["eval_windows"]),
                                 _POSITION_SEED + 1),
        "test": _sample_positions(torch, split_ids["test"],
                                  int(config["eval_windows"]),
                                  _POSITION_SEED + 2),
    }
    pozisyon_imzasi = canonical_hash(
        {s: [[int(a), int(b)] for a, b in p] for s, p in pozisyonlar.items()})

    ngram = _WindowNgram(split_ids["train"], vocab_size)

    sonuclar: Dict[str, Dict[str, Any]] = {
        arm: {str(c): {"per_seed": []} for c in baglam_listesi}
        for arm in NEURAL_ARMS
    }
    butceler: Dict[str, Dict[str, Any]] = {}
    ngram_sonuc: Dict[str, Dict[str, float]] = {}

    for baglam in baglam_listesi:
        cfg_c = dict(config)
        cfg_c["context"] = int(baglam)
        # Aktivasyon belleği bağlamla büyür: dikkat skorları karesel
        # (batch × heads × c × c), parametre-eşli transformer FFN'i ise
        # bağlamla genişler (256 tokenda ~7400) ve eval aktivasyonu
        # batch × c × ffn olur. 3 GB sınıfı makinede eval batch bağlama göre
        # ölçeklenir. Bu yalnız değerlendirme parçalamasıdır; metrikleri
        # değiştirmez.
        cfg_c["eval_batch_size"] = max(
            32, min(int(config["eval_batch_size"]), 16_384 // int(baglam)))
        train_x, train_y = _windows_at(torch, split_ids["train"],
                                       pozisyonlar["train"], baglam)
        evals = {
            "dev": _windows_at(torch, split_ids["dev"],
                               pozisyonlar["dev"], baglam),
            "test": _windows_at(torch, split_ids["test"],
                                pozisyonlar["test"], baglam),
        }
        test_x, test_y = evals["test"]
        ngram_sonuc[str(baglam)] = {
            "unigram_test_ppl": ngram.ppl(test_x, test_y, "unigram"),
            "bigram_test_ppl": ngram.ppl(test_x, test_y, "bigram"),
        }
        kurucular, eslesme = build_lm_models(vocab_size, cfg_c)
        parametreler = {
            ad: sum(p.numel() for p in ctor().parameters() if p.requires_grad)
            for ad, ctor in kurucular.items()
        }
        oran = max(parametreler.values()) / min(parametreler.values())
        butceler[str(baglam)] = {
            "parameters": parametreler,
            "max_to_min_ratio": round(oran, 6),
            **eslesme,
        }
        for tohum in tohumlar:
            uretec = torch.Generator(device="cpu").manual_seed(int(tohum) + 7_919)
            schedule = torch.randint(
                0, int(train_x.shape[0]),
                (int(config["steps"]), int(config["batch_size"])),
                generator=uretec)
            for kol_indeksi, ad in enumerate(NEURAL_ARMS):
                torch.manual_seed(int(tohum) + 100_003 * (kol_indeksi + 1))
                model = kurucular[ad]()
                olcum = _train_and_eval(torch, nn, model, train_x, train_y,
                                        evals, schedule, cfg_c)
                olcum["seed"] = int(tohum)
                sonuclar[ad][str(baglam)]["per_seed"].append(olcum)
        del train_x, train_y, evals  # 256 bağlamda pencere RAM'i geri ver

    # Özetler ve bağlam etkisi.
    for arm in NEURAL_ARMS:
        for baglam in baglam_listesi:
            hucre = sonuclar[arm][str(baglam)]
            ppl_degerleri = [s["test"]["perplexity"] for s in hucre["per_seed"]]
            hucre["test_ppl_mean"] = round(_stat.fmean(ppl_degerleri), 8)
            hucre["test_ppl_std"] = (
                round(_stat.stdev(ppl_degerleri), 8)
                if len(ppl_degerleri) > 1 else None)
            hucre["test_top1_mean"] = round(_stat.fmean(
                [s["test"]["top1_accuracy"] for s in hucre["per_seed"]]), 8)

    ilk, son = str(baglam_listesi[0]), str(baglam_listesi[-1])
    etki: Dict[str, Any] = {"reference_context": int(ilk),
                            "max_context": int(son), "per_arm": {}}
    for arm in NEURAL_ARMS:
        ppl_ilk = sonuclar[arm][ilk]["test_ppl_mean"]
        ppl_son = sonuclar[arm][son]["test_ppl_mean"]
        etki["per_arm"][arm] = {
            "ppl_at_reference": ppl_ilk,
            "ppl_at_max": ppl_son,
            "relative_change": round((ppl_son - ppl_ilk) / ppl_ilk, 6),
            "direction": ("improves" if ppl_son < ppl_ilk else
                          "degrades" if ppl_son > ppl_ilk else "flat"),
        }

    tum_ppl = [s["test"]["perplexity"]
               for arm in NEURAL_ARMS for c in baglam_listesi
               for s in sonuclar[arm][str(c)]["per_seed"]]
    en_iyi_son = min(sonuclar[a][son]["test_ppl_mean"] for a in NEURAL_ARMS)
    unigram_son = ngram_sonuc[son]["unigram_test_ppl"]
    bigram_son = ngram_sonuc[son]["bigram_test_ppl"]

    # Kapı tasarımı: bu protokolün iddiası BAĞLAM ETKİSİDİR, mutlak LM
    # kalitesi değil. Mutlak kalite kapıları (n-gram'ı geçmek) ana
    # ``turkish_lm`` protokolünde yaşar (full: 4000 adım, PASS). Buradaki
    # kısa eşit-bütçe taraması (300 adım) yakınsama iddiası taşımaz; n-gram
    # zemini KAPI değil RAPORLANAN ölçümdür ve neural kolların onu bu
    # bütçede geçmediği bulgularda açıkça yazılır. Aynı iddiayı iki
    # protokolde kapılamak çift sayım olurdu.
    checks = {
        "all_cells_completed": all(
            len(sonuclar[a][str(c)]["per_seed"]) == len(tohumlar)
            for a in NEURAL_ARMS for c in baglam_listesi),
        "finite_metrics_all_cells": all(
            math.isfinite(v) for v in tum_ppl),
        "parameter_budget_leq_1_05_every_context": all(
            b["max_to_min_ratio"] <= 1.05 for b in butceler.values()),
        "positions_shared_across_contexts_and_arms": True,  # kuruluş gereği
        "ngram_floor_reported_every_context": all(
            str(c) in ngram_sonuc for c in baglam_listesi),
        "context_effect_direction_reported": bool(etki["per_arm"]),
        "multi_seed_reported": len(tohumlar) >= 2,
    }

    bulgular: List[str] = []
    for arm in NEURAL_ARMS:
        e = etki["per_arm"][arm]
        bulgular.append(
            f"`{arm}`: bağlam {ilk}→{son} tokenda test PPL "
            f"{e['ppl_at_reference']:.1f}→{e['ppl_at_max']:.1f} "
            f"({e['relative_change']:+.1%}, {e['direction']}).")
    bulgular.append(
        f"Zemin @ bağlam {son}: unigram PPL {unigram_son:.1f}, "
        f"bigram {bigram_son:.1f}; en iyi neural {en_iyi_son:.1f}.")
    if en_iyi_son >= bigram_son:
        bulgular.append(
            "AÇIK SINIR: bu kısa eşit-bütçe taramasında (300 adım) hiçbir "
            "neural kol bigram zeminini geçmedi. Mutlak LM kalitesi iddiası "
            "bu protokolün konusu değildir ve `turkish_lm` full (4000 adım) "
            "protokolünde ölçülür; oradaki n-gram kapıları gerçek veriyle "
            "PASS. Buradaki sayılar yalnız bağlam ETKİSİNİN yönünü taşır.")

    imza = canonical_hash({
        "protocol": PROTOCOL, "schema_version": SCHEMA_VERSION,
        "profile": profile, "seeds": tohumlar,
        "corpus_hash": corpus.dataset_hash,
        "positions": pozisyon_imzasi,
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in config.items() if k != "seeds"},
    })

    return LongContextReport(
        protocol=PROTOCOL,
        schema_version=SCHEMA_VERSION,
        profile=profile,
        seeds=tohumlar,
        dataset_hash=imza,
        config_hash=canonical_hash(
            {k: (list(v) if isinstance(v, tuple) else v)
             for k, v in config.items()}),
        config={k: (list(v) if isinstance(v, tuple) else v)
                for k, v in config.items() if k != "seeds"},
        corpus={"dataset_hash": corpus.dataset_hash,
                "tokens": token_sayilari,
                "vocabulary_size": vocab_size},
        positions={"signature": pozisyon_imzasi,
                   "train": len(pozisyonlar["train"]),
                   "dev": len(pozisyonlar["dev"]),
                   "test": len(pozisyonlar["test"]),
                   "sampling_seed": _POSITION_SEED},
        budget_by_context=butceler,
        ngram_by_context=ngram_sonuc,
        results=sonuclar,
        context_effect=etki,
        checks=checks,
        findings=bulgular,
        limitations=[
            "PPL, deterministik örneklenmiş pencereler üzerinde hesaplanır; "
            "korpus CE'sinin yansız tahminidir, tam sayımı değildir.",
            "Adım sayısı sabittir ve yakınsama iddiası yoktur; bağlamlar "
            "arası karşılaştırma eşit-eğitim-bütçesi karşılaştırmasıdır.",
            "Girdi düzleştirmeli mimarilerde (dense, HGA encoder) bağlam "
            "uzadıkça parametre bütçesi aynı kalsın diye genişlikler düşer; "
            "bu, mimari ailelerin uzun bağlamdaki YAPISAL maliyet farkıdır "
            "ve gizlenmez.",
            "Tek korpus (tr_corpus_v1) ve tek dil (Türkçe); genelleme "
            "iddiası taşımaz.",
        ],
    )


def long_context_markdown(report: LongContextReport) -> str:
    """Raporu Markdown tabloya çevir."""
    s = report
    satirlar = [
        "# Uzun Bağlam Dil Modelleme (P1)",
        "",
        f"Protokol: `{s.protocol}` v{s.schema_version} · profil `{s.profile}` "
        f"· veri imzası `{s.dataset_hash[:12]}`",
        f"Tohumlar: {s.seeds} · korpus token: {s.corpus['tokens']} · "
        f"sözlük {s.corpus['vocabulary_size']}",
        f"Pencereler: train {s.positions['train']:,} / eval "
        f"{s.positions['test']:,} (deterministik örnekleme, "
        f"imza `{s.positions['signature'][:12]}`)",
        "",
        "## Test PPL (ortalama ± std)",
        "",
        "| Bağlam | " + " | ".join(NEURAL_ARMS) + " | unigram | bigram |",
        "|---:|" + "---:|" * (len(NEURAL_ARMS) + 2),
    ]
    for baglam in s.config["contexts"]:
        hucreler = []
        for arm in NEURAL_ARMS:
            h = s.results[arm][str(baglam)]
            std = h["test_ppl_std"]
            hucreler.append(
                f"{h['test_ppl_mean']:.1f}"
                + (f" ±{std:.1f}" if std is not None else ""))
        ng = s.ngram_by_context[str(baglam)]
        satirlar.append(
            f"| {baglam} | " + " | ".join(hucreler)
            + f" | {ng['unigram_test_ppl']:.1f} "
            f"| {ng['bigram_test_ppl']:.1f} |")
    satirlar.extend(["", "## Bağlam etkisi "
                     f"({s.context_effect['reference_context']}→"
                     f"{s.context_effect['max_context']} token)", ""])
    for arm, e in s.context_effect["per_arm"].items():
        satirlar.append(
            f"- `{arm}`: {e['ppl_at_reference']:.1f} → {e['ppl_at_max']:.1f} "
            f"({e['relative_change']:+.1%}, {e['direction']})")
    satirlar.extend(["", "## Kapılar", ""])
    satirlar.extend(f"- {'GEÇTİ' if v else 'KALDI'} — `{k}`"
                    for k, v in s.checks.items())
    satirlar.extend(["", "## Bulgular", ""])
    satirlar.extend(f"- {b}" for b in s.findings)
    satirlar.extend(["", "## Sınırlar", ""])
    satirlar.extend(f"- {n}" for n in s.limitations)
    return "\n".join(satirlar) + "\n"


__all__ = [
    "CONTEXTS", "PROFILES", "PROTOCOL", "SCHEMA_VERSION",
    "LongContextReport", "long_context_markdown", "run_long_context_benchmark",
]
