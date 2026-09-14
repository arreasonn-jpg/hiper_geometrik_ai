# -*- coding: utf-8 -*-
"""Gerçek Türkçe dil modelleme benchmarkı (karnenin son kanıtsız neural bölümü).

Karnede ``language_modeling`` bölümü bugüne kadar bilinçli olarak ``n/a``
kaldı: "``perplexity --tiny`` bir dil modeli iddiası değildir" ve öyle kalmalı.
Bu modül o bölümü sahte kanıtla değil, repo içinde zaten var olan GERÇEK
veriyle doldurur: TWT v1 (4.851 ham, insan-anotasyonlu Türkçe cümle,
hash-doğrulamalı). Görev klasik held-out dil modellemesidir: bir belgenin
önceki token'larından bir sonraki token'ı tahmin et.

Dürüstlük sözleşmesi:

* **Splitler belge-ayrıktır.** Aynı web sayfası/wiki maddesinin cümleleri asla
  hem train hem test'e düşmez; split belge kimliğinin SHA-256'sından türetilir.
* **Tokenizer yalnız train metninde eğitilir.** Test metni BPE merge'lerini
  etkileyemez.
* **N-gram kontrolleri zorunludur.** Add-α unigram ve interpolasyonlu bigram,
  "neural model gerçekten bir şey öğreniyor mu?" sorusunun zeminidir. Neural
  bir kol unigram'ı geçemiyorsa tablo bunu saklamaz.
* **Parametre bütçesi eşlenir.** Dense/Transformer kolları HGA'nın toplam
  parametre sayısına ±%5 içinde oturtulur; oturmuyorsa kapı FAIL olur.
* **Korpus boyutu kapısı serttir.** Milyon-kelime kapısı
  (``corpus_at_least_1m_words``) yalnız gerçek veriyle açılır: smoke profil
  TWT (~66K kelime) üzerinde koşar ve kapı FAIL kalır; full profil vendored
  ``tr_corpus_v1`` (≥1.1M kelime, UD r2.14 + Bible CC0 + TWT, hash doğrulamalı)
  üzerinde koşar ve kapıyı gerçek insan metniyle açar.

Bu modülün üretmediği şey: üretim kalitesi/sohbet iddiası. Ölçülen şey held-out
perplexity ve next-token doğruluğudur, o kadar.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .experiment import canonical_hash, file_sha256
from .real_turkish import TurkishWebTreebank
from .statistics import compare_paired, summarize_seed_metric

PROTOCOL = "turkish_lm_v1"
SCHEMA_VERSION = 1

#: Neural kollar; ``hga`` repo'nun gerçek mimarisidir (attention + geometrik
#: encoder + Kronecker zinciri + fraktal decoder), oyuncak bir kopya değil.
NEURAL_ARMS: Tuple[str, ...] = ("dense", "transformer", "hga")
NGRAM_ARMS: Tuple[str, ...] = ("unigram", "bigram")

#: Parametre eşleme toleransı: max/min toplam parametre oranı.
PARAMETER_TOLERANCE = 1.05

#: Bigram interpolasyon ağırlığı ve add-α düzleştirme.
BIGRAM_LAMBDA = 0.7
NGRAM_ALPHA = 1.0

PROFILES: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "corpus": "twt",
        "max_docs": 400, "max_vocab": 2000, "context": 16,
        "steps": 400, "batch_size": 128, "eval_batch_size": 1024,
        "learning_rate": 0.003, "gradient_clip_norm": 5.0,
        "seeds": (1, 2),
        "emb_dim": 16, "hga_n": 16, "heads": 2, "hga_layers": 1,
    },
    # full: milyon-kelime tr_corpus_v1 üzerinde koşar. Tek FAIL kapıyı
    # (corpus_at_least_1m_words) gerçek veriyle açabilen profil budur.
    # 4000 adım yakınsama probuyla seçildi: dense dev-PPL bigram'ı ~2100.
    # adımda geçer, 4000'de hâlâ düşmektedir (298 @ probe).
    "full": {
        "corpus": "tr_corpus_v1",
        "max_docs": None, "max_vocab": 8000, "context": 24,
        "steps": 4000, "batch_size": 256, "eval_batch_size": 4096,
        "learning_rate": 0.003, "gradient_clip_norm": 5.0,
        "seeds": (1, 2),
        "emb_dim": 32, "hga_n": 24, "heads": 4, "hga_layers": 2,
    },
}

#: Vendored milyon-kelime korpusun dizini ve provenance dosyası.
TR_CORPUS_DIR = Path(__file__).resolve().parent / "datasets" / "tr_corpus_v1"


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as error:  # pragma: no cover - ortam bağımlı
        raise ImportError("Türkçe LM benchmarkı için PyTorch gereklidir") from error
    return torch, nn


# ── Korpus hazırlığı (torch'suz) ────────────────────────────────────────────
@dataclass(frozen=True)
class LMDocument:
    doc_id: str
    split: str
    text: str
    sentence_count: int
    word_count: int


@dataclass(frozen=True)
class LMCorpus:
    """Belge-ayrık train/dev/test korpusu ve içerik imzası."""

    documents: Tuple[LMDocument, ...]
    dataset_hash: str
    config_hash: str

    def split_texts(self, split: str) -> List[str]:
        return [d.text for d in self.documents if d.split == split]

    def split_stats(self) -> Dict[str, Dict[str, int]]:
        stats: Dict[str, Dict[str, int]] = {}
        for split in ("train", "dev", "test"):
            docs = [d for d in self.documents if d.split == split]
            stats[split] = {
                "documents": len(docs),
                "sentences": sum(d.sentence_count for d in docs),
                "words": sum(d.word_count for d in docs),
            }
        return stats


def _doc_split(doc_id: str) -> str:
    bucket = int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest(), 16) % 10
    if bucket < 8:
        return "train"
    return "dev" if bucket == 8 else "test"


def _load_twt_raw_documents() -> List[Tuple[str, str]]:
    """TWT'den (doc_id, text) listesi; belge = sent_id'nin ilk iki parçası."""
    twt = TurkishWebTreebank()
    gruplar: Dict[str, List[Any]] = {}
    for sentence in twt.sentences:
        parcalar = sentence.sentence_id.split(":")
        doc_id = ":".join(parcalar[:2]) if len(parcalar) >= 2 else sentence.sentence_id
        gruplar.setdefault(doc_id, []).append(sentence)
    belgeler = []
    for doc_id in sorted(gruplar):
        cumleler = sorted(gruplar[doc_id], key=lambda s: s.sentence_id)
        belgeler.append((doc_id, "\n".join(c.text for c in cumleler)))
    return belgeler


def _load_tr_corpus_raw_documents() -> List[Tuple[str, str]]:
    """Vendored tr_corpus_v1'i PROVENANCE hash doğrulamasıyla yükle.

    İçerik hash'i gzip sarmalayıcısından bağımsız olarak canonical JSONL
    baytları üzerinden de doğrulanır; iki hash'ten biri tutmazsa korpus
    bozulmuş demektir ve YÜKLENMEZ.
    """
    provenance_path = TR_CORPUS_DIR / "PROVENANCE.json"
    corpus_path = TR_CORPUS_DIR / "corpus.jsonl.gz"
    if not provenance_path.exists() or not corpus_path.exists():
        raise FileNotFoundError(
            "tr_corpus_v1 bulunamadı. Korpus vendored olmalıdır; yeniden "
            "üretmek için: python hga/data/build_tr_corpus.py")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = provenance["files"]["corpus.jsonl.gz"]
    actual_file = file_sha256(corpus_path)
    if actual_file != expected["sha256"]:
        raise ValueError(
            f"tr_corpus_v1 dosya SHA-256 uyuşmazlığı: beklenen "
            f"{expected['sha256']} gerçek {actual_file}")
    with gzip.open(corpus_path, "rb") as handle:
        canonical = handle.read()
    actual_content = hashlib.sha256(canonical).hexdigest()
    if actual_content != expected["canonical_content_sha256"]:
        raise ValueError("tr_corpus_v1 içerik SHA-256 uyuşmazlığı")
    belgeler = []
    for line in canonical.decode("utf-8").splitlines():
        record = json.loads(line)
        belgeler.append((record["doc_id"], record["text"]))
    return belgeler


#: Desteklenen korpus kaynakları → yükleyici.
CORPUS_LOADERS = {
    "twt": _load_twt_raw_documents,
    "tr_corpus_v1": _load_tr_corpus_raw_documents,
}


def prepare_lm_corpus(max_docs: Optional[int] = None,
                      corpus: str = "twt") -> LMCorpus:
    """Ham belgeleri deterministik olarak belge-ayrık böl.

    ``corpus="twt"``: belge kimliği ``sent_id``'nin ilk iki parçasıdır
    (örn. ``tr-forum:00000222``); aynı kaynak sayfanın tüm cümleleri tek
    splitte kalır. ``corpus="tr_corpus_v1"``: vendored milyon-kelime korpus,
    hash doğrulamasıyla yüklenir. ``max_docs`` verilirse sıralı belge
    listesinin başı alınır — bu yalnız smoke/test hızlandırmasıdır ve
    hash'e dahildir.
    """
    if corpus not in CORPUS_LOADERS:
        raise ValueError(
            f"corpus şunlardan biri olmalı: {', '.join(CORPUS_LOADERS)}")
    ham_belgeler = CORPUS_LOADERS[corpus]()
    if max_docs is not None:
        ham_belgeler = ham_belgeler[: int(max_docs)]

    belgeler: List[LMDocument] = []
    for doc_id, metin in ham_belgeler:
        satirlar = metin.split("\n")
        belgeler.append(LMDocument(
            doc_id=doc_id,
            split=_doc_split(doc_id),
            text=metin,
            sentence_count=len(satirlar),
            word_count=len(metin.split()),
        ))

    config = {"protocol": PROTOCOL, "schema_version": SCHEMA_VERSION,
              "corpus": corpus, "max_docs": max_docs,
              "split_rule": "sha256(doc_id) % 10 -> 8/1/1"}
    imza = canonical_hash({
        "config": config,
        "documents": [
            [d.doc_id, d.split,
             hashlib.sha256(d.text.encode("utf-8")).hexdigest()]
            for d in belgeler
        ],
    })
    return LMCorpus(documents=tuple(belgeler), dataset_hash=imza,
                    config_hash=canonical_hash(config))


def document_splits_disjoint(corpus: LMCorpus) -> bool:
    gorulen: Dict[str, str] = {}
    for doc in corpus.documents:
        if doc.doc_id in gorulen and gorulen[doc.doc_id] != doc.split:
            return False
        gorulen[doc.doc_id] = doc.split
    return True


# ── N-gram kontrolleri (torch'suz) ──────────────────────────────────────────
class NgramBaselines:
    """Add-α unigram ve interpolasyonlu bigram; belge başı sanal BOS önceli.

    İnterpolasyon her iki bileşeni de add-α ile düzleştirir; böylece herhangi
    bir ``prev`` için dağılım gerçekten 1'e toplanır ve perplexity geçerli bir
    olasılık modelinden ölçülür.
    """

    def __init__(self, train_docs: Sequence[Sequence[int]], vocab_size: int,
                 bos_id: int = 2, alpha: float = NGRAM_ALPHA,
                 lam: float = BIGRAM_LAMBDA):
        self.vocab_size = int(vocab_size)
        self.bos_id = int(bos_id)
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.unigram: Counter = Counter()
        self.bigram: Counter = Counter()
        self.prev_totals: Counter = Counter()
        self.total = 0
        for doc in train_docs:
            prev = self.bos_id
            for token in doc:
                self.unigram[token] += 1
                self.bigram[(prev, token)] += 1
                self.prev_totals[prev] += 1
                self.total += 1
                prev = token

    def _p_unigram(self, token: int) -> float:
        return ((self.unigram.get(token, 0) + self.alpha)
                / (self.total + self.alpha * self.vocab_size))

    def _p_bigram(self, prev: int, token: int) -> float:
        pay = self.bigram.get((prev, token), 0) + self.alpha
        payda = self.prev_totals.get(prev, 0) + self.alpha * self.vocab_size
        return pay / payda

    def perplexity(self, docs: Sequence[Sequence[int]],
                   model: str) -> Dict[str, float]:
        if model not in NGRAM_ARMS:
            raise ValueError(f"bilinmeyen n-gram kolu: {model!r}")
        toplam_log, sayi = 0.0, 0
        for doc in docs:
            prev = self.bos_id
            for token in doc:
                if model == "unigram":
                    p = self._p_unigram(token)
                else:
                    p = (self.lam * self._p_bigram(prev, token)
                         + (1.0 - self.lam) * self._p_unigram(token))
                toplam_log += -math.log(p)
                sayi += 1
                prev = token
        if not sayi:
            raise ValueError("Perplexity için boş değerlendirme kümesi")
        ce = toplam_log / sayi
        return {"cross_entropy": round(ce, 8),
                "perplexity": round(math.exp(min(50.0, ce)), 8),
                "tokens": float(sayi)}

    def majority_accuracy(self, docs: Sequence[Sequence[int]]) -> float:
        """En sık train token'ını her yerde tahmin eden dejenere kolun skoru."""
        if not self.unigram:
            raise ValueError("Boş train korpusu")
        moda = self.unigram.most_common(1)[0][0]
        toplam = sum(len(d) for d in docs)
        dogru = sum(1 for d in docs for t in d if t == moda)
        return round(dogru / toplam, 8) if toplam else 0.0


# ── Neural kollar ───────────────────────────────────────────────────────────
def _build_windows(torch, docs: Sequence[Sequence[int]], context: int,
                   pad_id: int = 0):
    """Belge İÇİ kayan pencereler: bağlam asla belge sınırını aşmaz."""
    xs: List[List[int]] = []
    ys: List[int] = []
    for doc in docs:
        for i, target in enumerate(doc):
            baslangic = max(0, i - context)
            baglam = list(doc[baslangic:i])
            baglam = [pad_id] * (context - len(baglam)) + baglam
            xs.append(baglam)
            ys.append(int(target))
    if not xs:
        raise ValueError("Pencere üretilecek token yok")
    return (torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long))


def build_lm_models(vocab_size: int, config: Dict[str, Any]):
    """HGA'yı kur, dense/transformer'ı HGA'nın parametre bütçesine eşle.

    HGA kolu repo'nun GERÇEK bileşenlerinden kurulur (attention + geometrik
    encoder + Kronecker zinciri + fraktal decoder) — ``twt_baselines`` ile
    aynı desen. ``mimari.kuresel_model`` düz sibling import kullandığı için
    paket bağlamında bileşen bileşen import edilir; mimari aynıdır.
    """
    torch, nn = _torch()
    from mimari.decoder import FraktalDecoder
    from mimari.encoder import GeometrikVeriEncoder
    from mimari.hiper_attention import HiperGeometrikAttention
    from mimari.kuresel_bag import KureselZincir

    emb = int(config["emb_dim"])
    context = int(config["context"])
    flat = emb * context

    class HGALM(nn.Module):  # type: ignore[name-defined]
        """Gömme → nedensel attention → dış çarpım → bilinear zincir → logit."""

        def __init__(self):
            super().__init__()
            n = int(config["hga_n"])
            self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
            self.position = nn.Parameter(torch.empty(1, context, emb))
            nn.init.normal_(self.position, mean=0.0, std=0.02)
            self.attention = HiperGeometrikAttention(
                emb, int(config["heads"]), dropout=0.0, is_causal=True)
            self.encoder = GeometrikVeriEncoder(flat, n, aktivasyon="tanh")
            self.body = KureselZincir(
                n=n, katman_sayisi=int(config["hga_layers"]), dropout=0.0,
                checkpoint_kullan=False, aktivasyon="silu")
            self.norm = nn.LayerNorm(n)
            self.decoder = FraktalDecoder(n, vocab_size)

        def forward(self, token_ids):
            hidden = self.attention(self.embedding(token_ids) + self.position)
            return self.decoder(self.norm(self.body(self.encoder(hidden.flatten(1)))))

    def hga_ctor():
        return HGALM()

    def parametre(model) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    hedef = parametre(hga_ctor())

    class DenseLM(nn.Module):  # type: ignore[name-defined]
        def __init__(self, hidden: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
            self.body = nn.Sequential(
                nn.Linear(flat, hidden), nn.GELU(), nn.Linear(hidden, vocab_size))

        def forward(self, token_ids):
            return self.body(self.embedding(token_ids).flatten(1))

    class TransformerLM(nn.Module):  # type: ignore[name-defined]
        def __init__(self, ffn: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
            self.position = nn.Parameter(torch.empty(1, context, emb))
            nn.init.normal_(self.position, mean=0.0, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=emb, nhead=int(config["heads"]), dim_feedforward=ffn,
                dropout=0.0, activation="gelu", batch_first=True,
                norm_first=True)
            self.body = nn.TransformerEncoder(
                layer, num_layers=1, norm=nn.LayerNorm(emb),
                enable_nested_tensor=False)
            self.readout = nn.Linear(emb, vocab_size)

        def forward(self, token_ids):
            hidden = self.body(self.embedding(token_ids) + self.position)
            return self.readout(hidden.mean(dim=1))

    def _en_yakin(ctor, alt: int, ust: int) -> int:
        """Hedef bütçeye en yakın genişliği ikili aramayla bul."""
        while alt < ust:
            orta = (alt + ust) // 2
            if parametre(ctor(orta)) < hedef:
                alt = orta + 1
            else:
                ust = orta
        adaylar = [max(1, alt - 1), alt]
        return min(adaylar, key=lambda h: abs(parametre(ctor(h)) - hedef))

    dense_hidden = _en_yakin(DenseLM, 1, 4096)
    ffn = _en_yakin(TransformerLM, 1, 65536)

    return {
        "dense": lambda: DenseLM(dense_hidden),
        "transformer": lambda: TransformerLM(ffn),
        "hga": hga_ctor,
    }, {"dense_hidden_dim": dense_hidden, "transformer_ffn_dim": ffn,
        "target_parameters": hedef}


def _train_and_eval(torch, nn, model, train_x, train_y, evals: Dict[str, Any],
                    schedule, config: Dict[str, Any]) -> Dict[str, Any]:
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(config["learning_rate"]))
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    basla = time.perf_counter()
    for adim in range(schedule.shape[0]):
        indeksler = schedule[adim]
        optimizer.zero_grad()
        logits = model(train_x[indeksler])
        loss = loss_fn(logits, train_y[indeksler])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["gradient_clip_norm"]))
        optimizer.step()
    egitim_sn = time.perf_counter() - basla

    sonuc: Dict[str, Any] = {"training_seconds": round(egitim_sn, 4)}
    model.eval()
    for ad, (x, y) in evals.items():
        toplam_log, dogru1, dogru5, sayi = 0.0, 0, 0, 0
        with torch.no_grad():
            adim_boyu = int(config["eval_batch_size"])
            for ofset in range(0, int(x.shape[0]), adim_boyu):
                logits = model(x[ofset:ofset + adim_boyu])
                hedefler = y[ofset:ofset + adim_boyu]
                log_p = torch.log_softmax(logits, dim=-1)
                toplam_log += float(-log_p.gather(
                    1, hedefler.unsqueeze(1)).sum())
                k = min(5, logits.shape[-1])
                ilk5 = logits.topk(k, dim=-1).indices
                dogru1 += int((ilk5[:, 0] == hedefler).sum())
                dogru5 += int((ilk5 == hedefler.unsqueeze(1)).any(dim=1).sum())
                sayi += int(hedefler.shape[0])
        ce = toplam_log / max(1, sayi)
        sonuc[ad] = {
            "cross_entropy": round(ce, 8),
            "perplexity": round(math.exp(min(50.0, ce)), 8),
            "top1_accuracy": round(dogru1 / max(1, sayi), 8),
            "top5_accuracy": round(dogru5 / max(1, sayi), 8),
            "tokens": sayi,
        }
    return sonuc


# ── Rapor ───────────────────────────────────────────────────────────────────
@dataclass
class TurkishLMReport:
    protocol: str
    schema_version: int
    profile: str
    seeds: List[int]
    dataset_hash: str
    config_hash: str
    config: Dict[str, Any]
    corpus: Dict[str, Any]
    tokenizer: Dict[str, Any]
    budget: Dict[str, Any]
    ngram: Dict[str, Any]
    arms: Dict[str, Any]
    comparisons: List[Dict[str, Any]]
    perplexity: Dict[str, Any]
    checks: Dict[str, bool]
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_turkish_lm_benchmark(
    profile: str = "smoke",
    seeds: Optional[Sequence[int]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> TurkishLMReport:
    """Gerçek TWT metninde held-out dil modelleme protokolünü koş.

    Args:
        profile: ``smoke`` (küçük alt küme, hızlı) veya ``full`` (tüm TWT).
        seeds: Verilirse profil tohumlarını ezer; en az 2 tohum gerekir.
        overrides: Test/hız için profil alanı ezmeleri (hash'e dahil edilir).

    Raises:
        ValueError: bilinmeyen profil ya da yetersiz tohum.
    """
    if profile not in PROFILES:
        raise ValueError(f"profile şunlardan biri olmalı: {', '.join(PROFILES)}")
    config = dict(PROFILES[profile])
    config.update(dict(overrides or {}))
    tohumlar = [int(s) for s in (seeds if seeds is not None else config["seeds"])]
    if len(tohumlar) < 2:
        raise ValueError("çoklu-tohum raporu için en az 2 tohum gerekir")

    torch, nn = _torch()
    from mimari.bpe_tokenizer import BPETokenizer

    corpus_name = str(config.get("corpus", "twt"))
    corpus = prepare_lm_corpus(max_docs=config.get("max_docs"),
                               corpus=corpus_name)
    train_metin = "\n".join(corpus.split_texts("train"))

    tokenizer = BPETokenizer(baglam_penceresi=int(config["context"]),
                             max_vocab_size=int(config["max_vocab"]),
                             min_freq=2)
    tokenizer.fit_on_text(train_metin, verbose=False)
    vocab_size = max(int(tokenizer.sozluk_boyutu), 8)

    split_ids: Dict[str, List[List[int]]] = {}
    for split in ("train", "dev", "test"):
        split_ids[split] = [tokenizer.encode(d, pad=False)
                            for d in corpus.split_texts(split)]
    token_sayilari = {s: sum(len(d) for d in v) for s, v in split_ids.items()}

    # N-gram kontrolleri — neural sonuçların anlam zemini.
    ngram_model = NgramBaselines(split_ids["train"], vocab_size)
    ngram_sonuc = {
        arm: {split: ngram_model.perplexity(split_ids[split], arm)
              for split in ("dev", "test")}
        for arm in NGRAM_ARMS
    }
    majority_top1 = ngram_model.majority_accuracy(split_ids["test"])

    # Neural kollar: paylaşılan pencereler + tohum başına paylaşılan schedule.
    train_x, train_y = _build_windows(torch, split_ids["train"],
                                      int(config["context"]))
    evals = {
        "dev": _build_windows(torch, split_ids["dev"], int(config["context"])),
        "test": _build_windows(torch, split_ids["test"], int(config["context"])),
    }
    kurucular, eslesme = build_lm_models(vocab_size, config)

    parametreler: Dict[str, int] = {}
    kol_sonuclari: Dict[str, Any] = {name: {"per_seed": []} for name in NEURAL_ARMS}
    for tohum in tohumlar:
        uretec = torch.Generator(device="cpu").manual_seed(int(tohum) + 7_919)
        schedule = torch.randint(0, int(train_x.shape[0]),
                                 (int(config["steps"]), int(config["batch_size"])),
                                 generator=uretec)
        for kol_indeksi, name in enumerate(NEURAL_ARMS):
            torch.manual_seed(int(tohum) + 100_003 * (kol_indeksi + 1))
            model = kurucular[name]()
            parametreler[name] = sum(
                p.numel() for p in model.parameters() if p.requires_grad)
            sonuc = _train_and_eval(torch, nn, model, train_x, train_y,
                                    evals, schedule, config)
            sonuc["seed"] = int(tohum)
            kol_sonuclari[name]["per_seed"].append(sonuc)

    for name in NEURAL_ARMS:
        koseler = kol_sonuclari[name]["per_seed"]
        kol_sonuclari[name]["summary"] = {
            metric: summarize_seed_metric(
                [k["test"][metric] for k in koseler])
            for metric in ("perplexity", "top1_accuracy", "top5_accuracy")
        }

    def _ortalama(name: str, metric: str) -> float:
        return float(kol_sonuclari[name]["summary"][metric]["mean"])

    en_iyi_kol = min(NEURAL_ARMS, key=lambda a: _ortalama(a, "perplexity"))

    karsilastirmalar = [
        compare_paired(
            "test_perplexity",
            [k["test"]["perplexity"] for k in kol_sonuclari["hga"]["per_seed"]],
            [k["test"]["perplexity"] for k in kol_sonuclari[rakip]["per_seed"]],
            treatment_label="hga", baseline_label=rakip,
        ).to_dict()
        for rakip in ("dense", "transformer")
    ]

    oran = (max(parametreler.values()) / min(parametreler.values())
            if parametreler else float("inf"))
    tum_sonlu = all(
        math.isfinite(k["test"]["perplexity"]) and math.isfinite(k["test"]["cross_entropy"])
        for name in NEURAL_ARMS for k in kol_sonuclari[name]["per_seed"])

    checks = {
        # Her iki yükleyici de hash doğrular: TWT loader upstream SHA'ları,
        # tr_corpus_v1 loader dosya + canonical içerik SHA'larını.
        "real_human_corpus_hash_verified": True,
        "document_disjoint_splits": document_splits_disjoint(corpus),
        "tokenizer_fit_train_only": True,          # inşa gereği; test metni görülmez
        "parameter_budget_max_min_ratio_leq_1_05": oran <= PARAMETER_TOLERANCE,
        "finite_metrics_all_arms": tum_sonlu,
        "best_neural_beats_unigram_ppl": (
            _ortalama(en_iyi_kol, "perplexity")
            < ngram_sonuc["unigram"]["test"]["perplexity"]),
        "best_neural_beats_interpolated_bigram_ppl": (
            _ortalama(en_iyi_kol, "perplexity")
            < ngram_sonuc["bigram"]["test"]["perplexity"]),
        "best_neural_beats_majority_top1": (
            _ortalama(en_iyi_kol, "top1_accuracy") > majority_top1),
        "multi_seed_reported": len(tohumlar) >= 2,
        # Milyon-kelime kapısı: TOPLAM korpus kelimesi ≥ 1M (Brown Corpus
        # geleneğindeki "milyon-kelimelik korpus" iddiası toplam boyuttur).
        # Kelime (whitespace) sayılır, BPE token değil — BPE token sayısı
        # sözlük küçüldükçe şişer ve kapıyı yapay geçirtebilirdi. Train
        # payı ayrıca raporlanır; split oranı kapıyı geçmek için OYNANMAZ.
        "corpus_at_least_1m_words": sum(
            d.word_count for d in corpus.documents) >= 1_000_000,
    }

    perplexity_ozeti = {
        "best_arm": en_iyi_kol,
        "hga_test_ppl_mean": _ortalama("hga", "perplexity"),
        "best_neural_test_ppl_mean": _ortalama(en_iyi_kol, "perplexity"),
        "unigram_test_ppl": ngram_sonuc["unigram"]["test"]["perplexity"],
        "bigram_test_ppl": ngram_sonuc["bigram"]["test"]["perplexity"],
        "majority_top1": majority_top1,
    }

    if corpus_name == "twt":
        korpus_notu = (
            "Korpus TWT v1'dir (~66K kelime). Bu GERÇEK ve insan-anotasyonlu "
            "bir Türkçe korpustur ama milyon-kelime ölçeği değildir; "
            "`corpus_at_least_1m_words` kapısı bu yüzden bilinçli FAIL'dir.")
    else:
        korpus_notu = (
            "Korpus tr_corpus_v1'dir (UD r2.14 ×8 + Bible CC0 + TWT; "
            "≥1.1M kelime, tamamı gerçek insan metni, hash doğrulamalı). "
            "Tür dağılımı dengeli değildir: İncil çevirisi korpusun ~%40'ıdır "
            "ve dil/biçem olarak moderndir ama tematik olarak dardır. "
            "TWT hem burada hem twt_real_results_v1 arc benchmarkındadır; "
            "görevler farklıdır ve bu çapraz kullanım bilinçlidir.")
    limitations = [
        korpus_notu,
        "Ölçülen şey held-out perplexity ve next-token doğruluğudur; üretim "
        "kalitesi, talimat takibi ve sohbet kalitesi bu protokolün dışındadır.",
        "Bağlam pencereleri belge içidir; belgeler arası uzun-bağlam etkisi "
        "bu sürümde ölçülmez.",
        f"{len(tohumlar)} tohum dağılım raporlar; kesin anlamlılık iddiası "
        "için 20-tohum çekirdek protokolü gerekir.",
    ]

    return TurkishLMReport(
        protocol=PROTOCOL,
        schema_version=SCHEMA_VERSION,
        profile=profile,
        seeds=tohumlar,
        dataset_hash=corpus.dataset_hash,
        config_hash=canonical_hash({"corpus": corpus.config_hash,
                                    "config": config,
                                    "seeds": tohumlar}),
        config={k: v for k, v in config.items() if k != "seeds"},
        corpus={"splits": corpus.split_stats(), "tokens": token_sayilari,
                "document_count": len(corpus.documents)},
        tokenizer={"vocab_size": vocab_size,
                   "byte_fallback": bool(tokenizer.byte_fallback_var),
                   "vocab_hash": canonical_hash(tokenizer.kelime_to_id)},
        budget={"parameters": parametreler,
                "max_to_min_ratio": round(oran, 6),
                **eslesme},
        ngram=ngram_sonuc,
        arms=kol_sonuclari,
        comparisons=karsilastirmalar,
        perplexity=perplexity_ozeti,
        checks=checks,
        limitations=limitations,
    )


def turkish_lm_markdown(report: TurkishLMReport) -> str:
    korpus_adi = str(report.config.get("corpus", "twt"))
    korpus_basligi = ("tr_corpus_v1 — 1.11M kelime"
                      if korpus_adi == "tr_corpus_v1" else "TWT v1")
    lines = [
        f"# Gerçek Türkçe Dil Modelleme Benchmarkı ({korpus_basligi})",
        "",
        f"Protokol: `{report.protocol}` v{report.schema_version} · "
        f"korpus `{korpus_adi}` · "
        f"profil `{report.profile}` · veri imzası `{report.dataset_hash[:12]}` · "
        f"tohumlar {report.seeds}",
        "",
        "Görev: belge-ayrık held-out Türkçe metinde next-token tahmini. "
        "Tokenizer YALNIZ train metninde eğitildi.",
        "",
        "## Korpus",
        "",
        "| Split | Belge | Cümle | Kelime | BPE token |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "dev", "test"):
        istatistik = report.corpus["splits"][split]
        lines.append(
            f"| {split} | {istatistik['documents']:,} | "
            f"{istatistik['sentences']:,} | {istatistik['words']:,} | "
            f"{report.corpus['tokens'][split]:,} |")
    lines.extend([
        "",
        f"Tokenizer: BPE, sözlük {report.tokenizer['vocab_size']:,} "
        f"(byte fallback: {report.tokenizer['byte_fallback']})",
        "",
        "## Parametre bütçesi",
        "",
        "| Kol | Parametre |",
        "|---|---:|",
    ])
    for name in NEURAL_ARMS:
        lines.append(f"| `{name}` | {report.budget['parameters'][name]:,} |")
    lines.extend([
        f"| **max/min oranı** | **{report.budget['max_to_min_ratio']:.4f}** |",
        "",
        "## Test sonuçları",
        "",
        "| Kol | PPL ↓ (ort ± std) | top-1 ↑ | top-5 ↑ |",
        "|---|---:|---:|---:|",
    ])
    for name in NEURAL_ARMS:
        ozet = report.arms[name]["summary"]
        lines.append(
            f"| `{name}` | {ozet['perplexity']['mean']:.2f} "
            f"±{ozet['perplexity']['std_sample']:.2f} | "
            f"{ozet['top1_accuracy']['mean']:.4f} | "
            f"{ozet['top5_accuracy']['mean']:.4f} |")
    lines.extend([
        f"| `unigram` (add-α) | {report.ngram['unigram']['test']['perplexity']:.2f} | — | — |",
        f"| `bigram` (interp.) | {report.ngram['bigram']['test']['perplexity']:.2f} | — | — |",
        "",
        f"Dejenere çoğunluk kolu top-1: `{report.perplexity['majority_top1']:.4f}` — "
        f"en iyi neural kol: `{report.perplexity['best_arm']}`",
        "",
        "## Kabul kapıları",
        "",
    ])
    lines.extend(f"- {'PASS' if value else 'FAIL'} — `{key}`"
                 for key, value in report.checks.items())
    lines.extend(["", "## Sınırlar", ""])
    lines.extend(f"- {not_}" for not_ in report.limitations)
    return "\n".join(lines) + "\n"


__all__ = [
    "PROTOCOL", "PROFILES", "NEURAL_ARMS", "NGRAM_ARMS",
    "LMCorpus", "LMDocument", "NgramBaselines", "TurkishLMReport",
    "prepare_lm_corpus", "document_splits_disjoint", "build_lm_models",
    "run_turkish_lm_benchmark", "turkish_lm_markdown",
]
