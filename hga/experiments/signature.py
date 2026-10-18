"""HGA Signature Benchmark v1 — "HGA hangi görevde neden daha iyi?" (P0-4).

Bu repo bugüne kadar çok şeyi ayrı ayrı ölçtü: bellek, doğrulama,
kompozisyonellik, epistemik durum, Kronecker rank, Türkçe arc doğrulama.
Ama tek bir soruyu keskin biçimde cevaplamadı:

    HGA, mevcut standart mimarilerden **hangi görev sınıfında** anlamlı
    biçimde daha iyi ve bu fark hangi bileşeninden geliyor?

Signature Benchmark bu soruyu tek bir kontrollü protokolde sorar.

Görev ailesi (hepsi **bağlam-koşullu**)
--------------------------------------
Her örnek bir **bağlam** (olgu listesi) ve bir **sorgu** üçlüsünden oluşur.
Model, sorgunun bağlamdan çıkarılıp çıkarılamayacağına karar verir. Etiketler
üç sınıflıdır: ``NO`` (çürütülür), ``YES`` (çıkarılır), ``UNCERTAIN``
(bağlam karar vermeye yetmez). Çekimserlik gizlenmez, sınıftır.

===  ==========================================================
A    Uzun ilişkisel zincir (2–4 adım kompozisyon)
B    Yeni entity + bilinen relation
C    Bilinen entity + yeni relation
D    Yeni entity + yeni relation
E    Çelişki tespiti (bağlamda zıt olgular)
F    Bellek-bağımlı çıkarım (gereken olgu görünür pencerenin DIŞINDA)
G    KNOWN / UNKNOWN / UNCERTAIN ayrımı
H    Dolgu-ağırlıklı çıkarım (bağlamın çoğu alakasız)
===  ==========================================================

Kollar
------
``symbolic``          kural motoru; bilmiyorsa UNCERTAIN der, uydurmaz.
                      **Rakip değil KÂHİN tavandır**: gizli olguları ham
                      okur ve deterministik dünyada geçişli kapanış
                      hatasızdır (yapısal 1.0). İmza kapısı bu yüzden
                      öğrenen kollara bakar; kâhin farkı ``oracle_gap``
                      alanında ayrıca raporlanır.
``dense``             gömme + MLP (öğrenen rakip)
``transformer``       TransformerEncoder (öğrenen rakip; bağlamı okur)
``hga``               attention + geometrik encoder + Kronecker zinciri + bellek
``hga_no_memory``     bellek kanalı kapalı
``hga_no_kronecker``  Kronecker zinciri Identity
``hga_no_attention``  attention kapalı
``hybrid``            sembolik kesin konuşursa o kazanır, yoksa HGA

Adalet sözleşmesi
-----------------
Tüm nöral kollar **aynı** örnekleri, aynı tokenizasyonu, aynı batch sırasını,
aynı optimizer/adım/clip ayarını ve aynı çıkış başlığını kullanır. Yalnız
gövde mimarisi değişir. Parametre bütçesi oranı raporlanır ve bir adalet
kapısına bağlanır.

Görev F'nin özel kuralı
-----------------------
Görev F'de zincirin bir kenarı **görünür bağlam penceresinin dışındadır**;
yalnız seyrek belleğe yazılmıştır. Bellek kanalı olmayan hiçbir kol bu bilgiye
erişemez. Bu, "memory-grounded reasoning" iddiasının çürütülebilir testidir:
``hga`` ile ``hga_no_memory`` arasındaki fark **F'ye özgü** olmalıdır. Fark
her görevde ortaya çıkıyorsa bellek kanalı bilgi taşımıyor, sadece ek
parametre veriyor demektir — rapor bunu ayrı bir kapıyla denetler.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

TASKS = ("A_long_chain", "B_unseen_entity", "C_unseen_relation",
         "D_unseen_both", "E_conflict", "F_memory_dependent",
         "G_epistemic", "H_distractor")
NEURAL_ARMS = ("dense", "transformer", "hga", "hga_no_memory",
               "hga_no_kronecker", "hga_no_attention")
ARMS = ("symbolic",) + NEURAL_ARMS + ("hybrid",)
LABELS = ("NO", "YES", "UNCERTAIN")
NO, YES, UNCERTAIN = 0, 1, 2
PROTOCOL = "hga_signature_benchmark_v1"
RELEASE_GATE_PROTOCOL = "hga_signature_release_gate_v1"
SIGNATURE_RELEASE_MIN_SEEDS = 20
SIGNATURE_RELEASE_PROFILE = "standard"

#: Bellek kanalının fark yaratması BEKLENEN tek görev.
MEMORY_CRITICAL_TASKS = ("F_memory_dependent",)

PROFILES: Dict[str, Dict[str, Any]] = {
    "smoke": {"train_size": 256, "test_size": 128, "steps": 40,
              "batch_size": 64, "context_slots": 8, "entity_count": 40,
              "relation_count": 6},
    "standard": {"train_size": 2048, "test_size": 512, "steps": 300,
                 "batch_size": 128, "context_slots": 12, "entity_count": 120,
                 "relation_count": 10},
}

ARCH: Dict[str, Any] = {
    "embedding_dim": 16,
    "transformer_heads": 4,
    "transformer_layers": 1,
    "hga_n": 22,
    "hga_layers": 2,
    # Dense hidden ve transformer feed-forward genişliği SABİT DEĞİLDİR:
    # her görevde dizi uzunluğu değiştiği için bunlar HGA gövdesinin
    # parametre sayısına otomatik oturtulur (bkz. _fit_widths).
    "dense_hidden": None,
    "transformer_ff": None,
    "parameter_ratio_gate": 1.10,
}


def _torch() -> Tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as error:  # pragma: no cover - ortama bağlı
        raise ImportError(
            "Signature Benchmark nöral kolları için PyTorch gereklidir") from error
    return torch, nn


# ── Veri üretimi ────────────────────────────────────────────────────────────
@dataclass
class Instance:
    """Bağlam-koşullu tek örnek."""

    task: str
    context: List[Tuple[str, str, str]]       # görünür olgular
    hidden_facts: List[Tuple[str, str, str]]  # yalnız bellekte olanlar (görev F)
    query: Tuple[str, str, str]
    label: int
    unseen_entity: bool
    unseen_relation: bool
    chain_depth: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["context"] = [list(f) for f in self.context]
        d["hidden_facts"] = [list(f) for f in self.hidden_facts]
        d["query"] = list(self.query)
        return d


class _World:
    """Sentetik ilişkisel dünya: varlıklar, ilişkiler, train/test ayrımı.

    ``held_out_*`` kümeleri eğitimde HİÇ geçmez; B/C/D görevlerinin cold-start
    iddiası bu ayrıma dayanır ve ``leakage_report`` ile denetlenir.
    """

    def __init__(self, entity_count: int, relation_count: int, seed: int) -> None:
        rng = random.Random(seed)
        self.rng = rng
        self.entities = [f"e{i:04d}" for i in range(entity_count)]
        self.relations = [f"r{i:02d}" for i in range(relation_count)]
        held_e = max(4, entity_count // 5)
        held_r = max(2, relation_count // 5)
        self.held_out_entities = set(rng.sample(self.entities, held_e))
        self.held_out_relations = set(rng.sample(self.relations, held_r))
        self.train_entities = [e for e in self.entities
                               if e not in self.held_out_entities]
        self.train_relations = [r for r in self.relations
                                if r not in self.held_out_relations]
        if len(self.train_entities) < 8 or len(self.train_relations) < 2:
            raise ValueError("dünya çok küçük: held-out sonrası varlık/ilişki kalmadı")


def _chain_context(world: _World, entities: Sequence[str], relation: str,
                   depth: int, rng: random.Random) -> Tuple[List[Tuple[str, str, str]], str, str]:
    """``depth`` kenarlı zincir kur; (olgular, baş, son) döndür."""
    dugumler = rng.sample(list(entities), depth + 1)
    olgular = [(dugumler[i], relation, dugumler[i + 1]) for i in range(depth)]
    return olgular, dugumler[0], dugumler[-1]


def _distractors(world: _World, entities: Sequence[str],
                 relations: Sequence[str], count: int,
                 rng: random.Random,
                 signal_relation: Optional[str] = None,
                 protected_nodes: Collection[str] = ()) -> List[Tuple[str, str, str]]:
    """Etiketi DEĞİŞTİRMEYEN dolgu olgular üret.

    Dolgunun işi ilgisiz hacim olmaktır, sessizce cevabı değiştirmek değil.
    ``signal_relation`` verilirse, o ilişki üzerinden ``protected_nodes``
    içinden ÇIKAN kenar üretilmez: aksi halde rastgele bir dolgu ya zincire
    sahte bir yol ekler (NO → YES) ya da sorgu öznesinde sahte çelişki
    yaratır (YES → UNCERTAIN). Aynı ilişki üzerinden zincir DIŞI düğümlerden
    çıkan kenarlar serbesttir; gürültü böylece gerçek kalır.
    """
    korunan = set(protected_nodes)
    cikti: List[Tuple[str, str, str]] = []
    havuz = list(entities)
    iliski_havuzu = list(relations)
    alternatifler = [r for r in iliski_havuzu if r != signal_relation]
    for _ in range(count):
        a, b = rng.sample(havuz, 2)
        r = rng.choice(iliski_havuzu)
        if r == signal_relation and a in korunan:
            if alternatifler:
                r = rng.choice(alternatifler)
            else:
                disarisi = [e for e in havuz if e not in korunan and e != b]
                if not disarisi:
                    continue
                a = rng.choice(disarisi)
        cikti.append((a, r, b))
    return cikti


def _make_instance(task: str, world: _World, split: str,
                   context_slots: int, rng: random.Random) -> Instance:
    """Tek örnek üret. ``split`` train ise held-out varlık/ilişki KULLANILMAZ."""
    if split == "train":
        ents, rels = world.train_entities, world.train_relations
    else:
        ents, rels = world.entities, world.relations

    unseen_e = unseen_r = False
    hidden: List[Tuple[str, str, str]] = []

    if task == "A_long_chain":
        depth = rng.choice([2, 3, 4])
        rel = rng.choice(world.train_relations)
        olgular, bas, son = _chain_context(world, ents, rel, depth, rng)
        if rng.random() < 0.5:
            sorgu, etiket = (bas, rel, son), YES
        else:
            # Zincirin bir kenarını kaldır → çıkarılamaz.
            olgular = olgular[:-1]
            sorgu, etiket = (bas, rel, son), NO
        dolgu = max(0, context_slots - len(olgular))
        dugumler = {f[0] for f in olgular} | {f[2] for f in olgular} | {bas, son}
        baglam = olgular + _distractors(world, ents, rels, dolgu, rng,
                                        signal_relation=rel,
                                        protected_nodes=dugumler)
        rng.shuffle(baglam)
        return Instance(task, baglam, hidden, sorgu, etiket, False, False, depth)

    if task in ("B_unseen_entity", "C_unseen_relation", "D_unseen_both"):
        if task in ("B_unseen_entity", "D_unseen_both") and split == "test":
            havuz = sorted(world.held_out_entities)
            unseen_e = True
        else:
            havuz = world.train_entities
        if task in ("C_unseen_relation", "D_unseen_both") and split == "test":
            rel = rng.choice(sorted(world.held_out_relations))
            unseen_r = True
        else:
            rel = rng.choice(world.train_relations)
        if len(havuz) < 3:
            havuz = list(havuz) + world.train_entities
        depth = 2
        olgular, bas, son = _chain_context(world, havuz, rel, depth, rng)
        if rng.random() < 0.5:
            sorgu, etiket = (bas, rel, son), YES
        else:
            olgular = olgular[:-1]
            sorgu, etiket = (bas, rel, son), NO
        dolgu = max(0, context_slots - len(olgular))
        dugumler = {f[0] for f in olgular} | {f[2] for f in olgular} | {bas, son}
        baglam = olgular + _distractors(world, ents, rels, dolgu, rng,
                                        signal_relation=rel,
                                        protected_nodes=dugumler)
        rng.shuffle(baglam)
        return Instance(task, baglam, hidden, sorgu, etiket, unseen_e, unseen_r, depth)

    if task == "E_conflict":
        rel = rng.choice(world.train_relations)
        a, b, c = rng.sample(list(ents), 3)
        if rng.random() < 0.5:
            # Aynı özne-ilişki için iki farklı nesne → çelişki (UNCERTAIN).
            baglam = [(a, rel, b), (a, rel, c)]
            sorgu, etiket = (a, rel, b), UNCERTAIN
        else:
            baglam = [(a, rel, b)]
            sorgu, etiket = (a, rel, b), YES
        dolgu = max(0, context_slots - len(baglam))
        # Çelişki KASITLI olmalı: dolgu ne yeni çelişki yaratabilir ne de
        # var olanı bulanıklaştırabilir.
        baglam = baglam + _distractors(world, ents, rels, dolgu, rng,
                                       signal_relation=rel,
                                       protected_nodes={a, b, c})
        rng.shuffle(baglam)
        return Instance(task, baglam, hidden, sorgu, etiket, False, False, 1)

    if task == "F_memory_dependent":
        rel = rng.choice(world.train_relations)
        depth = 3
        olgular, bas, son = _chain_context(world, ents, rel, depth, rng)
        # Zincirin ORTA kenarı görünür bağlamdan çıkarılır; yalnız belleğe yazılır.
        gizli_index = depth // 2
        gizli = olgular[gizli_index]
        gorunur = [f for i, f in enumerate(olgular) if i != gizli_index]
        if rng.random() < 0.5:
            hidden = [gizli]
            etiket = YES
        else:
            # Gizli kenar hiç yok: ne bağlamda ne bellekte → çıkarılamaz.
            hidden = []
            etiket = NO
        dolgu = max(0, context_slots - len(gorunur))
        dugumler = {f[0] for f in olgular} | {f[2] for f in olgular} | {bas, son}
        baglam = gorunur + _distractors(world, ents, rels, dolgu, rng,
                                        signal_relation=rel,
                                        protected_nodes=dugumler)
        rng.shuffle(baglam)
        return Instance(task, baglam, hidden, (bas, rel, son), etiket,
                        False, False, depth)

    if task == "G_epistemic":
        rel = rng.choice(world.train_relations)
        a, b = rng.sample(list(ents), 2)
        secim = rng.random()
        if secim < 1 / 3:
            baglam = [(a, rel, b)]
            sorgu, etiket = (a, rel, b), YES
        elif secim < 2 / 3:
            c = rng.choice([e for e in ents if e not in (a, b)])
            baglam = [(a, rel, c)]
            sorgu, etiket = (a, rel, b), NO     # aynı özne, farklı nesne biliniyor
        else:
            baglam = []
            sorgu, etiket = (a, rel, b), UNCERTAIN   # hiçbir kanıt yok
        dolgu = max(0, context_slots - len(baglam))
        # Dolgular sorguyla İLGİSİZ olmalı; aksi halde UNCERTAIN etiketi bozulur.
        ilgisiz = [f for f in _distractors(world, ents, rels, dolgu * 2, rng,
                                           signal_relation=rel,
                                           protected_nodes={a, b})
                   if f[0] != sorgu[0] or f[1] != sorgu[1]][:dolgu]
        baglam = baglam + ilgisiz
        rng.shuffle(baglam)
        return Instance(task, baglam, hidden, sorgu, etiket, False, False, 1)

    if task == "H_distractor":
        rel = rng.choice(world.train_relations)
        depth = 2
        olgular, bas, son = _chain_context(world, ents, rel, depth, rng)
        if rng.random() < 0.5:
            sorgu, etiket = (bas, rel, son), YES
        else:
            olgular = olgular[:-1]
            sorgu, etiket = (bas, rel, son), NO
        # Bağlamın büyük çoğunluğu dolgu; sinyal/gürültü oranı kasıtla düşük.
        dolgu = max(0, context_slots * 3 - len(olgular))
        dugumler = {f[0] for f in olgular} | {f[2] for f in olgular} | {bas, son}
        baglam = olgular + _distractors(world, ents, rels, dolgu, rng,
                                        signal_relation=rel,
                                        protected_nodes=dugumler)
        rng.shuffle(baglam)
        return Instance(task, baglam, hidden, sorgu, etiket, False, False, depth)

    raise ValueError(f"bilinmeyen görev: {task}")


def build_dataset(task: str, seed: int, train_size: int, test_size: int,
                  context_slots: int, entity_count: int,
                  relation_count: int) -> Tuple[List[Instance], List[Instance], _World]:
    """Bir görev için train/test üret; test held-out varlık/ilişki içerebilir."""
    if task not in TASKS:
        raise ValueError(f"bilinmeyen görev: {task}")
    world = _World(entity_count, relation_count, seed)
    rng = random.Random(seed * 7919 + 13)
    train = [_make_instance(task, world, "train", context_slots, rng)
             for _ in range(train_size)]
    test = [_make_instance(task, world, "test", context_slots, rng)
            for _ in range(test_size)]
    return train, test, world


def leakage_report(task: str, train: Sequence[Instance],
                   test: Sequence[Instance], world: _World) -> Dict[str, Any]:
    """Held-out varlık/ilişkilerin eğitim tarafına sızmadığını denetle."""
    def semboller(ornekler: Sequence[Instance]) -> Tuple[set, set]:
        e, r = set(), set()
        for ornek in ornekler:
            for f in list(ornek.context) + list(ornek.hidden_facts) + [ornek.query]:
                e.add(f[0])
                e.add(f[2])
                r.add(f[1])
        return e, r

    train_e, train_r = semboller(train)
    sizinti_e = train_e & world.held_out_entities
    sizinti_r = train_r & world.held_out_relations
    test_e, test_r = semboller(test)
    return {
        "task": task,
        "train_entities": len(train_e),
        "train_relations": len(train_r),
        "held_out_entity_leak": sorted(sizinti_e),
        "held_out_relation_leak": sorted(sizinti_r),
        "clean": not sizinti_e and not sizinti_r,
        "test_uses_held_out_entities": bool(test_e & world.held_out_entities),
        "test_uses_held_out_relations": bool(test_r & world.held_out_relations),
    }


# ── Sembolik kol (öğrenmez; kural motoru) ───────────────────────────────────
def symbolic_predict(instance: Instance, use_memory: bool = True) -> int:
    """Bağlam (+ istenirse bellek) üzerinde geçişli kapanış ile karar ver.

    UNCERTAIN yalnız iki durumda döner: sorgunun öznesi/ilişkisi hakkında
    hiçbir kayıt yoksa, ya da bağlam çelişkiliyse. Aksi halde kesin konuşur.
    """
    olgular = list(instance.context) + (list(instance.hidden_facts) if use_memory else [])
    ozne, iliski, nesne = instance.query

    # Çelişki: aynı (özne, ilişki) için birden fazla farklı nesne doğrudan kayıtlı.
    dogrudan = {f[2] for f in olgular if f[0] == ozne and f[1] == iliski}
    if len(dogrudan) > 1:
        return UNCERTAIN
    if nesne in dogrudan:
        return YES

    # Geçişli kapanış: aynı ilişki üzerinden yol var mı?
    kenarlar: Dict[str, set] = {}
    for a, r, b in olgular:
        if r == iliski:
            kenarlar.setdefault(a, set()).add(b)
    gorulen = {ozne}
    sinir = [ozne]
    while sinir:
        dugum = sinir.pop()
        for komsu in kenarlar.get(dugum, ()):
            if komsu == nesne:
                return YES
            if komsu not in gorulen:
                gorulen.add(komsu)
                sinir.append(komsu)

    # Özne hakkında bu ilişkide hiçbir kayıt yoksa "bilmiyorum" demek dürüsttür.
    if not any(f[0] == ozne and f[1] == iliski for f in olgular):
        return UNCERTAIN
    return NO


# ── Nöral kollar ────────────────────────────────────────────────────────────
def _vocabulary(train: Sequence[Instance]) -> Dict[str, int]:
    """Sözlük YALNIZ eğitim örneklerinden kurulur; test OOV görür (0 = UNK)."""
    semboller: set = set()
    for ornek in train:
        for f in list(ornek.context) + [ornek.query]:
            semboller.update(f)
    return {s: i + 1 for i, s in enumerate(sorted(semboller))}


def _encode(instances: Sequence[Instance], vocab: Dict[str, int],
            max_facts: int) -> Tuple[List[List[int]], List[List[float]], List[int]]:
    """(token dizisi, bellek kanalı, etiket) — tüm kollar aynı kodlamayı görür.

    Bellek kanalı iki skaler taşır: (a) sorgunun öznesinden çıkan gizli bir
    kenar bellekte var mı, (b) bellek zinciri tamamlıyor mu. Bu kanal
    ``hga_no_memory`` kolunda sıfırlanır — mimari fark değil, **bilgi** farkı
    ölçülsün diye.
    """
    diziler: List[List[int]] = []
    bellek: List[List[float]] = []
    etiketler: List[int] = []
    for ornek in instances:
        tokenlar: List[int] = []
        for f in ornek.context[:max_facts]:
            tokenlar.extend(vocab.get(s, 0) for s in f)
        while len(tokenlar) < max_facts * 3:
            tokenlar.append(0)
        tokenlar.extend(vocab.get(s, 0) for s in ornek.query)
        diziler.append(tokenlar)
        gizli_var = 1.0 if ornek.hidden_facts else 0.0
        tamamlar = 1.0 if (ornek.hidden_facts
                           and symbolic_predict(ornek, use_memory=True) == YES
                           and symbolic_predict(ornek, use_memory=False) != YES) else 0.0
        bellek.append([gizli_var, tamamlar])
        etiketler.append(ornek.label)
    return diziler, bellek, etiketler


def _body_parameters(model, nn) -> int:
    """Paylaşılan gömme/pozisyon hariç, yalnız gövde parametreleri."""
    paylasilan = 0
    for ad, parametre in model.named_parameters():
        if ad.startswith("embedding.") or ad == "position":
            paylasilan += parametre.numel()
    toplam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(toplam - paylasilan)


def _fit_widths(vocab_size: int, seq_len: int, torch, nn,
                tolerance: float = 0.02) -> Dict[str, int]:
    """Dense ve Transformer genişliğini HGA gövdesine oturt.

    Referans HGA'dır: onun gövde parametre sayısı hedeftir. Diğer kolların
    tek serbest genişlik parametresi ikili arama ile bu hedefe yaklaştırılır.
    Böylece "HGA daha iyi" sonucu daha fazla parametreden gelemez.
    """
    hedef = _body_parameters(
        _build_neural_arm("hga", vocab_size, seq_len, torch, nn,
                          widths={"dense_hidden": 8, "transformer_ff": 8}), nn)

    def ara(arm: str, anahtar: str) -> int:
        alt, ust = 1, 8
        while True:
            model = _build_neural_arm(arm, vocab_size, seq_len, torch, nn,
                                      widths={anahtar: ust,
                                              "dense_hidden": ust,
                                              "transformer_ff": ust})
            if _body_parameters(model, nn) >= hedef or ust > 1 << 20:
                break
            alt, ust = ust, ust * 2
        while alt < ust:
            orta = (alt + ust) // 2
            model = _build_neural_arm(arm, vocab_size, seq_len, torch, nn,
                                      widths={anahtar: orta,
                                              "dense_hidden": orta,
                                              "transformer_ff": orta})
            if _body_parameters(model, nn) < hedef:
                alt = orta + 1
            else:
                ust = orta
        # İkili arama hedefi AŞAN en küçük genişliği verir; granülerlik kabaysa
        # bu %10 kapısını tek başına deler. Bir alt genişlik hedefe oransal
        # olarak daha yakınsa onu seç — amaç eşitlik, üstünlük değil.
        def sapma(genislik: int) -> float:
            model = _build_neural_arm(arm, vocab_size, seq_len, torch, nn,
                                      widths={anahtar: genislik,
                                              "dense_hidden": genislik,
                                              "transformer_ff": genislik})
            oran = _body_parameters(model, nn) / max(1, hedef)
            return abs(math.log(max(oran, 1e-9)))

        en_iyi = max(1, alt)
        if en_iyi > 1 and sapma(en_iyi - 1) < sapma(en_iyi):
            en_iyi -= 1
        return en_iyi

    return {
        "dense_hidden": ara("dense", "dense_hidden"),
        "transformer_ff": ara("transformer", "transformer_ff"),
    }


def _build_neural_arm(arm: str, vocab_size: int, seq_len: int, torch, nn,
                      widths: Optional[Dict[str, int]] = None):
    from mimari.decoder import FraktalDecoder
    from mimari.encoder import GeometrikVeriEncoder
    from mimari.hiper_attention import HiperGeometrikAttention
    from mimari.kuresel_bag import KureselZincir

    emb = int(ARCH["embedding_dim"])
    flat = emb * seq_len
    memory_dim = 2
    widths = widths or {}
    dense_hidden = int(widths.get("dense_hidden") or 128)
    transformer_ff = int(widths.get("transformer_ff") or 232)

    class Dense(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, emb)
            hidden = dense_hidden
            self.body = nn.Sequential(
                nn.Linear(flat + memory_dim, hidden), nn.GELU(),
                nn.Linear(hidden, 3))

        def forward(self, ids, mem):
            return self.body(torch.cat([self.embedding(ids).flatten(1), mem], dim=-1))

    class Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, emb)
            self.position = nn.Parameter(torch.zeros(1, seq_len, emb))
            nn.init.normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=emb, nhead=int(ARCH["transformer_heads"]),
                dim_feedforward=transformer_ff, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(
                layer, num_layers=int(ARCH["transformer_layers"]),
                norm=nn.LayerNorm(emb), enable_nested_tensor=False)
            self.readout = nn.Linear(emb + memory_dim, 3)

        def forward(self, ids, mem):
            hidden = self.body(self.embedding(ids) + self.position)
            return self.readout(torch.cat([hidden.mean(dim=1), mem], dim=-1))

    class HGA(nn.Module):
        def __init__(self, ablation: str) -> None:
            super().__init__()
            self.ablation = ablation
            n = int(ARCH["hga_n"])
            self.n = n
            self.embedding = nn.Embedding(vocab_size, emb)
            self.position = nn.Parameter(torch.zeros(1, seq_len, emb))
            nn.init.normal_(self.position, std=0.02)
            self.attention = (None if ablation == "no_attention"
                              else HiperGeometrikAttention(
                                  emb, int(ARCH["transformer_heads"]),
                                  dropout=0.0, is_causal=False))
            self.encoder = GeometrikVeriEncoder(flat, n, aktivasyon="tanh")
            self.body = (nn.Identity() if ablation == "no_kronecker"
                         else KureselZincir(n=n, katman_sayisi=int(ARCH["hga_layers"]),
                                            dropout=0.0, checkpoint_kullan=False,
                                            aktivasyon="silu"))
            self.norm = nn.LayerNorm(n)
            self.memory_gate = nn.Linear(memory_dim, n)
            self.decoder = FraktalDecoder(n, 3)

        def forward(self, ids, mem):
            hidden = self.embedding(ids) + self.position
            if self.attention is not None:
                hidden = self.attention(hidden)
            matrix = self.encoder(hidden.flatten(1))
            matrix = self.norm(self.body(matrix))
            # Bellek kanalı satır ölçeği olarak girer; no_memory kolunda mem=0'dır.
            matrix = matrix + self.memory_gate(mem).unsqueeze(-1)
            return self.decoder(matrix)

    if arm == "dense":
        return Dense()
    if arm == "transformer":
        return Transformer()
    if arm == "hga":
        return HGA("full")
    if arm == "hga_no_memory":
        return HGA("full")           # mimari aynı; bellek girdisi sıfırlanır
    if arm == "hga_no_kronecker":
        return HGA("no_kronecker")
    if arm == "hga_no_attention":
        return HGA("no_attention")
    raise ValueError(f"bilinmeyen nöral kol: {arm}")


def _metrics(gold: Sequence[int], pred: Sequence[int]) -> Dict[str, float]:
    toplam = len(gold)
    if toplam == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "uncertain_rate": 0.0,
                "hallucination_rate": 0.0}
    dogru = sum(1 for g, p in zip(gold, pred) if g == p)
    f1s = []
    for sinif in (NO, YES, UNCERTAIN):
        tp = sum(1 for g, p in zip(gold, pred) if g == sinif and p == sinif)
        fp = sum(1 for g, p in zip(gold, pred) if g != sinif and p == sinif)
        fn = sum(1 for g, p in zip(gold, pred) if g == sinif and p != sinif)
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    # Halüsinasyon: gerçekte çıkarılamayan bir şeye kesin "YES" demek.
    halusinasyon = sum(1 for g, p in zip(gold, pred)
                       if p == YES and g in (NO, UNCERTAIN))
    return {
        "accuracy": round(dogru / toplam, 6),
        "macro_f1": round(statistics.fmean(f1s), 6) if f1s else 0.0,
        "uncertain_rate": round(sum(1 for p in pred if p == UNCERTAIN) / toplam, 6),
        "hallucination_rate": round(halusinasyon / toplam, 6),
    }


def _majority_baseline(train: Sequence[Instance],
                       test: Sequence[Instance]) -> Dict[str, float]:
    """Negatif kontrol: bağlamı hiç okumayan çoğunluk sınıfı kolu."""
    sayim: Dict[int, int] = {}
    for ornek in train:
        sayim[ornek.label] = sayim.get(ornek.label, 0) + 1
    cogunluk = max(sayim, key=lambda k: sayim[k]) if sayim else NO
    return _metrics([o.label for o in test], [cogunluk] * len(test))


@dataclass
class SignatureReport:
    protocol: str
    profile: str
    seeds: List[int]
    tasks: List[str]
    arms: List[str]
    dataset_hash: str
    config: Dict[str, Any]
    parameters: Dict[str, int]
    body_parameters: Dict[str, int]
    parameter_ratio: float
    body_parameter_ratio: float
    leakage: Dict[str, Dict[str, Any]]
    results: Dict[str, Dict[str, Dict[str, float]]]   # görev → kol → metrik
    majority_baseline: Dict[str, Dict[str, float]]
    signature: Dict[str, Any]
    checks: Dict[str, bool]
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        return signature_markdown(self)


@dataclass
class SignatureReleaseGateReport:
    protocol: str
    source_protocol: str
    source_dataset_hash: str
    status: str
    required_profile: str
    min_seed_count: int
    checks: Dict[str, bool]
    failed_checks: List[str]
    evidence: Dict[str, Any]
    limitations: List[str] = field(default_factory=list)

    @property
    def release_ready(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> Dict[str, Any]:
        veri = asdict(self)
        veri["release_ready"] = self.release_ready
        return veri

    def markdown(self) -> str:
        return signature_release_gate_markdown(self)


def _mean(values: Sequence[float]) -> float:
    return round(statistics.fmean(values), 6) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return round(statistics.stdev(values), 6) if len(values) > 1 else 0.0


def run_signature_benchmark(
    profile: str = "smoke",
    seeds: Sequence[int] = (1, 2, 3),
    tasks: Sequence[str] = TASKS,
    arms: Sequence[str] = ARMS,
    learning_rate: float = 0.003,
    device: str = "cpu",
) -> SignatureReport:
    """Signature Benchmark'ı koş: görev × kol × tohum ızgarası."""
    if profile not in PROFILES:
        raise ValueError(f"bilinmeyen profil: {profile}")
    tasks = list(tasks)
    arms = list(arms)
    seeds = [int(s) for s in seeds]
    if not tasks or not arms or not seeds:
        raise ValueError("tasks, arms ve seeds boş olamaz")
    bilinmeyen = [a for a in arms if a not in ARMS]
    if bilinmeyen:
        raise ValueError(f"bilinmeyen kollar: {bilinmeyen}")
    bilinmeyen_t = [t for t in tasks if t not in TASKS]
    if bilinmeyen_t:
        raise ValueError(f"bilinmeyen görevler: {bilinmeyen_t}")

    ayar = PROFILES[profile]
    neural = [a for a in arms if a in NEURAL_ARMS]
    torch: Any = None
    nn: Any = None
    if neural or "hybrid" in arms:
        torch, nn = _torch()
    hedef = torch.device(device) if torch is not None else None

    ham: Dict[str, Dict[str, List[Dict[str, float]]]] = {
        t: {a: [] for a in arms} for t in tasks}
    cogunluk: Dict[str, List[Dict[str, float]]] = {t: [] for t in tasks}
    sizinti: Dict[str, Dict[str, Any]] = {}
    parametreler: Dict[str, int] = {}
    govde_parametreleri: Dict[str, int] = {}

    for task in tasks:
        for seed in seeds:
            train, test, world = build_dataset(
                task, seed, ayar["train_size"], ayar["test_size"],
                ayar["context_slots"], ayar["entity_count"], ayar["relation_count"])
            if seed == seeds[0]:
                sizinti[task] = leakage_report(task, train, test, world)
            cogunluk[task].append(_majority_baseline(train, test))
            gold = [o.label for o in test]

            sembolik_tahmin: Optional[List[int]] = None
            if "symbolic" in arms or "hybrid" in arms:
                sembolik_tahmin = [symbolic_predict(o) for o in test]
            if "symbolic" in arms and sembolik_tahmin is not None:
                ham[task]["symbolic"].append(_metrics(gold, sembolik_tahmin))

            if not neural and "hybrid" not in arms:
                continue

            vocab = _vocabulary(train)
            max_facts = max(len(o.context) for o in train + test)
            seq_len = max_facts * 3 + 3
            tr_ids, tr_mem, tr_y = _encode(train, vocab, max_facts)
            te_ids, te_mem, te_y = _encode(test, vocab, max_facts)
            tr_ids_t = torch.tensor(tr_ids, dtype=torch.long, device=hedef)
            tr_mem_t = torch.tensor(tr_mem, dtype=torch.float32, device=hedef)
            tr_y_t = torch.tensor(tr_y, dtype=torch.long, device=hedef)
            te_ids_t = torch.tensor(te_ids, dtype=torch.long, device=hedef)
            te_mem_t = torch.tensor(te_mem, dtype=torch.float32, device=hedef)

            genislikler = _fit_widths(len(vocab) + 1, seq_len, torch, nn)
            hga_tahmin: Optional[List[int]] = None
            for arm in neural + (["hga"] if "hybrid" in arms and "hga" not in neural else []):
                torch.manual_seed(seed)
                model = _build_neural_arm(
                    arm, len(vocab) + 1, seq_len, torch, nn,
                    widths=genislikler).to(hedef)
                parametreler[arm] = sum(p.numel() for p in model.parameters()
                                        if p.requires_grad)
                govde_parametreleri[arm] = _body_parameters(model, nn)
                mem_tr = torch.zeros_like(tr_mem_t) if arm == "hga_no_memory" else tr_mem_t
                mem_te = torch.zeros_like(te_mem_t) if arm == "hga_no_memory" else te_mem_t
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                              weight_decay=0.01)
                kayip_fn = nn.CrossEntropyLoss()
                gen = torch.Generator(device="cpu").manual_seed(seed)
                for _ in range(int(ayar["steps"])):
                    idx = torch.randint(0, len(tr_y), (int(ayar["batch_size"]),),
                                        generator=gen).to(hedef)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(tr_ids_t[idx], mem_tr[idx])
                    loss = kayip_fn(logits, tr_y_t[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    tahmin = model(te_ids_t, mem_te).argmax(dim=-1).tolist()
                if arm in ham[task]:
                    ham[task][arm].append(_metrics(te_y, tahmin))
                if arm == "hga":
                    hga_tahmin = tahmin

            if "hybrid" in arms and sembolik_tahmin is not None and hga_tahmin is not None:
                karma = [s if s != UNCERTAIN else h
                         for s, h in zip(sembolik_tahmin, hga_tahmin)]
                ham[task]["hybrid"].append(_metrics(te_y, karma))

    # ── özet ───────────────────────────────────────────────────────────────
    sonuclar: Dict[str, Dict[str, Dict[str, float]]] = {}
    for task in tasks:
        sonuclar[task] = {}
        for arm in arms:
            kosular = ham[task][arm]
            if not kosular:
                continue
            ozet: Dict[str, Any] = {}
            for metrik in kosular[0]:
                degerler = [k[metrik] for k in kosular]
                ozet[f"{metrik}_mean"] = _mean(degerler)
                ozet[f"{metrik}_std"] = _std(degerler)
                # Tohum-başı ham değerler: mean/std tek başına güven aralığı
                # ya da eşleşmiş anlamlılık testi üretmeye yetmez.
                ozet[f"per_seed_{metrik}"] = [round(d, 9) for d in degerler]
            sonuclar[task][arm] = ozet

    cogunluk_ozet = {
        task: {m: _mean([k[m] for k in kosular]) for m in kosular[0]}
        for task, kosular in cogunluk.items() if kosular
    }

    # ── imza analizi: HGA nerede kazanıyor, fark nereden geliyor? ──────────
    # İKİ AYRI soru ölçülür ve karıştırılmaz:
    #
    # 1. ÖĞRENEN rakipler (dense, transformer — aynı bilgi, aynı parametre
    #    bütçesi, aynı eğitim) arasında HGA'nın imza görevi var mı?
    # 2. Sembolik KÂHİNE (oracle) uzaklık ne kadar?
    #
    # `symbolic` bir rakip DEĞİL, tavandır: gizli olguları HAM biçimde okur
    # (nöral kollara yalnız 2 skalerlik özet kanal verilir) ve deterministik
    # dünyada geçişli kapanış hatasızdır — yapısal 1.0. Kâhini "en iyi
    # rakip" saymak, imza kapısını hiçbir öğrenen sistemin geçemeyeceği bir
    # tanıma bağlar: ölçüm yanlış soruyu sorar. Kâhin farkı `oracle_gap`
    # alanında AYRICA raporlanır; gizlenmez.
    ogrenen_rakipler = [a for a in ("dense", "transformer") if a in arms]
    imza_kazanim: Dict[str, Any] = {}
    for task in tasks:
        if "hga" not in sonuclar[task]:
            continue
        hga_acc = sonuclar[task]["hga"]["accuracy_mean"]
        en_iyi_rakip = max(
            ((a, sonuclar[task][a]["accuracy_mean"]) for a in ogrenen_rakipler
             if a in sonuclar[task]),
            key=lambda item: item[1], default=(None, 0.0))
        imza_kazanim[task] = {
            "hga_accuracy": hga_acc,
            "best_competitor": en_iyi_rakip[0],
            "best_competitor_accuracy": en_iyi_rakip[1],
            "margin": round(hga_acc - en_iyi_rakip[1], 6),
            "beats_majority": hga_acc > cogunluk_ozet.get(task, {}).get("accuracy", 0.0),
        }
        if "symbolic" in sonuclar[task]:
            imza_kazanim[task]["oracle_accuracy"] = (
                sonuclar[task]["symbolic"]["accuracy_mean"])
            imza_kazanim[task]["oracle_gap"] = round(
                sonuclar[task]["symbolic"]["accuracy_mean"] - hga_acc, 6)
        for bilesen, kol in (("memory", "hga_no_memory"),
                             ("kronecker", "hga_no_kronecker"),
                             ("attention", "hga_no_attention")):
            if kol in sonuclar[task]:
                imza_kazanim[task][f"{bilesen}_contribution"] = round(
                    hga_acc - sonuclar[task][kol]["accuracy_mean"], 6)

    kazanilan = [t for t, v in imza_kazanim.items() if v["margin"] > 0.0]
    bellek_katkisi = {t: v.get("memory_contribution", 0.0)
                      for t, v in imza_kazanim.items()}
    bellek_kritik = [t for t in MEMORY_CRITICAL_TASKS if t in bellek_katkisi]
    bellek_digerleri = [t for t in bellek_katkisi if t not in MEMORY_CRITICAL_TASKS]

    p_degerleri = [v for v in parametreler.values() if v > 0]
    oran = round(max(p_degerleri) / min(p_degerleri), 6) if p_degerleri else 1.0
    g_degerleri = [v for v in govde_parametreleri.values() if v > 0]
    govde_orani = (round(max(g_degerleri) / min(g_degerleri), 6)
                   if g_degerleri else 1.0)

    kontroller = {
        "parameter_budget_within_gate": oran <= float(ARCH["parameter_ratio_gate"]),
        "body_parameter_budget_within_gate": (
            govde_orani <= float(ARCH["parameter_ratio_gate"])),
        "no_held_out_leak": all(v["clean"] for v in sizinti.values()),
        # Her kol çoğunluk kolunu yenmeli; yenmiyorsa görev öğrenilmemiştir.
        "hga_beats_majority_everywhere": all(
            v["beats_majority"] for v in imza_kazanim.values()),
        # HGA en az bir görevde en iyi rakibini geçmeli — "signature" budur.
        "hga_has_signature_task": bool(kazanilan),
        # Nöral kollar görevleri gerçekten öğrenebiliyor mu? Hepsi şans
        # seviyesinde kalıyorsa mimari karşılaştırması anlamsızdır: ölçülen
        # şey mimari farkı değil, ortak öğrenilemezliktir.
        "neural_arms_learn_above_chance": any(
            sonuclar[t].get(a, {}).get("accuracy_mean", 0.0)
            > cogunluk_ozet.get(t, {}).get("accuracy", 0.0) + 0.10
            for t in tasks for a in NEURAL_ARMS if a in sonuclar[t]),
        # Bellek katkısı F'ye ÖZGÜ olmalı: F'de pozitif, diğerlerinde ihmal
        # edilebilir. Aksi halde kanal bilgi değil kapasite taşıyor demektir.
        "memory_gain_is_task_specific": (
            all(bellek_katkisi.get(t, 0.0) > 0.05 for t in bellek_kritik)
            and all(abs(bellek_katkisi.get(t, 0.0)) <= 0.05
                    for t in bellek_digerleri)
            if bellek_kritik else False),
    }

    imza = hashlib.sha256(json.dumps({
        "protocol": PROTOCOL, "profile": profile, "seeds": seeds,
        "tasks": tasks, "arms": arms, "config": ayar, "arch": ARCH,
    }, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    bulgular = [
        f"Profil `{profile}`: {len(tasks)} görev × {len(arms)} kol × "
        f"{len(seeds)} tohum.",
    ]
    if kazanilan:
        detay = ", ".join(
            f"{t} (+{imza_kazanim[t]['margin']:.4f} vs "
            f"{imza_kazanim[t]['best_competitor']})" for t in kazanilan)
        bulgular.append(f"HGA'nın rakiplerini geçtiği görevler: {detay}.")
    else:
        bulgular.append(
            "HGA hiçbir görevde en iyi rakibini geçmedi — bu profilde imza "
            "görev bulunamadı. Sonuç gizlenmiyor.")
    if bellek_kritik:
        bulgular.append(
            "Bellek kanalının katkısı: "
            + ", ".join(f"{t}: {bellek_katkisi[t]:+.4f}"
                        for t in sorted(bellek_katkisi))
            + ". Katkı yalnız F'de belirgin değilse kanal bilgi değil kapasite "
              "taşıyor demektir.")

    # Nöral kolların şans seviyesine yakınlığı açıkça bulgu olarak yazılır.
    sans_ustu: Dict[str, List[str]] = {}
    for task in tasks:
        taban = cogunluk_ozet.get(task, {}).get("accuracy", 0.0)
        sans_ustu[task] = [a for a in NEURAL_ARMS
                           if a in sonuclar[task]
                           and sonuclar[task][a]["accuracy_mean"] > taban + 0.10]
    ogrenilemeyen = [t for t, kollar in sans_ustu.items() if not kollar]
    if ogrenilemeyen:
        bulgular.append(
            f"Şu görevlerde HİÇBİR parameter-matched nöral kol çoğunluk "
            f"tabanını +0.10'dan fazla geçemedi: {ogrenilemeyen}. Bu, mimari "
            f"farkından önce gelen bir sonuçtur: bu bütçe ve adım sayısında "
            f"görev nöral kollar için öğrenilemiyor; sembolik kolla "
            f"karşılaştırma paradigma farkı olarak okunmalıdır.")

    sinirlar = [
        "Görevler sentetik ilişkisel dünyalardır; doğal dil veya gerçek bilgi "
        "grafiği sonucu DEĞİLDİR.",
        "Sembolik kol öğrenmez; doğruluğu görev tanımının kendisinden gelir ve "
        "nöral kollarla aynı anlamda 'model performansı' değildir.",
        "Bellek kanalı iki skaler özetle temsil edilir; tam bir bellek geri "
        "çağırma mimarisi değildir. Ölçtüğü şey bilginin ERİŞİLEBİLİRLİĞİdir.",
        f"Parametre bütçesi oranı {oran:.3f} (gövde {govde_orani:.3f}); "
        f"gömme katmanı sözlük boyutuna bağlı olduğu için tam eşitlik değil, "
        f"kapı ({ARCH['parameter_ratio_gate']}) hedeflenir.",
        f"Tohum sayısı {len(seeds)}; çekirdek iddia için 20 tohum ayrıca "
        "koşulmalıdır.",
    ]

    return SignatureReport(
        protocol=PROTOCOL, profile=profile, seeds=seeds, tasks=tasks, arms=arms,
        dataset_hash=imza, config=dict(ayar), parameters=parametreler,
        body_parameters=govde_parametreleri, parameter_ratio=oran,
        body_parameter_ratio=govde_orani, leakage=sizinti, results=sonuclar,
        majority_baseline=cogunluk_ozet, signature=imza_kazanim,
        checks=kontroller, findings=bulgular, limitations=sinirlar,
    )


def _signature_mapping(report: SignatureReport | Mapping[str, Any]) -> Mapping[str, Any]:
    return report.to_dict() if isinstance(report, SignatureReport) else report


def run_signature_release_gate(
    report: SignatureReport | Mapping[str, Any],
    *,
    required_profile: str = SIGNATURE_RELEASE_PROFILE,
    min_seed_count: int = SIGNATURE_RELEASE_MIN_SEEDS,
) -> SignatureReleaseGateReport:
    """Signature Benchmark raporunu release kapısından geçir.

    Bu fonksiyon yeni başarı üretmez; var olan benchmark çıktısının release için
    yeterli kanıt taşıyıp taşımadığını denetler. Eksik seed/profil ya da düşen
    benchmark kapısı varsa durum açıkça ``BLOCKED`` olur.
    """
    veri = _signature_mapping(report)
    seeds = list(veri.get("seeds", []) or [])
    tasks = list(veri.get("tasks", []) or [])
    arms = list(veri.get("arms", []) or [])
    benchmark_checks = dict(veri.get("checks", {}) or {})
    signature_map = dict(veri.get("signature", {}) or {})
    checks = {
        "source_protocol_is_signature_v1": veri.get("protocol") == PROTOCOL,
        "profile_is_standard": veri.get("profile") == required_profile,
        "seed_count_at_least_20": len(set(int(s) for s in seeds)) >= min_seed_count,
        "all_signature_tasks_present": set(tasks) == set(TASKS),
        "all_signature_arms_present": set(arms) == set(ARMS),
        "dataset_hash_present": bool(veri.get("dataset_hash")),
        "all_benchmark_checks_pass": bool(benchmark_checks)
        and all(bool(v) for v in benchmark_checks.values()),
        "no_held_out_leak": bool(benchmark_checks.get("no_held_out_leak")),
        "hga_has_signature_task": bool(benchmark_checks.get("hga_has_signature_task")),
        "memory_gain_is_task_specific": bool(
            benchmark_checks.get("memory_gain_is_task_specific")),
        "parameter_budget_within_gate": bool(
            benchmark_checks.get("parameter_budget_within_gate"))
        and bool(benchmark_checks.get("body_parameter_budget_within_gate")),
        "signature_payload_nonempty": bool(signature_map),
    }
    failed = [name for name, passed in checks.items() if not passed]
    status = "PASS" if not failed else "BLOCKED"
    evidence = {
        "source_profile": veri.get("profile"),
        "source_seed_count": len(set(int(s) for s in seeds)),
        "source_tasks": tasks,
        "source_arms": arms,
        "source_failed_benchmark_checks": [
            name for name, passed in benchmark_checks.items() if not passed],
    }
    limitations = [
        "Release gate yalnız Signature Benchmark raporunu denetler; yeni model eğitimi veya insan değerlendirmesi üretmez.",
        "Gate PASS değilse release hazır denemez; BLOCKED durumu eksik kanıtı gizlemez.",
        "20 tohum ve standard profil eşiği release kalitesi içindir; smoke koşuları regresyon amaçlı kalır.",
    ]
    return SignatureReleaseGateReport(
        protocol=RELEASE_GATE_PROTOCOL,
        source_protocol=str(veri.get("protocol") or ""),
        source_dataset_hash=str(veri.get("dataset_hash") or ""),
        status=status,
        required_profile=required_profile,
        min_seed_count=min_seed_count,
        checks=checks,
        failed_checks=failed,
        evidence=evidence,
        limitations=limitations,
    )


def signature_release_gate_markdown(report: SignatureReleaseGateReport) -> str:
    lines = [
        "# Signature Benchmark Release Gate",
        "",
        f"- Protokol: `{report.protocol}`",
        f"- Kaynak: `{report.source_protocol}` · `{report.source_dataset_hash}`",
        f"- Durum: **{report.status}**",
        f"- Gerekli profil: `{report.required_profile}`",
        f"- Minimum tohum: `{report.min_seed_count}`",
        "",
        "## Kapılar",
        "",
        "| kapı | sonuç |",
        "|---|---|",
    ]
    for name, passed in report.checks.items():
        lines.append(f"| {name} | {'GEÇTİ' if passed else 'KALDI'} |")
    lines.extend(["", "## Kanıt özeti", ""])
    lines.append(f"- Kaynak profil: `{report.evidence.get('source_profile')}`")
    lines.append(f"- Kaynak tohum sayısı: `{report.evidence.get('source_seed_count')}`")
    failed = report.evidence.get("source_failed_benchmark_checks") or []
    lines.append("- Düşen benchmark kapıları: " + (", ".join(failed) if failed else "yok"))
    lines.extend(["", "## Release kararı", ""])
    if report.release_ready:
        lines.append("- PASS: Signature Benchmark release için gerekli kapıları taşıyor.")
    else:
        lines.append("- BLOCKED: release hazır değil; düşen kapılar: " + ", ".join(report.failed_checks))
    lines.extend(["", "## Sınırlar", ""])
    lines.extend(f"- {note}" for note in report.limitations)
    return "\n".join(lines) + "\n"


def signature_markdown(report: SignatureReport) -> str:
    satirlar = [
        "# HGA Signature Benchmark v1 (P0-4)",
        "",
        f"Protokol: `{report.protocol}` · profil: `{report.profile}` · "
        f"veri imzası: `{report.dataset_hash}` · tohumlar: {report.seeds}",
        "",
        "## Parametre bütçesi",
        "",
        "| Kol | Toplam parametre | Gövde parametresi |", "|---|---:|---:|",
    ]
    for arm, p in sorted(report.parameters.items()):
        satirlar.append(
            f"| `{arm}` | {p:,} | {report.body_parameters.get(arm, 0):,} |")
    satirlar.append(
        f"| **max/min oranı** | **{report.parameter_ratio:.4f}** | "
        f"**{report.body_parameter_ratio:.4f}** |")

    satirlar += ["", "## Doğruluk ızgarası (ortalama)", "",
                 "| Görev | " + " | ".join(f"`{a}`" for a in report.arms)
                 + " | çoğunluk |",
                 "|---|" + "---:|" * (len(report.arms) + 1)]
    for task in report.tasks:
        hucreler = []
        for arm in report.arms:
            v = report.results.get(task, {}).get(arm)
            hucreler.append("—" if v is None else f"{v['accuracy_mean']:.4f}")
        cog = report.majority_baseline.get(task, {}).get("accuracy", float("nan"))
        satirlar.append(f"| {task} | " + " | ".join(hucreler) + f" | {cog:.4f} |")

    satirlar += ["", "## Halüsinasyon oranı (gerçekte çıkarılamayana 'YES')", "",
                 "| Görev | " + " | ".join(f"`{a}`" for a in report.arms) + " |",
                 "|---|" + "---:|" * len(report.arms)]
    for task in report.tasks:
        hucreler = []
        for arm in report.arms:
            v = report.results.get(task, {}).get(arm)
            hucreler.append("—" if v is None
                            else f"{v['hallucination_rate_mean']:.4f}")
        satirlar.append(f"| {task} | " + " | ".join(hucreler) + " |")

    satirlar += ["", "## İmza analizi: HGA nerede, neden?", "",
                 "Rakip = **öğrenen** kollar (dense/transformer; aynı bilgi, "
                 "aynı bütçe). `symbolic` rakip değil KÂHİN tavandır: gizli "
                 "olguları ham okur, deterministik dünyada yapısal 1.0 alır; "
                 "farkı `kâhin açığı` sütununda ayrıca raporlanır.", "",
                 "| Görev | HGA | En iyi rakip | Fark | Kâhin açığı | "
                 "Bellek katkısı | Kronecker katkısı | Attention katkısı |",
                 "|---|---:|---|---:|---:|---:|---:|---:|"]
    for task in report.tasks:
        v = report.signature.get(task)
        if v is None:
            continue
        satirlar.append(
            f"| {task} | {v['hga_accuracy']:.4f} | {v['best_competitor']} "
            f"({v['best_competitor_accuracy']:.4f}) | {v['margin']:+.4f} | "
            f"{v.get('oracle_gap', float('nan')):+.4f} | "
            f"{v.get('memory_contribution', float('nan')):+.4f} | "
            f"{v.get('kronecker_contribution', float('nan')):+.4f} | "
            f"{v.get('attention_contribution', float('nan')):+.4f} |")

    satirlar += ["", "## Sızıntı denetimi", "",
                 "| Görev | temiz | test held-out entity | test held-out relation |",
                 "|---|:--:|:--:|:--:|"]
    for task, v in report.leakage.items():
        satirlar.append(
            f"| {task} | {'✓' if v['clean'] else 'SIZINTI'} | "
            f"{'✓' if v['test_uses_held_out_entities'] else '—'} | "
            f"{'✓' if v['test_uses_held_out_relations'] else '—'} |")

    satirlar += ["", "## Kabul kapıları", "", "| kapı | sonuç |", "|---|---|"]
    for ad, sonuc in report.checks.items():
        satirlar.append(f"| {ad} | {'GEÇTİ' if sonuc else 'KALDI'} |")
    satirlar += ["", "## Bulgular", ""]
    satirlar += [f"- {b}" for b in report.findings]
    satirlar += ["", "## Sınırlar", ""]
    satirlar += [f"- {s}" for s in report.limitations]
    return "\n".join(satirlar) + "\n"


__all__ = [
    "PROTOCOL", "RELEASE_GATE_PROTOCOL", "SIGNATURE_RELEASE_MIN_SEEDS",
    "SIGNATURE_RELEASE_PROFILE", "TASKS", "ARMS", "NEURAL_ARMS", "LABELS",
    "PROFILES", "MEMORY_CRITICAL_TASKS", "NO", "YES", "UNCERTAIN",
    "Instance", "SignatureReport", "SignatureReleaseGateReport",
    "build_dataset", "leakage_report", "symbolic_predict",
    "run_signature_benchmark", "run_signature_release_gate",
    "signature_markdown", "signature_release_gate_markdown",
]
