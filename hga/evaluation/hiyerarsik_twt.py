# -*- coding: utf-8 -*-
"""TWT'de flat HGA vs hierarchical HGA karşılaştırması.

Bu script mevcut twt_baselines.py altyapısını (vocabulary, batch schedule,
metrics) kullanır ama SADECE 2 modeli eğitir:
1. Flat HGA (mevcut, hga_n=26)
2. Hierarchical HGA (piramit)

İki senaryo test edilir:
- A) "Zengin": hiyerarşik aynı n_base=26 → 2× parametre
- B) "Eşit":  hiyerarşik n_base≈18 → aynı parametre bütçesi
"""
from __future__ import annotations

import json
import statistics as _stat
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn


def _make_flat_hga(vocabulary_size: int, n: int, layers: int,
                   embedding_dim: int, seq_len: int):
    """Mevcut flat HGA (twt_baselines.py'deki HGAClassifier ile aynı)."""
    import sys
    from pathlib import Path as _P
    _M = _P(__file__).resolve().parents[2] / "mimari"
    if str(_M) not in sys.path:
        sys.path.insert(0, str(_M))

    from decoder import FraktalDecoder
    from encoder import GeometrikVeriEncoder
    from hiper_attention import HiperGeometrikAttention
    from kuresel_bag import KureselZincir

    flat_dim = embedding_dim * seq_len

    class FlatHGA(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
            self.position = nn.Parameter(torch.empty(1, seq_len, embedding_dim))
            nn.init.normal_(self.position, mean=0.0, std=0.02)
            self.attention = HiperGeometrikAttention(
                embedding_dim, 4, dropout=0.0, is_causal=False)
            self.encoder = GeometrikVeriEncoder(flat_dim, n, aktivasyon="tanh")
            self.body = KureselZincir(
                n=n, katman_sayisi=layers, dropout=0.0,
                checkpoint_kullan=False, aktivasyon="silu")
            self.norm = nn.LayerNorm(n)
            self.decoder = FraktalDecoder(n, 2)

        def forward(self, token_ids):
            hidden = self.embedding(token_ids) + self.position
            hidden = self.attention(hidden)
            flat = hidden.flatten(1)
            matrix = self.encoder(flat)
            return self.decoder(self.norm(self.body(matrix)))

    return FlatHGA


def _make_hiyerarsik_hga(vocabulary_size: int, n: int, layers: int,
                         expansion: int, embedding_dim: int, seq_len: int):
    """Hiyerarşik piramit HGA."""
    import sys
    from pathlib import Path as _P
    _M = _P(__file__).resolve().parents[2] / "mimari"
    if str(_M) not in sys.path:
        sys.path.insert(0, str(_M))

    from decoder import FraktalDecoder
    from encoder import GeometrikVeriEncoder
    from hiper_attention import HiperGeometrikAttention
    from hiyerarsik_bag import HiyerarsikZincir

    flat_dim = embedding_dim * seq_len

    class HiyerarsikHGA(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
            self.position = nn.Parameter(torch.empty(1, seq_len, embedding_dim))
            nn.init.normal_(self.position, mean=0.0, std=0.02)
            self.attention = HiperGeometrikAttention(
                embedding_dim, 4, dropout=0.0, is_causal=False)
            self.encoder = GeometrikVeriEncoder(flat_dim, n, aktivasyon="tanh")
            self.body = HiyerarsikZincir(
                n_base=n, katman_sayisi=layers, expansion=expansion,
                dropout=0.0, aktivasyon="silu")
            self.norm = nn.LayerNorm(n)
            self.decoder = FraktalDecoder(n, 2)

        def forward(self, token_ids):
            hidden = self.embedding(token_ids) + self.position
            hidden = self.attention(hidden)
            flat = hidden.flatten(1)
            matrix = self.encoder(flat)
            return self.decoder(self.norm(self.body(matrix)))

    return HiyerarsikHGA


def _eval(model, x, y, batch_size=2048, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = x[i:i+batch_size].to(device)
            yb = y[i:i+batch_size].to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += yb.shape[0]
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(yb.cpu().tolist())
    acc = correct / max(total, 1)
    # F1
    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    far = fp / max(fp + (sum(1 for l in all_labels if l == 0)), 1)
    frr = fn / max(fn + tp, 1)
    return {"accuracy": acc, "f1": f1, "precision": precision,
            "recall": recall, "far": far, "frr": frr}


def _train_one(model, train_x, train_y, steps, batch, lr, seed, device="cpu"):
    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    n = train_x.shape[0]

    # Batch schedule (aynı seed → aynı sıra)
    gen = torch.Generator().manual_seed(seed)
    schedule = []
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,), generator=gen)
        schedule.append(idx)

    losses = []
    model.train()
    started = time.perf_counter()
    for idx in schedule:
        xb = train_x[idx].to(device)
        yb = train_y[idx].to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.item()))
    train_time = time.perf_counter() - started
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, params, losses[-1] if losses else float("nan"), train_time


def run_hiyerarsik_twt(
    seeds: List[int] = [1, 2, 3],
    profile: str = "smoke",
    steps: int = 32,
    batch: int = 512,
    lr: float = 3e-3,
    hga_n: int = 26,
    layers: int = 2,
    expansion: int = 2,
    hiy_eq_n: int = 18,
    min_distance: int = 0,
    out: str = "artifacts/hiyerarsik_twt_sonuclari.json",
    markdown: str = "artifacts/hiyerarsik_twt_sonuclari.md",
) -> Dict[str, Any]:
    """TWT'de flat vs hierarchical HGA karşılaştırması."""
    import sys
    from pathlib import Path as _P
    _ROOT = _P(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from hga.evaluation.real_turkish import prepare_real_turkish_task
    from hga.evaluation.twt_baselines import ArcFeatureVocabulary

    print("[1/4] TWT verisi yükleniyor...")
    task_data = prepare_real_turkish_task()
    print(f"    (filtresiz) train={len(task_data.train)}, dev={len(task_data.dev)}, test={len(task_data.test)}")

    # Uzun mesafe filtresi (frozen dataclass güvenli)
    if min_distance > 0:
        from dataclasses import replace
        def _dist_ok(c):
            return abs(c.dependent_id - c.head_id) >= min_distance
        task_data = replace(
            task_data,
            train=[c for c in task_data.train if _dist_ok(c)],
            dev=[c for c in task_data.dev if _dist_ok(c)],
            test=[c for c in task_data.test if _dist_ok(c)],
        )
        print(f"    (min_dist>={min_distance}) train={len(task_data.train)}, "
              f"dev={len(task_data.dev)}, test={len(task_data.test)}")

        if len(task_data.train) < 100:
            print("    ⚠️ Filtre çok agresif, train seti küçük!")

    print("[2/4] Vocabulary kuruluyor...")
    vocab = ArcFeatureVocabulary.fit(task_data.train)
    train_x_list, _ = vocab.encode(task_data.train)
    test_x_list, _ = vocab.encode(task_data.test)
    train_x = torch.tensor(train_x_list, dtype=torch.long)
    test_x = torch.tensor(test_x_list, dtype=torch.long)
    train_y = torch.tensor([int(c.expected_valid) for c in task_data.train], dtype=torch.long)
    test_y = torch.tensor([int(c.expected_valid) for c in task_data.test], dtype=torch.long)
    vocab_size = vocab.size
    print(f"    vocab_size={vocab_size}")

    # Config (twt_baselines ile aynı)
    EMB_DIM = 16
    SEQ_LEN = 6

    # Senaryo A: Zengin hiyerarşik (n_base=26, 2x param)
    flat_cls = _make_flat_hga(vocab_size, n=hga_n, layers=layers,
                               embedding_dim=EMB_DIM, seq_len=SEQ_LEN)
    hiy_rich_cls = _make_hiyerarsik_hga(vocab_size, n=hga_n, layers=layers,
                                         expansion=expansion, embedding_dim=EMB_DIM, seq_len=SEQ_LEN)
    hiy_eq_cls = _make_hiyerarsik_hga(vocab_size, n=hiy_eq_n, layers=layers,
                                       expansion=expansion, embedding_dim=EMB_DIM, seq_len=SEQ_LEN)

    models = {
        "flat_hga": flat_cls,
        "hiyerarsik_rich": hiy_rich_cls,
        "hiyerarsik_eq": hiy_eq_cls,
    }

    print(f"[3/4] Eğitim başlıyor ({len(seeds)} seed × {len(models)} model)...")
    results = {name: [] for name in models}
    for seed in seeds:
        print(f"\n  --- seed={seed} ---")
        for name, cls in models.items():
            # Aynı embedding başlangıcı
            model = cls()
            with torch.no_grad():
                gen = torch.Generator().manual_seed(seed + 31337)
                common_emb = torch.randn(vocab_size, EMB_DIM, generator=gen)
                model.embedding.weight.copy_(common_emb)

            model, params, final_loss, t_train = _train_one(
                model, train_x, train_y,
                steps=steps, batch=batch, lr=lr, seed=seed)
            metrics = _eval(model, test_x, test_y)
            results[name].append({
                "seed": seed,
                "params": params,
                "final_loss": final_loss,
                "train_time": t_train,
                **metrics,
            })
            print(f"    {name:20s} params={params:>7,}  "
                  f"test_acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  "
                  f"süre={t_train:.1f}s")

    print("[4/4] Rapor üretiliyor...")
    # Aggregate
    aggregate = {}
    for name, rows in results.items():
        accs = [r["accuracy"] for r in rows]
        f1s = [r["f1"] for r in rows]
        aggregate[name] = {
            "params": rows[0]["params"],
            "accuracy_mean": round(_stat.mean(accs), 4),
            "accuracy_std": round(_stat.stdev(accs) if len(accs) > 1 else 0, 4),
            "f1_mean": round(_stat.mean(f1s), 4),
            "f1_std": round(_stat.stdev(f1s) if len(f1s) > 1 else 0, 4),
            "raw": rows,
        }

    report = {
        "protocol": "hiyerarsik_twt_v1",
        "seeds": seeds,
        "profile": profile,
        "min_distance": min_distance,
        "vocab_size": vocab_size,
        "train_size": len(task_data.train),
        "test_size": len(task_data.test),
        "aggregate": aggregate,
    }

    # JSON yaz
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown yaz
    lines = [
        "# TWT Karşılaştırma — Flat vs Hierarchical HGA",
        "",
        f"- Seeds: {seeds}",
        f"- Profile: {profile}",
        f"- Min distance: {min_distance}",
        f"- Train/Test: {len(task_data.train)}/{len(task_data.test)}",
        f"- Vocab: {vocab_size}",
        "",
        "## Özet",
        "",
        "| Model | Params | Test Acc | Test F1 |",
        "|---|---:|---:|---:|",
    ]
    for name in ["flat_hga", "hiyerarsik_eq", "hiyerarsik_rich"]:
        agg = aggregate[name]
        lines.append(
            f"| {name} | {agg['params']:,} | "
            f"{agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f} | "
            f"{agg['f1_mean']:.4f} ± {agg['f1_std']:.4f} |"
        )
    lines.append("")
    lines.append("## Yorum")
    lines.append("")
    flat_acc = aggregate["flat_hga"]["accuracy_mean"]
    hiy_eq_acc = aggregate["hiyerarsik_eq"]["accuracy_mean"]
    hiy_rich_acc = aggregate["hiyerarsik_rich"]["accuracy_mean"]
    if hiy_eq_acc > flat_acc:
        lines.append(f"✅ Eşit parametreli hiyerarşik KAZANIYOR: {hiy_eq_acc:.4f} vs {flat_acc:.4f}")
    else:
        lines.append(f"❌ Flat kazanıyor: {flat_acc:.4f} vs {hiy_eq_acc:.4f}")
    lines.append("")
    if hiy_rich_acc > flat_acc:
        lines.append(f"✅ Zengin hiyerarşik de kazanıyor: {hiy_rich_acc:.4f}")
    else:
        lines.append(f"❌ Zengin hiyerarşik kaybediyor: {hiy_rich_acc:.4f}")

    md_p = Path(markdown)
    md_p.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n✅ JSON: {out_p}")
    print(f"✅ Markdown: {md_p}")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--hga-n", type=int, default=26)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--expansion", type=int, default=2)
    parser.add_argument("--min-distance", type=int, default=0)
    parser.add_argument("--out", default="artifacts/hiyerarsik_twt_sonuclari.json")
    parser.add_argument("--markdown", default="artifacts/hiyerarsik_twt_sonuclari.md")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    run_hiyerarsik_twt(seeds=seeds, steps=args.steps, profile=args.profile,
                       hga_n=args.hga_n, layers=args.layers, expansion=args.expansion,
                       min_distance=args.min_distance,
                       out=args.out, markdown=args.markdown)
