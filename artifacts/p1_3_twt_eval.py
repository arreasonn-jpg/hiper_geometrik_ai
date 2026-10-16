# -*- coding: utf-8 -*-
"""P1-3: Turkce semantik cikarim — TWT dis gecerlilik testi.

Fixture (136 küratörlü cümle, F1=1.0) sonucunun TWT (4.851 gercek
web/wiki cümlesi, UD anotasyonlu) üzerinde ne kadar düştüğünü ölçer.

Yöntem:
- UD conllu'dan cümleleri ve altın etiketleri çıkar
- Silver ground truth: NOUN/PROPN → entity, nsubj-verb-obj → relation
- cumle_coz() ile çalıştır, precision/recall/F1 hesapla
- Fixture F1=1.0 ile karşılaştır
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hga.experience.semantik_ayiklayici import cumle_coz

TWT_DIR = Path("hga/evaluation/datasets/twt_v1")

#: Cümle başına maksimum (hız için örnekleme)
MAX_CUMLE = 2000


def conllu_oku(yol: Path) -> list[dict[str, Any]]:
    """TWT conllu dosyasini cumle listesine cevir."""
    cumleler: list[dict[str, Any]] = []
    mevcut: dict[str, Any] = {"tokens": [], "text": None, "sent_id": None}
    for satir in yol.read_text(encoding="utf-8").splitlines():
        if satir.startswith("# text = "):
            mevcut["text"] = satir[len("# text = "):].strip()
        elif satir.startswith("# sent_id = "):
            mevcut["sent_id"] = satir[len("# sent_id = "):].strip()
        elif satir.strip() == "":
            if mevcut["tokens"]:
                cumleler.append(mevcut)
            mevcut = {"tokens": [], "text": None, "sent_id": None}
        elif not satir.startswith("#"):
            parcalar = satir.split("\t")
            if len(parcalar) >= 8 and parcalar[0].isdigit():
                mevcut["tokens"].append({
                    "id": int(parcalar[0]),
                    "form": parcalar[1],
                    "lemma": parcalar[2],
                    "upos": parcalar[3],
                    "feats": parcalar[5],
                    "head": int(parcalar[6]) if parcalar[6].isdigit() else 0,
                    "deprel": parcalar[7].split(":")[0],
                })
    if mevcut["tokens"]:
        cumleler.append(mevcut)
    return cumleler


def _silver_ground_truth(satir: dict[str, Any]) -> dict[str, set]:
    """UD anotasyonundan silver ground truth cikar.

    - entity: NOUN, PROPN tokenlari (lemma)
    - relation: (nsubj_lemma, verb_lemma, obj_lemma) ucluleri
    """
    tokens = satir["tokens"]

    # Entity: icerik sozcukleri
    entities = set()
    for t in tokens:
        if t["upos"] in ("NOUN", "PROPN"):
            entities.add(t["lemma"].lower())

    # Relation: her fiil icin nsubj/obj
    relations = set()
    fiiller = [t for t in tokens if t["upos"] == "VERB"]
    for fiil in fiiller:
        ozne = None
        nesne = None
        for t in tokens:
            if t["head"] == fiil["id"]:
                if t["deprel"] in ("nsubj",):
                    ozne = t["lemma"].lower()
                elif t["deprel"] in ("obj", "dobj", "iobj"):
                    nesne = t["lemma"].lower()
        if ozne and nesne:
            relations.add((ozne, fiil["lemma"].lower(), nesne))

    return {"entities": entities, "relations": relations}


def _tahmin_setleri(cikarim) -> dict[str, set]:
    """cumle_coz cikitisindan karsilastirilabilir setler."""
    entities = {v.lemma.lower() for v in cikarim.entities}
    relations = {
        (r.subject.lower(), r.predicate.lower(), r.object.lower())
        for r in cikarim.relations
    }
    return {"entities": entities, "relations": relations}


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


def degerlendir(max_cumle: int = MAX_CUMLE) -> dict[str, Any]:
    """TWT üzerinde cumle_coz'u çalıştır, F1 hesapla."""
    tum_satirlar = []
    for conllu in ("web.conllu", "wiki.conllu"):
        yol = TWT_DIR / conllu
        if yol.exists():
            tum_satirlar.extend(conllu_oku(yol))

    # Deterministik örnekleme
    if len(tum_satirlar) > max_cumle:
        adim = len(tum_satirlar) / max_cumle
        tum_satirlar = [tum_satirlar[int(i * adim)]
                        for i in range(max_cumle)]

    ent_tp = ent_fp = ent_fn = 0
    rel_tp = rel_fp = rel_fn = 0
    bos_cumle = 0

    for satir in tum_satirlar:
        if not satir.get("text"):
            continue
        altin = _silver_ground_truth(satir)
        try:
            cikarim = cumle_coz(satir["text"])
        except Exception:
            bos_cumle += 1
            continue
        if cikarim.bos_mu:
            bos_cumle += 1
        tahmin = _tahmin_setleri(cikarim)

        # Entity
        ent_tp += len(tahmin["entities"] & altin["entities"])
        ent_fp += len(tahmin["entities"] - altin["entities"])
        ent_fn += len(altin["entities"] - tahmin["entities"])
        # Relation
        rel_tp += len(tahmin["relations"] & altin["relations"])
        rel_fp += len(tahmin["relations"] - altin["relations"])
        rel_fn += len(altin["relations"] - tahmin["relations"])

    return {
        "protokol": "p1_3_twt_eval_v1",
        "kaynak": "twt_v1 (web + wiki)",
        "n_cumle_degerlendirilen": len(tum_satirlar),
        "bos_cumle_orani": round(bos_cumle / max(1, len(tum_satirlar)), 4),
        "entity": _prf(ent_tp, ent_fp, ent_fn),
        "relation": _prf(rel_tp, rel_fp, rel_fn),
    }


def main() -> int:
    print("═══ P1-3: TWT Dış Geçerlilik Testi ═══\n")
    rapor = degerlendir()

    print(f"Cümle: {rapor['n_cumle_degerlendirilen']}")
    print(f"Boş çıkarım oranı: {rapor['bos_cumle_orani']}\n")
    print("Entity:")
    for k, v in rapor["entity"].items():
        print(f"  {k}: {v}")
    print("\nRelation:")
    for k, v in rapor["relation"].items():
        print(f"  {k}: {v}")

    out = Path("artifacts/w7_runs/p1_3_twt_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rapor, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nSonuç: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
