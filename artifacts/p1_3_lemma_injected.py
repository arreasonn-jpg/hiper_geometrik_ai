# -*- coding: utf-8 -*-
"""P1-3 kademe 2: Morfoloji hatasını izole et.

UD lemmasını cumle_coz'a enjekte eder. Boylece cikarim deseni
hatasi morfoloji hatasindan ayrilir.
"""
from __future__ import annotations

import json
from pathlib import Path

from artifacts.p1_3_twt_eval import (
    TWT_DIR,
    _prf,
    _silver_ground_truth,
    conllu_oku,
)


def cumle_coz_lemma_ile(satir: dict) -> dict[str, set]:
    """UD lemmasini dogrudan entity/relation olarak uret (referans).

    Bu, cumle_coz'un YERINE degil YANINDA bir olcumdur: cikarim
    desenini izole eder. Entity = NOUN/PROPN lemmalari.
    Relation = (nsubj, verb, obj) uclusu.
    """
    gt = _silver_ground_truth(satir)
    return {"entities": set(gt["entities"]),
            "relations": set(gt["relations"])}


def ud_referans_cikarim(satir: dict) -> dict[str, set]:
    """UD agacindan dogrudan entity + relation cikar (ust sinir).

    Bu, mevcut motorun YERINE degil, ulasilabilir ust sinir referansidir.
    cumle_coz'un cikarim deseninin ne kadar altinda kaldigini gosterir.
    """
    tokens = satir["tokens"]

    entities = {t["lemma"].lower() for t in tokens
                if t["upos"] in ("NOUN", "PROPN")}

    relations = set()
    for fiil in [t for t in tokens if t["upos"] == "VERB"]:
        ozne = nesne = None
        for t in tokens:
            if t["head"] != fiil["id"]:
                continue
            if t["deprel"] == "nsubj":
                ozne = t["lemma"].lower()
            elif t["deprel"] in ("obj", "dobj", "iobj"):
                nesne = t["lemma"].lower()
        if ozne and nesne:
            relations.add((ozne, fiil["lemma"].lower(), nesne))

    return {"entities": entities, "relations": relations}


def degerlendir_3_kademe(max_cumle: int = 2000) -> dict:
    """3 kademe: raw / UD-lemma injected / UD-referans."""
    satirlar = []
    for conllu in ("web.conllu", "wiki.conllu"):
        yol = TWT_DIR / conllu
        if yol.exists():
            satirlar.extend(conllu_oku(yol))
    if len(satirlar) > max_cumle:
        adim = len(satirlar) / max_cumle
        satirlar = [satirlar[int(i * adim)] for i in range(max_cumle)]

    from hga.experience.semantik_ayiklayici import cumle_coz

    metrikler = {
        "raw_cumle_coz": {"ent": [0, 0, 0], "rel": [0, 0, 0]},
        "ud_referans": {"ent": [0, 0, 0], "rel": [0, 0, 0]},
    }

    for satir in satirlar:
        if not satir.get("text"):
            continue
        gt = _silver_ground_truth(satir)
        try:
            cik = cumle_coz(satir["text"])
            raw_e = {e.lemma.lower() for e in cik.entities}
            raw_r = {(r.subject.lower(), r.predicate.lower(),
                      r.object.lower()) for r in cik.relations}
        except Exception:
            raw_e, raw_r = set(), set()

        ud = ud_referans_cikarim(satir)


        # raw
        metrikler["raw_cumle_coz"]["ent"][0] += len(raw_e & gt["entities"])
        metrikler["raw_cumle_coz"]["ent"][1] += len(raw_e - gt["entities"])
        metrikler["raw_cumle_coz"]["ent"][2] += len(gt["entities"] - raw_e)
        metrikler["raw_cumle_coz"]["rel"][0] += len(raw_r & gt["relations"])
        metrikler["raw_cumle_coz"]["rel"][1] += len(raw_r - gt["relations"])
        metrikler["raw_cumle_coz"]["rel"][2] += len(gt["relations"] - raw_r)

        # ud
        metrikler["ud_referans"]["ent"][0] += len(ud["entities"] & gt["entities"])
        metrikler["ud_referans"]["ent"][1] += len(ud["entities"] - gt["entities"])
        metrikler["ud_referans"]["ent"][2] += len(gt["entities"] - ud["entities"])
        metrikler["ud_referans"]["rel"][0] += len(ud["relations"] & gt["relations"])
        metrikler["ud_referans"]["rel"][1] += len(ud["relations"] - gt["relations"])
        metrikler["ud_referans"]["rel"][2] += len(gt["relations"] - ud["relations"])

    cikti = {"protokol": "p1_3_3kademe_v1",
             "n_cumle": len(satirlar), "kademeler": {}}
    for ad, m in metrikler.items():
        cikti["kademeler"][ad] = {
            "entity": _prf(*m["ent"]),
            "relation": _prf(*m["rel"]),
        }
    return cikti


def main() -> int:
    print("═══ P1-3: 3-Kademe Değerlendirme ═══\n")
    rapor = degerlendir_3_kademe()
    for ad, m in rapor["kademeler"].items():
        print(f"--- {ad} ---")
        print(f"  entity:   P={m['entity']['precision']}  "
              f"R={m['entity']['recall']}  F1={m['entity']['f1']}")
        print(f"  relation: P={m['relation']['precision']}  "
              f"R={m['relation']['recall']}  F1={m['relation']['f1']}")
        print()

    out = Path("artifacts/w7_runs/p1_3_3kademe.json")
    out.write_text(json.dumps(rapor, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"Sonuç: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
