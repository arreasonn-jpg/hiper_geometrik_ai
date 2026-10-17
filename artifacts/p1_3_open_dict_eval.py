"""P1-3 Deney 1: TWT train lemmasindan acik sozluk uretip olc."""
from __future__ import annotations

import json
from pathlib import Path

from artifacts.p1_3_twt_eval import TWT_DIR, _prf, _silver_ground_truth, conllu_oku
from hga.experience.semantik_ayiklayici import cumle_coz


def build_open_dict():
    """TWT'den isim ve fiil lemma sozlugu uret."""
    satirlar = []
    for c in ("web.conllu", "wiki.conllu"):
        if (TWT_DIR / c).exists():
            satirlar.extend(conllu_oku(TWT_DIR / c))

    # Train/test split: ilk %80 train, son %20 test (deterministik)
    n_train = int(len(satirlar) * 0.8)
    train_set = satirlar[:n_train]
    test_set = satirlar[n_train:]

    isimler = set()
    fiiller = set()
    for s in train_set:
        for t in s["tokens"]:
            if t["upos"] in ("NOUN", "PROPN"):
                isimler.add(t["lemma"].lower())
            elif t["upos"] == "VERB":
                fiiller.add(t["lemma"].lower())

    return isimler, fiiller, test_set


def evaluate_open_dict():
    isimler, fiiller, test_set = build_open_dict()

    import hga.experience.semantik_ayiklayici as sa

    # Orijinalleri sakla
    orig_tipler = dict(sa.VARSAYILAN_TIPLER)
    orig_fiiller = dict(sa.VARSAYILAN_FIILLER)

    # Açık sözlük enjeksiyonu
    for isim in isimler:
        if isim not in sa.VARSAYILAN_TIPLER:
            sa.VARSAYILAN_TIPLER[isim] = "nesne"  # basit tip ataması
    for fiil in fiiller:
        if fiil not in sa.VARSAYILAN_FIILLER:
            sa.VARSAYILAN_FIILLER[fiil] = fiil

    ent_tp = ent_fp = ent_fn = 0
    rel_tp = rel_fp = rel_fn = 0

    for satir in test_set:
        if not satir.get("text"):
            continue
        gt = _silver_ground_truth(satir)
        try:
            cik = cumle_coz(satir["text"])
        except Exception:
            continue

        raw_e = {e.lemma.lower() for e in cik.entities}
        raw_r = {(r.subject.lower(), r.predicate.lower(),
                  r.object.lower()) for r in cik.relations}

        ent_tp += len(raw_e & gt["entities"])
        ent_fp += len(raw_e - gt["entities"])
        ent_fn += len(gt["entities"] - raw_e)

        rel_tp += len(raw_r & gt["relations"])
        rel_fp += len(raw_r - gt["relations"])
        rel_fn += len(gt["relations"] - raw_r)

    # Geri yükle
    sa.VARSAYILAN_TIPLER.clear()
    sa.VARSAYILAN_TIPLER.update(orig_tipler)
    sa.VARSAYILAN_FIILLER.clear()
    sa.VARSAYILAN_FIILLER.update(orig_fiiller)

    return {
        "n_test": len(test_set),
        "n_train_isim": len(isimler),
        "n_train_fiil": len(fiiller),
        "entity": _prf(ent_tp, ent_fp, ent_fn),
        "relation": _prf(rel_tp, rel_fp, rel_fn),
    }


if __name__ == "__main__":
    print("═══ P1-3 Deney 1: Açık Sözlük Enjeksiyonu ═══\n")
    rapor = evaluate_open_dict()
    print(f"Test seti: {rapor['n_test']} cümle")
    print(f"Train sözlüğü: {rapor['n_train_isim']} isim, "
          f"{rapor['n_train_fiil']} fiil\n")
    print(f"Entity:   F1={rapor['entity']['f1']} "
          f"(P={rapor['entity']['precision']}, "
          f"R={rapor['entity']['recall']})")
    print(f"Relation: F1={rapor['relation']['f1']} "
          f"(P={rapor['relation']['precision']}, "
          f"R={rapor['relation']['recall']})")

    out = Path("artifacts/w7_runs/p1_3_open_dict.json")
    out.write_text(json.dumps(rapor, indent=2), encoding="utf-8")
    print(f"\nSonuç: {out}")
