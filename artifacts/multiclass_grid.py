# -*- coding: utf-8 -*-
"""Multiclass grid: K ve hidden_ratio degisirken formul testi."""
import statistics as st
import sys
sys.path.insert(0, '.')

from artifacts.multiclass_paradigm import (
    gorev_uret, symbolic_pred, neural_train
)


def main():
    print("="*75)
    print("MULTICLASS GRID — FORMUL ROBUSTNESS")
    print("="*75)
    print()

    K_values = [2, 4, 8]
    hidden_ratios = [0.1, 0.3, 0.5]
    seeds = [1, 2, 3]

    print(f"{'K':>3} {'hidden':>8} {'cov':>8} {'sym_acc':>9} {'neu':>8} "
          f"{'obs':>9} {'pred':>9} {'err':>8}")
    print("-"*75)

    results = []
    for K in K_values:
        for hidden in hidden_ratios:
            errs = []
            covs = []
            sym_accs = []
            for seed in seeds:
                gorev = gorev_uret(
                    K=K, hidden_ratio=hidden,
                    train_size=600, test_size=200,  # kucuk: hizli
                    seed=seed,
                )
                sym_preds = symbolic_pred(gorev, gorev["test"])
                sym_n_ans = sum(1 for p in sym_preds if p is not None)
                sym_correct = sum(1 for p, e in zip(sym_preds, gorev["test"])
                                  if p is not None and p == e.label)
                sym_cov = sym_n_ans / len(gorev["test"])
                sym_acc = sym_correct / max(sym_n_ans, 1)

                neu_preds = neural_train(gorev, seed=seed, epochs=100)
                neu_acc = sum(1 for p, e in zip(neu_preds, gorev["test"])
                              if p == e.label) / len(gorev["test"])

                hyb_preds = [
                    s if s is not None else n
                    for s, n in zip(sym_preds, neu_preds)
                ]
                hyb_acc = sum(1 for p, e in zip(hyb_preds, gorev["test"])
                              if p == e.label) / len(gorev["test"])

                obs = hyb_acc - neu_acc
                pred = sym_cov * (sym_acc - neu_acc)
                err = abs(obs - pred)
                errs.append(err)
                covs.append(sym_cov)
                sym_accs.append(sym_acc)

            mean_cov = st.mean(covs)
            mean_sym = st.mean(sym_accs)
            mean_err = st.mean(errs)

            results.append({
                "K": K, "hidden": hidden,
                "cov": mean_cov, "sym_acc": mean_sym,
                "err": mean_err,
            })

            print(f"{K:>3} {hidden:>8.2f} {mean_cov:>8.4f} {mean_sym:>9.4f} "
                  f"{'':>8} {'':>9} {'':>9} {mean_err:>8.4f}")

    print()
    print("="*75)
    print("OZET")
    print("="*75)
    print()
    print(f"Konfig sayisi: {len(results)}")
    print(f"Ortalama hata: {st.mean(r['err'] for r in results):.5f}")
    print(f"Maksimum hata: {max(r['err'] for r in results):.5f}")
    print(f"Tum hatalar < 0.01: {all(r['err'] < 0.01 for r in results)}")
    print(f"Tum hatalar < 0.02: {all(r['err'] < 0.02 for r in results)}")
    print()

    # K'ya gore
    print("K bazinda ortalama hata:")
    for K in K_values:
        ks = [r for r in results if r["K"] == K]
        print(f"  K={K}: {st.mean(r['err'] for r in ks):.5f}")

    print()
    print("Hidden_ratio bazinda ortalama hata:")
    for h in hidden_ratios:
        hs = [r for r in results if r["hidden"] == h]
        print(f"  hidden={h}: {st.mean(r['err'] for r in hs):.5f}")


if __name__ == "__main__":
    main()
