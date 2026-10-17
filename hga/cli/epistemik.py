"""HGA CLI komutu (Faz 2 refactor)."""
from __future__ import annotations

import json
import os


def _epistemik(seeds=None, out=None, markdown=None):
    """P0-007: KNOWN/UNKNOWN/UNCERTAIN/CONFLICT/FALSE epistemik benchmarkı."""
    from hga.evaluation.epistemic import (
        EpistemicDataset,
        run_epistemic_baselines,
        run_epistemic_benchmark,
    )

    veri = EpistemicDataset()
    rapor = run_epistemic_benchmark(veri)
    tohumlar = [int(v.strip()) for v in (seeds or "1").split(",") if v.strip()]

    print("Epistemik Benchmark — bilmediğini biliyor mu? (P0-007)\n")
    print(f"  protokol       : {rapor.protocol}")
    print(f"  veri kümesi    : {rapor.dataset_hash[:16]}…  ({rapor.total_cases} vaka)")
    print(f"  tohumlar       : {tohumlar} (protokol deterministik)\n")

    print("  ASIL METRİKLER")
    print(f"    yanlış güven oranı (↓)   : {rapor.false_confidence_rate:.3f} "
          f"({rapor.false_confidence_cases} vaka)")
    print(f"    bilinmeyen doğruluğu (↑) : {rapor.unknown_accuracy:.3f}")
    print(f"    sessiz kabul oranı (↓)   : {rapor.silent_failure_rate:.3f}")
    print(f"    genel doğruluk           : {rapor.accuracy:.3f}\n")

    print("  SINIF BAZINDA")
    print(f"    {'sınıf':<12}{'n':>4}{'doğru':>7}{'isabet':>9}  gözlenen durumlar")
    for sinif in rapor.per_class:
        durumlar = ", ".join(f"{k}×{v}" for k, v in sinif["observed_states"].items())
        print(f"    {sinif['epistemic_class']:<12}{sinif['total']:>4}"
              f"{sinif['correct']:>7}{sinif['accuracy']:>9.3f}  {durumlar}")

    print("\n  NEGATİF KONTROL (dejenere politikalar)")
    print(f"    {'kol':<18}{'doğruluk':>9}{'bilinen':>9}{'bilinmeyen':>12}{'y.güven':>9}")
    for kol in run_epistemic_baselines(veri):
        print(f"    {kol.arm:<18}{kol.accuracy:>9.3f}{kol.known_accuracy:>9.3f}"
              f"{kol.unknown_accuracy:>12.3f}{kol.false_confidence_rate:>9.3f}")
    print(f"    {'EVALUATOR':<18}{rapor.accuracy:>9.3f}{rapor.known_accuracy:>9.3f}"
          f"{rapor.unknown_accuracy:>12.3f}{rapor.false_confidence_rate:>9.3f}")

    ayrim = rapor.epistemic_resolution
    print("\n  EPİSTEMİK ÇÖZÜNÜRLÜK (dürüstlük notu)")
    print(f"    UNKNOWN ↔ UNCERTAIN ayrılabilir mi? : "
          f"{'EVET' if ayrim['distinguishable'] else 'HAYIR'}")
    print(f"    {ayrim['note']}")

    print("\n  KAPILAR")
    for ad, deger in rapor.checks.items():
        print(f"    [{'GEÇTİ' if deger else 'KALDI'}] {ad}")

    print("\n  SINIRLAR")
    for sinir in rapor.limitations:
        print(f"    - {sinir}")

    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(rapor.to_dict(), handle, ensure_ascii=False, indent=2,
                      sort_keys=True)
        print(f"\n  report: {out}")
    if markdown:
        os.makedirs(os.path.dirname(os.path.abspath(markdown)) or ".", exist_ok=True)
        with open(markdown, "w", encoding="utf-8") as handle:
            handle.write("# Epistemik Benchmark (P0-007)\n\n")
            handle.write(f"Veri kümesi hash: `{rapor.dataset_hash}`\n\n")
            handle.write(f"- Yanlış güven oranı (↓): **{rapor.false_confidence_rate:.3f}**\n")
            handle.write(f"- Bilinmeyen doğruluğu (↑): **{rapor.unknown_accuracy:.3f}**\n")
            handle.write(f"- Sessiz kabul oranı (↓): **{rapor.silent_failure_rate:.3f}**\n")
            handle.write(f"- Genel doğruluk: **{rapor.accuracy:.3f}**\n\n")
            handle.write("## Sınıf bazında\n\n")
            handle.write("| Sınıf | n | Doğru | İsabet |\n|---|---:|---:|---:|\n")
            for sinif in rapor.per_class:
                handle.write(f"| {sinif['epistemic_class']} | {sinif['total']} | "
                             f"{sinif['correct']} | {sinif['accuracy']:.3f} |\n")
            handle.write("\n## Epistemik çözünürlük\n\n")
            handle.write(f"{ayrim['note']}\n\n")
            handle.write("## Sınırlar\n\n")
            for sinir in rapor.limitations:
                handle.write(f"- {sinir}\n")
        print(f"  markdown: {markdown}")
