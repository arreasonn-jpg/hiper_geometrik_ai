"""HGA CLI komutu (Faz 2 refactor)."""
from __future__ import annotations

import json
import os


def _provenance(out=None, markdown=None):
    """Faz 27/28: köken denetimi ve belge hash doğrulaması demosu."""
    from hga.evaluation.provenance import (
        audit_provenance,
        build_provenance_chain,
        ingest_with_provenance,
        verify_document_hashes,
    )
    from hga.knowledge import KaynakTuru, KnowledgeStore

    store = KnowledgeStore()
    belge = ("Ali ata bindi. Ayşe kitabı okudu. Mehmet topu attı. "
             "Kuş gökyüzünde uçtu.")
    cumleler = [c.strip() + "." for c in belge.split(".") if c.strip()]
    bilgi = ingest_with_provenance(store, cumleler,
                                   source_url="ornek://turkce-mini-korpus",
                                   content=belge)
    print("Köken damgalı aktarım (Faz 27/28):")
    for anahtar, deger in bilgi.items():
        print(f"  {anahtar}: {deger}")

    zincir = None
    if store.relations.olgular():
        zincir = build_provenance_chain(store, store.relations.olgular()[0])
        print("\nÖrnek provenance zinciri:")
        print("  " + " → ".join(zincir.kinds()))
        print(f"  tam dış iz: {zincir.complete_external_trace}; eksikler: {zincir.missing}")

    # Kökensiz bir olgu ekleyip denetimin bunu yakaladığını göster.
    store.varlik_ekle("yetim_ozne", entity_id="E_YETIM_S")
    store.varlik_ekle("yetim_nesne", entity_id="E_YETIM_O")
    store.iliski_tanimla("yetim_iliski", relation_id="R_YETIM")
    store.olgu_kaydet("E_YETIM_S", "R_YETIM", "E_YETIM_O", 1.0,
                      source=KaynakTuru.REAL_DATA, confidence=1.0)

    rapor = audit_provenance(store)
    print("\nKöken denetimi:")
    print(rapor.markdown())
    print()
    for bulgu in rapor.findings:
        print(f"  - {bulgu}")
    print(f"  temiz mi: {rapor.clean} (kökensiz olgu bilerek eklendi)")

    dogrulama = verify_document_hashes(
        store, {"ornek://turkce-mini-korpus": belge})
    print(f"\nHash doğrulaması: kontrol={dogrulama.checked} "
          f"eşleşen={dogrulama.matched} tutmayan={dogrulama.mismatched}")
    bozuk = verify_document_hashes(
        store, {"ornek://turkce-mini-korpus": belge + " Belge değişti."})
    print(f"  belge değiştirildiğinde yakalanan: {bozuk.mismatched}")

    cikti = {"ingest": bilgi, "audit": rapor.to_dict(),
             "provenance_chain_example": zincir.to_dict() if zincir else None,
             "hash_check": dogrulama.to_dict(),
             "tampered_check": bozuk.to_dict()}
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(cikti, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"  report: {out}")
    if markdown:
        os.makedirs(os.path.dirname(os.path.abspath(markdown)) or ".", exist_ok=True)
        with open(markdown, "w", encoding="utf-8") as handle:
            handle.write("# Faz 27/28 — Köken (Provenance) Denetimi\n\n")
            handle.write(rapor.markdown() + "\n\n")
            if zincir is not None:
                handle.write("## Örnek Provenance Zinciri\n\n")
                handle.write("`" + " → ".join(zincir.kinds()) + "`\n\n")
                handle.write(f"- Tam dış iz: `{zincir.complete_external_trace}`\n")
                handle.write(f"- Eksikler: `{zincir.missing}`\n\n")
            for bulgu in rapor.findings:
                handle.write(f"- {bulgu}\n")
        print(f"  markdown: {markdown}")
