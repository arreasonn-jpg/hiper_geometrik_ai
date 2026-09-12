# -*- coding: utf-8 -*-
"""
v0.1 Deneyim Döngüsü Demosu — §14 + §15 senaryosu
===================================================
Hiper Geometrik AI: Experience Engine / Self-Expanding Knowledge Architecture

Çalıştırma:
    python experiments/experience_loop/run_demo.py

Ne gösterir:
    1. KnowledgeStore kurulur (Ali, Ata, Araba, Gökyüzü + Binmek ilişkisi).
    2. ExperienceGenerator KONTROLLÜ adaylar üretir (Ali × {Ata, Araba, Gökyüzü}).
    3. ExperienceEvaluator her adayı VALID / CONFLICT / INVALID olarak karar ağacıyla
       değerlendirir; MODEL_GENERATED hiçbir zaman otomatik VERIFIED edilmez.
    4. Consolidator sonuçları konsolide eder; VALID'ler belleğe aday yazılır,
       INVALID reddedilir; deney metrikleri raporlanır.
    5. Seyrek slot köprüsü deneyimleri hash'ler (mevcut seyrek belleğin v0.6 köprüsü).
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.knowledge import KnowledgeStore, DeneyimDurumu  # noqa: E402
from hga.experience import (ExperienceEvaluator, ExperienceGenerator,  # noqa: E402
                            Consolidator, ConflictResolver)
from hga.memory import DeneyimSlotlari  # noqa: E402


def main():
    print("=" * 72)
    print("HGA Experience Engine — v0.1 Deneyim Döngüsü Demosu")
    print("=" * 72)

    # ── 1. Bilgi tabanı (rapor §15) ──────────────────────────────────────
    k = KnowledgeStore()
    k.varlik_ekle("Ali", entity_type="insan", properties={"canli": 1}, entity_id="E_001")
    k.varlik_ekle("Ata", entity_type="hayvan", properties={"binilebilir": 1}, entity_id="E_002")
    k.varlik_ekle("Araba", entity_type="tasit", properties={"binilebilir": 1}, entity_id="E_003")
    k.varlik_ekle("Gökyüzü", entity_type="mekan", properties={"binilebilir": 0}, entity_id="E_004")
    k.iliski_tanimla("Binmek", relation_id="R_001",
                     subject_types=["insan"],
                     requires_object_props={"binilebilir": 1.0})

    print("\n[1] Bilgi tabanı:")
    for e in k.entities.hepsi():
        ozellikler = {ad: pv.deger for ad, pv in k.properties.hepsi(e.entity_id).items()}
        print(f"    {e.entity_id}  {e.token:<8} tip={e.entity_type:<7} özellik={ozellikler}")

    # ── 2. Kontrollü üretim (§7) ─────────────────────────────────────────
    gen = ExperienceGenerator()
    adaylar = gen.uret(k, relation_ids=["R_001"])
    print(f"\n[2] Üretilen aday sayısı (kontrollü kombinasyon): {len(adaylar)}")

    # ── 3. Değerlendirme (§8/§9) ─────────────────────────────────────────
    ev = ExperienceEvaluator()
    print("\n[3] Değerlendirme sonuçları:")
    for a in adaylar:
        ev.degerlendir(a, k)
        nesne = k.entities.getir(a.object_id).token
        ozet = (f"    {a.experience_id}  Ali --Binmek--> {nesne:<8} "
                f"→ {a.state.value}")
        if a.scores:
            ozet += f"   (skor={a.scores['weighted']})"
        print(ozet)

    # ── 4. Konsolidasyon (§12, EK-C) ─────────────────────────────────────
    cons = Consolidator()
    rapor = cons.konsolide_et(k, adaylar)
    print("\n[4] Konsolidasyon raporu:")
    for anahtar, deger in rapor.to_dict().items():
        print(f"    {anahtar:<18}: {deger}")

    # ── 5. Çelişki → keşif (§11) ─────────────────────────────────────────
    k2 = KnowledgeStore()
    k2.varlik_ekle("Ali", entity_type="insan", entity_id="E_001")
    k2.varlik_ekle("Ata", entity_type="hayvan", properties={"binilebilir": 1}, entity_id="E_002")
    k2.varlik_ekle("Halı", entity_type="esya", entity_id="E_003")  # özellik BİLİNMİYOR
    k2.iliski_tanimla("Binmek", relation_id="R_001",
                      subject_types=["insan"],
                      requires_object_props={"binilebilir": 1.0})
    from hga.knowledge import ExperienceCandidate
    belirsiz = ExperienceCandidate(experience_id="X_PROBE", subject_id="E_001",
                                   relation_id="R_001", object_id="E_003")
    ev.degerlendir(belirsiz, k2)
    print(f"\n[5] Çelişki → keşif: Ali --Binmek--> Halı → {belirsiz.state.value}")
    cr = ConflictResolver(evaluator=ev)
    cozum = cr.coz(k2, belirsiz)
    print("    nedenler:", *cozum.nedenler, sep="\n      ")
    print("    alternatifler:", [alt.object_id for alt in cozum.alternatifler])
    for t in cozum.test_sonuclari:
        print(f"    • {t}")
    print(f"    yeniden değerlendirme → {cozum.son_durum.value}")

    # ── 6. Seyrek slot köprüsü (v0.6 hazırlığı, §22 commit 6) ────────────
    slotlar = DeneyimSlotlari(slot_sayisi=64)
    for a in adaylar:
        if a.state == DeneyimDurumu.VALID:
            slotlar.yaz(a.experience_id, a.uclusu)
    print("\n[6] Seyrek slot köprüsü:")
    for anahtar, deger in slotlar.kapasite().items():
        print(f"    {anahtar:<18}: {deger}")

    print("\n" + "=" * 72)
    print("Döngü tamam: üret → değerlendir → çelişkiyi ayır → konsolide et")
    print("=" * 72)


if __name__ == "__main__":
    main()
