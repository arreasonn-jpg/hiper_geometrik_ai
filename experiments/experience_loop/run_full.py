# -*- coding: utf-8 -*-
"""
HGA Experience Engine — Uçtan Uca Yol Haritası Demosu (v0.1 → v1.0)
=====================================================================
Rapor §19'daki tüm aşamaların tek demosu:

    v0.1  Knowledge/Index + Evaluator + VALID/CONFLICT/INVALID
    v0.2  Generator → metin/olay üretimi (TextGenerator)
    v0.3  Novelty + information-gain skorları
    v0.4  Experience Replay + consolidation ↔ seyrek bellek
    v0.5  Deterministik mini-environment (aritmetik doğrulayıcı)
    v0.6  Deneyim ↔ geometrik seyrek bellek köprüsü (torch varsa)
    v1.0  Sürekli öğrenme döngüsü + kontrollü metrikler

Çalıştırma:
    python experiments/experience_loop/run_full.py
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.knowledge import KnowledgeStore, DeneyimDurumu, KaynakTuru  # noqa: E402
from hga.experience import (ExperienceEvaluator, ExperienceGenerator,  # noqa: E402
                            TextGenerator, Consolidator, DeneyimDongusu,
                            AritmetikOrtam, ConflictResolver)
from hga.memory import BellekEntegrasyonu, torch_var_mi  # noqa: E402

CIZGI = "=" * 74


def v01_bilgi_temelli():
    print(CIZGI)
    print("v0.1 — Bilgi tabanı + değerlendirme + durum makinesi")
    print(CIZGI)
    k = KnowledgeStore()
    k.varlik_ekle("Ali", entity_type="insan", properties={"canli": 1}, entity_id="E_001")
    k.varlik_ekle("Ata", entity_type="hayvan", properties={"binilebilir": 1}, entity_id="E_002")
    k.varlik_ekle("Araba", entity_type="tasit", properties={"binilebilir": 1}, entity_id="E_003")
    k.varlik_ekle("Gökyüzü", entity_type="mekan", properties={"binilebilir": 0}, entity_id="E_004")
    k.iliski_tanimla("Binmek", relation_id="R_001",
                     subject_types=["insan"],
                     requires_object_props={"binilebilir": 1.0})

    ev = ExperienceEvaluator()
    adaylar = ExperienceGenerator().uret(k, relation_ids=["R_001"])
    for a in adaylar:
        ev.degerlendir(a, k)
        nesne = k.entities.getir(a.object_id).token
        print(f"  Ali --Binmek--> {nesne:<8} → {a.state.value:<9} skor={a.scores.get('weighted', '-'):}")
    return k, adaylar


def v02_metin(k, adaylar):
    print("\n" + CIZGI)
    print("v0.2 — Generator'dan metin/olay üretimi (TextGenerator)")
    print(CIZGI)
    tg = TextGenerator()
    for a in adaylar:
        print(f"  {a.experience_id}: \"{tg.cumle(k, a)}\"  [{a.state.value}]")


def v03_bilgi_kazanci(k, adaylar):
    print("\n" + CIZGI)
    print("v0.3 — Novelty + information-gain skorları")
    print(CIZGI)
    for a in adaylar:
        if a.scores:
            print(f"  {a.experience_id}: novelty={a.scores['novelty']} "
                  f"information_gain={a.scores['information_gain']} "
                  f"weighted={a.scores['weighted']}")


def v04_bellek(k, adaylar):
    print("\n" + CIZGI)
    print("v0.4 — Replay + consolidation ↔ seyrek bellek")
    print(CIZGI)
    bellek = BellekEntegrasyonu(slot_sayisi=64, replay_kapasitesi=16)
    for a in adaylar:
        if a.state in (DeneyimDurumu.VALID, DeneyimDurumu.VERIFIED):
            bellek.yaz(a)
    for anahtar, deger in bellek.rapor().items():
        print(f"  {anahtar:<18}: {deger}")
    ornekler = bellek.ornek_oynat(2)
    print(f"  replay örneği: {[o.experience_id for o in ornekler]}")


def v05_mini_env():
    print("\n" + CIZGI)
    print("v0.5 — Deterministik mini-environment (aritmetik)")
    print(CIZGI)
    ortam = AritmetikOrtam()
    for iddia in ("2+3=5", "2+3=6", "4/0=1", "(2+3)*2=10"):
        print(f"  {iddia:<12} → {ortam.iddia_dogrula(iddia)}")


def v06_kopru():
    print("\n" + CIZGI)
    print("v0.6 — Deneyim ↔ geometrik seyrek bellek köprüsü")
    print(CIZGI)
    if torch_var_mi():
        from hga.memory import TorchKoprusu
        kopru = TorchKoprusu(tablo_boyutu=1024, boyut=8)
        vektor = kopru.vektor(("E_001", "R_001", "E_002"))
        print(f"  torch köprüsü kuruldu; üçlü vektörü şekli: {tuple(vektor.shape)} "
              f"(boş küme → sıfır)")
    else:
        print("  torch kurulu değil → köprü hazır; gerçek tablo bağlantısı "
              "torch kurulu ortamda aktifleşir.")


def v10_dongu():
    print("\n" + CIZGI)
    print("v1.0 — Sürekli öğrenme döngüsü + kontrollü metrikler")
    print(CIZGI)
    k = KnowledgeStore()
    k.varlik_ekle("1+1", entity_type="ifade", entity_id="E_001")
    k.varlik_ekle("2", entity_type="sayi", entity_id="E_002")
    k.varlik_ekle("3", entity_type="sayi", entity_id="E_003")
    k.iliski_tanimla("eşittir", relation_id="R_001")

    ortam = AritmetikOrtam()
    dongu = DeneyimDongusu(
        store=k,
        generator=ExperienceGenerator(tip_filtresi=False),
        evaluator=ExperienceEvaluator(),
        consolidator=Consolidator(),
        bellek=BellekEntegrasyonu(slot_sayisi=64, replay_kapasitesi=16),
        dogrulayici=ortam.aday_dogrula,
    )
    rapor = dongu.adim(relation_ids=["R_001"])
    for anahtar, deger in rapor.to_dict().items():
        print(f"  {anahtar:<18}: {deger}")


def main():
    k, adaylar = v01_bilgi_temelli()
    v02_metin(k, adaylar)
    v03_bilgi_kazanci(k, adaylar)
    v04_bellek(k, adaylar)
    v05_mini_env()
    v06_kopru()
    v10_dongu()
    print("\n" + CIZGI)
    print("Tüm aşamalar tamam: öğren → temsil et → üret → değerlendir → "
          "çelişkiyi ayır → konsolide et → yeniden üret")
    print(CIZGI)


if __name__ == "__main__":
    main()
