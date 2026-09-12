# -*- coding: utf-8 -*-
"""
HGA Experience Engine — Komut Satırı Arayüzü
=============================================
Kullanım:
    python -m hga bilgi                  # bilgi tabanı demosu (Ali/Ata/Araba/Gökyüzü)
    python -m hga gercek-veri            # gerçek veri → temsil → deneyim → doğrulama
    python -m hga benchmark              # kontrollü benchmark (metrik tablosu)
    python -m hga dogrulama              # kapalı doğrulama hattı (false accept 24→0)
    python -m hga ozet [bilgi.json]      # bilgi tabanı özeti (opsiyonel yükleme)
"""
import argparse
import os
import sys


def _bilgi_demo():
    from hga.engine import ExperienceEngine
    e = ExperienceEngine()
    e.store.varlik_ekle("Ali", entity_type="insan", properties={"canli": 1},
                        entity_id="E_001", ozel_isim=True)
    e.store.varlik_ekle("Ata", entity_type="hayvan", properties={"binilebilir": 1},
                        entity_id="E_002")
    e.store.varlik_ekle("Araba", entity_type="tasit", properties={"binilebilir": 1},
                        entity_id="E_003")
    e.store.varlik_ekle("Gökyüzü", entity_type="mekan", properties={"binilebilir": 0},
                        entity_id="E_004")
    e.store.iliski_tanimla("Binmek", relation_id="R_001",
                           subject_types=["insan"],
                           requires_object_props={"binilebilir": 1.0})
    adaylar = e.uret(["R_001"])
    e.degerlendir(adaylar)
    print("Deneyim değerlendirmesi (durum makinesi):")
    for a in adaylar:
        nesne = e.store.entities.getir(a.object_id).token
        print(f"  Ali --Binmek--> {nesne:<8} → {a.state.value}")
    print("Metin üretimi:")
    for a in adaylar:
        if a.state.value == "VALID":
            print(f"  \"{e.metin(a)}\"")


def _gercek_veri():
    from hga.engine import ExperienceEngine
    e = ExperienceEngine()
    aktarilan = e.gercek_veri(
        ["Ali ataya bindi.", "Ali arabaya bindi.",
         "Ali gökyüzüne bindi.", "Ali gökyüzüne baktı."],
        iliski_kisitlari={"Binmek": {"subject_types": ["insan"],
                                     "requires_object_props": {"binilebilir": 1.0}}})
    print(f"REAL_DATA aktarılan üçlü: {len(aktarilan)}")
    rid = next(r.relation_id for r in e.store.relations.iliskiler()
               if r.token == "Binmek")
    adaylar = e.uret([rid])
    e.degerlendir(adaylar)
    for a in adaylar:
        nesne = e.store.entities.getir(a.object_id).token
        print(f"  Ali --Binmek--> {nesne:<8} → {a.state.value}")
    print("Bilgi özeti:", e.store.ozet())


def _benchmark():
    from hga.engine import ExperienceEngine
    from hga.experience import AritmetikOrtam, aritmetik_etki_alani
    store = aritmetik_etki_alani()
    e = ExperienceEngine(store=store, dogrulayici=AritmetikOrtam().aday_dogrula)
    raporlar = e.dongu(1)
    r = raporlar[0]
    print("Benchmark (1 adım, aritmetik alan):")
    for k, v in r.to_dict().items():
        print(f"  {k:<18}: {v}")


def _dogrulama():
    from hga.engine import ExperienceEngine
    from hga.experience import AritmetikOrtam, aritmetik_etki_alani
    store = aritmetik_etki_alani()
    e = ExperienceEngine(store=store, dogrulayici=AritmetikOrtam().aday_dogrula)
    adaylar = e.uret()
    e.degerlendir(adaylar)
    print(f"Değerlendirme sonrası: {len(adaylar)} aday, hepsi VALID (MODEL_GENERATED)")
    rapor = e.dogrula(adaylar)
    print("Doğrulama hattı sonucu:")
    for k, v in rapor.to_dict().items():
        print(f"  {k:<18}: {v}")


def _ozet(yol):
    from hga.knowledge import KnowledgeStore
    if yol:
        k = KnowledgeStore.yukle(yol)
    else:
        k = KnowledgeStore()
    print("Bilgi tabanı özeti:")
    for k, v in k.ozet().items():
        print(f"  {k:<10}: {v}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m hga",
                                description="HGA Experience Engine CLI")
    p.add_argument("komut", choices=["bilgi", "gercek-veri", "benchmark",
                                     "dogrulama", "ozet"])
    p.add_argument("yol", nargs="?", default=None,
                   help="'ozet' için JSON dosya yolu (opsiyonel)")
    args = p.parse_args(argv)
    {"bilgi": _bilgi_demo, "gercek-veri": _gercek_veri,
     "benchmark": _benchmark, "dogrulama": _dogrulama,
     "ozet": lambda: _ozet(args.yol)}[args.komut]()


if __name__ == "__main__":
    main()
