"""HGA CLI komutu (Faz 2 refactor)."""
from __future__ import annotations

import os


def _observability_demo(out=None, markdown=None, html_yol=None):
    """Torch gerektirmeyen gözlemlenebilirlik demoları."""
    from hga.knowledge import DeneyimDurumu, ExperienceCandidate
    from hga.memory import DeneyimSlotlari
    from hga.observability import gozlem_paneli_kaydet, gozlem_paneli_olustur
    adaylar = [
        ExperienceCandidate("X1", "E1", "R1", "E2", state=DeneyimDurumu.VALID),
        ExperienceCandidate("X2", "E1", "R1", "E3", state=DeneyimDurumu.INVALID),
        ExperienceCandidate("X3", "E1", "R1", "E4", state=DeneyimDurumu.CONFLICT),
    ]
    slot = DeneyimSlotlari(slot_sayisi=16)
    slot.yaz("X1", ("E1", "R1", "E2"))
    slot.yaz("X3", ("E1", "R1", "E4"))
    panel = gozlem_paneli_olustur(adaylar, slot, [([1, 0], [1, 0]), ([0, 1], [1, 0])])
    yollar = gozlem_paneli_kaydet(panel, json_yol=out, markdown_yol=markdown, html_yol=html_yol)
    if yollar:
        for tur, yol in yollar.items():
            print(f"Gözlem paneli ({tur}) yazıldı: {yol}")
    else:
        print("Deneyim akışı:", panel["deneyim_akisi"])
        print("Bellek doluluk haritası:")
        print(panel["bellek_ascii"])
        print("Katman benzerliği:", panel["geometri"])

def _halusinasyon():
    """Aritmetik alanda somut hallucination/factual consistency metriği."""
    from hga.experience import (
        AritmetikOrtam,
        ExperienceEvaluator,
        ExperienceGenerator,
        aritmetik_etki_alani,
    )
    from hga.experiments import hallucination_metrics
    store = aritmetik_etki_alani()
    gen = ExperienceGenerator(tip_filtresi=False)
    ev = ExperienceEvaluator()
    adaylar = gen.uret(store)
    for a in adaylar:
        ev.degerlendir(a, store)
    r = hallucination_metrics(adaylar, store=store,
                              validator=AritmetikOrtam().aday_dogrula)
    print("Halüsinasyon / factual consistency raporu:")
    for k, v in r.to_dict().items():
        print(f"  {k:<28}: {v}")

def _sweep():
    """n/K/context hızlı kapasite taraması."""
    from hga.config import model_config_yukle
    from hga.experiments import nk_taramasi
    mcfg = model_config_yukle()["model"]
    nler = sorted({max(8, mcfg.n // 2), mcfg.n, mcfg.n * 2})
    satirlar = nk_taramasi(n_degerleri=nler, vocab=mcfg.sozluk_boyutu,
                           emb=mcfg.emb_dim, heads=mcfg.num_heads,
                           seyrek_satir=mcfg.seyrek_tablo_boyutu,
                           seyrek_boyut=mcfg.seyrek_boyut)
    print("n/K/context taraması (tahmini):")
    print("  n     K   ctx   parametre(M)  fp32_ram(MB)  zincir_ops      attn_ops")
    for r in satirlar:
        print(f"  {r['n']:<5} {r['K']:<3} {r['baglam']:<5} "
              f"{r['tahmini_parametre']/1e6:>10.2f}  "
              f"{r['tahmini_ram_mb_fp32']:>11.1f}  "
              f"{r['zincir_ops_yaklasik']:>10.2e}  "
              f"{r['attention_ops_yaklasik']:>9.2e}")

def _defter(out=None):
    """Immutable experience ledger demosu (Faz 24)."""
    from hga.experience import ExperienceLedger
    from hga.knowledge import DeneyimDurumu, ExperienceCandidate, KaynakTuru

    defter = ExperienceLedger()
    ornekler = [
        ("E_001|R_001|E_002", DeneyimDurumu.VERIFIED, "bağımsız doğrulayıcı onayladı"),
        ("E_001|R_001|E_003", DeneyimDurumu.INVALID, "binilebilir=0 kuralı ihlal edildi"),
        ("E_001|R_002|E_009", DeneyimDurumu.UNCERTAIN, "kanıt yetersiz (Mars bilinmiyor)"),
        ("E_001|R_001|E_004", DeneyimDurumu.CONFLICT, "kayıtlı kanıtla çelişiyor"),
    ]
    for index, (uclu, durum, neden) in enumerate(ornekler):
        s, r, o = uclu.split("|")
        aday = ExperienceCandidate(f"LDG-{index:03d}", s, r, o,
                                   source=KaynakTuru.MODEL_GENERATED,
                                   state=durum)
        if durum == DeneyimDurumu.VERIFIED:
            aday.verified_by = "deterministik-dogrulayici"
        defter.kaydet(aday, knowledge_version="K12", reason=neden, cycle=1)

    print("Immutable Experience Ledger (Faz 24):")
    for kayit in defter:
        print(f"  #{kayit.seq} {kayit.status:<10} {kayit.experience:<22} "
              f"{kayit.reason}")
    print("\nÖzet:")
    for k, v in defter.ozet().items():
        print(f"  {k:<20}: {v}")
    print("  not: reddedilen deneyim de silinmez; 'model nerede hata yaptı?' "
          "sorusu ancak böyle cevaplanır.")
    if out:
        defter.kaydet_jsonl(out)
        print(f"  report: {os.path.abspath(out)}")

def _ozet(yol):
    from hga.knowledge import KnowledgeStore
    if yol:
        k = KnowledgeStore.yukle(yol)
    else:
        k = KnowledgeStore()
    print("Bilgi tabanı özeti:")
    for k, v in k.ozet().items():
        print(f"  {k:<10}: {v}")

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

def _kesif_demo():
    from hga.engine import ExperienceEngine
    e = ExperienceEngine()
    e.store.varlik_ekle("Ali", entity_type="insan", properties={"canli": 1},
                        entity_id="E_001", ozel_isim=True)
    e.store.varlik_ekle("Ata", entity_type="hayvan", properties={"binilebilir": 1},
                        entity_id="E_002")
    e.store.varlik_ekle("Araba", entity_type="tasit", properties={"binilebilir": 1},
                        entity_id="E_003")
    e.store.iliski_tanimla("Binmek", relation_id="R_001",
                           subject_types=["insan"],
                           requires_object_props={"binilebilir": 1.0})
    e.store.olgu_kaydet("E_001", "R_001", "E_002", score=1.0)
    adaylar = e.uret(["R_001"])
    e.degerlendir(adaylar)
    harita = e.uzay_haritasi(adaylar)
    print("Exploration Uzay Haritası (Faz 24):")
    for k, v in harita.to_dict().items():
        print(f"  {k:<20}: {v}")
    print("\nAktif Öğrenme ile En Bilgilendirici Deneyim Seçimi (Faz 25):")
    secilenler = e.aktif_ogrenme_sec(adaylar, k=2)
    for aday, puan in secilenler:
        nesne = e.store.entities.getir(aday.object_id).token
        print(f"  Ali --Binmek--> {nesne:<8} [InfoGain Opt Puanı: {puan:.4f}]")

def _graf_demo():
    from hga.engine import ExperienceEngine
    e = ExperienceEngine()
    e.store.varlik_ekle("Ali", entity_type="insan", properties={"canli": 1},
                        entity_id="E_001", ozel_isim=True)
    e.store.varlik_ekle("Ata", entity_type="hayvan", properties={"binilebilir": 1},
                        entity_id="E_002")
    e.store.varlik_ekle("Araba", entity_type="tasit", properties={"binilebilir": 1},
                        entity_id="E_003")
    e.store.iliski_tanimla("Binmek", relation_id="R_001",
                           subject_types=["insan"],
                           requires_object_props={"binilebilir": 1.0})
    adaylar = e.uret(["R_001"])
    e.degerlendir(adaylar)
    print("Experience Graph Özeti (Faz 23):")
    print("  ", e.graph.ozet())
    for a in adaylar:
        print("\n" + e.aciklama(a.experience_id))

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
