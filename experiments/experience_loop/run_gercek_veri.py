# -*- coding: utf-8 -*-
"""
HGA Experience Engine — "Gerçek Veri → Temsil → Deneyim → Doğrulama" Demosu
============================================================================
Yol haritası döngüsünün ilk aşamasını uçtan uca gösterir (rapor §2, §12):

    1. GERÇEK VERİ   — Türkçe cümleler lexiconsuz pattern ile üçlülere ayrılır
                        ve REAL_DATA kaynağıyla KnowledgeStore'a aktarılır.
    2. TEMSİL         — varlıklar (tip + özellik) ve ilişki kanıtları kurulur.
    3. DENEYİM        — Generator kontrollü adaylar üretir; Evaluator karar verir.
    4. DOĞRULAMA      — CONFLICT adaylar deterministik kanıtla araştırılır
                        (ArastirmaKuyrugu), kesin duruma iner.
    5. METİN          — TextGenerator morfolojik ek uyumuyla Türkçe cümle üretir.

Çalıştırma:
    python experiments/experience_loop/run_gercek_veri.py
"""
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.experience import (  # noqa: E402
    ArastirmaKuyrugu,
    ExperienceEvaluator,
    ExperienceGenerator,
    TextGenerator,
    cumlelerden_bilgi_aktar,
)
from hga.knowledge import DeneyimDurumu, KnowledgeStore  # noqa: E402

CIZGI = "=" * 74


def main():
    print(CIZGI)
    print("1) GERÇEK VERİ → TEMSİL (REAL_DATA aktarımı)")
    print(CIZGI)
    k = KnowledgeStore()
    cumleler = [
        "Ali ataya bindi.",
        "Ali arabaya bindi.",
        "Ali gökyüzüne bindi.",
        "Ali gökyüzüne baktı.",
    ]
    aktarilan = cumlelerden_bilgi_aktar(
        k, cumleler,
        iliski_kisitlari={
            "Binmek": {"subject_types": ["insan"],
                       "requires_object_props": {"binilebilir": 1.0}},
        })
    for u in aktarilan:
        print(f"  '{u.ozne} {u.iliski.lower()} {u.nesne}' → üçlü eklendi")
    print(f"  Bilgi özeti: {k.ozet()}")

    print("\n" + CIZGI)
    print("2) DENEYİM ÜRET + DEĞERLENDİR (durum makinesi)")
    print(CIZGI)
    ev = ExperienceEvaluator()
    adaylar = ExperienceGenerator().uret(k, relation_ids=[
        next(r.relation_id for r in k.relations.iliskiler() if r.token == "Binmek")])
    for a in adaylar:
        ev.degerlendir(a, k)
        nesne = k.entities.getir(a.object_id).token
        print(f"  Ali --Binmek--> {nesne:<8} → {a.state.value:<9} "
              f"skor={a.scores.get('weighted', '-'):}")

    print("\n" + CIZGI)
    print("3) DOĞRULAMA (araştırma kuyruğu + deterministik kanıt)")
    print(CIZGI)
    # Bilinmeyen özellikli bir nesne: CONFLICT → araştır
    k.varlik_ekle("Halı", entity_type="esya", entity_id="E_HALI")
    from hga.knowledge import ExperienceCandidate
    ali = k.entities.ad_bul("Ali")
    rid = next(r.relation_id for r in k.relations.iliskiler() if r.token == "Binmek")
    belirsiz = ExperienceCandidate(experience_id="X_PROBE", subject_id=ali,
                                   relation_id=rid, object_id="E_HALI")
    ev.degerlendir(belirsiz, k)
    print(f"  Ali --Binmek--> Halı → {belirsiz.state.value} (özellik bilinmiyor)")

    kuyruk = ArastirmaKuyrugu(
        evaluator=ev,
        dogrulayici=lambda store, aday: False)   # kanıt: Halı binilebilir DEĞİL
    kuyruk.ekle(belirsiz)
    rapor = kuyruk.isle(k)
    print(f"  Araştırma raporu: {rapor.to_dict()}")
    print(f"  Yeniden değerlendirme → {belirsiz.state.value}")

    print("\n" + CIZGI)
    print("4) METİN ÜRETİMİ (morfolojik ek uyumu)")
    print(CIZGI)
    tg = TextGenerator()
    for a in adaylar:
        if a.state == DeneyimDurumu.VALID:
            print(f"  \"{tg.cumle(k, a)}\"")

    print("\n" + CIZGI)
    print("Döngü: gerçek veri → temsil → deneyim → doğrulama → konsolide et")
    print(CIZGI)


if __name__ == "__main__":
    main()
