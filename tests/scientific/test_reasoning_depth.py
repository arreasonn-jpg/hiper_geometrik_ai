# -*- coding: utf-8 -*-
"""P0-7: C_R / C_RD güvenilir çıkarım derinliği testleri."""
import pytest

from hga.evaluation.reasoning_depth import (
    PROFILES,
    measure_reasoning_depth,
    reasoning_depth_markdown,
)


def test_smoke_profili_temel_alanlari_uretir():
    rapor = measure_reasoning_depth("smoke")
    assert rapor.c_r >= 1
    assert set(rapor.c_rd) == {16, 64}
    assert set(rapor.accuracy_grid) == set(rapor.hops)


def test_dolgu_sifir_seviyesi_zorunlu():
    """C_R tanımı dolgu=0 gerektirir; eksikse açık hata."""
    with pytest.raises(ValueError, match="dolgu=0"):
        measure_reasoning_depth("smoke", distractor_levels=[16, 64])


def test_bilinmeyen_profil_ve_esik_acik_hata():
    with pytest.raises(ValueError):
        measure_reasoning_depth("yok")
    with pytest.raises(ValueError):
        measure_reasoning_depth("smoke", reliability_threshold=0.0)
    with pytest.raises(ValueError):
        measure_reasoning_depth("smoke", reliability_threshold=1.5)


def test_c_r_izgara_tavani_isaretlenir():
    """C_R en derin hop'a eşitse bu bir sınır değil tavandır; bayrak şart."""
    rapor = measure_reasoning_depth("smoke", hops=[1, 2], distractor_levels=[0])
    if rapor.c_r == 2:
        assert rapor.c_r_grid_limited
        assert not rapor.checks["c_r_not_grid_limited"]
        assert "tavana çarptı" in " ".join(rapor.findings)


def test_derinlik_kesintisiz_tanimli():
    """2 hopta düşüp 8'de yükselmek '8 hop güvenilir' anlamına gelmemeli."""
    rapor = measure_reasoning_depth("smoke", hops=[1, 2, 4],
                                    distractor_levels=[0], slot_sayisi=4096)
    izgara = rapor.accuracy_grid
    for hop in sorted(rapor.hops):
        if izgara[hop][0] < 1.0:
            assert rapor.c_r < hop
            break


def test_bellek_kucuklugu_derinligi_dusurur():
    """Bellek slotu daraltılırsa derinlik düşmeli — metrik ölü olmamalı."""
    genis = measure_reasoning_depth("smoke", hops=[1, 2, 4],
                                    distractor_levels=[0, 64],
                                    slot_sayisi=65536, tablo_sayisi=2)
    dar = measure_reasoning_depth("smoke", hops=[1, 2, 4],
                                  distractor_levels=[0, 64],
                                  slot_sayisi=8, tablo_sayisi=1)
    assert dar.c_r <= genis.c_r
    en_dusuk_dar = min(v for alt in dar.memory_recall_grid.values()
                       for v in alt.values())
    en_dusuk_genis = min(v for alt in genis.memory_recall_grid.values()
                         for v in alt.values())
    assert en_dusuk_dar <= en_dusuk_genis


def test_esik_gevsetmek_derinligi_artirabilir_ve_raporlanir():
    sert = measure_reasoning_depth("smoke", reliability_threshold=1.0)
    gevsek = measure_reasoning_depth("smoke", reliability_threshold=0.5)
    assert gevsek.c_r >= sert.c_r
    # Eşik rapora yazılmalı ki tablo eşiksiz okunmasın.
    assert sert.reliability_threshold == 1.0
    assert "1" in reasoning_depth_markdown(sert)


def test_derinlik_dolgu_ile_monoton():
    rapor = measure_reasoning_depth("standard")
    assert rapor.checks["depth_monotone_in_distractors"]


def test_c_rd_c_r_den_buyuk_olamaz():
    rapor = measure_reasoning_depth("standard")
    for deger in rapor.c_rd.values():
        assert deger <= rapor.c_r


def test_serilestirme_string_anahtarli():
    import json
    rapor = measure_reasoning_depth("smoke")
    veri = json.loads(json.dumps(rapor.to_dict()))
    assert all(isinstance(k, str) for k in veri["c_rd"])
    assert all(isinstance(k, str) for k in veri["accuracy_grid"])


def test_markdown_c_r_ve_kapilari_icerir():
    rapor = measure_reasoning_depth("smoke")
    md = reasoning_depth_markdown(rapor)
    assert "C_R" in md and "C_RD" in md
    assert "Bellek geri çağırma ızgarası" in md
    assert "Kabul kapıları" in md


def test_profil_tanimlari_tutarli():
    for ad, ayar in PROFILES.items():
        assert 0 in ayar["distractors"], f"{ad}: dolgu=0 zorunlu"
        assert min(ayar["hops"]) == 1, f"{ad}: hop=1 negatif kontrol olarak şart"
