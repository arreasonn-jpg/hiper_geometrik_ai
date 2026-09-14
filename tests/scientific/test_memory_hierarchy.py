# -*- coding: utf-8 -*-
"""P0-8: Hot/Warm/Cold/Archive hiyerarşik bellek testleri."""
import os

import pytest

from hga.evaluation.memory_hierarchy import (
    PROFILES,
    memory_hierarchy_markdown,
    run_memory_hierarchy_benchmark,
)
from hga.memory.hierarchical import TIERS, HierarchicalMemory


@pytest.fixture
def depo(tmp_path):
    m = HierarchicalMemory(root=str(tmp_path / "mem"), hot_capacity=10,
                           warm_capacity=20, cold_capacity=30,
                           archive_segment_records=5, durable=False)
    yield m
    m.close()


# ── Yapılandırma ────────────────────────────────────────────────────────────
def test_artan_olmayan_kapasite_acik_hata(tmp_path):
    with pytest.raises(ValueError, match="artan"):
        HierarchicalMemory(root=str(tmp_path / "a"), hot_capacity=100,
                           warm_capacity=10, cold_capacity=1000)


def test_sifir_kapasite_acik_hata(tmp_path):
    with pytest.raises(ValueError):
        HierarchicalMemory(root=str(tmp_path / "b"), hot_capacity=0)
    with pytest.raises(ValueError):
        HierarchicalMemory(root=str(tmp_path / "c"), archive_segment_records=0)


def test_kapatilmis_depoya_yazilamaz(tmp_path):
    m = HierarchicalMemory(root=str(tmp_path / "d"))
    m.close()
    with pytest.raises(RuntimeError):
        m.put("k", 1)


# ── Taşma zinciri ───────────────────────────────────────────────────────────
def test_kayitlar_kapasiteye_gore_asagi_iner(depo):
    for i in range(100):
        depo.put(f"k{i}", {"v": i})
    sayim = depo.counts()
    assert sayim["hot"] == 10
    assert sayim["warm"] == 20
    assert sayim["cold"] == 30
    assert sayim["archive"] == 40


def test_kapasite_sinirlari_asilmaz(depo):
    for i in range(500):
        depo.put(f"k{i}", i)
        sayim = depo.counts()
        assert sayim["hot"] <= 10
        assert sayim["warm"] <= 20
        assert sayim["cold"] <= 30


def test_tahliye_veri_kaybetmez(depo):
    """Hiyerarşinin tüm varlık sebebi bu: aşağı inmek silinmek değildir."""
    for i in range(100):
        depo.put(f"k{i}", {"v": i})
    for i in range(100):
        deger, katman = depo.get(f"k{i}")
        assert deger is not None, f"k{i} kayboldu"
        assert deger["v"] == i
    assert depo.dropped == 0


def test_arsiv_kapaliyken_kayip_gercekten_olur(tmp_path):
    """Kayıpsızlık iddiası ancak kaybın mümkün olduğu gösterilirse anlamlı."""
    m = HierarchicalMemory(root=str(tmp_path / "e"), hot_capacity=5,
                           warm_capacity=10, cold_capacity=15,
                           archive_enabled=False, durable=False)
    try:
        for i in range(100):
            m.put(f"k{i}", i)
        assert m.dropped > 0
        assert m.get("k0")[0] is None
    finally:
        m.close()


def test_ayni_anahtar_tekrar_yazilinca_cogalmaz(depo):
    for _ in range(50):
        depo.put("tek", {"v": 1})
    toplam = sum(depo.counts()[t] for t in ("hot", "warm", "cold", "archive"))
    assert toplam == 1
    assert depo.get("tek")[0] == {"v": 1}


def test_guncelleme_eski_surumu_birakmaz(depo):
    """Eski sürüm alt katmanda kalırsa okuma yanlış değer döndürebilir."""
    for i in range(60):
        depo.put(f"k{i}", {"v": i})
    assert depo.tier_of("k0") in ("cold", "archive")
    depo.put("k0", {"v": 999})
    assert depo.tier_of("k0") == "hot"
    assert depo.get("k0")[0] == {"v": 999}


# ── Terfi ───────────────────────────────────────────────────────────────────
def test_okuma_alt_katmandan_terfi_ettirir(depo):
    for i in range(100):
        depo.put(f"k{i}", {"v": i})
    soguk = next(f"k{i}" for i in range(100) if depo.tier_of(f"k{i}") == "cold")
    depo.get(soguk)
    assert depo.tier_of(soguk) == "hot"


def test_arsiv_de_terfi_eder(depo):
    """Arşiv terfi etmezse sık okunan kayıt sonsuza dek en yavaş yolda kalır."""
    for i in range(100):
        depo.put(f"k{i}", {"v": i})
    arsiv = next(f"k{i}" for i in range(100)
                 if depo.tier_of(f"k{i}") == "archive")
    depo.get(arsiv)
    assert depo.tier_of(arsiv) == "hot"


def test_terfi_kapatilabilir(tmp_path):
    m = HierarchicalMemory(root=str(tmp_path / "f"), hot_capacity=5,
                           warm_capacity=10, cold_capacity=100,
                           promote_on_read=False, durable=False)
    try:
        for i in range(50):
            m.put(f"k{i}", i)
        soguk = next(f"k{i}" for i in range(50) if m.tier_of(f"k{i}") == "cold")
        m.get(soguk)
        assert m.tier_of(soguk) == "cold"
    finally:
        m.close()


def test_lru_en_az_kullanilani_dusurur(depo):
    for i in range(10):
        depo.put(f"k{i}", i)
    for _ in range(3):
        depo.get("k0")           # k0 sürekli okunuyor
    for i in range(10, 19):
        depo.put(f"yeni{i}", i)  # hot'a baskı
    assert depo.tier_of("k0") == "hot"


# ── Dayanıklılık ────────────────────────────────────────────────────────────
def test_cokme_ram_katmanlarini_silip_diski_birakir(depo):
    for i in range(100):
        depo.put(f"k{i}", {"v": i})
    depo.simulate_crash()
    sayim = depo.counts()
    assert sayim["hot"] == 0 and sayim["warm"] == 0
    assert sayim["cold"] > 0


def test_kurtarma_tum_kayitlari_geri_getirir(depo):
    for i in range(100):
        depo.put(f"k{i}", {"v": i})
    depo.simulate_crash()
    bilgi = depo.recover()
    assert bilgi["wal_records"] == 100
    for i in range(100):
        deger, _ = depo.get(f"k{i}")
        assert deger == {"v": i}, f"k{i} kurtarılamadı"


def test_kurtarma_diskte_olani_tekrar_yuklemez(depo):
    """Aksi halde cold'daki her kayıt hot'a dolar ve hiyerarşi bozulur."""
    for i in range(100):
        depo.put(f"k{i}", {"v": i})
    depo.simulate_crash()
    bilgi = depo.recover()
    assert bilgi["restored_to_hot"] < bilgi["wal_records"]


def test_bozuk_wal_satiri_cokme_yaratmaz(depo, tmp_path):
    for i in range(5):
        depo.put(f"k{i}", i)
    depo.flush()
    with open(os.path.join(depo.root, "wal.log"), "a", encoding="utf-8") as f:
        f.write('{"k": "yarim", "v":')   # çökme anında kesilmiş satır
    depo.simulate_crash()
    bilgi = depo.recover()
    assert bilgi["wal_records"] == 5


def test_wal_her_yazmada_buyur(depo):
    for i in range(10):
        depo.put(f"k{i}", i)
    assert depo.wal_appends == 10


# ── Ölçüm yüzeyi ────────────────────────────────────────────────────────────
def test_tier_of_terfi_ettirmez(depo):
    for i in range(100):
        depo.put(f"k{i}", {"v": i})
    soguk = next(f"k{i}" for i in range(100) if depo.tier_of(f"k{i}") == "cold")
    for _ in range(5):
        assert depo.tier_of(soguk) == "cold"


def test_disk_baytlari_gercek_dosyadan(depo):
    for i in range(100):
        depo.put(f"k{i}", {"v": i, "pad": "x" * 50})
    bayt = depo.disk_bytes()
    assert bayt["cold_sqlite"] > 0
    assert bayt["wal"] > 0
    assert bayt["total"] == (bayt["cold_sqlite"] + bayt["archive_gzip"]
                             + bayt["wal"])


def test_snapshot_tum_katmanlari_kapsar(depo):
    depo.put("k", 1)
    anlik = depo.snapshot()
    assert set(anlik["tiers"]) == set(TIERS)
    assert anlik["wal_appends"] == 1


def test_context_manager_kapatir(tmp_path):
    with HierarchicalMemory(root=str(tmp_path / "g"), durable=False) as m:
        m.put("k", 1)
        assert m.get("k")[0] == 1
    with pytest.raises(RuntimeError):
        m.put("k2", 2)


# ── Benchmark ───────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def rapor():
    return run_memory_hierarchy_benchmark("smoke", seed=1)


def test_bilinmeyen_profil_acik_hata():
    with pytest.raises(ValueError):
        run_memory_hierarchy_benchmark("yok")
    with pytest.raises(ValueError):
        run_memory_hierarchy_benchmark("smoke", payload_size=-1)


def test_profiller_artan_kapasiteli():
    for ad, ayar in PROFILES.items():
        assert ayar["hot"] <= ayar["warm"] <= ayar["cold"], ad
        assert ayar["records"] > ayar["hot"], f"{ad}: taşma tetiklenmiyor"


def test_tum_kapilar_gecer(rapor):
    for ad, sonuc in rapor.checks.items():
        assert sonuc, f"kapı düştü: {ad}"


def test_recall_tam_ve_bozulmasiz(rapor):
    assert rapor.recall["recall"] == 1.0
    assert rapor.recall["value_corruptions"] == 0


def test_tum_katmanlar_calisir(rapor):
    for katman in ("hot", "warm", "cold", "archive"):
        assert rapor.tier_distribution[katman] > 0


def test_gecikme_katman_basina_ve_siralı(rapor):
    """Tek ortalama gecikme çok modlu dağılımı temsil etmez."""
    hot = rapor.read_latency_by_tier["hot"]
    cold = rapor.read_latency_by_tier["cold"]
    assert hot["n"] and cold["n"]
    assert hot["p50"] < cold["p50"]
    for katman in TIERS:
        m = rapor.read_latency_by_tier[katman]
        if m["n"]:
            assert m["p50"] <= m["p95"] <= m["p99"] <= m["max"]


def test_cokme_kurtarma_tam(rapor):
    assert rapor.crash_recovery["post_recovery_recall"] == 1.0


def test_kontrol_grubu_kaybi_gosterir(rapor):
    assert rapor.drop_behavior["drop_expected_at_this_scale"]
    assert rapor.drop_behavior["archive_disabled_dropped"] > 0


def test_disk_ve_ram_olculur(rapor):
    assert rapor.disk["total"] > 0
    assert rapor.ram["rss_after_write_bytes"] is not None


def test_markdown_tum_bolumleri_icerir(rapor):
    md = memory_hierarchy_markdown(rapor)
    for baslik in ("Katman dağılımı", "Okuma gecikmesi", "Recall",
                   "Tahliye politikası", "Çökme kurtarma",
                   "Kontrol grubu", "Kabul kapıları"):
        assert baslik in md


def test_rapor_serilestirilebilir(rapor):
    import json
    assert json.loads(json.dumps(rapor.to_dict()))["protocol"]
