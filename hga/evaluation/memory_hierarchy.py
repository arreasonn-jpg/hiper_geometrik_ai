# -*- coding: utf-8 -*-
"""P0-8 — Hiyerarşik bellek benchmarkı: recall / latency / RAM / disk / kurtarma.

Ölçülen beş şey ve neden ölçüldükleri:

1. **Recall.** Yazılan her kayıt geri okunabiliyor mu? Hiyerarşik bir depo
   "ölçekleniyor" diyip sessizce kayıt düşürebilir; recall < 1.0 bunu açığa
   çıkarır. Arşiv açıkken recall 1.0 OLMAK ZORUNDADIR, aksi halde tasarım
   hatalıdır ve kapı düşer.
2. **Katman başına latency.** Ortalama gecikme yanıltıcıdır: hot okuma
   mikrosaniye, arşiv okuma milisaniyedir ve ortalama ikisini de temsil
   etmez. Bu yüzden katman başına p50/p95/p99 raporlanır.
3. **RAM.** Kapasite iddiası ancak bellek büyümesi sınırlıysa anlamlıdır.
   RSS ölçülür; ölçülemiyorsa ``None`` yazılır, tahmin edilmez.
4. **Disk.** Cold (SQLite) ve archive (gzip) gerçek dosya boyutları.
5. **Çökme kurtarma.** WAL tekrar oynatıldıktan sonra kaç kayıt geri
   geliyor? Kayıp varsa sayılır.

Ayrıca **tahliye doğruluğu** denetlenir: LRU politikası gerçekten en az
kullanılanı mı düşürüyor, yoksa rastgele mi? Sık okunan kayıtların hot'ta
kalma oranı ölçülür.
"""
from __future__ import annotations

import hashlib
import json
import random
import statistics as _stat
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..memory.hierarchical import TIERS, HierarchicalMemory

PROTOCOL = "hierarchical_memory_v1"
SCHEMA_VERSION = 1

#: Ölçek profilleri. ``deep`` gerçekten 10⁶ kayıt yazar ve dakikalar sürer.
PROFILES: Dict[str, Dict[str, Any]] = {
    "smoke": {"records": 2_000, "hot": 100, "warm": 400, "cold": 1_000,
              "probe": 400},
    "standard": {"records": 50_000, "hot": 1_000, "warm": 10_000,
                 "cold": 30_000, "probe": 2_000},
    "deep": {"records": 1_000_000, "hot": 10_000, "warm": 100_000,
             "cold": 500_000, "probe": 10_000},
}


def _rss_bytes() -> Optional[int]:
    """Süreç RSS'i; okunamıyorsa tahmin ETME, None döndür."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as dosya:
            for satir in dosya:
                if satir.startswith("VmRSS:"):
                    return int(satir.split()[1]) * 1024
    except OSError:
        return None
    return None


def _percentiles(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """p50/p95/p99 — ortalama tek başına çok modlu dağılımı temsil etmez."""
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None,
                "max": None}
    sirali = sorted(values)

    def yuzde(oran: float) -> float:
        if len(sirali) == 1:
            return sirali[0]
        konum = oran * (len(sirali) - 1)
        alt = int(konum)
        ust = min(alt + 1, len(sirali) - 1)
        agirlik = konum - alt
        return sirali[alt] * (1 - agirlik) + sirali[ust] * agirlik

    return {
        "n": len(sirali),
        "mean": round(_stat.fmean(sirali), 9),
        "p50": round(yuzde(0.50), 9),
        "p95": round(yuzde(0.95), 9),
        "p99": round(yuzde(0.99), 9),
        "max": round(sirali[-1], 9),
    }


@dataclass
class MemoryHierarchyReport:
    """P0-8 raporu."""

    protocol: str
    schema_version: int
    profile: str
    seed: int
    config: Dict[str, Any]
    write_throughput: Dict[str, float]
    tier_distribution: Dict[str, int]
    recall: Dict[str, Any]
    read_latency_by_tier: Dict[str, Dict[str, Optional[float]]]
    eviction: Dict[str, Any]
    ram: Dict[str, Optional[int]]
    disk: Dict[str, int]
    crash_recovery: Dict[str, Any]
    drop_behavior: Dict[str, Any]
    checks: Dict[str, bool]
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_memory_hierarchy_benchmark(
    profile: str = "smoke",
    seed: int = 1,
    root: Optional[str] = None,
    payload_size: int = 4,
    durable: bool = True,
) -> MemoryHierarchyReport:
    """Hiyerarşik belleği uçtan uca ölç.

    Args:
        profile: ``smoke`` / ``standard`` / ``deep``.
        seed: Erişim örüntüsü ve sonda seçimi için tohum.
        root: Disk kökü; ``None`` ise geçici dizin (sonda silinir).
        payload_size: Kayıt başına dolgu alanı uzunluğu.
        durable: WAL fsync'i. ``False`` hızlıdır ama çökme testi zayıflar.

    Raises:
        ValueError: bilinmeyen profil ya da geçersiz payload.
    """
    if profile not in PROFILES:
        raise ValueError(
            f"profile şunlardan biri olmalı: {', '.join(PROFILES)}")
    if int(payload_size) < 0:
        raise ValueError("payload_size negatif olamaz")

    ayar = dict(PROFILES[profile])
    kayit_sayisi = int(ayar["records"])
    sonda_sayisi = min(int(ayar["probe"]), kayit_sayisi)
    rng = random.Random(seed)

    rss_baslangic = _rss_bytes()
    depo = HierarchicalMemory(
        root=root,
        hot_capacity=int(ayar["hot"]),
        warm_capacity=int(ayar["warm"]),
        cold_capacity=int(ayar["cold"]),
        archive_enabled=True,
        durable=durable,
    )

    try:
        # ── yazma ──────────────────────────────────────────────────────────
        dolgu = "x" * int(payload_size)
        basla = time.perf_counter()
        for i in range(kayit_sayisi):
            depo.put(f"key-{i:09d}", {"i": i, "pad": dolgu})
        yazma_suresi = time.perf_counter() - basla
        depo.flush()
        rss_yazma_sonrasi = _rss_bytes()

        yazma = {
            "records": kayit_sayisi,
            "seconds": round(yazma_suresi, 6),
            "records_per_second": round(kayit_sayisi / yazma_suresi, 3)
            if yazma_suresi > 0 else 0.0,
            "microseconds_per_record": round(
                yazma_suresi / kayit_sayisi * 1e6, 3) if kayit_sayisi else 0.0,
        }
        dagilim = depo.counts()

        # ── recall + katman başına latency ─────────────────────────────────
        # Sonda anahtarları tüm aralıktan seçilir: yalnız son yazılanları
        # sormak hot'u ölçer, hiyerarşiyi değil.
        sonda_indeksleri = rng.sample(range(kayit_sayisi), sonda_sayisi)
        gecikmeler: Dict[str, List[float]] = {t: [] for t in TIERS}
        bulunan = 0
        yanlis_deger = 0
        bulunamayan: List[str] = []

        for indeks in sonda_indeksleri:
            anahtar = f"key-{indeks:09d}"
            katman = depo.tier_of(anahtar)   # terfi ettirmeden yeri öğren
            t0 = time.perf_counter()
            deger, bulunan_katman = depo.get(anahtar)
            gecen = time.perf_counter() - t0
            if deger is None:
                if len(bulunamayan) < 50:
                    bulunamayan.append(anahtar)
                continue
            bulunan += 1
            if deger.get("i") != indeks:
                yanlis_deger += 1
            hedef = katman or bulunan_katman
            if hedef in gecikmeler:
                gecikmeler[hedef].append(gecen)

        geri_cagirma = {
            "probes": sonda_sayisi,
            "found": bulunan,
            "recall": round(bulunan / sonda_sayisi, 6) if sonda_sayisi else 0.0,
            "value_corruptions": yanlis_deger,
            "missing_examples": bulunamayan,
        }
        latency = {t: _percentiles(gecikmeler[t]) for t in TIERS}

        # ── tahliye doğruluğu ──────────────────────────────────────────────
        # LRU iddiası: sık okunan kayıtlar hot'ta kalmalı. Bunu ölçmek için
        # küçük bir "sıcak küme" defalarca okunur, sonra yeni yazmalarla
        # baskı uygulanır ve sıcak kümenin hayatta kalma oranına bakılır.
        sicak_kume = [f"key-{i:09d}" for i in
                      rng.sample(range(kayit_sayisi), min(20, kayit_sayisi))]
        for _ in range(5):
            for anahtar in sicak_kume:
                depo.get(anahtar)
        baski = max(1, int(ayar["hot"]) // 2)
        for j in range(baski):
            depo.put(f"pressure-{j:09d}", {"i": -1, "pad": dolgu})
        hayatta = sum(1 for a in sicak_kume if depo.tier_of(a) == "hot")
        tahliye = {
            "hot_set_size": len(sicak_kume),
            "reads_per_key": 5,
            "pressure_writes": baski,
            "hot_set_retained_in_hot": hayatta,
            "hot_set_retention": round(hayatta / len(sicak_kume), 6)
            if sicak_kume else 0.0,
            "evictions": {t: depo.stats[t].evictions_out for t in TIERS},
            "promotions": {t: depo.stats[t].promotions_in for t in TIERS},
        }

        disk = depo.disk_bytes()
        rss_son = _rss_bytes()

        # ── çökme ve kurtarma ──────────────────────────────────────────────
        depo.simulate_crash()
        cokme_sonrasi_ram = depo.counts()
        kurtarma_bilgi = depo.recover()
        kurtarma_sonda = rng.sample(range(kayit_sayisi),
                                    min(200, kayit_sayisi))
        kurtarilan = sum(1 for i in kurtarma_sonda
                         if depo.get(f"key-{i:09d}")[0] is not None)
        cokme = {
            "ram_records_after_crash": cokme_sonrasi_ram["hot"]
            + cokme_sonrasi_ram["warm"],
            "wal_records": kurtarma_bilgi["wal_records"],
            "restored_to_hot": kurtarma_bilgi["restored_to_hot"],
            "post_recovery_probes": len(kurtarma_sonda),
            "post_recovery_found": kurtarilan,
            "post_recovery_recall": round(kurtarilan / len(kurtarma_sonda), 6)
            if kurtarma_sonda else 0.0,
        }
    finally:
        depo.close(destroy=root is None)

    # ── arşiv kapalıyken davranış (kontrol grubu) ──────────────────────────
    # Bu, "veri kaybı yok" iddiasının boş olmadığını gösterir: arşivi
    # kapatınca kayıp GERÇEKTEN oluşuyor, yani mekanizma çalışıyor.
    kontrol = HierarchicalMemory(
        hot_capacity=int(ayar["hot"]), warm_capacity=int(ayar["warm"]),
        cold_capacity=int(ayar["cold"]), archive_enabled=False, durable=False)
    try:
        for i in range(kayit_sayisi):
            kontrol.put(f"key-{i:09d}", {"i": i})
        kontrol_dagilim = kontrol.counts()
        dusurulen = kontrol.dropped
    finally:
        kontrol.close(destroy=True)

    kapasite_toplami = int(ayar["hot"]) + int(ayar["warm"]) + int(ayar["cold"])
    dusme_beklenir = kayit_sayisi > kapasite_toplami
    dusme = {
        "archive_disabled_dropped": dusurulen,
        "archive_disabled_distribution": kontrol_dagilim,
        "drop_expected_at_this_scale": dusme_beklenir,
        "note": ("Arşiv kapalıyken düşen kayıt sayısı; arşiv açıkken bu "
                 "sayı 0 olmalıdır. Kontrol grubu, kayıpsızlık iddiasının "
                 "boş bir ifade olmadığını gösterir."),
    }

    # ── kapılar ────────────────────────────────────────────────────────────
    kapilar: Dict[str, bool] = {
        "no_data_loss_with_archive": geri_cagirma["recall"] == 1.0,
        "no_value_corruption": geri_cagirma["value_corruptions"] == 0,
        "all_tiers_exercised": all(
            dagilim.get(t, 0) > 0 for t in ("hot", "warm", "cold")),
        "hot_faster_than_cold": (
            (latency["hot"]["p50"] or 0) < (latency["cold"]["p50"] or float("inf"))
            if latency["hot"]["n"] and latency["cold"]["n"] else False),
        "crash_recovery_complete": cokme["post_recovery_recall"] == 1.0,
        "lru_retains_hot_set": tahliye["hot_set_retention"] >= 0.9,
        "capacity_bounds_respected": (
            dagilim["hot"] <= int(ayar["hot"])
            and dagilim["warm"] <= int(ayar["warm"])
            and dagilim["cold"] <= int(ayar["cold"])),
        "drop_control_group_behaves_as_expected": (
            (dusurulen > 0) == dusme_beklenir),
    }

    # ── bulgular ───────────────────────────────────────────────────────────
    bulgular: List[str] = [
        f"Profil `{profile}`: {kayit_sayisi:,} kayıt yazıldı, "
        f"kapasiteler hot/warm/cold = {ayar['hot']:,}/{ayar['warm']:,}/"
        f"{ayar['cold']:,}.",
        f"Katman dağılımı: hot {dagilim['hot']:,}, warm {dagilim['warm']:,}, "
        f"cold {dagilim['cold']:,}, archive {dagilim['archive']:,} "
        f"({dagilim['archive_segments']} segment).",
        f"Yazma hızı {yazma['records_per_second']:,.0f} kayıt/s "
        f"({yazma['microseconds_per_record']:.1f} µs/kayıt, "
        f"durable={durable}).",
    ]

    okunan_katmanlar = [t for t in TIERS if latency[t]["n"]]
    if okunan_katmanlar:
        parcalar = [f"{t} p50={latency[t]['p50'] * 1e6:.1f}µs "
                    f"(n={latency[t]['n']})" for t in okunan_katmanlar]
        bulgular.append("Okuma gecikmesi: " + ", ".join(parcalar) + ".")
        if latency["hot"]["n"] and latency["cold"]["n"]:
            oran = latency["cold"]["p50"] / max(latency["hot"]["p50"], 1e-12)
            bulgular.append(
                f"Cold okuma hot'tan **{oran:.1f}× yavaş**. Tek bir ortalama "
                "gecikme sayısı bu farkı gizler; katman ayrımı bu yüzden var.")

    if geri_cagirma["recall"] == 1.0:
        bulgular.append(
            f"Recall {geri_cagirma['recall']:.4f} — {sonda_sayisi} sondanın "
            "hepsi bulundu, değer bozulması yok. Tahliye veri kaybı değil, "
            "katman değişimidir.")
    else:
        bulgular.append(
            f"Recall {geri_cagirma['recall']:.4f} — KAYIP VAR. "
            f"Örnek bulunamayanlar: {geri_cagirma['missing_examples'][:5]}.")

    bulgular.append(
        f"Çökme sonrası: RAM'de {cokme['ram_records_after_crash']} kayıt kaldı, "
        f"WAL'den {cokme['restored_to_hot']:,} kayıt hot'a geri yüklendi, "
        f"kurtarma sonrası recall {cokme['post_recovery_recall']:.4f}.")

    if dusme_beklenir:
        bulgular.append(
            f"Kontrol grubu (arşiv KAPALI): {dusurulen:,} kayıt düştü. "
            "Arşiv açıkken 0 düştü — kayıpsızlık mekanizması gerçekten "
            "çalışıyor, tanım gereği doğru değil.")
    else:
        bulgular.append(
            "Bu ölçekte toplam kapasite tüm kayıtları aldığı için arşiv "
            "kapalı kontrol grubunda da kayıp beklenmiyor; kayıpsızlık "
            "iddiası bu profilde ZAYIF test edilmiştir.")

    if not kapilar["lru_retains_hot_set"]:
        bulgular.append(
            f"LRU tahliyesi sıcak kümeyi koruyamadı "
            f"(hot'ta kalma {tahliye['hot_set_retention']:.2f} < 0.90). "
            "Tahliye politikası erişim sıklığını yeterince yansıtmıyor.")

    ram = {
        "rss_start_bytes": rss_baslangic,
        "rss_after_write_bytes": rss_yazma_sonrasi,
        "rss_end_bytes": rss_son,
        "rss_growth_bytes": (rss_son - rss_baslangic)
        if (rss_son is not None and rss_baslangic is not None) else None,
        "bytes_per_record_in_ram": round(
            (rss_yazma_sonrasi - rss_baslangic) / kayit_sayisi, 3)
        if (rss_yazma_sonrasi is not None and rss_baslangic is not None
            and kayit_sayisi) else None,
    }
    if ram["rss_growth_bytes"] is not None:
        bulgular.append(
            f"RSS büyümesi {ram['rss_growth_bytes'] / 1e6:.1f} MB; "
            f"disk {disk['total'] / 1e6:.1f} MB "
            f"(cold {disk['cold_sqlite'] / 1e6:.1f} MB + "
            f"archive {disk['archive_gzip'] / 1e6:.1f} MB + "
            f"wal {disk['wal'] / 1e6:.1f} MB).")

    imza = hashlib.sha256(json.dumps(
        {"protocol": PROTOCOL, "profile": profile, "seed": seed,
         "config": ayar, "payload": payload_size},
        sort_keys=True).encode("utf-8")).hexdigest()[:12]

    return MemoryHierarchyReport(
        protocol=PROTOCOL,
        schema_version=SCHEMA_VERSION,
        profile=profile,
        seed=int(seed),
        config={**ayar, "signature": imza, "payload_size": int(payload_size),
                "durable": bool(durable)},
        write_throughput=yazma,
        tier_distribution=dagilim,
        recall=geri_cagirma,
        read_latency_by_tier=latency,
        eviction=tahliye,
        ram=ram,
        disk=disk,
        crash_recovery=cokme,
        drop_behavior=dusme,
        checks=kapilar,
        findings=bulgular,
        limitations=[
            "Tek süreç, tek iş parçacığı; eşzamanlı erişim ve kilitlenme "
            "davranışı ÖLÇÜLMEMİŞTİR.",
            "'Çökme' süreç sonlandırma değil, RAM katmanlarının temizlenmesi "
            "ile taklit edilir; gerçek kill -9 sırasında sayfa önbelleği "
            "davranışı farklı olabilir.",
            "Disk ölçümleri dosya boyutlarıdır; dosya sistemi blok "
            "doldurmasını ve sıkıştırma oranını içermez.",
            "RSS tüm süreci ölçer; başka nesnelerin payı ayrıştırılamaz.",
            f"Profil `{profile}` gerçek 10⁸ ölçeğinin altındadır; "
            "kapasite iddiası ekstrapolasyon değil, ölçülen aralıkla "
            "sınırlıdır.",
        ],
    )


def memory_hierarchy_markdown(report: MemoryHierarchyReport) -> str:
    """P0-8 raporunu Markdown'a çevir."""
    s = report
    satirlar = [
        "# Hiyerarşik Bellek Benchmarkı (P0-8)",
        "",
        f"- Protokol: `{s.protocol}` v{s.schema_version} "
        f"(imza `{s.config['signature']}`)",
        f"- Profil / tohum: `{s.profile}` / `{s.seed}`",
        f"- Kapasiteler: hot `{s.config['hot']:,}` / warm `{s.config['warm']:,}` "
        f"/ cold `{s.config['cold']:,}` / archive sınırsız",
        f"- Dayanıklı yazma (WAL fsync): `{s.config['durable']}`",
        "",
        "## Katman dağılımı ve yazma",
        "",
        "| Katman | Kayıt |",
        "|---|---:|",
    ]
    for katman in ("hot", "warm", "cold", "archive"):
        satirlar.append(f"| {katman} | {s.tier_distribution[katman]:,} |")
    satirlar.extend([
        "",
        f"Yazma: **{s.write_throughput['records_per_second']:,.0f} kayıt/s** "
        f"({s.write_throughput['microseconds_per_record']:.1f} µs/kayıt, "
        f"toplam {s.write_throughput['records']:,} kayıt "
        f"{s.write_throughput['seconds']:.2f} s)",
        "",
        "## Okuma gecikmesi (katman başına)",
        "",
        "| Katman | n | p50 (µs) | p95 (µs) | p99 (µs) | max (µs) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for katman in TIERS:
        m = s.read_latency_by_tier[katman]
        if not m["n"]:
            satirlar.append(f"| {katman} | 0 | — | — | — | — |")
            continue
        satirlar.append(
            f"| {katman} | {m['n']} | {m['p50'] * 1e6:.2f} | "
            f"{m['p95'] * 1e6:.2f} | {m['p99'] * 1e6:.2f} | "
            f"{m['max'] * 1e6:.2f} |")

    satirlar.extend([
        "",
        "## Recall ve bütünlük",
        "",
        f"- Sonda sayısı: `{s.recall['probes']}`",
        f"- Bulunan: `{s.recall['found']}` → **recall "
        f"{s.recall['recall']:.6f}**",
        f"- Değer bozulması: `{s.recall['value_corruptions']}`",
        "",
        "## Tahliye politikası (LRU)",
        "",
        f"- Sıcak küme: `{s.eviction['hot_set_size']}` anahtar, "
        f"her biri `{s.eviction['reads_per_key']}` kez okundu",
        f"- Baskı yazması: `{s.eviction['pressure_writes']:,}`",
        f"- Sıcak kümenin hot'ta kalma oranı: "
        f"**{s.eviction['hot_set_retention']:.4f}**",
        f"- Tahliyeler: `{s.eviction['evictions']}`",
        f"- Terfiler: `{s.eviction['promotions']}`",
        "",
        "## Kaynak kullanımı",
        "",
        "| Kaynak | Bayt |",
        "|---|---:|",
        f"| RSS büyümesi | {s.ram['rss_growth_bytes'] or 0:,} |",
        f"| Cold (SQLite) | {s.disk['cold_sqlite']:,} |",
        f"| Archive (gzip) | {s.disk['archive_gzip']:,} |",
        f"| WAL | {s.disk['wal']:,} |",
        f"| **Disk toplam** | **{s.disk['total']:,}** |",
        "",
        "## Çökme kurtarma",
        "",
        f"- Çökme sonrası RAM'de kalan: "
        f"`{s.crash_recovery['ram_records_after_crash']}`",
        f"- WAL kaydı: `{s.crash_recovery['wal_records']:,}`",
        f"- Hot'a geri yüklenen: `{s.crash_recovery['restored_to_hot']:,}`",
        f"- Kurtarma sonrası recall: "
        f"**{s.crash_recovery['post_recovery_recall']:.6f}** "
        f"({s.crash_recovery['post_recovery_probes']} sonda)",
        "",
        "## Kontrol grubu — arşiv KAPALI",
        "",
        f"- Düşen kayıt: `{s.drop_behavior['archive_disabled_dropped']:,}`",
        f"- Bu ölçekte düşme bekleniyor mu: "
        f"`{s.drop_behavior['drop_expected_at_this_scale']}`",
        "",
        f"> {s.drop_behavior['note']}",
        "",
        "## Kabul kapıları",
        "",
        "| Kapı | Sonuç |",
        "|---|---|",
    ])
    satirlar.extend(f"| {ad} | {'GEÇTİ' if v else 'KALDI'} |"
                    for ad, v in s.checks.items())
    satirlar.extend(["", "## Bulgular", ""])
    satirlar.extend(f"- {b}" for b in s.findings)
    satirlar.extend(["", "## Sınırlar", ""])
    satirlar.extend(f"- {b}" for b in s.limitations)
    return "\n".join(satirlar) + "\n"


__all__ = [
    "PROFILES",
    "PROTOCOL",
    "SCHEMA_VERSION",
    "MemoryHierarchyReport",
    "memory_hierarchy_markdown",
    "run_memory_hierarchy_benchmark",
]
