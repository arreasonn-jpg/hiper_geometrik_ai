# -*- coding: utf-8 -*-
"""P0-8 — Hot/Warm/Cold/Archive hiyerarşik bellek.

Tek düzlemli ``DeneyimSlotlari`` sabit boyutludur: kapasite dolunca ya
çakışma üretir ya da eski kaydı düşürür. 10⁸ ölçeğinde bu iki davranış da
kabul edilemez — biri yanlış cevap, diğeri sessiz veri kaybıdır.

Bu modül dört katmanlı bir depo kurar:

======== ============ ================= ==========================
Katman   Ortam        Tipik kapasite    Erişim maliyeti
======== ============ ================= ==========================
hot      RAM (dict)   10⁴               ~O(1), mikrosaniye
warm     RAM (kompakt)10⁶               ~O(1), mikrosaniye
cold     Disk SQLite  10⁸+              indeksli arama, milisaniye
archive  Disk gzip    sınırsız          segment taraması, yavaş
======== ============ ================= ==========================

Tasarım kararları ve gerekçeleri:

* **Tahliye kaybetmez, düşürür.** Hot dolunca en az kullanılan kayıt warm'a,
  warm dolunca cold'a, cold dolunca archive'a iner. Hiçbir katman veriyi
  silmez; ``drop`` yalnız ``archive_enabled=False`` ise olur ve sayılır.
* **Okuma terfi ettirir.** Cold'da bulunan bir kayıt hot'a taşınır; erişim
  örüntüsü yerelse ikinci okuma ucuzdur. Bu, ölçülen latency dağılımının
  neden çift tepeli olduğunu açıklar.
* **Dayanıklılık WAL ile.** Her yazma önce append-only log'a gider, sonra
  katmana. Çökme sonrası ``recover()`` log'u tekrar oynatır. Bu yüzden
  "crash recovery" bir iddia değil, ölçülebilir bir işlemdir.
* **Sayaçlar tam, örnekler sınırlı.** 10⁸ kayıtta her olayı listelemek
  belleği ikinci kez tüketir; sayaçlar eksiksiz tutulur, örnekler kırpılır.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import sqlite3
import tempfile
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

TIERS: Tuple[str, ...] = ("hot", "warm", "cold", "archive")

#: Varsayılan kapasiteler — kullanıcı isteğindeki 10⁴/10⁶/10⁸ hedefi.
DEFAULT_CAPACITY: Dict[str, int] = {
    "hot": 10_000,
    "warm": 1_000_000,
    "cold": 100_000_000,
}

WAL_FILE = "wal.log"
COLD_FILE = "cold.sqlite3"
ARCHIVE_DIR = "archive"
ARCHIVE_SEGMENT_RECORDS = 10_000


@dataclass
class TierStats:
    """Tek katmanın sayaçları."""

    name: str
    capacity: int
    records: int = 0
    writes: int = 0
    hits: int = 0
    misses: int = 0
    evictions_out: int = 0
    promotions_in: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HierarchicalMemory:
    """Dört katmanlı, çökmeye dayanıklı deneyim deposu.

    Args:
        root: Disk katmanlarının kök dizini. ``None`` ise geçici dizin
            oluşturulur ve ``close(destroy=True)`` ile silinir.
        hot_capacity / warm_capacity / cold_capacity: Katman kapasiteleri.
        archive_enabled: ``False`` ise cold taşınca kayıt DÜŞÜRÜLÜR ve
            ``dropped`` sayacı artar. Varsayılan ``True``: veri kaybı yok.
        durable: ``True`` ise her yazma WAL'e fsync'lenir. Yavaştır ama
            çökme testinin anlamlı olması için gereklidir.
        promote_on_read: Okumada alt katmandan hot'a terfi.

    Raises:
        ValueError: kapasiteler artan sırada değilse ya da pozitif değilse.
    """

    def __init__(self,
                 root: Optional[str] = None,
                 hot_capacity: int = DEFAULT_CAPACITY["hot"],
                 warm_capacity: int = DEFAULT_CAPACITY["warm"],
                 cold_capacity: int = DEFAULT_CAPACITY["cold"],
                 archive_enabled: bool = True,
                 durable: bool = True,
                 promote_on_read: bool = True,
                 archive_segment_records: int = ARCHIVE_SEGMENT_RECORDS) -> None:
        for ad, deger in (("hot_capacity", hot_capacity),
                          ("warm_capacity", warm_capacity),
                          ("cold_capacity", cold_capacity)):
            if int(deger) < 1:
                raise ValueError(f"{ad} >= 1 olmalı")
        if not (hot_capacity <= warm_capacity <= cold_capacity):
            raise ValueError(
                "kapasiteler artan olmalı: hot <= warm <= cold "
                f"(verilen: {hot_capacity}, {warm_capacity}, {cold_capacity})")
        if int(archive_segment_records) < 1:
            raise ValueError("archive_segment_records >= 1 olmalı")

        self._gecici = root is None
        self.root = root or tempfile.mkdtemp(prefix="hga-mem-")
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(os.path.join(self.root, ARCHIVE_DIR), exist_ok=True)

        self.archive_enabled = bool(archive_enabled)
        self.durable = bool(durable)
        self.promote_on_read = bool(promote_on_read)
        self.archive_segment_records = int(archive_segment_records)

        self.stats: Dict[str, TierStats] = {
            "hot": TierStats("hot", int(hot_capacity)),
            "warm": TierStats("warm", int(warm_capacity)),
            "cold": TierStats("cold", int(cold_capacity)),
            "archive": TierStats("archive", -1),
        }
        self.dropped = 0
        self.wal_appends = 0
        self.recoveries = 0

        # Hot ve warm: LRU sırası korunan sözlükler.
        self._hot: "OrderedDict[str, Any]" = OrderedDict()
        self._warm: "OrderedDict[str, Any]" = OrderedDict()

        self._cold = sqlite3.connect(os.path.join(self.root, COLD_FILE))
        self._cold.execute(
            "CREATE TABLE IF NOT EXISTS kayit ("
            " anahtar TEXT PRIMARY KEY, deger TEXT NOT NULL, tik INTEGER)")
        self._cold.execute(
            "CREATE INDEX IF NOT EXISTS kayit_tik ON kayit(tik)")
        self._cold.commit()

        self._wal_path = os.path.join(self.root, WAL_FILE)
        self._wal = open(self._wal_path, "a", encoding="utf-8")
        self._arsiv_tampon: List[Tuple[str, Any]] = []
        self._tik = 0
        self._kapali = False

    # ── WAL ────────────────────────────────────────────────────────────────
    def _wal_yaz(self, anahtar: str, deger: Any) -> None:
        """Yazmayı önce log'a al; katmanlar sonra güncellenir.

        Sıra önemlidir: log'a yazılmamış bir kayıt çökme sonrası
        kurtarılamaz, katmana yazılmamış ama log'da olan kayıt kurtarılır.
        Bu yüzden log ÖNCE gelir.
        """
        self._wal.write(json.dumps({"k": anahtar, "v": deger},
                                   ensure_ascii=False) + "\n")
        self.wal_appends += 1
        if self.durable:
            self._wal.flush()
            os.fsync(self._wal.fileno())

    # ── yazma ──────────────────────────────────────────────────────────────
    def put(self, key: str, value: Any) -> None:
        """Kaydı hot katmana yaz; taşma zinciri gerekiyorsa tetiklenir.

        Raises:
            RuntimeError: depo kapatılmışsa.
        """
        if self._kapali:
            raise RuntimeError("kapatılmış depoya yazılamaz")
        anahtar = str(key)
        self._tik += 1
        self._wal_yaz(anahtar, value)

        # Aynı anahtar alt katmanlarda varsa, oradaki eski sürüm ölü kalmasın.
        self._alt_katmanlardan_sil(anahtar)

        self._hot[anahtar] = value
        self._hot.move_to_end(anahtar)
        self.stats["hot"].writes += 1
        self._hot_tasir()

    def _alt_katmanlardan_sil(self, anahtar: str) -> None:
        self._warm.pop(anahtar, None)
        self._cold.execute("DELETE FROM kayit WHERE anahtar = ?", (anahtar,))

    def _hot_tasir(self) -> None:
        while len(self._hot) > self.stats["hot"].capacity:
            anahtar, deger = self._hot.popitem(last=False)  # en eski (LRU)
            self.stats["hot"].evictions_out += 1
            self._warm[anahtar] = deger
            self._warm.move_to_end(anahtar)
            self.stats["warm"].writes += 1
            self._warm_tasir()

    def _warm_tasir(self) -> None:
        while len(self._warm) > self.stats["warm"].capacity:
            anahtar, deger = self._warm.popitem(last=False)
            self.stats["warm"].evictions_out += 1
            self._cold.execute(
                "INSERT OR REPLACE INTO kayit(anahtar, deger, tik) VALUES (?,?,?)",
                (anahtar, json.dumps(deger, ensure_ascii=False), self._tik))
            self.stats["cold"].writes += 1
            self._cold_tasir()

    def _cold_tasir(self) -> None:
        (sayi,) = self._cold.execute("SELECT COUNT(*) FROM kayit").fetchone()
        fazla = sayi - self.stats["cold"].capacity
        if fazla <= 0:
            return
        satirlar = self._cold.execute(
            "SELECT anahtar, deger FROM kayit ORDER BY tik ASC LIMIT ?",
            (fazla,)).fetchall()
        for anahtar, ham in satirlar:
            self._cold.execute("DELETE FROM kayit WHERE anahtar = ?", (anahtar,))
            self.stats["cold"].evictions_out += 1
            if self.archive_enabled:
                self._arsive_yaz(anahtar, json.loads(ham))
            else:
                # Arşiv kapalıysa bu GERÇEK bir veri kaybıdır ve sayılır.
                self.dropped += 1

    def _arsive_yaz(self, anahtar: str, deger: Any) -> None:
        self._arsiv_tampon.append((anahtar, deger))
        self.stats["archive"].writes += 1
        if len(self._arsiv_tampon) >= self.archive_segment_records:
            self._arsiv_bosalt()

    def _arsiv_bosalt(self) -> None:
        if not self._arsiv_tampon:
            return
        dizin = os.path.join(self.root, ARCHIVE_DIR)
        yol = os.path.join(dizin, f"segment-{len(os.listdir(dizin)):06d}.jsonl.gz")
        with gzip.open(yol, "wt", encoding="utf-8") as dosya:
            for anahtar, deger in self._arsiv_tampon:
                dosya.write(json.dumps({"k": anahtar, "v": deger},
                                       ensure_ascii=False) + "\n")
        self._arsiv_tampon.clear()

    # ── okuma ──────────────────────────────────────────────────────────────
    def get(self, key: str) -> Tuple[Optional[Any], Optional[str]]:
        """Kaydı ara; ``(değer, bulunduğu_katman)`` döndür.

        Bulunamazsa ``(None, None)``. Alt katmanda bulunan kayıt
        ``promote_on_read`` ise hot'a terfi eder.
        """
        anahtar = str(key)

        if anahtar in self._hot:
            self._hot.move_to_end(anahtar)
            self.stats["hot"].hits += 1
            return self._hot[anahtar], "hot"
        self.stats["hot"].misses += 1

        if anahtar in self._warm:
            deger = self._warm[anahtar]
            self.stats["warm"].hits += 1
            if self.promote_on_read:
                del self._warm[anahtar]
                self._hot[anahtar] = deger
                self._hot.move_to_end(anahtar)
                self.stats["hot"].promotions_in += 1
                self._hot_tasir()
            return deger, "warm"
        self.stats["warm"].misses += 1

        satir = self._cold.execute(
            "SELECT deger FROM kayit WHERE anahtar = ?", (anahtar,)).fetchone()
        if satir is not None:
            deger = json.loads(satir[0])
            self.stats["cold"].hits += 1
            if self.promote_on_read:
                self._cold.execute("DELETE FROM kayit WHERE anahtar = ?",
                                   (anahtar,))
                self._hot[anahtar] = deger
                self._hot.move_to_end(anahtar)
                self.stats["hot"].promotions_in += 1
                self._hot_tasir()
            return deger, "cold"
        self.stats["cold"].misses += 1

        deger = self._arsivde_ara(anahtar)
        if deger is not None:
            self.stats["archive"].hits += 1
            if self.promote_on_read:
                # Arşiv de terfi ETMELİ. Aksi halde sık okunan bir arşiv
                # kaydı sonsuza dek en yavaş yolda kalır ve terfi zinciri
                # cold'da kopar — katman hiyerarşisi tek yönlü bir çöp
                # kutusuna dönüşür. Segment dosyası yerinde düzenlenmez
                # (gzip append-only); kayıt hot'a kopyalanır ve bir sonraki
                # tahliyede güncel sürümüyle tekrar iner.
                self._hot[anahtar] = deger
                self._hot.move_to_end(anahtar)
                self.stats["hot"].promotions_in += 1
                self._hot_tasir()
            return deger, "archive"
        self.stats["archive"].misses += 1
        return None, None

    def _arsivde_ara(self, anahtar: str) -> Optional[Any]:
        """Arşiv segmentlerini tara. Kasıtla yavaştır — arşiv sıcak yol değil."""
        for kayit_anahtar, deger in reversed(self._arsiv_tampon):
            if kayit_anahtar == anahtar:
                return deger
        dizin = os.path.join(self.root, ARCHIVE_DIR)
        for ad in sorted(os.listdir(dizin), reverse=True):
            with gzip.open(os.path.join(dizin, ad), "rt", encoding="utf-8") as dosya:
                for satir in dosya:
                    kayit = json.loads(satir)
                    if kayit["k"] == anahtar:
                        return kayit["v"]
        return None

    def __contains__(self, key: str) -> bool:
        return self.get(key)[0] is not None

    # ── dayanıklılık ───────────────────────────────────────────────────────
    def flush(self) -> None:
        """Tüm tamponları kalıcı ortama indir."""
        self._wal.flush()
        if self.durable:
            os.fsync(self._wal.fileno())
        self._arsiv_bosalt()
        self._cold.commit()

    def simulate_crash(self) -> None:
        """Süreç çökmesini taklit et: RAM katmanları kaybolur, disk kalır.

        Gerçek bir kill -9 ile aynı SEMANTİĞİ hedefler: hot/warm uçar,
        sqlite ve WAL dosyası diskte kalır. ``recover()`` bundan sonra
        çağrılmalıdır.
        """
        self._wal.flush()
        os.fsync(self._wal.fileno())
        self._cold.commit()
        self._hot.clear()
        self._warm.clear()

    def recover(self) -> Dict[str, Any]:
        """WAL'i tekrar oynatarak RAM katmanlarını yeniden kur.

        Yalnız cold/archive'da BULUNMAYAN kayıtlar geri yüklenir; zaten
        diske inmiş olanı tekrar hot'a çekmek çökme öncesi durumu taklit
        etmez, onu bozar.
        """
        self._cold.commit()
        geri_yuklenen = 0
        gorulen: Dict[str, Any] = {}
        with open(self._wal_path, "r", encoding="utf-8") as dosya:
            for satir in dosya:
                satir = satir.strip()
                if not satir:
                    continue
                try:
                    kayit = json.loads(satir)
                except json.JSONDecodeError:
                    # Çökme anında yarım yazılmış son satır: atlanır, sayılmaz.
                    continue
                gorulen[kayit["k"]] = kayit["v"]
        for anahtar, deger in gorulen.items():
            satir = self._cold.execute(
                "SELECT 1 FROM kayit WHERE anahtar = ?", (anahtar,)).fetchone()
            if satir is not None:
                continue
            if self._arsivde_ara(anahtar) is not None:
                continue
            self._hot[anahtar] = deger
            geri_yuklenen += 1
        self._hot_tasir()
        self.recoveries += 1
        return {
            "wal_records": len(gorulen),
            "restored_to_hot": geri_yuklenen,
            "recoveries": self.recoveries,
        }

    # ── ölçüm ──────────────────────────────────────────────────────────────
    def tier_of(self, key: str) -> Optional[str]:
        """Terfi ETTİRMEDEN kaydın hangi katmanda olduğunu söyle."""
        anahtar = str(key)
        if anahtar in self._hot:
            return "hot"
        if anahtar in self._warm:
            return "warm"
        if self._cold.execute("SELECT 1 FROM kayit WHERE anahtar = ?",
                              (anahtar,)).fetchone():
            return "cold"
        if self._arsivde_ara(anahtar) is not None:
            return "archive"
        return None

    def counts(self) -> Dict[str, int]:
        (soguk,) = self._cold.execute("SELECT COUNT(*) FROM kayit").fetchone()
        dizin = os.path.join(self.root, ARCHIVE_DIR)
        arsiv = self.stats["archive"].writes
        return {"hot": len(self._hot), "warm": len(self._warm),
                "cold": int(soguk), "archive": int(arsiv),
                "archive_segments": len(os.listdir(dizin))}

    def disk_bytes(self) -> Dict[str, int]:
        """Gerçek disk kullanımı — tahmin değil, ``os.stat`` ölçümü."""
        self.flush()
        def boyut(yol: str) -> int:
            try:
                return os.path.getsize(yol)
            except OSError:
                return 0
        dizin = os.path.join(self.root, ARCHIVE_DIR)
        arsiv = sum(boyut(os.path.join(dizin, ad)) for ad in os.listdir(dizin))
        soguk = boyut(os.path.join(self.root, COLD_FILE))
        wal = boyut(self._wal_path)
        return {"cold_sqlite": soguk, "archive_gzip": arsiv, "wal": wal,
                "total": soguk + arsiv + wal}

    def snapshot(self) -> Dict[str, Any]:
        return {
            "counts": self.counts(),
            "disk_bytes": self.disk_bytes(),
            "tiers": {ad: s.to_dict() for ad, s in self.stats.items()},
            "dropped": self.dropped,
            "wal_appends": self.wal_appends,
            "recoveries": self.recoveries,
            "archive_enabled": self.archive_enabled,
            "durable": self.durable,
            "promote_on_read": self.promote_on_read,
        }

    def close(self, destroy: bool = False) -> None:
        """Depoyu kapat; ``destroy`` ise disk izlerini de sil."""
        if self._kapali:
            return
        self.flush()
        self._wal.close()
        self._cold.close()
        self._kapali = True
        if destroy or self._gecici:
            shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> "HierarchicalMemory":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def write_many(memory: HierarchicalMemory,
               items: Iterable[Tuple[str, Any]]) -> Dict[str, float]:
    """Toplu yazma; toplam ve kayıt başına süreyi ölç."""
    basla = time.perf_counter()
    sayi = 0
    for anahtar, deger in items:
        memory.put(anahtar, deger)
        sayi += 1
    sure = time.perf_counter() - basla
    return {
        "records": sayi,
        "seconds": round(sure, 6),
        "seconds_per_record": round(sure / sayi, 9) if sayi else 0.0,
        "records_per_second": round(sayi / sure, 3) if sure > 0 else 0.0,
    }


__all__ = [
    "ARCHIVE_SEGMENT_RECORDS",
    "DEFAULT_CAPACITY",
    "TIERS",
    "HierarchicalMemory",
    "TierStats",
    "write_many",
]
