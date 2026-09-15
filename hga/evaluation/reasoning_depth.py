"""Güvenilir çıkarım derinliği: ``C_R`` ve ``C_RD`` (P0-7).

``docs/COK_ADIMLI_VE_UZUN_BAGLAM.md`` tek bir sayı bildiriyordu: "en derin
güvenilir zincir 4 adım". Bu sayı iki farklı şeyi karıştırıyor:

* sistemin **temiz** koşulda ne kadar derin zincir takip edebildiği,
* **dolgu baskısı altında** bu derinliğin nasıl çöktüğü.

Bu modül ikisini ayırır ve HGA Capability Vector'ün iki bileşenini üretir::

    C_R      = distractor yokken (dolgu=0) güvenilir kalınan en derin hop
    C_RD(d)  = d dolgu olgu altında güvenilir kalınan en derin hop

"Güvenilir" eşiği açıkça parametredir (varsayılan 1.0 = tam doğruluk) ve
rapora yazılır; eşik gevşetilerek derinlik şişirilemesin diye kullanılan
eşik her tabloda görünür.

Derinlik ızgarası 1→32 hop, dolgu ızgarası 0→16384'e kadar taranabilir.
Tarama pahalıdır; bu yüzden üç profil vardır (``smoke`` / ``standard`` /
``deep``) ve her profil rapora yazılır — bir sayının hangi bütçeyle elde
edildiği gizlenmez.

Dürüstlük
---------
Zincir takibi deterministik bir genişlik-öncelikli aramadır; öğrenilmiş bir
yetenek değildir. Ölçülen şey, **seyrek belleğin** zincir kenarlarını dolgu
baskısı altında hayatta tutup tutamadığıdır. Bellek kenarı kaybederse zincir
kopar ve derinlik düşer — C_RD'nin ölçtüğü tam olarak budur.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .multi_hop import MultiHopReport, run_multi_hop_benchmark

PROTOCOL = "reliable_reasoning_depth_v1"

#: Izgaralar, seyrek bellek adresleme düzeltmesi (bkz.
#: docs/SPARSE_ADDRESSING_FIX.md) SONRASINA göre boyutlandırıldı. Eski
#: "Bloom tarzı" iki tablo sıfır bağımsızlık sağlıyordu ve icerir() AND
#: semantiği kaybı büyütüyordu; kırılmalar o kusurun eseriydi. Düzeltme
#: sonrası her profil ızgarası, kırılma noktasını İÇERECEK kadar derin
#: seçildi ki C_R tavana çarpmasın (c_r_not_grid_limited kapısı).
PROFILES: Dict[str, Dict[str, Any]] = {
    # Hızlı CI kapısı. slot=4096'da kırılma ~512 hoptadır; ızgara 512'yi
    # içerir, dolayısıyla C_R=256 ölçümdür, tavan değil.
    "smoke": {"hops": (1, 2, 4, 64, 256, 512), "distractors": (0, 16, 64),
              "seeds": (1,), "slot_sayisi": 4096, "tablo_sayisi": 2},
    # Varsayılan bilimsel tarama: 2^16 slotta kırılma ~2048 hoptadır.
    "standard": {"hops": (1, 2, 8, 64, 256, 1024, 2048),
                 "distractors": (0, 64, 256, 1024),
                 "seeds": (1, 2, 3), "slot_sayisi": 65536, "tablo_sayisi": 2},
    # Tam hedef ızgara (pahalı). Slot sayısı dolgu yüküne göre boyutlandırılır:
    # 2^18 slotta 16384 dolgu tabloyu DOYURUYORDU ve "dolgu direnci" kapısı
    # aslında kapasite taşmasını ölçüyordu (kök neden teşhisi: memory_capacity,
    # log-log eğim ~1.33 — derinlik slotla ölçekleniyor). 2^20 slotta dolgusuz
    # kırılma ~32768 hopta, 16384 dolgu altında ~16384 hoptadır; ikisi de
    # ızgaranın içindedir ve kapı artık doygunluğu değil girişim direncini ölçer.
    "deep": {"hops": (1, 2, 8, 64, 256, 1024, 4096, 8192, 16384, 32768),
             "distractors": (0, 64, 256, 1024, 4096, 16384),
             "seeds": (1, 2, 3), "slot_sayisi": 1048576, "tablo_sayisi": 2},
}


@dataclass
class DepthReport:
    protocol: str
    profile: str
    reliability_threshold: float
    hops: List[int]
    distractor_levels: List[int]
    seeds: List[int]
    slot_sayisi: int
    tablo_sayisi: int
    dataset_hash: str
    c_r: int
    c_r_grid_limited: bool
    c_rd: Dict[int, int]                       # dolgu → güvenilir derinlik
    accuracy_grid: Dict[int, Dict[int, float]]  # hop → dolgu → doğruluk
    memory_recall_grid: Dict[int, Dict[int, float]]
    depth_retention: Dict[int, float]           # dolgu → C_RD/C_R
    collapse_distractor: Optional[int]          # C_RD'nin ilk düştüğü seviye
    checks: Dict[str, bool]
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # JSON anahtarları string olmalı.
        d["c_rd"] = {str(k): v for k, v in self.c_rd.items()}
        d["depth_retention"] = {str(k): v for k, v in self.depth_retention.items()}
        d["accuracy_grid"] = {
            str(h): {str(dd): v for dd, v in sub.items()}
            for h, sub in self.accuracy_grid.items()
        }
        d["memory_recall_grid"] = {
            str(h): {str(dd): v for dd, v in sub.items()}
            for h, sub in self.memory_recall_grid.items()
        }
        return d

    def markdown(self) -> str:
        return reasoning_depth_markdown(self)


def _derinlik(grid: Dict[int, Dict[int, float]], hops: Sequence[int],
              distractor: int, esik: float) -> int:
    """Verilen dolgu seviyesinde eşiği geçen en derin **kesintisiz** hop.

    Kesintisizlik önemlidir: 2 hopta başarısız olup 8 hopta tesadüfen
    başarılı olmak "8 hop güvenilir" anlamına gelmez. hop=1 geri çağırmadır;
    çıkarım derinliği yine de oradan başlatılır çünkü zincirin temel taşıdır.
    """
    derinlik = 0
    for hop in sorted(hops):
        if grid.get(hop, {}).get(distractor, 0.0) + 1e-12 >= esik:
            derinlik = hop
        else:
            break
    return derinlik


def measure_reasoning_depth(
    profile: str = "standard",
    reliability_threshold: float = 1.0,
    hops: Optional[Sequence[int]] = None,
    distractor_levels: Optional[Sequence[int]] = None,
    seeds: Optional[Sequence[int]] = None,
    slot_sayisi: Optional[int] = None,
    tablo_sayisi: Optional[int] = None,
) -> DepthReport:
    """``C_R`` ve ``C_RD`` ölç; alt katman olarak multi-hop ızgarasını koşar."""
    if profile not in PROFILES:
        raise ValueError(f"bilinmeyen profil: {profile}; "
                         f"seçenekler {sorted(PROFILES)}")
    if not 0.0 < reliability_threshold <= 1.0:
        raise ValueError("reliability_threshold (0,1] aralığında olmalı")
    ayar = PROFILES[profile]
    hops = [int(h) for h in (hops if hops is not None else ayar["hops"])]
    distractor_levels = [int(d) for d in (distractor_levels
                                          if distractor_levels is not None
                                          else ayar["distractors"])]
    seeds = [int(s) for s in (seeds if seeds is not None else ayar["seeds"])]
    slot_sayisi = int(slot_sayisi if slot_sayisi is not None else ayar["slot_sayisi"])
    tablo_sayisi = int(tablo_sayisi if tablo_sayisi is not None
                       else ayar["tablo_sayisi"])
    if 0 not in distractor_levels:
        raise ValueError("C_R tanımı gereği dolgu=0 seviyesi zorunludur")

    alt: MultiHopReport = run_multi_hop_benchmark(
        hops=hops, distractor_levels=distractor_levels, seeds=seeds,
        slot_sayisi=slot_sayisi, tablo_sayisi=tablo_sayisi,
    )

    grid: Dict[int, Dict[int, float]] = {h: {} for h in hops}
    bellek: Dict[int, Dict[int, float]] = {h: {} for h in hops}
    for hucre in alt.cells:
        grid[hucre.hop][hucre.distractors] = hucre.accuracy
        bellek[hucre.hop][hucre.distractors] = hucre.memory_recall_rate

    c_r = _derinlik(grid, hops, 0, reliability_threshold)
    c_rd = {d: _derinlik(grid, hops, d, reliability_threshold)
            for d in distractor_levels if d != 0}
    tutma = {d: round(v / c_r, 6) if c_r else 0.0 for d, v in c_rd.items()}
    cokme = next((d for d in sorted(c_rd) if c_rd[d] < c_r), None)
    # C_R ızgaranın en derin hop'una eşitse ölçüm TAVANA ÇARPMIŞTIR: gerçek
    # derinlik daha büyük olabilir. Bu sayıyı "sistemin sınırı" diye okumak
    # yanlış olur, bu yüzden ayrı bir bayrakla işaretlenir.
    c_r_grid_limited = bool(hops) and c_r == max(hops)

    imza = hashlib.sha256(json.dumps({
        "protocol": PROTOCOL, "hops": hops, "distractors": distractor_levels,
        "seeds": seeds, "slots": slot_sayisi, "tables": tablo_sayisi,
        "threshold": reliability_threshold,
    }, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    kontroller = {
        # Hedef: 8+ hop. Bu kapı kasıtlı olarak zordur ve geçmeyebilir.
        "c_r_at_least_8": c_r >= 8,
        # Dolgu altında derinliğin en az yarısı korunuyor mu?
        "retains_half_depth_under_max_distractors": (
            bool(c_rd) and tutma[max(c_rd)] >= 0.5),
        # Çıkarım (hop>=2) tek adımlı geri çağırmadan ayrışıyor mu?
        "multi_hop_beats_degenerate": alt.checks.get("beats_degenerate", False),
        # Derinlik dolgu ile monoton azalıyor mu (beklenen davranış)?
        # C_R ızgara tavanına dayandıysa "bu sistemin derinlik sınırı" denemez.
        "c_r_not_grid_limited": not c_r_grid_limited,
        "depth_monotone_in_distractors": all(
            c_rd[a] >= c_rd[b]
            for a, b in zip(sorted(c_rd), sorted(c_rd)[1:])
        ) if len(c_rd) > 1 else True,
    }

    bulgular = [
        f"C_R = {c_r} (dolgu yok, eşik {reliability_threshold:g})."
        + (f" UYARI: bu değer ızgaranın en derin hop'una ({max(hops)}) eşit — "
           "ölçüm tavana çarptı, gerçek derinlik daha büyük olabilir."
           if c_r_grid_limited else ""),
        "C_RD: " + ", ".join(f"{d}→{v}" for d, v in sorted(c_rd.items())) + ".",
    ]
    if cokme is not None:
        bulgular.append(
            f"Derinlik ilk olarak {cokme} dolgu seviyesinde düşüyor "
            f"({c_r} → {c_rd[cokme]}). Bu ölçülen bir sınırdır; eşik "
            f"gevşetilerek gizlenmedi.")
    else:
        bulgular.append(
            f"Taranan en yüksek dolgu seviyesinde ({max(distractor_levels)}) "
            f"bile derinlik düşmedi — bu ızgarada çökme noktası bulunamadı.")
    en_dusuk_bellek = min(
        (v for sub in bellek.values() for v in sub.values()), default=1.0)
    bulgular.append(
        f"En düşük bellek geri çağırma oranı {en_dusuk_bellek:.4f}; "
        f"{slot_sayisi} slot × {tablo_sayisi} tablo ile ölçüldü. Derinlik "
        f"kaybı ile bellek kaybı bu sayede ayrı okunabilir.")

    sinirlar = list(alt.limitations) + [
        "C_R/C_RD deterministik zincir takibini ölçer; öğrenilmiş akıl "
        "yürütme değildir.",
        "Güvenilirlik eşiği rapora yazılır; eşik düşürülürse derinlik yapay "
        "olarak artar — tablolar eşiksiz okunmamalıdır.",
        f"Profil `{profile}` ızgarası sonlu: tarananın ötesinde bir çökme "
        "noktası olabilir, bulunmaması yokluğu kanıtlamaz.",
    ]

    return DepthReport(
        protocol=PROTOCOL, profile=profile,
        reliability_threshold=float(reliability_threshold),
        hops=hops, distractor_levels=distractor_levels, seeds=seeds,
        slot_sayisi=slot_sayisi, tablo_sayisi=tablo_sayisi,
        dataset_hash=imza, c_r=c_r, c_r_grid_limited=c_r_grid_limited, c_rd=c_rd, accuracy_grid=grid,
        memory_recall_grid=bellek, depth_retention=tutma,
        collapse_distractor=cokme, checks=kontroller,
        findings=bulgular, limitations=sinirlar,
    )


def reasoning_depth_markdown(report: DepthReport) -> str:
    satirlar = [
        "# Güvenilir Çıkarım Derinliği — C_R ve C_RD (P0-7)",
        "",
        f"Protokol: `{report.protocol}` · profil: `{report.profile}` · "
        f"veri imzası: `{report.dataset_hash}`",
        f"Eşik: {report.reliability_threshold:g} · tohumlar: {report.seeds} · "
        f"bellek: {report.slot_sayisi} slot × {report.tablo_sayisi} tablo",
        "",
        f"**C_R = {report.c_r}**"
        + ("  *(ızgara tavanı — gerçek derinlik daha büyük olabilir)*"
           if report.c_r_grid_limited else ""),
        "",
        "| Dolgu | C_RD | C_RD / C_R |",
        "|---:|---:|---:|",
    ]
    satirlar.append(f"| 0 | {report.c_r} | 1.000 |")
    for d in sorted(report.c_rd):
        satirlar.append(
            f"| {d} | {report.c_rd[d]} | {report.depth_retention[d]:.3f} |")

    satirlar += ["", "## Doğruluk ızgarası", "",
                 "| hop \\ dolgu | " + " | ".join(
                     str(d) for d in report.distractor_levels) + " |",
                 "|---|" + "---:|" * len(report.distractor_levels)]
    for hop in report.hops:
        etiket = f"{hop} adım" + (" (geri çağırma)" if hop == 1 else "")
        satirlar.append(
            f"| {etiket} | " + " | ".join(
                f"{report.accuracy_grid[hop].get(d, float('nan')):.4f}"
                for d in report.distractor_levels) + " |")

    satirlar += ["", "## Bellek geri çağırma ızgarası", "",
                 "| hop \\ dolgu | " + " | ".join(
                     str(d) for d in report.distractor_levels) + " |",
                 "|---|" + "---:|" * len(report.distractor_levels)]
    for hop in report.hops:
        satirlar.append(
            f"| {hop} adım | " + " | ".join(
                f"{report.memory_recall_grid[hop].get(d, float('nan')):.4f}"
                for d in report.distractor_levels) + " |")

    satirlar += ["", "## Kabul kapıları", "", "| kapı | sonuç |", "|---|---|"]
    for ad, sonuc in report.checks.items():
        satirlar.append(f"| {ad} | {'GEÇTİ' if sonuc else 'KALDI'} |")
    satirlar += ["", "## Bulgular", ""]
    satirlar += [f"- {b}" for b in report.findings]
    satirlar += ["", "## Sınırlar", ""]
    satirlar += [f"- {s}" for s in report.limitations]
    return "\n".join(satirlar) + "\n"


__all__ = ["PROTOCOL", "PROFILES", "DepthReport", "measure_reasoning_depth",
           "reasoning_depth_markdown"]
