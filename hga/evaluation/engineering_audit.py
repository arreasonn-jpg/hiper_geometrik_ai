# -*- coding: utf-8 -*-
"""Mühendislik ve yeniden üretilebilirlik denetimi (karnenin iki boş bölümü).

Karnede ``reproducibility`` ve ``engineering`` bölümleri uzun süre ``n/a``
kaldı. Bunun sebebi kapıların olmaması değildi — CI zaten lint, mypy,
paketleme ve 4 Python sürümünde test koşuyordu. Sebep, bu kapıların
**makine-okunur bir rapor üretmemesiydi**: karne yalnız protokol raporlarından
beslenir, YAML dosyası okuyamaz.

Bu modül o boşluğu kapatır. İki şeyi denetler:

* ``audit_engineering`` — CI iş akışının ve paketleme sözleşmesinin
  gerçekten var olduğunu **dosyadan okuyarak** doğrular. "CI var" demek
  yetmez; hangi kapıların hangi sürümlerde koştuğu çıkarılır.
* ``audit_reproducibility`` — bir deneyin tekrar üretilebilmesi için gereken
  şeyleri denetler: manifest üretimi, veri/konfig hash'i, tohum kontrolü,
  determinizm ve protokollerin çoklu tohum tamamlaması.

Önemli sınır: bu modül **CI'ı koşturmaz**. Koşan CI'ın sonucunu değil,
sözleşmenin repoda tanımlı olduğunu doğrular. Yeşil bir CI rozeti bu
raporun yerine geçmez ve rapor bunu açıkça yazar.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROTOCOL_ENGINEERING = "engineering_contract_audit_v1"
PROTOCOL_REPRODUCIBILITY = "reproducibility_audit_v1"
SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_PATH = Path(".github") / "workflows" / "ci.yml"

#: Paketleme sözleşmesinde bulunması beklenen alanlar.
REQUIRED_PROJECT_FIELDS: Tuple[str, ...] = (
    "name", "version", "description", "requires-python", "license",
)

#: CI'da bulunması beklenen kalite kapıları ve hangi komutla tanındıkları.
EXPECTED_CI_GATES: Dict[str, Tuple[str, ...]] = {
    "lint": ("ruff check",),
    "type_check": ("mypy",),
    "tests": ("pytest",),
    "package_build": ("python -m build", "pip install"),
}


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _load_pyproject(root: Path) -> Optional[Dict[str, Any]]:
    """pyproject.toml'u oku; tomllib yoksa tomli'ye düş, ikisi de yoksa None."""
    ham = _read(root / "pyproject.toml")
    if ham is None:
        return None
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return None
    try:
        veri: Dict[str, Any] = tomllib.loads(ham)
        return veri
    except Exception:
        return None


@dataclass
class EngineeringReport:
    """Mühendislik sözleşmesi denetim raporu."""

    protocol: str
    schema_version: int
    repo_root: str
    ci: Dict[str, Any]
    packaging: Dict[str, Any]
    test_suite: Dict[str, Any]
    tooling: Dict[str, Any]
    checks: Dict[str, bool]
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def audit_engineering(root: Optional[Path] = None) -> EngineeringReport:
    """CI, paketleme ve test sözleşmesini repodan okuyarak denetle.

    Args:
        root: Repo kökü. ``None`` ise modülün bulunduğu repo kullanılır.

    Raises:
        FileNotFoundError: kök dizin yoksa.
    """
    kok = Path(root) if root is not None else REPO_ROOT
    if not kok.is_dir():
        raise FileNotFoundError(f"repo kökü bulunamadı: {kok}")

    # ── CI ─────────────────────────────────────────────────────────────────
    ci_ham = _read(kok / CI_PATH)
    ci_var = ci_ham is not None
    python_surumleri: List[str] = []
    ci_kapilari: Dict[str, bool] = {}
    ci_komutlari: List[str] = []
    if ci_ham:
        eslesme = re.search(r"python-version:\s*\[([^\]]+)\]", ci_ham)
        if eslesme:
            python_surumleri = [p.strip().strip('"\'')
                                for p in eslesme.group(1).split(",")]
        for kapi, isaretler in EXPECTED_CI_GATES.items():
            ci_kapilari[kapi] = any(i in ci_ham for i in isaretler)
        ci_komutlari = sorted(set(re.findall(r"python -m hga ([a-z0-9-]+)",
                                             ci_ham)))

    ci = {
        "workflow_path": str(CI_PATH),
        "exists": ci_var,
        "python_matrix": python_surumleri,
        "matrix_size": len(python_surumleri),
        "gates_present": ci_kapilari,
        "cli_commands_smoke_tested": ci_komutlari,
        "cli_command_count": len(ci_komutlari),
    }

    # ── paketleme ──────────────────────────────────────────────────────────
    pyproject = _load_pyproject(kok)
    proje = (pyproject or {}).get("project", {})
    eksik_alanlar = [a for a in REQUIRED_PROJECT_FIELDS if a not in proje]
    paketleme = {
        "pyproject_parsed": pyproject is not None,
        "name": proje.get("name"),
        "version": proje.get("version"),
        "requires_python": proje.get("requires-python"),
        "missing_fields": eksik_alanlar,
        "has_optional_test_extra": bool(
            (proje.get("optional-dependencies") or {}).get("test")),
        "build_backend": ((pyproject or {}).get("build-system") or {})
        .get("build-backend"),
    }

    # ── test paketi ────────────────────────────────────────────────────────
    test_dosyalari = sorted(str(p.relative_to(kok))
                            for p in kok.glob("tests/**/test_*.py"))
    bilimsel = [p for p in test_dosyalari if p.startswith("tests/scientific/")]
    test_paketi = {
        "test_files": len(test_dosyalari),
        "scientific_test_files": len(bilimsel),
        "pytest_configured": bool(
            ((pyproject or {}).get("tool") or {}).get("pytest")),
        "markers_registered": bool(
            (((pyproject or {}).get("tool") or {}).get("pytest") or {})
            .get("ini_options", {}).get("markers")),
    }

    # ── araçlar ────────────────────────────────────────────────────────────
    arac = ((pyproject or {}).get("tool") or {})
    araclar = {
        "ruff_configured": "ruff" in arac,
        "mypy_configured": "mypy" in arac,
        "ruff_line_length": (arac.get("ruff") or {}).get("line-length"),
        "mypy_strict_flags": sorted(
            k for k, v in (arac.get("mypy") or {}).items() if v is True),
    }

    kapilar: Dict[str, bool] = {
        "ci_workflow_exists": ci_var,
        "ci_matrix_covers_multiple_pythons": len(python_surumleri) >= 2,
        "ci_runs_lint": ci_kapilari.get("lint", False),
        "ci_runs_type_check": ci_kapilari.get("type_check", False),
        "ci_runs_tests": ci_kapilari.get("tests", False),
        "ci_verifies_package_build": ci_kapilari.get("package_build", False),
        "ci_smoke_tests_cli": len(ci_komutlari) >= 10,
        "packaging_contract_complete": (
            pyproject is not None and not eksik_alanlar),
        "test_extra_declared": bool(paketleme["has_optional_test_extra"]),
        "lint_and_type_tools_configured": bool(
            araclar["ruff_configured"] and araclar["mypy_configured"]),
        "scientific_tests_present": len(bilimsel) >= 10,
    }

    bulgular: List[str] = [
        f"CI matrisi: {python_surumleri or 'YOK'} "
        f"({len(python_surumleri)} Python sürümü).",
        f"CI'da smoke koşulan CLI komutu: {len(ci_komutlari)}.",
        f"Test dosyası: {len(test_dosyalari)} "
        f"(bilimsel: {len(bilimsel)}).",
    ]
    eksik_kapilar = [k for k, v in ci_kapilari.items() if not v]
    if eksik_kapilar:
        bulgular.append(
            f"CI'da BULUNAMAYAN kalite kapıları: {eksik_kapilar}.")
    if eksik_alanlar:
        bulgular.append(
            f"pyproject.toml'da eksik alanlar: {eksik_alanlar}.")

    # Denetimi girdisine bağlayan deterministik imza: incelenen dosyaların
    # içerik özeti. Aynı CI/pyproject/test yerleşimi → aynı imza.
    ci["signature"] = hashlib.sha256(json.dumps({
        "protocol": PROTOCOL_ENGINEERING,
        "ci_gates": {k: bool(v) for k, v in ci_kapilari.items()},
        "packaging": {k: paketleme.get(k) for k in sorted(paketleme)
                      if isinstance(paketleme.get(k), (str, bool, int))},
        "test_files": test_paketi.get("test_files"),
    }, ensure_ascii=False, sort_keys=True, default=str)
        .encode("utf-8")).hexdigest()[:12]

    return EngineeringReport(
        protocol=PROTOCOL_ENGINEERING,
        schema_version=SCHEMA_VERSION,
        repo_root=str(kok),
        ci=ci,
        packaging=paketleme,
        test_suite=test_paketi,
        tooling=araclar,
        checks=kapilar,
        findings=bulgular,
        limitations=[
            "Bu denetim CI'ı KOŞTURMAZ; iş akışı dosyasının ne tanımladığını "
            "okur. Yeşil bir CI koşusu ayrı bir kanıttır.",
            "Kapı varlığı metin eşlemesiyle tespit edilir; bir kapının "
            "tanımlı olması doğru yapılandırıldığı anlamına gelmez.",
            "Test SAYISI kalite değildir; bu rapor kapsamı ölçmez.",
        ],
    )


# ── Yeniden üretilebilirlik ─────────────────────────────────────────────────
@dataclass
class ReproducibilityReport:
    """Yeniden üretilebilirlik denetim raporu."""

    protocol: str
    schema_version: int
    manifest: Dict[str, Any]
    determinism: Dict[str, Any]
    seed_discipline: Dict[str, Any]
    hashing: Dict[str, Any]
    environment: Dict[str, Any]
    checks: Dict[str, bool]
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


#: Çekirdek bilimsel iddia için gereken tohum sayısı (kullanıcı kuralı).
CORE_SEED_REQUIREMENT = 20


def audit_reproducibility(
    reports: Optional[Dict[str, Dict[str, Any]]] = None,
    root: Optional[Path] = None,
) -> ReproducibilityReport:
    """Deney kaydı, determinizm ve tohum disiplinini denetle.

    Args:
        reports: Protokol adı → rapor sözlüğü. Her rapordan tohum sayısı ve
            veri imzası çıkarılır. ``None`` ise yalnız altyapı denetlenir.
        root: Repo kökü.

    Determinizm gerçekten ÖLÇÜLÜR: aynı tohumla iki kez koşulan bir
    protokolün bayt-eş çıktı üretip üretmediğine bakılır. Bu, "deterministik"
    iddiasının tek geçerli kanıtıdır.
    """
    kok = Path(root) if root is not None else REPO_ROOT
    raporlar = dict(reports or {})

    # ── manifest altyapısı ─────────────────────────────────────────────────
    manifest_bilgi: Dict[str, Any] = {"can_create_run": False}
    try:
        from .experiment import ExperimentRun, canonical_hash

        with tempfile.TemporaryDirectory() as gecici:
            kosu = ExperimentRun.create(
                root=gecici,
                config={"protocol": "reproducibility_selftest", "n": 1},
                seed=1,
                dataset_hash=canonical_hash({"veri": "sentetik"}),
            )
            man = kosu.manifest
            manifest_bilgi = {
                "can_create_run": True,
                "experiment_id": kosu.experiment_id,
                "fields": sorted(man.keys()),
                "has_config_hash": "config_hash" in man,
                "has_dataset_hash": "dataset_hash" in man,
                "has_seed": "seed" in man,
                # Manifest ortam alanlarını iç içe değil DÜZ tutar
                # (cpu, platform, torch_version...). İç içe bir "runtime"
                # anahtarı aramak bu yüzden yanlış negatif verirdi.
                "has_git_metadata": any(
                    a in man for a in ("git_commit", "git_dirty")),
                "has_runtime_metadata": all(
                    a in man for a in ("python_version", "platform", "cpu")),
                "recorded_environment_fields": sorted(
                    a for a in ("python_version", "python_implementation",
                                "platform", "cpu", "cpu_count",
                                "ram_total_bytes", "device", "gpu",
                                "torch_version", "cuda_version")
                    if a in man),
                "config_file_written": (
                    Path(kosu.directory) / "config.yaml").exists(),
            }
    except Exception as hata:  # pragma: no cover - altyapı hatası
        manifest_bilgi = {"can_create_run": False, "error": repr(hata)}

    # ── determinizm: aynı tohum, bayt-eş çıktı mı? ─────────────────────────
    determinizm: Dict[str, Any] = {}
    try:
        from .compositional_v2 import run_compositional_v2_benchmark
        from .semantic_extraction import run_semantic_extraction_benchmark

        for ad, fonksiyon in (
                ("compositional_v2", run_compositional_v2_benchmark),
                ("semantic_extraction", run_semantic_extraction_benchmark)):
            bir = json.dumps(fonksiyon().to_dict(), sort_keys=True)
            iki = json.dumps(fonksiyon().to_dict(), sort_keys=True)
            determinizm[ad] = {
                "byte_identical": bir == iki,
                "digest": hashlib.sha256(bir.encode("utf-8")).hexdigest()[:16],
            }
    except Exception as hata:  # pragma: no cover
        determinizm["error"] = repr(hata)

    # ── tohum disiplini ────────────────────────────────────────────────────
    tohum_satirlari: Dict[str, Any] = {}
    for ad, rapor in raporlar.items():
        if not isinstance(rapor, dict):
            continue
        tohumlar = rapor.get("seeds")
        if tohumlar is None:
            tohum = rapor.get("seed")
            tohumlar = [tohum] if tohum is not None else []
        sayi = len(tohumlar) if isinstance(tohumlar, (list, tuple)) else 0
        tohum_satirlari[ad] = {
            "seeds": list(tohumlar) if isinstance(tohumlar, (list, tuple))
            else [],
            "count": sayi,
            "meets_core_requirement": sayi >= CORE_SEED_REQUIREMENT,
            "tier": ("core" if sayi >= CORE_SEED_REQUIREMENT
                     else "engineering" if sayi >= 2 else "smoke"),
        }
    cekirdek_karsilayan = [a for a, v in tohum_satirlari.items()
                           if v["meets_core_requirement"]]
    tohum_disiplini = {
        "per_protocol": tohum_satirlari,
        "protocols_examined": len(tohum_satirlari),
        "protocols_meeting_core_requirement": len(cekirdek_karsilayan),
        "core_requirement": CORE_SEED_REQUIREMENT,
    }

    # ── hash kapsaması ─────────────────────────────────────────────────────
    def _imza_var(rapor: Dict[str, Any]) -> bool:
        """Protokol verisini bağlayan bir imza/hash taşıyor mu?

        Protokoller imzayı farklı yerlere koyar (üst düzey ``dataset_hash``,
        ``config.signature``, ya da ``contamination.signature`` gibi bir alt
        bölüm). Hepsi aynı işi görür: raporu ürettiği veriye bağlamak. Bu
        yüzden bir düzey derinliğe kadar taranır.
        """
        if rapor.get("dataset_hash") or rapor.get("config_hash"):
            return True
        if rapor.get("signature"):
            return True
        for deger in rapor.values():
            if isinstance(deger, dict) and deger.get("signature"):
                return True
        return False

    hashli = {a: _imza_var(r)
              for a, r in raporlar.items() if isinstance(r, dict)}
    hashleme = {
        "per_protocol": hashli,
        "protocols_with_signature": sum(1 for v in hashli.values() if v),
        "protocols_total": len(hashli),
    }

    # ── ortam ──────────────────────────────────────────────────────────────
    ortam: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "lockfile_present": any(
            (kok / ad).exists()
            for ad in ("requirements.lock", "poetry.lock", "uv.lock")),
    }
    try:
        import torch
        ortam["torch_version"] = torch.__version__
        ortam["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        ortam["torch_version"] = None
        ortam["cuda_available"] = False
    try:
        ortam["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(kok),
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        ortam["git_commit"] = None

    determinizm_temiz = bool(determinizm) and all(
        v.get("byte_identical") for v in determinizm.values()
        if isinstance(v, dict))

    kapilar: Dict[str, bool] = {
        "manifest_infrastructure_works": bool(
            manifest_bilgi.get("can_create_run")),
        "manifest_records_config_hash": bool(
            manifest_bilgi.get("has_config_hash")),
        "manifest_records_dataset_hash": bool(
            manifest_bilgi.get("has_dataset_hash")),
        "manifest_records_environment": bool(
            manifest_bilgi.get("has_runtime_metadata")),
        "manifest_records_git_commit": bool(
            manifest_bilgi.get("has_git_metadata")),
        "deterministic_protocols_are_byte_identical": determinizm_temiz,
        "all_protocols_carry_data_signature": bool(hashli) and all(
            hashli.values()),
        "at_least_one_protocol_meets_20_seeds": len(cekirdek_karsilayan) >= 1,
        "git_commit_recoverable": ortam["git_commit"] is not None,
    }

    bulgular: List[str] = []
    if manifest_bilgi.get("can_create_run"):
        bulgular.append(
            f"Manifest altyapısı çalışıyor; kayıt alanları: "
            f"{manifest_bilgi.get('fields')}.")
    else:
        bulgular.append("Manifest altyapısı ÇALIŞMADI — deney kaydı alınamıyor.")

    if determinizm_temiz:
        bulgular.append(
            f"Determinizm ölçüldü: {sorted(determinizm)} protokolleri iki "
            "koşuda bayt-eş çıktı üretti.")
    elif determinizm:
        bozuk = [a for a, v in determinizm.items()
                 if isinstance(v, dict) and not v.get("byte_identical")]
        bulgular.append(
            f"Determinizm İHLALİ: {bozuk} aynı girdiyle farklı çıktı verdi.")

    if tohum_satirlari:
        bulgular.append(
            f"Tohum disiplini: {len(cekirdek_karsilayan)}/"
            f"{len(tohum_satirlari)} protokol 20 tohum eşiğini karşılıyor. "
            f"Karşılamayanlar ENGINEERING kanıtı sayılır: "
            f"{sorted(a for a in tohum_satirlari if a not in cekirdek_karsilayan)}.")

    if not ortam["lockfile_present"]:
        bulgular.append(
            "Bağımlılık kilit dosyası (lockfile) YOK. Sürüm aralıkları "
            "gelecekte farklı çözülebilir; tam ortam yeniden üretimi "
            "garanti edilemez.")

    return ReproducibilityReport(
        protocol=PROTOCOL_REPRODUCIBILITY,
        schema_version=SCHEMA_VERSION,
        manifest=manifest_bilgi,
        determinism=determinizm,
        seed_discipline=tohum_disiplini,
        hashing=hashleme,
        environment=ortam,
        checks=kapilar,
        findings=bulgular,
        limitations=[
            "Determinizm YALNIZ aynı makinede, aynı sürümlerle ölçüldü. "
            "Farklı donanım veya BLAS kütüphanesi kayan nokta sonuçlarını "
            "değiştirebilir.",
            "Nöral protokoller bu denetime dahil edilmedi; GPU/threading "
            "kaynaklı non-determinizm ayrıca ölçülmelidir.",
            "Lockfile yokluğunda 'yeniden üretilebilir' iddiası bağımlılık "
            "çözümüne bağımlıdır.",
        ],
    )


def engineering_markdown(report: EngineeringReport) -> str:
    """Mühendislik denetimini Markdown'a çevir."""
    s = report
    satirlar = [
        "# Mühendislik Sözleşmesi Denetimi",
        "",
        f"- Protokol: `{s.protocol}` v{s.schema_version}",
        "",
        "## CI",
        "",
        f"- İş akışı: `{s.ci['workflow_path']}` "
        f"({'var' if s.ci['exists'] else 'YOK'})",
        f"- Python matrisi: `{s.ci['python_matrix']}`",
        f"- Smoke koşulan CLI komutu: `{s.ci['cli_command_count']}`",
        "",
        "| Kapı | Durum |",
        "|---|---|",
    ]
    for kapi, durum in s.ci["gates_present"].items():
        satirlar.append(f"| {kapi} | {'VAR' if durum else 'YOK'} |")
    satirlar.extend([
        "",
        "## Paketleme",
        "",
        f"- Ad / sürüm: `{s.packaging['name']}` / `{s.packaging['version']}`",
        f"- Python gereksinimi: `{s.packaging['requires_python']}`",
        f"- Build backend: `{s.packaging['build_backend']}`",
        f"- Eksik alanlar: `{s.packaging['missing_fields'] or 'yok'}`",
        "",
        "## Test paketi",
        "",
        f"- Test dosyası: `{s.test_suite['test_files']}` "
        f"(bilimsel `{s.test_suite['scientific_test_files']}`)",
        f"- Marker kaydı: `{s.test_suite['markers_registered']}`",
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


def reproducibility_markdown(report: ReproducibilityReport) -> str:
    """Yeniden üretilebilirlik denetimini Markdown'a çevir."""
    s = report
    satirlar = [
        "# Yeniden Üretilebilirlik Denetimi",
        "",
        f"- Protokol: `{s.protocol}` v{s.schema_version}",
        f"- Git commit: `{s.environment.get('git_commit') or 'bilinmiyor'}`",
        f"- Python / torch: `{s.environment['python_version']}` / "
        f"`{s.environment.get('torch_version')}`",
        "",
        "## Manifest altyapısı",
        "",
        f"- Deney kaydı oluşturulabiliyor: "
        f"`{s.manifest.get('can_create_run')}`",
        f"- Kaydedilen alanlar: `{s.manifest.get('fields')}`",
        "",
        "## Determinizm (ölçülen)",
        "",
        "| Protokol | Bayt-eş | Özet |",
        "|---|---|---|",
    ]
    for ad, veri in s.determinism.items():
        if not isinstance(veri, dict):
            continue
        satirlar.append(
            f"| {ad} | {'EVET' if veri.get('byte_identical') else 'HAYIR'} | "
            f"`{veri.get('digest')}` |")

    satirlar.extend([
        "",
        "## Tohum disiplini",
        "",
        f"Çekirdek eşiği: **{s.seed_discipline['core_requirement']} tohum**. "
        "Altındaki her sonuç engineering/smoke kanıtıdır.",
        "",
        "| Protokol | Tohum | Sınıf |",
        "|---|---:|---|",
    ])
    for ad, veri in s.seed_discipline["per_protocol"].items():
        satirlar.append(f"| {ad} | {veri['count']} | {veri['tier']} |")

    satirlar.extend(["", "## Kabul kapıları", "", "| Kapı | Sonuç |", "|---|---|"])
    satirlar.extend(f"| {ad} | {'GEÇTİ' if v else 'KALDI'} |"
                    for ad, v in s.checks.items())
    satirlar.extend(["", "## Bulgular", ""])
    satirlar.extend(f"- {b}" for b in s.findings)
    satirlar.extend(["", "## Sınırlar", ""])
    satirlar.extend(f"- {b}" for b in s.limitations)
    return "\n".join(satirlar) + "\n"


__all__ = [
    "CORE_SEED_REQUIREMENT",
    "EXPECTED_CI_GATES",
    "PROTOCOL_ENGINEERING",
    "PROTOCOL_REPRODUCIBILITY",
    "REQUIRED_PROJECT_FIELDS",
    "EngineeringReport",
    "ReproducibilityReport",
    "audit_engineering",
    "audit_reproducibility",
    "engineering_markdown",
    "reproducibility_markdown",
]
