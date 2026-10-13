"""P0-1 regresyon testi: torch.load çağrılarında ``weights_only=True`` zorunlu.

İki katmanlı koruma:

1. **Statik (AST):** ``hga/``, ``egitim/``, ``mimari/`` ağacındaki tüm
   ``torch.load(...)`` çağrıları ``weights_only=True`` argümanını içermeli.
2. **Davranışsal:** ``weights_only=True`` ile zararlı bir pickle payload'ı
   reddedilmeli; aksi halde güvensiz deserialization RCE'ye açık olur.

Referans: P0-1 (audit raporu), bkz. docs/audit/.
"""
from __future__ import annotations

import ast
import io
import pickle
from pathlib import Path

import pytest


# --- Repo kökü: bu dosya tests/security/ altında; iki seviye yukarısı ---
_REPO_KOK = Path(__file__).resolve().parents[2]
_TARANACAK_DIZINLER = ("hga", "egitim", "mimari")


def _torch_load_cagrilari(py_dosyasi: Path) -> list[ast.Call]:
    """Bir Python dosyasındaki ``torch.load(...)`` çağrılarını döndürür."""
    agac = ast.parse(py_dosyasi.read_text(encoding="utf-8"), filename=str(py_dosyasi))
    bulunanlar: list[ast.Call] = []
    for dugum in ast.walk(agac):
        if not isinstance(dugum, ast.Call):
            continue
        f = dugum.func
        # torch.load(...) veya alias.load(...) biçimleri
        if isinstance(f, ast.Attribute) and f.attr == "load":
            if isinstance(f.value, ast.Name) and f.value.id == "torch":
                bulunanlar.append(dugum)
    return bulunanlar


def _weights_only_true_mu(cagri: ast.Call) -> bool:
    for kw in cagri.keywords:
        if kw.arg == "weights_only":
            # weights_only=True (NameConstant) veya True literal
            return isinstance(kw.value, ast.Constant) and kw.value.value is True
    return False


def test_tum_torch_load_cagrilari_weights_only_kullaniyor():
    """hga/, egitim/, mimari/ altındaki tüm torch.load çağrıları güvenli olmalı."""
    eksik: list[str] = []
    toplam = 0

    for dizin in _TARANACAK_DIZINLER:
        kok = _REPO_KOK / dizin
        if not kok.exists():
            continue
        for py in kok.rglob("*.py"):
            for cagri in _torch_load_cagrilari(py):
                toplam += 1
                if not _weights_only_true_mu(cagri):
                    eksik.append(f"{py.relative_to(_REPO_KOK)}:{cagri.lineno}")

    assert toplam > 0, (
        "Hiç torch.load çağrısı bulunamadı; test yanlış dizini tarıyor olabilir."
    )
    assert not eksik, (
        f"weights_only=True eksik ({len(eksik)} çağrı):\n  "
        + "\n  ".join(eksik)
    )


def test_zararli_pickle_reddedilir():
    """weights_only=True, __reduce__ tabanlı zararlı payload'ı reddetmeli."""
    torch = pytest.importorskip("torch")

    class Zararli:
        def __reduce__(self):
            # Bu çağrı, güvensiz yüklemede RCE tetikler.
            return (eval, ("__import__('os').system('echo PWNED')",))

    buf = io.BytesIO()
    pickle.dump({"zararli": Zararli()}, buf)
    buf.seek(0)

    with pytest.raises(Exception):
        torch.load(buf, weights_only=True)


def test_temiz_checkpoint_weights_only_ile_yuklenir():
    """Optimizer state dahil temiz bir checkpoint weights_only=True ile yüklenebilmeli."""
    torch = pytest.importorskip("torch")

    ckpt = {
        "model_state_dict": {"agirlik": torch.randn(4, 4)},
        "optimizer_state_dict": {
            "state": {},
            "param_groups": [{"lr": 1e-3, "weight_decay": 0.0}],
        },
        "epoch": 7,
    }
    buf = io.BytesIO()
    torch.save(ckpt, buf)
    buf.seek(0)

    yuklenen = torch.load(buf, weights_only=True)
    assert yuklenen["epoch"] == 7
    assert "agirlik" in yuklenen["model_state_dict"]
    assert yuklenen["model_state_dict"]["agirlik"].shape == (4, 4)
