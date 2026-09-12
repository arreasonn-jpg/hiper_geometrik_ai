# -*- coding: utf-8 -*-
"""Yeni HGA CLI rapor komutları."""
import importlib.util
import os
import subprocess
import sys
import tempfile

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIMARI = os.path.join(KOK, "mimari")
if MIMARI not in sys.path:
    sys.path.insert(0, MIMARI)


def _run(cmd):
    return subprocess.check_output([sys.executable, "-m", "hga", cmd],
                                   cwd=KOK, text=True)


def test_cli_sweep():
    out = _run("sweep")
    assert "n/K/context" in out
    assert "parametre" in out


def test_cli_tokenizer():
    out = _run("tokenizer")
    assert "Mini Türkçe" in out
    assert "unk_orani" in out


def test_cli_halusinasyon():
    out = _run("halusinasyon")
    assert "hallucination_rate" in out
    assert "factual_consistency_score" in out


def test_cli_perplexity_smoke():
    out = subprocess.check_output([sys.executable, "-m", "hga", "perplexity", "--tiny"],
                                  cwd=KOK, text=True)
    assert "Perplexity" in out or "perplexity" in out
    assert "torch gerekli" in out or "token_sayisi" in out


def test_cli_observability():
    out = _run("observability")
    assert "Deneyim akışı" in out
    assert "Bellek doluluk" in out


def test_cli_benchmark_rapor_ve_observability_cikti():
    with tempfile.TemporaryDirectory() as tmp:
        bj = os.path.join(tmp, "benchmark.json")
        bm = os.path.join(tmp, "benchmark.md")
        out = subprocess.check_output([sys.executable, "-m", "hga", "benchmark-rapor",
                                       "--out", bj, "--markdown", bm],
                                      cwd=KOK, text=True)
        assert "Benchmark raporu" in out
        assert os.path.exists(bj) and os.path.exists(bm)
        oj = os.path.join(tmp, "observability.json")
        oh = os.path.join(tmp, "observability.html")
        out = subprocess.check_output([sys.executable, "-m", "hga", "observability",
                                       "--out", oj, "--html", oh],
                                      cwd=KOK, text=True)
        assert "Gözlem paneli" in out
        assert os.path.exists(oj) and os.path.exists(oh)


def test_cli_veri_canli_smoke_bos_konu_ag_yok():
    out = subprocess.check_output([sys.executable, "-m", "hga", "veri-canli-smoke",
                                   "--konular", ","], cwd=KOK, text=True)
    assert "Canlı veri smoke" in out
    assert "konu listesi boş" in out


def test_cli_veri_canli_smoke_kontrollu_cikti():
    with tempfile.TemporaryDirectory() as tmp:
        out_yol = os.path.join(tmp, "veri.json")
        temiz_yol = os.path.join(tmp, "temiz.txt")
        out = subprocess.check_output([sys.executable, "-m", "hga", "veri-canli-smoke",
                                       "--kontrollu", "--konular", "Türkçe,İstanbul",
                                       "--out", out_yol, "--cikis", temiz_yol],
                                      cwd=KOK, text=True)
        assert "Canlı veri smoke" in out
        assert os.path.exists(out_yol)
        assert os.path.exists(temiz_yol)


def test_cli_manifest():
    fd, yol = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("Türkçe veri\n")
        out = subprocess.check_output([sys.executable, "-m", "hga", "manifest", yol],
                                      cwd=KOK, text=True)
        assert "sha256" in out
        assert "bytes" in out
    finally:
        if os.path.exists(yol):
            os.remove(yol)


def test_cli_checkpoint_rapor():
    if importlib.util.find_spec("torch") is None:
        out = subprocess.check_output([sys.executable, "-m", "hga", "checkpoint-rapor", "dummy.pt"],
                                      cwd=KOK, text=True)
        assert "torch gerekli" in out
        return

    import torch
    from kuresel_model import HiperGeometrikAI

    model = HiperGeometrikAI(n=8, katman_sayisi=1, baglam_penceresi=4, emb_dim=8,
                             num_heads=2, sozluk_boyutu=32, dropout=0.0,
                             seyrek_tablo_boyutu=0, bilgilendir=False)
    fd, yol = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save({"checkpoint_version": "cli-test",
                    "model_meta": {"n": 8, "baglam_penceresi": 4},
                    "model_state_dict": model.state_dict()}, yol)
        out = subprocess.check_output([
            sys.executable, "-m", "hga", "checkpoint-rapor", yol,
            "--n", "8", "--katman", "1", "--baglam", "4", "--vocab", "32",
            "--emb", "8", "--heads", "2", "--seyrek-satir", "0",
        ], cwd=KOK, text=True)
        assert "Checkpoint uyumluluk" in out
        assert "ok                  : True" in out
        assert "cli-test" in out
    finally:
        if os.path.exists(yol):
            os.remove(yol)


if __name__ == "__main__":
    testler = [(ad, fn) for ad, fn in sorted(globals().items())
               if ad.startswith("test_") and callable(fn)]
    basarisiz = 0
    for ad, fn in testler:
        try:
            fn()
            print(f"  ✅ {ad}")
        except Exception as e:  # noqa: BLE001
            basarisiz += 1
            print(f"  ❌ {ad}: {type(e).__name__}: {e}")
    print("\nTÜM TESTLER GEÇTİ" if basarisiz == 0 else f"{basarisiz} test başarısız")
    sys.exit(1 if basarisiz else 0)
