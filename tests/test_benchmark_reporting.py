# -*- coding: utf-8 -*-
import json
import os
import sys
import tempfile

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)
MIMARI = os.path.join(KOK, "mimari")
if MIMARI not in sys.path:
    sys.path.insert(0, MIMARI)

from hga.evaluation.reporting import (
    benchmark_raporu_kaydet,
    benchmark_raporu_markdown,
    benchmark_raporu_olustur,
)


def test_benchmark_raporu_yapisi_ve_markdown():
    rapor = benchmark_raporu_olustur()
    assert rapor["rapor_tipi"] == "hga-mini-benchmark-v1"
    assert "cihaz" in rapor and "cuda_available" in rapor["cihaz"]
    assert rapor["tokenizer"]["turkce_karakter_tam"] is True
    assert "hallucination_rate" in rapor["halusinasyon"]
    md = benchmark_raporu_markdown(rapor)
    assert "HGA Mini Benchmark Raporu" in md
    assert "Held-out mini perplexity" in md


def test_benchmark_raporu_kaydet():
    rapor = {"rapor_tipi": "x", "olusturma_zamani_utc": "z", "sinirlar": {"not": "n"},
             "cihaz": {"device": "cpu"}, "tokenizer": {}, "perplexity": {"notlar": []},
             "halusinasyon": {}, "seyrek_bellek": {}}
    with tempfile.TemporaryDirectory() as tmp:
        jyol = os.path.join(tmp, "rapor.json")
        myol = os.path.join(tmp, "rapor.md")
        yollar = benchmark_raporu_kaydet(rapor, jyol, myol)
        assert set(yollar) == {"json", "markdown"}
        with open(jyol, "r", encoding="utf-8") as f:
            assert json.load(f)["rapor_tipi"] == "x"
        with open(myol, "r", encoding="utf-8") as f:
            assert "HGA Mini Benchmark" in f.read()


if __name__ == "__main__":
    test_benchmark_raporu_yapisi_ve_markdown()
    test_benchmark_raporu_kaydet()
    print("OK")
