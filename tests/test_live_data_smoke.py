# -*- coding: utf-8 -*-
import os
import sys
import tempfile

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.data import canli_wikipedia_smoke


def test_canli_wikipedia_smoke_fake_fetcher_manifest():
    def fake(konu):
        return (f"{konu} Türkçe bir ansiklopedi maddesidir. Öğrenciler bu metni kalite testi için okur. "
                f"İstanbul ve Iğdır gibi örnekler Türkçe karakterleri korur.")

    with tempfile.TemporaryDirectory() as tmp:
        cikis = os.path.join(tmp, "wiki_smoke.txt")
        r = canli_wikipedia_smoke(["Türkçe", "İstanbul"], proje_kok=tmp,
                                  cikis=cikis, fetcher=fake)
        assert r.durum == "ok"
        assert r.cekilen_konu == 2
        assert r.temiz_cumle >= 2
        assert os.path.exists(cikis)
        assert r.manifest and r.manifest["bytes"] > 0
        assert len(r.manifest["sha256"]) == 64


def test_canli_wikipedia_smoke_bos_konu():
    r = canli_wikipedia_smoke([], fetcher=lambda _: "")
    assert r.durum == "error"


if __name__ == "__main__":
    test_canli_wikipedia_smoke_fake_fetcher_manifest()
    test_canli_wikipedia_smoke_bos_konu()
    print("OK")
