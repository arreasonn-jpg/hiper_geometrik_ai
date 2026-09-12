# -*- coding: utf-8 -*-
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from bilgi_katmani import BilgiKatmani, KATMAN_ACIK, KATMAN_KISMI, KATMAN_TAM, norm


def test_bilgi_katmani_torchsuz_arama():
    bk = BilgiKatmani([{"soru": "Ali neye bindi?", "cevap": "Ali ata bindi."}])
    katman, cevap, skor, _ = bk.ara("Ali neye bindi?")
    assert katman == KATMAN_TAM
    assert "ata" in cevap
    assert skor == 1.0

    katman, _, skor, eslesme = bk.ara("Ali bindi mi?")
    assert katman in (KATMAN_KISMI, KATMAN_ACIK)
    if katman == KATMAN_KISMI:
        assert skor >= 0.4
        assert eslesme is not None


def test_norm_turkce_kesme_ve_karakter():
    assert norm("Türkiye'nin başkenti neresidir?") == "turkiyenin baskenti neresi"


if __name__ == "__main__":
    test_bilgi_katmani_torchsuz_arama()
    test_norm_turkce_kesme_ve_karakter()
    print("OK")
