# -*- coding: utf-8 -*-
import json
import os
import sys
import tempfile

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

from hga.knowledge import DeneyimDurumu, ExperienceCandidate
from hga.memory import DeneyimSlotlari
from hga.observability import gozlem_paneli_html, gozlem_paneli_kaydet, gozlem_paneli_olustur


def test_gozlem_paneli_json_markdown_html():
    adaylar = [
        ExperienceCandidate("X1", "E1", "R", "E2", state=DeneyimDurumu.VALID),
        ExperienceCandidate("X2", "E1", "R", "E3", state=DeneyimDurumu.INVALID),
    ]
    slot = DeneyimSlotlari(slot_sayisi=8)
    slot.yaz("X1", ("E1", "R", "E2"))
    panel = gozlem_paneli_olustur(adaylar, slot, [([1, 0], [1, 0]), ([0, 1], [1, 0])])
    assert panel["rapor_tipi"] == "hga-observability-panel-v1"
    assert panel["deneyim_akisi"]["accepted"] == 1
    assert panel["bellek"]["dolu"] == 1
    assert panel["geometri"]["katman_sayisi"] == 2
    html = gozlem_paneli_html(panel)
    assert "Ham JSON" in html
    with tempfile.TemporaryDirectory() as tmp:
        jyol = os.path.join(tmp, "panel.json")
        hyol = os.path.join(tmp, "panel.html")
        yollar = gozlem_paneli_kaydet(panel, json_yol=jyol, html_yol=hyol)
        assert set(yollar) == {"json", "html"}
        with open(jyol, "r", encoding="utf-8") as f:
            assert json.load(f)["rapor_tipi"] == "hga-observability-panel-v1"
        assert os.path.exists(hyol)


if __name__ == "__main__":
    test_gozlem_paneli_json_markdown_html()
    print("OK")
