# -*- coding: utf-8 -*-
import os
import sys
import tempfile

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

import egitim.veri_toplayici as vt


def test_import_torchsuz_ve_opsiyonel_ag():
    eski_requests, eski_pq = vt.requests, vt.pq
    vt.requests = None
    vt.pq = None
    try:
        assert vt._requests_gerekli.__name__ == "_requests_gerekli"
        try:
            vt._requests_gerekli()
            raise AssertionError("requests yokken ImportError bekleniyordu")
        except ImportError as e:
            assert "requests" in str(e)

        with tempfile.TemporaryDirectory() as tmp:
            toplayici = vt.OtomatikVeriToplayici(tmp)
            assert toplayici.wikipedia_cek("İstanbul") == ""
            assert toplayici._dosya_oku_ve_ayikla("repo", "dosya.txt", 10, 0) == ("", 0)
            assert toplayici._parquet_akilli_oku("https://example.com/a.parquet", 10, 0) == ("", 0)
    finally:
        vt.requests, vt.pq = eski_requests, eski_pq


if __name__ == "__main__":
    test_import_torchsuz_ve_opsiyonel_ag()
    print("OK")
