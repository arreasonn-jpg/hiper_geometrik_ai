# -*- coding: utf-8 -*-
"""
Seyrek bellek collision/determinizm/bellek muhasebesi testleri.
Torch yoksa güvenli atlanır.
"""
import importlib.util
import os
import sys

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIMARI = os.path.join(KOK, "mimari")
for _p in [KOK, MIMARI]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _torch_yoksa_atla():
    return importlib.util.find_spec("torch") is None


def test_hash_determinizmi_ve_collision_metriği():
    if _torch_yoksa_atla():
        print("  (torch yok — seyrek metrik testi atlandı)")
        return
    import torch
    from seyrek_tablo import HashlenmisKureselTablo

    t = HashlenmisKureselTablo(tablo_boyutu=4096, boyut=8, tablo_sayisi=2)
    x = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 4]], dtype=torch.long)
    assert torch.equal(t.anahtar(x), t.anahtar(x))
    assert torch.equal(t.adres_imzalari(x), t.adres_imzalari(x))
    rapor = t.carpisma_istatistigi(x)
    assert rapor["benzersiz_pencere"] == 2
    assert 0.0 <= rapor["collision_rate"] <= 1.0
    assert rapor["tablo_sayisi"] == 2


def test_bellek_kullanimi():
    if _torch_yoksa_atla():
        return
    from seyrek_tablo import HashlenmisKureselTablo

    t = HashlenmisKureselTablo(tablo_boyutu=1024, boyut=32, tablo_sayisi=1)
    m = t.bellek_kullanimi()
    assert m["toplam_bayt"] == 1024 * 32 * m["parametre_bayt"]
    assert t.kapasite()["ram_mb"] == m["ram_mb"]


def test_lru_decay_erisim_izleme():
    if _torch_yoksa_atla():
        return
    import torch
    from seyrek_tablo import HashlenmisKureselTablo

    t = HashlenmisKureselTablo(tablo_boyutu=32, boyut=4, erisim_izleme=True)
    x1 = torch.tensor([[1, 2, 3]], dtype=torch.long)
    x2 = torch.tensor([[4, 5, 6]], dtype=torch.long)
    t(x1)
    # Dokunulan satırı elle doldur: LRU temizliğinin etkisi net görülsün.
    adr1 = int(t.adres(t.anahtar(x1), 0).item())
    t.tablolar[0].weight.data[adr1].fill_(1.0)
    t(x2)
    assert t.erisim_sayisi[0, adr1].item() == 1
    assert t.decay_uygula(0.5) >= 1
    temizlenen = t.lru_temizle(max_yas=0)
    assert temizlenen >= 1
    assert t.tablolar[0].weight.data[adr1].abs().sum().item() == 0.0


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
