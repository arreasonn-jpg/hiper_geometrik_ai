# P3 — HGA Türkçe LM Ablasyonları (Neden 3× Geride?)

**Tarih:** 2026-10-18
**Scriptler:** `artifacts/p3_lm_ablasyon.py`, `p3_lm_derin_analiz.py`, `p3_lm_egitim_egrisi.py`
**Veri:** TWT Smoke (16K cümle, 400-2000 step)

## Özet Bulgular

1. **HGA öğreniyor ama yavaş.** 2000 step'te PPL 99→56'ya düşüyor. Dense 39, Transformer 40.
2. **Attention ölü yük.** HGA içindeki `HiperGeometrikAttention` kapatıldığında PPL 61.13 → 60.35 (neredeyse hiç fark yok).
3. **Kronecker asıl düzenlileştirici.** Kronecker kapatıldığında PPL 61.13 → 67.92 (kötüleşiyor). Yapısal kısıt, ezberlemeyi engelliyor.
4. **Derinlik (K) zararlı.** K=1 (61.13) vs K=2 (65.32). Preprint 3 bulgularıyla uyumlu.

## Eğitim Eğrisi (2000 step)

| Step | Dense | Transformer | HGA |
|---:|---:|---:|---:|
| 200 | 82.8 | 101.6 | 99.9 |
| 1000 | 46.0 | 52.1 | 69.9 |
| 2000 | **39.6** | **40.2** | **56.6** |

## Hiperparametre Duyarlılığı

| Varyant | PPL | Sonuç |
|---|---:|---|
| HGA (LR=0.003) | 61.13 | Baz |
| HGA (LR=0.01) | 113.53 | **Iraksıyor** |
| HGA (LR=0.0001) | 109.85 | **Öğrenemiyor** |
| HGA (n=32) | 58.89 | Genişlik yardımcı |
| HGA (n=8) | 81.66 | Çok dar |
| HGA (K=2) | 65.32 | Derinlik zararlı |

## Bilimsel Sonuç

**HGA genel bir dil modeli değildir.** 248K parametreyle 61 PPL alırken, Dense 108K parametreyle 45 PPL alıyor.
- **HGA'nın yeri:** Gürültülü, yapısal kısıt gerektiren, düşük verili görevler (W6/W7).
- **Dense/Transformer'ın yeri:** Saf dil modelleme.
- Bu, **Preprint 3'ün "implicit regulariser" tezini** doğrular ve HGA'nın sınırlarını dürüstçe çizer.

## Referans
- `docs/TURKISH_LM_20SEED.md` (ana LM sonucu: 621 vs 208 PPL)
- W6/W7 mekanizma çalışmaları
