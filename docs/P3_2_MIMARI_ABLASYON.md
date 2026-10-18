# P3-2 — HGA Derin Mimari Ablasyonu (Asıl Sorun Bulundu!)

**Tarih:** 2026-10-18
**Script:** `artifacts/p3_2_mimari_ablation.py`
**Sonuç:** `artifacts/w7_runs/p3_2_mimari_ablation.json`
**Protokol:** Smoke (1000 step, 42 seed, TWT)

## Amaç
HGA'yı oluşturan alt bileşenleri tek tek kapatarak (pozisyon, embedding, encoder, decoder) hangisinin performansı sınırladığını bulmak.

## Sonuçlar

| Varyant | Test PPL | Parametre | Süre | Yorum |
|---|---:|---:|---:|---|
| hga_full | 70.40 | 248K | 12s | Referans |
| hga_no_pos | 72.24 | 248K | 12s | Pozisyon hafif yardımcı |
| hga_frozen_emb | 81.58 | 216K | 32s | Gömme katmanı kritik |
| hga_linear_decoder | 72.82 | 696K | 26s | Basit decoder 3x pahalı ve daha kötü |
| **hga_no_encoder** | **58.32** | **306K** | **13s** | 🚨 **EN İYİ HGA** |

## Kritik Keşif: GeometrikVeriEncoder (0→1 katmanı) Zararlı!

HGA'nın içindeki `GeometrikVeriEncoder` (tanh(u ⊗ v) dış çarpım köprüsü) modelin en zayıf halkasıdır.

- Onu basit bir `nn.Linear` projeksiyonu ile değiştirdiğimizde, HGA **70.40 → 58.32 PPL**'e düşüyor (12 puan iyileşme).
- Bu, modelin Kronecker zincirinin ve Fraktal Decoder'ının aslında **iyi çalıştığını**, ama karmaşık encoder'ın gradyan akışını sabote ettiğini gösteriyor.

## Diğer Önemli Bulgular

1. **FraktalDecoder aslında çok verimli:** Onu basit bir Linear ile değiştirmek PPL'i hafif kötüleştiriyor (72.82) ve parametre sayısını 3 kat artırıyor (696K). Fraktal yapı korunmalı.
2. **Gömme katmanı kritik:** Dondurulduğunda PPL 70 → 81'e fırlıyor. HGA'nın kendi temsillerini öğrenmesi şart.
3. **Pozisyon kodlaması hafif yardımcı:** Sıfırlandığında PPL 70 → 72 oluyor. Etkisi küçük ama pozitif.

## Bilimsel Sonuç

HGA genel bir dil modeli olarak hala Dense'in (45 PPL) gerisinde (58 PPL). Ancak bu ablasyon, **mimari bir kusuru** ortaya çıkardı: `GeometrikVeriEncoder` gereksiz karmaşık ve zararlı. Onu daha basit bir projeksiyonla değiştirmek, HGA'nın performansını ciddi şekilde artırıyor.

## Referans
- `docs/P3_LM_ABLASYON.md` (ilk ablasyonlar)
- `docs/TURKISH_LM_20SEED.md` (ana LM sonucu)
