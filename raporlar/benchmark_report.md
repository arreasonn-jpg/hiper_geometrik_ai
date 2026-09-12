# HGA Mini Benchmark Raporu

- Rapor tipi: `hga-mini-benchmark-v1`
- Zaman (UTC): `2026-09-12T22:46:13.693405+00:00`
- Sınır: Rapor smoke/held-out sağlık ölçümüdür; eğitimli kalite iddiası değildir.

## Cihaz
- torch_available: `True`
- torch_version: `2.3.1+cu121`
- cuda_available: `False`
- device: `cpu`
- note: `CUDA yok; VRAM metrikleri CPU fallback olarak raporlandı`

## Ana Metrikler
- Tokenizer karakter kapsama: `1.0`
- Tokenizer bilinmeyen oranı: `0.0070921985815602835`
- Held-out mini perplexity: `165.54400899606878`
- Hallucination oranı: `0.8`
- Bellek doluluk oranı: `0.25`

## Notlar
- unk_orani=0.0093
- tokenizer yalnız train split üzerinde fit edildi; held-out mini smoke
- checkpoint yok: rastgele ağırlıklar; kalite iddiası değildir
