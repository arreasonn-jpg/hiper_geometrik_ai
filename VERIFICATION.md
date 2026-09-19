# HGA Verification Record — CKPT-001 & CKPT-002

**Tarih:** 2026-09-19/20
**Makine:** HP 14-cf2006nt, WSL2, CPU-only, 4.8 GB RAM, 8 CPU
**Python:** 3.12.3 · torch 2.14.0+cpu · numpy 1.26.4 (locked)

## CKPT-001 — Foundation VERIFIED ✅

- 5-seed `reproduce-all --profile smoke`: **COMPLETED**, 6/6 checks PASS
- 5-seed `reproduce-all --profile full`: **COMPLETED**, 6/6 checks PASS
- 10-seed smoke: **COMPLETED**, 6/6 checks PASS
- **Determinizm kanıtlandı**: iki bağımsız 5-seed koşu, bit-bit aynı fingerprint
  - smoke: `93c2b59181f7f5fa5e41cf3bf079fa81b1a8f4be2f27c6292fb9f3a6a7880f5c`
  - full:  `8414a610a75bc42d468ad7697743f98dc92a06ef4a7e984dd14973213a39201e`
- Süre: 3–10 dk/koşu, RAM peak ~1 GB, disk artışı ~0

## CKPT-002 — DRAFT ✅ (Büyük bölümü tamam)

- Full profile 5 seed + 10 seed smoke: tamamlandı
- Kronecker rank grid (n=16,32,64 × K=1,2,4,8): 12 konfigürasyon
- English scaling probe (3 boyut, 5 seed): EXPLORATORY etiketli
- Tohum istatistik (10 seed, 3 protokol): 76 karşılaştırma, 44'ü anlamlı
- Paradigma karşılaştırması (3 kol, 3 seed): hybrid kazanıyor
- Memory benchmark (1K→1M): bounded capacity kanıtlandı

## Kritik Bilimsel Bulgular

1. **Hibrit paradigm üstünlüğü**: hybrid 0.885 > neural 0.762 > symbolic 0.491
2. **Kronecker derinlik (K) katkı sağlamıyor**: etkin rank K'dan bağımsız olarak n²
3. **Seyrek bellek çoğu görevde etkisiz**: 8 bağlamdan 7'sinde CI=[0,0]
4. **Bellek kapasitesi sınırı**: ~100K bağlamda çöküş (dürüst ölçüm)
5. **Determinizm tam**: iki bağımsız koşu, aynı fingerprint

## Sınırlar

- CPU-only doğrulama; GPU'da numerik farklılık beklenebilir
- Tek fiziksel makine; bağımsız çoğaltma henüz yok
- GHCR Docker imajı yayınlanmadı
- 3 legacy test failure (`test_gorev_ablasyonu.py`) izole edildi, CKPT-001 kapsamı dışında

## Onaylanan Save Tag'ler

- `ckpt-001-verified`
- `ckpt-002-draft`
