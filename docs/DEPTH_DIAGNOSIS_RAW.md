# Çıkarım Derinliği Çöküşü — Kök Neden Teşhisi

- Protokol: `reasoning_depth_root_cause_v1` v1 (imza `7429c620b3cb`)
- Profil: `deep` · tohumlar `[1, 2, 3]` · dolgu `16384`

## Tek değişkenli tasarım

Yalnız slot sayısı değişir; hop ızgarası, dolgu, tohum ve eşik sabittir.

| Slot | Güvenilir derinlik |
|---:|---:|
| 131072 (2^17) | 128 |
| 262144 (2^18) | 512 |
| 524288 (2^19) | 2048 |
| 1048576 (2^20) | 2048 |

## Geri çağırma ızgarası (dolgu = 16384)

| Slot \ hop | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| 131072 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 262144 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| 524288 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1048576 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Kontrol kolu (dolgu = 0)

- Tüm derinlikler güvenilir mi: **EVET**

> Dolgu=0'da tüm derinlikler güvenilirse, zincirin kendisi bozulmuyor demektir; kaybın kaynağı dolgunun belleğe yaptığı baskıdır.

## Ölçekleme

- Slot çarpanı: `8.0×`
- Derinlik çarpanı: `16.0×`
- log-log eğim: `1.3333`
- Slotlarda monoton: `True`

> Eğim ≈ 1.0 ise desteklenen derinlik slot sayısıyla doğrusal ölçekleniyor demektir: sınır bellektedir.

## Teşhis

**Kök neden: `memory_capacity`**

Slot sayısı arttıkça desteklenen derinlik ölçekleniyor ve dolgusuz kontrolde tüm derinlikler zaten güvenilir. Çöküş bir ÇIKARIM sınırı değil, BELLEK KAPASİTESİ sınırıdır.

> P0-7'de 16384 dolgudaki 32→8 düşüşü 'dolgu dayanıklılığı kaybı' diye yorumlanmıştı; bu teşhis o yorumu DÜZELTİR.

## Kabul kapıları

| Kapı | Sonuç |
|---|---|
| single_variable_design | GEÇTİ |
| control_arm_without_distractors_clean | GEÇTİ |
| depth_monotone_in_slots | GEÇTİ |
| root_cause_identified | GEÇTİ |
| scaling_measured | GEÇTİ |
| collapse_is_engineering_not_reasoning | GEÇTİ |

## Bulgular

- Tek değişken: slot sayısı 131072 → 1048576 (8×). Hop ızgarası, dolgu (16384), tohumlar ([1, 2, 3]) ve eşik sabit tutuldu.
-   slots=131072 (2^17) → güvenilir derinlik 128
-   slots=262144 (2^18) → güvenilir derinlik 512
-   slots=524288 (2^19) → güvenilir derinlik 2048
-   slots=1048576 (2^20) → güvenilir derinlik 2048
- Slot 8× büyüyünce derinlik 16× büyüdü; log-log eğim 1.333.
- Slot sayısı arttıkça desteklenen derinlik ölçekleniyor ve dolgusuz kontrolde tüm derinlikler zaten güvenilir. Çöküş bir ÇIKARIM sınırı değil, BELLEK KAPASİTESİ sınırıdır.
- Kontrol kolu: dolgu=0'da TÜM derinlikler %100 güvenilir. Zincirin kendisi hiçbir zaman bozulmadı.
- SONUÇ: C_RD tablosundaki düşüş bir yetenek eksikliği değil, yapılandırma seçimidir. Slot bütçesi verildiğinde derinlik geri gelir; bu bir ölçeklendirme parametresidir.
- Bu, P0-7'nin 'dolgu dayanıklılığı' yorumunu düzeltir. Eski yorum ölçümü değil, ölçümün SEBEBİNİ yanlış okuyordu.

## Sınırlar

- Bellek sınırı slot sayısıyla aşılıyor ama slotlar bedava değildir: RAM maliyeti doğrusal büyür (bkz. docs/MEMORY_HIERARCHY.md).
- Teşhis sentetik multi-hop ızgarası üzerindedir; gerçek metin üzerinde dolgunun etkisi farklı olabilir.
- Slot ölçeklemesi sonsuza kadar sürmez; bu ızgarada henüz bir çıkarım tavanına ULAŞILMADI, yani gerçek çıkarım sınırı hâlâ ölçülmemiştir.
- Eşik 1.0 (tam geri çağırma) seçilmiştir; gevşek bir eşik daha büyük derinlikler raporlardı.

