# Hiyerarşik Bellek 100M Streaming Harness

- Protokol: `hierarchical_memory_streaming_harness_v1` v1
- Hedef kayıt: `100,000,000`
- Gerçek smoke yazması: `10,000` kayıt

> Bu artifact 100M kaydın tamamının yazıldığını iddia etmez; 100M-ready planı, checkpoint/resume manifestini ve sınırlı smoke ingest ölçümünü üretir.

## Plan

| Alan | Değer |
|---|---:|
| Shard başına kayıt | 1,000,000 |
| Planlanan shard | 100 |
| Checkpoint aralığı | 1,000 |
| Örnek fraksiyon | 0.0001000000 |

Resume anahtarı: `key-000000010000` (format `key-{i:012d}`)

## Smoke ingest

- Yazma hızı: **70,322 kayıt/s** (14.22 µs/kayıt)
- Süre: `0.1422` s
- Checkpoint sayısı: `10`
- Final katmanlar: hot `500`, warm `2,000`, cold `5,000`, archive `2,500`
- Düşen kayıt: `0`

## Recall audit

- Probe: `200`
- Bulunan: `200`
- Recall: **1.000000**
- Değer bozulması: `0`

## Projeksiyonlar (ölçüm değil)

- Tür: `linear_projection_from_sample_not_measured_100m`
- Hedef duvar-saat projeksiyonu: `0.395008` saat
- Hedef disk projeksiyonu: `10,973,980,000` bayt
- Örnek disk/kayıt: `109.7398` bayt

## Kabul kapıları

| Kapı | Sonuç |
|---|---|
| target_is_100m_or_more | GEÇTİ |
| sample_stream_completed | GEÇTİ |
| sample_is_bounded_not_full_target | GEÇTİ |
| checkpoint_manifest_written | GEÇTİ |
| resume_key_monotonic | GEÇTİ |
| archive_enabled_no_drop | GEÇTİ |
| sample_recall_complete | GEÇTİ |
| projection_marked_not_measured | GEÇTİ |

## Bulgular

- 100M hedef planı: 100 shard × 1,000,000 kayıt.
- Smoke yazma: 10,000 kayıt, 70,322 kayıt/s; resume anahtarı `key-000000010000`.
- Örnek recall 1.0000; düşen kayıt 0.
- Disk/zaman hedef değerleri lineer projeksiyondur; 100M koşu ölçümü değildir.

## Sınırlar

- Bu harness 100M kaydı gerçekten yazmaz; 100M için plan, checkpoint ve smoke ölçümü üretir.
- Projeksiyonlar lineerdir; SQLite indeks büyümesi, dosya sistemi, sıkıştırma ve cache etkileri gerçek büyük koşuda değişebilir.
- Tek süreç/tek writer ölçülür; paralel ingest ve uzaktan nesne deposu bu smoke içinde yoktur.
