# İnsan Değerlendirme CSV Import/Aggregation Pipeline

- Protokol: `human_evaluation_csv_import_v1` v1 (imza `cb57a1eec46f`)
- Durum: **CSV_COMPLETE_UNATTESTED_NA**
- Gerçek insan beyanı: **YOK**

## Import özeti

| Alan | Değer |
|---|---:|
| CSV/değerlendirici | 10 |
| Öğe | 200 |
| Dolu hücre | 10000 |
| Toplam hücre | 10000 |
| Doluluk | 1.0000 |

## Kabul kapıları

| Kapı | Sonuç |
|---|---|
| csv_files_found | GEÇTİ |
| rating_values_in_scale | GEÇTİ |
| min_raters_present | GEÇTİ |
| all_cells_filled | GEÇTİ |
| balanced_rows_per_rater | GEÇTİ |
| rater_attestation_present | KALDI |
| reliability_computed_when_any_rating_present | GEÇTİ |
| arm_summary_available_when_complete | GEÇTİ |

## Krippendorff α

| Boyut | α | Hüküm |
|---|---:|---|
| dogruluk | 0.9839 | KABUL EDİLEBİLİR: α=0.9839 ≥ 0.8; sonuçlar güvenilir kabul edilebilir. |
| tutarlilik | 0.9839 | KABUL EDİLEBİLİR: α=0.9839 ≥ 0.8; sonuçlar güvenilir kabul edilebilir. |
| dil_kalitesi | 0.9839 | KABUL EDİLEBİLİR: α=0.9839 ≥ 0.8; sonuçlar güvenilir kabul edilebilir. |
| belirsizlik_durustlugu | 0.9839 | KABUL EDİLEBİLİR: α=0.9839 ≥ 0.8; sonuçlar güvenilir kabul edilebilir. |
| halusinasyon_var | 0.9436 | KABUL EDİLEBİLİR: α=0.9436 ≥ 0.8; sonuçlar güvenilir kabul edilebilir. |

## Kör açma sonrası kol özeti

| Kol | n/boyut | Halüsinasyon oranı |
|---|---:|---:|
| hga | 500 | 99.800% |
| dense | 500 | 99.800% |
| transformer | 500 | 99.600% |
| symbolic | 500 | 3.400% |

## Bulgular

- 10 CSV dosyası okundu; doluluk 1.0000.
- Tam/beyanlı gerçek insan puanı yok; ana raporda bu bölüm n/a kalmalıdır.

## Sınırlar

- CSV import gerçek insan kalitesini garanti etmez; yalnız format, ölçek ve agregasyon hattını doğrular.
- rater_attestation.json olmadan dolu CSV'ler gerçek insan sonucu sayılmaz ve ana raporda n/a kalır.
- Kör açma anahtarı yoksa veya paket eksikse kol sonuçları üretilmez.
