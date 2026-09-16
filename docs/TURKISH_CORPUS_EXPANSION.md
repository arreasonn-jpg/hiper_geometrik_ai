# Türkçe Korpus Genişletme Pipeline

- Protokol: `turkish_corpus_expansion_pipeline_v1` v1
- Mod: `embedded_smoke_candidates`
- Hedef: `2,000,000` kelime

> Bu rapor yeni büyük korpus release'i değildir; aday alımı, lisans, Türkçe filtre, dedup ve split/release-gate hattını kanıtlar.

## Projeksiyon

| Alan | Değer |
|---|---:|
| Mevcut kelime | 1,110,509 |
| Kabul edilen aday kelime | 26 |
| Projeksiyon toplam | 1,110,535 |
| Hedef boşluğu | 889,465 |
| Hedef tamamlanma oranı | 0.555268 |

## Aday özeti

| Alan | Değer |
|---|---:|
| Aday belge | 3 |
| Kabul edilen belge | 2 |
| Reddedilen belge | 1 |
| Duplicate satır | 0 |

## Split projeksiyonu (yalnız kabul edilen adaylar)

| Split | Belge | Kelime |
|---|---:|---:|
| train | 2 | 26 |
| dev | 0 | 0 |
| test | 0 | 0 |

## Release gate'leri

| Gate | Sonuç |
|---|---|
| external_candidates_supplied | KALDI |
| no_license_rejections | KALDI |
| accepted_candidate_words_positive | GEÇTİ |
| expanded_corpus_reaches_target_words | KALDI |
| provenance_urls_present_for_accepted | GEÇTİ |

## Pipeline kontrolleri

| Kontrol | Sonuç |
|---|---|
| base_provenance_loaded | GEÇTİ |
| candidate_schema_valid | GEÇTİ |
| accepted_licenses_allowed | GEÇTİ |
| turkish_filter_applied | GEÇTİ |
| dedup_against_current_corpus_applied | GEÇTİ |
| deterministic_split_assigned | GEÇTİ |
| smoke_mode_not_marked_as_release | GEÇTİ |

## Bulgular

- Mevcut tr_corpus_v1: 4,365 belge, 1,110,509 kelime.
- Aday mod: embedded_smoke_candidates; kabul edilen 2/3 belge, 26 kelime.
- Genişletilmiş projeksiyon 1,110,535/2,000,000 kelime; durum PIPELINE_READY_TARGET_NOT_REACHED.
- Smoke adayları yalnız pipeline kanıtıdır; tr_corpus_v2 release'i ya da gerçek genişleme değildir.
- Reddedilen adaylar: {'license_not_allowed': 1}.

## Sınırlar

- Bu pipeline dış kaynak indirmez; aday JSONL dizini ayrı sağlanmalıdır.
- Türkçe filtresi hafif bir sezgiseldir; nihai release için örneklemeli insan/otomatik kalite denetimi gerekir.
- Dedup normalize-exact düzeydedir; semantik yakın tekrarlar bu smoke içinde yakalanmaz.
- Release gate hedef kelime sayısı ve provenance koşulları sağlanmadan PASS olmaz.
