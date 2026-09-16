# Signature Benchmark Release Gate

- Protokol: `hga_signature_release_gate_v1`
- Kaynak: `hga_signature_benchmark_v1` · `d699088469e4`
- Durum: **BLOCKED**
- Gerekli profil: `standard`
- Minimum tohum: `20`

## Kapılar

| kapı | sonuç |
|---|---|
| source_protocol_is_signature_v1 | GEÇTİ |
| profile_is_standard | GEÇTİ |
| seed_count_at_least_20 | KALDI |
| all_signature_tasks_present | GEÇTİ |
| all_signature_arms_present | GEÇTİ |
| dataset_hash_present | GEÇTİ |
| all_benchmark_checks_pass | KALDI |
| no_held_out_leak | GEÇTİ |
| hga_has_signature_task | GEÇTİ |
| memory_gain_is_task_specific | GEÇTİ |
| parameter_budget_within_gate | GEÇTİ |
| signature_payload_nonempty | GEÇTİ |

## Kanıt özeti

- Kaynak profil: `standard`
- Kaynak tohum sayısı: `1`
- Düşen benchmark kapıları: hga_beats_majority_everywhere

## Release kararı

- BLOCKED: release hazır değil; düşen kapılar: seed_count_at_least_20, all_benchmark_checks_pass

## Sınırlar

- Release gate yalnız Signature Benchmark raporunu denetler; yeni model eğitimi veya insan değerlendirmesi üretmez.
- Gate PASS değilse release hazır denemez; BLOCKED durumu eksik kanıtı gizlemez.
- 20 tohum ve standard profil eşiği release kalitesi içindir; smoke koşuları regresyon amaçlı kalır.
