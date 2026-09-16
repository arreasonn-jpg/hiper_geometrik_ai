# Türkçe Semantik Çıkarım Benchmarkı (P0-5)

Protokol: `turkish_semantic_extraction_v2_curated_gold` · veri imzası: `564e616de621` · anotasyon imzası: `1c971ea4ff01` · 136 cümle (128 kapsam içi / 8 dışı)

| Katman | Precision | Recall | F1 | TP | Tahmin | Altın |
|---|---:|---:|---:|---:|---:|---:|
| entity | 1.0000 | 1.0000 | 1.0000 | 356 | 356 | 356 |
| relation | 1.0000 | 1.0000 | 1.0000 | 140 | 140 | 140 |
| property | 1.0000 | 1.0000 | 1.0000 | 108 | 108 | 108 |
| temporal | 1.0000 | 1.0000 | 1.0000 | 94 | 94 | 94 |
| negation | 1.0000 | 1.0000 | 1.0000 | 28 | 28 | 28 |

- İlişki tam eşleşme: **1.0000**
- Extraction yield: **1.0000**
- Relation coverage: 1.0000
- Aşırı çıkarım oranı: 0.0000
- Kapsam dışı yanlış pozitif: 0.0000
- Çekimserlik (ilişki çıkaramama): 0.0588
- Ortalama ilişki güveni: 0.9000

## Veri kapsamı

- Splitler: calibration=45, heldout=46, stress=45
- Yüklem kapsamı: almak=4, binmek=10, gelmek=28, gitmek=60, görmek=5, okumak=29, sevmek=4
- Rol kapsamı: hedef=80, kaynak=18, konum=12, nesne=30
- Fenomenler: case:ayrilma=18, case:belirtme=18, case:bulunma=12, case:yonelme=80, causative_voice=3, embedded_clause=1, instrument=88, location=12, movement=98, negation=29, nominalized_clause=1, object_relation=30, out_of_scope=8, passive_or_causative_voice=1, passive_subordinate_clause=1, polarity:positive=88, property=20, subordinate_clause=1, temporal=48

## Split metrikleri

| split | cümle | relation F1 | exact/yield | out-scope FP |
|---|---:|---:|---:|---:|
| calibration | 45 | 1.0000 | 1.0000 | 0.0000 |
| heldout | 46 | 1.0000 | 1.0000 | 0.0000 |
| stress | 45 | 1.0000 | 1.0000 | 0.0000 |

## Kabul kapıları

| kapı | sonuç |
|---|---|
| relation_f1_at_least_0_80 | GEÇTİ |
| entity_f1_at_least_0_90 | GEÇTİ |
| negation_recall_perfect | GEÇTİ |
| temporal_recall_perfect | GEÇTİ |
| no_out_of_scope_hallucination | GEÇTİ |
| property_precision_at_least_0_80 | GEÇTİ |
| extraction_yield_at_least_threshold | GEÇTİ |
| v2_gold_size_at_least_100 | GEÇTİ |
| v2_unique_sentences | GEÇTİ |
| v2_split_metadata_present | GEÇTİ |
| v2_out_of_scope_negatives_present | GEÇTİ |
| v2_extraction_yield_at_least_0_85 | GEÇTİ |

## Hatalı cümleler

- Yok: her cümlede ilişki kümesi altınla birebir.

## Bulgular

- 136 etiketli cümle (128 kapsam içi, 8 kapsam dışı).
- Katman F1: entity=1.0000, relation=1.0000, property=1.0000, temporal=1.0000, negation=1.0000.
- İlişki tam eşleşme / extraction_yield: 1.0000.
- Aşırı çıkarım oranı 0.0000; kapsam dışı cümlelerde yanlış ilişki 0/8.
- Kapsam özeti: almak=4, binmek=10, gelmek=28, gitmek=60, görmek=5, okumak=29, sevmek=4.

## Sınırlar ve dış geçerlilik

- Bu protokol istatistiksel bir Türkçe NER/RE değerlendirmesi DEĞİLDİR. Sonuçlar sınırlı, kural-tabanlı hattın kapsamını gösterir; genel Türkçe performansı veya dış geçerlilik iddiası değildir.
- v2 genişletmesi gerçek dış korpus veya bağımsız insan anotasyonu değil, kaynakta denetlenebilir şablon/altın etiket sözleşmesidir.
- Ontolojiden türetilen özellikler (canlı, binilebilir) altın sette yer almaz ve precision hesabına katılmaz; onlar metinden değil tip varsayımından gelir ve düşük güvenle yazılır.
- Hat kural tabanlıdır: sözlük büyüdükçe recall artar, bu bir öğrenme sonucu değildir.
- Kapsam dışı örnekler precision güvenliği için eklidir ama fuzzing veya geniş haber/sosyal medya dağılımı yerine geçmez.
