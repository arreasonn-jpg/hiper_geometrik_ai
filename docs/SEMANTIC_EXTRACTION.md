# Türkçe Semantik Çıkarım Benchmarkı (P0-5)

Protokol: `turkish_semantic_extraction_v1` · veri imzası: `525654cb978c` · 12 cümle (10 kapsam içi / 2 dışı)

| Katman | Precision | Recall | F1 | TP | Tahmin | Altın |
|---|---:|---:|---:|---:|---:|---:|
| entity | 1.0000 | 1.0000 | 1.0000 | 24 | 24 | 24 |
| relation | 1.0000 | 1.0000 | 1.0000 | 11 | 11 | 11 |
| property | 1.0000 | 1.0000 | 1.0000 | 5 | 5 | 5 |
| temporal | 1.0000 | 1.0000 | 1.0000 | 3 | 3 | 3 |
| negation | 1.0000 | 1.0000 | 1.0000 | 2 | 2 | 2 |

- İlişki tam eşleşme: **1.0000**
- Aşırı çıkarım oranı: 0.0000
- Kapsam dışı yanlış pozitif: 0.0000
- Çekimserlik (ilişki çıkaramama): 0.1667
- Ortalama ilişki güveni: 0.9000

## Kabul kapıları

| kapı | sonuç |
|---|---|
| relation_f1_at_least_0_80 | GEÇTİ |
| entity_f1_at_least_0_90 | GEÇTİ |
| negation_recall_perfect | GEÇTİ |
| temporal_recall_perfect | GEÇTİ |
| no_out_of_scope_hallucination | GEÇTİ |
| property_precision_at_least_0_80 | GEÇTİ |

## Hatalı cümleler

- Yok: her cümlede ilişki kümesi altınla birebir.

## Bulgular

- 12 elle etiketli cümle (10 kapsam içi, 2 kapsam dışı).
- Katman F1: entity=1.0000, relation=1.0000, property=1.0000, temporal=1.0000, negation=1.0000.
- İlişki tam eşleşme (özne+yüklem+nesne+rol+kutup): 1.0.
- Aşırı çıkarım oranı 0.0000; kapsam dışı cümlelerde yanlış ilişki 0/2.

## Sınırlar

- Altın set küçüktür ve elle yazılmıştır; istatistiksel bir Türkçe NER/RE değerlendirmesi DEĞİLDİR. Sonuçlar hattın kapsamını gösterir, genel Türkçe performansını değil.
- Ontolojiden türetilen özellikler (canlı, binilebilir) altın sette yer almaz ve precision hesabına katılmaz; onlar metinden değil tip varsayımından gelir ve düşük güvenle yazılır.
- Hat kural tabanlıdır: sözlük büyüdükçe recall artar, bu bir öğrenme sonucu değildir.
- Kapsam dışı örnek sayısı azdır; precision tahmini geniş güven aralığına sahiptir.
