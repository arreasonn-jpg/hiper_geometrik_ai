# C_G v2 — Ham Metinden Keşif ve Kompozisyon (P0-6)

Protokol: `compositional_generalization_v2_raw_text` · veri imzası: `f59f990d1591`

Eğitimde keşfedilen varlıklar: `['ali', 'araba', 'at', 'ayse', 'can', 'ev', 'fatma', 'mehmet', 'okul', 'otobus', 'tren', 'veli']`
Eğitimde keşfedilen ilişkiler: `['binmek', 'gitmek']`

**Sisteme hiçbir entity id, ontoloji kaydı veya relation şeması önceden verilmedi.**

| Eksen | n | Kompozisyon | Entity keşfi | Relation indüksiyonu | Tip | Özellik | Zaman | C_G v2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seen | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| unseen_entity | 4 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 1.0000 | 1.0000 | 1.0000 |
| unseen_relation | 4 | 0.5000 | 1.0000 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 |
| unseen_both | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| unseen_wording | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**Genel C_G v2 = 0.8571**

## Şema sızıntısı denetimi

- Sızan varlık: []
- Sızan ilişki: []
- Temiz mi: EVET

## Kabul kapıları

| kapı | sonuç |
|---|---|
| no_schema_leakage | GEÇTİ |
| unseen_entity_composition_works | GEÇTİ |
| unseen_relation_induction_works | KALDI |
| unseen_wording_property_and_time | GEÇTİ |
| unseen_both_composition_works | GEÇTİ |
| hard_subset_included | GEÇTİ |

## Bulgular

- Eğitim korpusundan keşfedilen 12 varlık, 2 ilişki: hiçbiri elle verilmedi.
- Eksen bazında C_G v2: seen=1.0000, unseen_entity=1.0000, unseen_relation=0.5000, unseen_both=1.0000, unseen_wording=1.0000.
- Genel C_G v2 = 0.8571 (kompozisyon × keşif).
- Sözlük-dışı ZOR alt kümede (4 cümle) kompozisyon 0.5000, tip doğruluğu 0.5000. Kural tabanlı keşfin sınırı buradadır ve gizlenmemiştir.
- Kompozisyonu kurulamayan cümleler: ['Ali kitabı inceledi.', 'Ayşe arabayı tamir etti.'].

## Sınırlar

- Çıkarım hattı kural tabanlıdır; 'keşif' istatistiksel öğrenme değil, morfolojik analiz + sözlük araması sonucudur.
- Sözlükte olmayan kök UNKNOWN tiple keşfedilir; type_accuracy bunu ayrı ölçer ve kompozisyon doğruluğuyla karıştırılmamalıdır.
- Test seti küçüktür (elle küratörlü); güven aralıkları geniştir.
- 'Görülmemiş ilişki' hattın fiil sözlüğünde OLABİLİR; görülmemişlik EĞİTİM KORPUSUNA göredir, hattın kapsamına göre değil. Bu ayrım schema_leakage bölümünde açıkça raporlanır.
