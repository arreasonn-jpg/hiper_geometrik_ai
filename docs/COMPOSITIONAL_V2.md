# C_G v2 — Ham Metinden Keşif ve Kompozisyon (P0-6)

Protokol: `compositional_generalization_v2_raw_text` · veri imzası: `cddb95cb8513`

Eğitimde keşfedilen varlıklar: `['ali', 'araba', 'at', 'ayse', 'can', 'ev', 'fatma', 'mehmet', 'okul', 'otobus', 'tren', 'veli']`
Eğitimde keşfedilen ilişkiler: `['binmek', 'gitmek']`

**Sisteme hiçbir entity id, ontoloji kaydı veya relation şeması önceden verilmedi.**

| Eksen | n | Kompozisyon | Entity keşfi | Relation indüksiyonu | Tip | Özellik | Zaman | C_G v2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seen | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| unseen_entity | 4 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 1.0000 | 1.0000 | 1.0000 |
| unseen_relation | 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| unseen_both | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| unseen_wording | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**Genel C_G v2 = 1.0000**

## Şema sızıntısı denetimi

- Sızan varlık: []
- Sızan ilişki: []
- Temiz mi: EVET

## Kabul kapıları

| kapı | sonuç |
|---|---|
| no_schema_leakage | GEÇTİ |
| unseen_entity_composition_works | GEÇTİ |
| unseen_relation_induction_works | GEÇTİ |
| unseen_wording_property_and_time | GEÇTİ |
| unseen_both_composition_works | GEÇTİ |
| hard_subset_included | GEÇTİ |
| induction_abstains_on_ambiguous_voice | GEÇTİ |

## Çekimserlik vakaları (ilişki üretmek YANLIŞ)

| cümle | çekimser mi | neden |
|---|---|---|
| Öğretmen öğrencilere kitabı okuttu. | EVET | cati_eki_uye_yapisi_belirsiz:okut<-oku |
| Ali kitabı sattırdı. | EVET | cati_eki_uye_yapisi_belirsiz:sattir<-sat |

## Bulgular

- Eğitim korpusundan keşfedilen 12 varlık, 2 ilişki: hiçbiri elle verilmedi.
- Eksen bazında C_G v2: seen=1.0000, unseen_entity=1.0000, unseen_relation=1.0000, unseen_both=1.0000, unseen_wording=1.0000.
- Genel C_G v2 = 1.0000 (kompozisyon × keşif).
- Sözlük-dışı ZOR alt kümede (4 cümle) kompozisyon 1.0000, tip doğruluğu 0.5000. Kural tabanlı keşfin sınırı buradadır ve gizlenmemiştir.
- Çekimserlik vakaları (ettirgen çatı vb.): 2/2 doğru çekimser. İlişki üretmek bu cümlelerde YANLIŞ olurdu; indüksiyonun 'her fiile mastar tak' dejenerasyonuna kaymadığının kanıtıdır.

## Sınırlar

- Çıkarım hattı kural tabanlıdır; 'keşif' istatistiksel öğrenme değil, morfolojik analiz + sözlük araması sonucudur.
- Sözlükte olmayan kök UNKNOWN tiple keşfedilir; type_accuracy bunu ayrı ölçer ve kompozisyon doğruluğuyla karıştırılmamalıdır.
- Test seti küçüktür (elle küratörlü); güven aralıkları geniştir.
- 'Görülmemiş ilişki' hattın fiil sözlüğünde OLABİLİR; görülmemişlik EĞİTİM KORPUSUNA göredir, hattın kapsamına göre değil. Bu ayrım schema_leakage bölümünde açıkça raporlanır.
- Mastar indüksiyonu kanıt-tabanlıdır (kök ≥4 harf + yalın özne + durum ekli nesne) ve DÜŞÜK güvenle (0.55) işaretlenir; ünlü uyumu ascii üzerinde yaklaşıktır (ı→i eşlemesi kimi art ünlülü köklerde -mek seçtirir). Çatı ekli fiillerde (ettirgen/edilgen) hat KASITLI çekimserdir: üye yapısı yüzey durumlardan çıkarılamaz.
