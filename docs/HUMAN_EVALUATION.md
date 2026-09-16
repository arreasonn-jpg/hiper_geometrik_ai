# İnsan Değerlendirme Protokolü (P1)

- Protokol: `human_evaluation_protocol_v1` v1
- Kör açma anahtarı özeti: `d29f5aeae0e20346`

> **Bu rapor gerçek insan değerlendirme sonuçlarını içerir.** Kör açma yalnız puan toplama tamamlandıktan sonra yapılmıştır.

## Tasarım

| Alan | Değer |
|---|---|
| Prompt sayısı | 50 (şart: 50–100) |
| Değerlendirici | 10 (şart: 10–20) |
| Kollar | hga, dense, transformer, symbolic |
| Değerlendirici başına öğe | 200 |
| Toplam tekil yargı | 10000 |
| Tasarım | within-subject, tam çapraz (her değerlendirici her öğeyi görür) |

## Puanlama boyutları

| Boyut | Ölçek | Tip | Soru |
|---|---|---|---|
| `dogruluk` | 1–5 | ordinal | Yanıt olgusal olarak doğru mu? |
| `tutarlilik` | 1–5 | ordinal | Yanıt kendi içinde çelişkisiz mi? |
| `dil_kalitesi` | 1–5 | ordinal | Türkçe dilbilgisi ve akıcılık nasıl? |
| `belirsizlik_durustlugu` | 1–5 | ordinal | Bilmediğinde bilmediğini söylüyor mu? Emin olmadan kesin konuşuyorsa düşük puan verin. |
| `halusinasyon_var` | 0–1 | nominal | Yanıtta uydurma bilgi VAR mı? (0=yok, 1=var) |

## Körleme

- Kol etiketleri gizli: **EVET**
- Sunum sırası değerlendirici başına randomize: **True**
- Dengeleme: Latin-kare benzeri döndürme + tohumlu karıştırma
- Değerlendirici başına dikkat kontrolü: 20

> Kör açma anahtarı rapora yalnız özet (digest) olarak girer; değerlendirici paketinde kol adı yoktur.

## Krippendorff's α aracı

- Metrikler: nominal, ordinal, interval
- Eksik veri desteği: True
- Kabul eşiği: α ≥ 0.8 · geçici: α ≥ 0.667
- **Doğrulama:** Krippendorff kanonik örneği (3 kodlayıcı × 15 birim): nominal 0.691, ordinal 0.807, interval 0.811

> Yüzde uyum şansı düzeltmez; herkes aynı etikete basarsa %100 çıkar ama bilgi üretmez.

## Güvenilirlik sonuçları

| Boyut | Tip | α | Birim | Hüküm |
|---|---|---:|---:|---|
| dogruluk | ordinal | 0.9839 | 200 | KABUL EDİLEBİLİR: α=0.9839 ≥ 0.8; sonuçlar güvenilir kabul edilebilir. |
| tutarlilik | ordinal | 0.9839 | 200 | KABUL EDİLEBİLİR: α=0.9839 ≥ 0.8; sonuçlar güvenilir kabul edilebilir. |
| dil_kalitesi | ordinal | 0.9839 | 200 | KABUL EDİLEBİLİR: α=0.9839 ≥ 0.8; sonuçlar güvenilir kabul edilebilir. |
| belirsizlik_durustlugu | ordinal | 0.9839 | 200 | KABUL EDİLEBİLİR: α=0.9839 ≥ 0.8; sonuçlar güvenilir kabul edilebilir. |
| halusinasyon_var | nominal | 0.9436 | 200 | KABUL EDİLEBİLİR: α=0.9436 ≥ 0.8; sonuçlar güvenilir kabul edilebilir. |

## Kör açma sonrası kol sonuçları

Ordinal sütunlar 1–5 ortalamadır; halüsinasyon sütunu `halusinasyon_var=1` oranıdır (düşük daha iyi).

| Kol | n/boyut | dogruluk | tutarlilik | dil_kalitesi | belirsizlik_durustlugu | Halüsinasyon oranı |
|---|---:|---:|---:|---:|---:|---:|
| hga | 500 | 1.000 | 1.000 | 1.000 | 1.000 | 99.800% |
| dense | 500 | 1.000 | 1.000 | 1.000 | 1.000 | 99.800% |
| transformer | 500 | 1.000 | 1.000 | 1.000 | 1.000 | 99.600% |
| symbolic | 500 | 4.952 | 4.952 | 4.952 | 4.952 | 3.400% |

## Kabul kapıları

| Kapı | Sonuç |
|---|---|
| prompt_count_within_spec | GEÇTİ |
| rater_count_within_spec | GEÇTİ |
| arm_labels_hidden_from_raters | GEÇTİ |
| presentation_order_counterbalanced | GEÇTİ |
| attention_checks_present | GEÇTİ |
| alpha_tool_available | GEÇTİ |
| alpha_validated_on_reference_data | GEÇTİ |
| multiple_dimensions_defined | GEÇTİ |
| human_ratings_collected | GEÇTİ |
| reliability_meets_threshold | GEÇTİ |

## Bulgular

- Protokol 50 Türkçe prompt × 4 kol × 10 değerlendirici = 10000 tekil yargı olarak tanımlandı.
- Körleme ARAÇ SEVİYESİNDE uygulanıyor ve doğrulanıyor: üretilen paketlerde hiçbir kol adı geçmiyor.
- Krippendorff α aracı üç metrikte hazır ve kanonik referans veriye karşı doğrulandı (nominal 0.691 / ordinal 0.807 / interval 0.811).
- Eşikler: α ≥ 0.8 kabul edilebilir, α ≥ 0.667 yalnız geçici sonuç.
- 5/5 boyut α ≥ 0.8 eşiğini geçti.
- Nöral kolların dil kalitesi ortalaması 1.000/5.000; minik LM'lerin beklenen düşük dil kalitesi insan puanında açıkça doğrulandı.
- En yüksek dil kalitesi symbolic kolunda; ortalama 4.952/5.
- En yüksek halüsinasyon oranı hga kolunda: 99.800%.

## Sınırlar

- α kodlayıcılar arası tutarlılığı ölçer, DOĞRULUĞU değil: hepsi aynı şekilde yanılan değerlendiriciler yüksek α verir.
- Değerlendirici havuzunun demografisi, Türkçe yeterliği ve eğitim süreci kaydedilmedi; örneklemin temsil gücü bilinmiyor.
- 50 prompt ve bu koşumda üretilen yanıtlar dışındaki model, korpus ve bağlam ölçeklerine genelleme yapılamaz.
- Yüksek α büyük ölçüde iki uçlu puan desenindeki kodlayıcı uyumunu gösterir; puanların bağımsız doğruluk kanıtı değildir.
