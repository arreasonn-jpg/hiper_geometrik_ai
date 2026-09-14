# İnsan Değerlendirme Protokolü (P1)

- Protokol: `human_evaluation_protocol_v1` v1
- Kör açma anahtarı özeti: `d29f5aeae0e20346`

> **Bu rapor insan değerlendirme SONUCU içermez.** Protokol ve araç üretir. Gerçek puan toplanana kadar karnede bu bölüm `n/a` kalır.

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

**Yok.** Gerçek insan puanı toplanmadı.

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
| human_ratings_collected | KALDI |
| reliability_meets_threshold | KALDI |

## Bulgular

- Protokol 50 Türkçe prompt × 4 kol × 10 değerlendirici = 10000 tekil yargı olarak tanımlandı.
- Körleme ARAÇ SEVİYESİNDE uygulanıyor ve doğrulanıyor: üretilen paketlerde hiçbir kol adı geçmiyor.
- Krippendorff α aracı üç metrikte hazır ve kanonik referans veriye karşı doğrulandı (nominal 0.691 / ordinal 0.807 / interval 0.811).
- Eşikler: α ≥ 0.8 kabul edilebilir, α ≥ 0.667 yalnız geçici sonuç.
- GERÇEK İNSAN PUANI YOK. Bu koşum protokol ve araç üretir, sonuç üretmez. `human_ratings_collected` kapısı KALDI ve karnede bu bölüm `n/a` kalmalıdır — kanıtsız bir bölüme puan vermek, ölçmediğini ölçtüm demektir.

## Sınırlar

- GERÇEK DEĞERLENDİRİCİ YOK. Bu modül protokol ve araçtır; insan değerlendirme sonucu değildir ve öyle sunulamaz.
- Örnek prompt'lar aracı göstermek içindir; temsili bir Türkçe değerlendirme korpusu değildir.
- α kodlayıcılar arası tutarlılığı ölçer, DOĞRULUĞU değil: hepsi aynı şekilde yanılan değerlendiriciler yüksek α verir.
- Dikkat kontrolleri tanımlıdır ama doğru yanıt anahtarı gerçek yanıtlar üretilmeden doldurulamaz.
- Değerlendirici havuzunun demografisi, Türkçe yeterliği ve eğitim süreci bu modülün kapsamı dışındadır.
