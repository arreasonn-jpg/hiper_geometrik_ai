# Faz 3/6 — Ölçeklendirilmiş Golden Benchmark (100 / 1.000 / 10.000)

Üretim komutu:

```bash
python -m hga olcekli-golden --sizes 100,1000,10000 --seeds 1,2,3,4,5 \
    --out raporlar/scaled_golden.json
python -m hga olcekli-golden --sizes 100,1000,10000 --seeds 1,2,3,4,5 --hard
```

Ham çıktı: `raporlar/scaled_golden.json` · Kod: `hga/evaluation/scaled_golden.py`
· Testler: `tests/test_scaled_golden.py` (10 test)

## Neden bu deney?

`golden_dataset/` elle sabitlenmiş **8 kayıt** içeriyor; test bölümü 5 örnek.
Bu, regresyon koruması olarak değerli ama istatistiksel bir iddia taşıyamaz:
5 örnekte `FAR = 0` görmek gürültüden ayırt edilemez. Kullanıcının "en büyük 5
eksik" listesindeki iki madde (büyük bağımsız golden benchmark, 100+/1000+
ölçekli verification istatistiği) tam olarak buydu.

## Tautoloji riski ve nasıl kaçınıldı

En büyük tehlike, beklenen etiketi evaluator'ın karar ağacını taklit ederek
üretmektir — o zaman test kendini doğrular. Burada **inşa yöntemi ground
truth'tur**: her örnek, önceden seçilmiş bir hedef sınıfın *kanıt durumuna*
göre kurulur, evaluator'ın o durumdan doğru epistemik sınıfı çıkarması ölçülür.

| Hedef | Nasıl inşa edilir |
|---|---|
| `VALID` | Gerekli özellikler bilinir, yüksek güvenli ve doğru |
| `INVALID` | Gerekli bir özellik bilinir, yüksek güvenli ve **yanlış** |
| `UNCERTAIN` | Gerekli özellik **yok** ya da güveni eşiğin altında |
| `CONFLICT` | Yapısal kural olumlu, kayıtlı yüksek güvenli kanıt olumsuz |

Her ölçekte `audit_partitions` ile train/test semantik parmak izi denetimi
yapılır; test üçlüleri bilgi tabanına olgu olarak yazılmaz (`CONFLICT`
vakalarındaki karşı-kanıt hariç, ki o sınıfın tanımıdır). Bu sözleşme ayrı bir
testle (`test_test_ucluleri_bilgi_tabanina_olgu_olarak_yazilmaz`) korunur.

## Zor mod

Kolay modda doğruluk 1.000 çıkıyordu; bu evaluator'ın gücünü değil görevin
kolaylığını ölçer. `--hard` sınır vakaları ekler:

| Sınır vakası | Test ettiği ayrım |
|---|---|
| Eşik sınırı | Güven tam `0.5`, değer tam `0.5` — "yanlış mı, belirsiz mi?" |
| Kısıt önceliği | Bir kısıt bilinmiyor, diğeri kesin ihlal → kesin ihlal baskın gelmeli (`INVALID`, `UNCERTAIN` değil) |
| Kaynak çatışması | Düşük güvenli `MODEL_GENERATED` vs yüksek güvenli `REAL_DATA` → gerçek veri kazanmalı |
| Zayıf kanıt | Olumsuz kanıt var ama güveni çelişki eşiğinin altında → `CONFLICT` **olmamalı** |

## Sonuçlar (5 seed, mean ± std)

### Doğruluk — kolay ve zor mod, her iki modda da aynı

| Ölçek | Accuracy | Precision | Recall | F1 | FAR | FRR |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 | 0.0000 |
| 1.000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 | 0.0000 |
| 10.000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 | 0.0000 |

Sınıf bazında da dört epistemik durumun tamamı her ölçekte 1.0000 ± 0.0000.
Toplam 30 koşu (2 mod × 3 ölçek × 5 seed), sızıntı denetimi hepsinde temiz.

### Maliyet — asıl bulgu burada

| Ölçek | Süre, kolay (sn) | Örnek başına (ms) | Süre, zor (sn) | Örnek başına (ms) |
|---:|---:|---:|---:|---:|
| 100 | 0.02 ± 0.00 | 0.178 | 0.01 ± 0.00 | 0.122 |
| 1.000 | 1.17 ± 0.01 | 1.169 | 0.76 ± 0.01 | 0.759 |
| 10.000 | 177.41 ± 14.51 | **17.741** | 98.10 ± 2.28 | **9.810** |

## Bulgular

**1. Verification doğruluğu 5 örnekten 10.000 örneğe taşındı ve bozulmadı.**
Artık "FAR=0" iddiası 5 değil 50.000 karar üzerinden veriliyor
(2 mod × 3 ölçek × 5 seed, en büyüğü 10.000 örnek). Standart sapma tüm
ölçeklerde 0.0000: sonuç seed'e duyarsız, yani deterministik.

**2. Bu mükemmel skor evaluator'ın zaferi değil, görevin sınırı.**
Sentetik oracle tamdır: kanıt durumu ne ise doğru cevap da odur. Zor mod dört
sınır vakası eklemesine rağmen skor düşmedi — karar ağacı bu ayrımları
gerçekten doğru yapıyor, fakat gerçek dilde bu kadar temiz kanıt durumu
bulunmaz. Beklenen düşüş gerçek Türkçe veride görülecektir (Faz 27/28).

**3. Asıl kırılma noktası doğrulukta değil, maliyette.**
Örnek başına süre ölçek 100× büyürken **99.7× arttı** — yani toplam maliyet
ölçeğin karesiyle büyüyor. Kaynak profillemeyle bulundu:
`Scoring.information_gain` her aday için bilgi tabanındaki **tüm** varlıkları
tarıyordu.

**4. Bir ölçeklenebilirlik düzeltmesi yapıldı (semantik değişmeden).**
Kosinüs benzerliği ortak özelliği olmayan iki vektör için tanım gereği 0'dır;
bu yüzden yalnız en az bir özellik paylaşan varlıkları taramak aynı sonucu
verir. `PropertyIndex.sahip_olanlar_kume()` (sıralamasız ters indeks) eklendi.
Sabit çarpan düştü, mevcut 369 testin tamamı geçmeye devam ediyor — ama
**asimptotik sınıf metriğin tanımı gereği O(N²) kalıyor**: bu veri setinde tüm
nesneler aynı özellik adlarını paylaşır, dolayısıyla ters indeks aday kümesini
daraltamaz.

**5. Pratik sınır.** 10.000 örnek ~3 dakika sürüyor. 100.000 örnek bu eğriyle
~5 saat eder. Bu ölçeğin ötesi için `information_gain`'in örnekleme tabanlı
veya yaklaşık-en-yakın-komşu (ANN) bir yeniden tanımına ihtiyaç var. Bu
**yapılmadı** — feature freeze kapsamında ölçüldü ve raporlandı.
