# Epistemik Benchmark — Bilmediğini Biliyor mu? (P0-007)

`epistemic-known-unknown-uncertain-conflict-false-v1`

## Kapsam

Bir sistemin "doğru cevap verme" oranı tek başına yanıltıcıdır. Bir model
her soruya kendinden emin cevap verip yüksek doğruluk alabilir; asıl soru
şudur:

> Kanıt yokken susmayı biliyor mu?

Bu protokol, `ExperienceEvaluator` karar ağacını beş epistemik sınıfa karşı
ölçer ve iki metriği **birlikte** raporlar: yanlış güven (↓) ve bilinmeyen
doğruluğu (↑). Tek metrik kandırılabilir olduğu için ikisi ayrılmaz.

Bu bir neural üretim değerlendirmesi değildir; ölçülen şey karar ağacının
epistemik davranışıdır. Yeni model, yeni tokenizer veya yeni attention
varyantı eklemez.

## Beş epistemik sınıf

| Sınıf | Anlamı | Beklenen durum |
|---|---|---|
| `KNOWN` | Kısıtlar sağlanır, kabul edilmeli | `VALID` |
| `FALSE` | Kural/tip ihlali kayıtlı kanıtla bilinir | `INVALID` |
| `UNKNOWN` | Varlık/ilişki bilgi tabanında hiç yok | `UNCERTAIN` |
| `UNCERTAIN` | Varlık var ama gerekli özellik hiç yazılmamış | `UNCERTAIN` |
| `CONFLICT` | Yapısal kural ile kayıtlı kanıt zıt yönde | `CONFLICT` |

`UNKNOWN` ile `UNCERTAIN` ayrımı kasıtlıdır: birincisi *bilmiyorum*,
ikincisi *emin değilim*. İkisi de "kesin karar verme" demektir ama
epistemik kaynakları farklıdır.

## Veri kümesi

`hga/evaluation/datasets/epistemic_tr_v1.json` — 28 vaka, 16 varlık,
2 ilişki. Elle yazılmış, sürümlü ve SHA-256 ile hash'lenmiş depo fixture'ıdır
(`curation.kind = manually_authored_repository_fixture`). Çalışma anında
üretilmez; beklenen durumlar elle sabitlenmiştir.

Dağılım: KNOWN 8, FALSE 8, UNKNOWN 5, UNCERTAIN 4, CONFLICT 3.

### Fixture bütünlük kuralları

Yükleme sırasında iki değişmez zorunlu tutulur:

1. **Vaka üçlüleri benzersizdir.** Aynı `(özne, ilişki, nesne)` iki vakada
   geçemez.
2. **Çelişki kanıtı yalnız `CONFLICT` vakalarına yazılır.**

İkincisi teorik bir kural değil, ölçülmüş bir hatadan doğdu: ilk sürümde
`KNOWN-008` ile `CONFLICT-003` aynı üçlüydü, dolayısıyla çelişki kanıtı
KNOWN vakasını kirletip sessizce `CONFLICT`'e düşürüyordu. Kabul kapısı
bunu yakaladı. Artık `EpistemicDataset` yüklemede her iki kuralı da
doğrular ve ihlalde `ValueError` yükseltir.

## Metrikler

- **`false_confidence_rate` (↓)** — Çekimser kalınması gereken
  (`UNKNOWN` + `UNCERTAIN`) örneklerde kesin karar (`VALID`/`INVALID`)
  verme oranı. Halüsinasyonun doğrudan ölçüsüdür.
- **`unknown_accuracy` (↑)** — Aynı örneklerde doğru şekilde çekimser
  kalma oranı.
- **`silent_failure_rate` (↓)** — `FALSE` örneğine `VALID` deme oranı.
  En tehlikeli hata sınıfı olduğu için ayrı raporlanır.
- Sınıf bazında isabet, karışıklık matrisi ve vaka bazında gerekçe.

## Negatif kontrol

Benchmark ilk koşuda %100 aldı. Tam puan, ölçümün mü yoksa fixture'ın
kolaylığının mı sonucu olduğunu ayırt etmek için üç dejenere politika aynı
veri kümesinde puanlanır:

| Kol | Doğruluk | Bilinen | Bilinmeyen | Yanlış güven |
|---|---:|---:|---:|---:|
| `always_valid` (her şeye "doğru") | 0.286 | 1.000 | 0.000 | 1.000 |
| `always_abstain` (her şeye "bilmiyorum") | 0.321 | 0.000 | 1.000 | 0.000 |
| `always_invalid` (her şeyi reddet) | 0.286 | 0.000 | 0.000 | 1.000 |
| **`ExperienceEvaluator`** | **1.000** | **1.000** | **1.000** | **0.000** |

Tablo, tek metriğe bakmanın neden yeterli olmadığını gösterir:
`always_abstain` yalnız `unknown_accuracy`'ye bakılsa **mükemmel** görünür,
oysa hiçbir bilineni kabul edemez. `always_valid` ise yalnız
`known_accuracy`'ye bakılsa mükemmeldir ama her yanlışı sessizce kabul eder.

`beats_degenerate_baselines` kapısı, gerçek değerlendiricinin her kolu
**dört eksende birden** domine etmesini zorunlu kılar.

## Kabul kapıları

- `known_cases_accepted` — bilinen doğru kabul edilir;
- `no_silent_acceptance_of_false` — bilinen yanlış asla kabul edilmez;
- `no_false_confidence_on_unknown` — kanıt yokluğunda kesin karar verilmez;
- `conflicts_flagged` — çelişki çelişki olarak işaretlenir;
- `all_epistemic_classes_present` — beş sınıf da temsil edilir;
- `beats_degenerate_baselines` — dejenere politikalar geçemez.

## Kapatılan mimari sınır: UNKNOWN ↔ UNCERTAIN

`ExperienceEvaluator` hem "kayıt hiç yok" hem "özellik yazılmamış" durumunu
tek bir `DeneyimDurumu.UNCERTAIN`'e indiriyordu. Bu iki şey epistemik olarak
farklıdır: birincisi *bilmiyorum*, ikincisi *emin değilim*. İlk ölçümde bu
sınır gizlenmedi, `distinguishable = false` olarak raporlandı ve bir testle
sabitlendi.

**Sınır artık kapatıldı.** Çözüm, `DeneyimDurumu`'na yeni bir üye eklemek
*değil* — bu, durum makinesi geçiş tablosunu ve 30'dan fazla `UNCERTAIN`
karşılaştırma noktasını kırardı. Bunun yerine `ExperienceCandidate`'e
makine-okunur bir sebep alanı eklendi:

```python
class BelirsizlikSebebi(str, Enum):
    YOK = "YOK"                  # durum UNCERTAIN değil
    KAYIT_YOK = "KAYIT_YOK"      # bilmiyorum: varlık/ilişki kaydı hiç yok
    OZELLIK_YOK = "OZELLIK_YOK"  # emin değilim: gerekli özellik yazılmamış
    DOGRULAYICI_KARARSIZ = "DOGRULAYICI_KARARSIZ"  # doğrulayıcı karar veremedi
```

Üçüncü üye ayrı bir kaynaktır: `DogrulamaHatti` bağımsız doğrulayıcıdan
`None` aldığında olgu ne doğrulanır ne çürütülür. Bu belirsizlik bilgi
tabanının eksikliğinden değil, doğrulayıcının kararsızlığından gelir ve
karıştırılmamalıdır.

Rapor iki ayrı çözünürlük seviyesi bildirir:

| Alan | Değer | Anlamı |
|---|---|---|
| `distinguishable_by_state` | `false` | Durum kodu hâlâ tek: ikisi de `UNCERTAIN` |
| `distinguishable_by_reason` | `true` | Sebep alanı kaynağı ayırıyor |
| `unknown_reasons` | `["KAYIT_YOK"]` | UNKNOWN sınıfı tek ve doğru sebep üretiyor |
| `uncertain_reasons` | `["OZELLIK_YOK"]` | UNCERTAIN sınıfı tek ve doğru sebep üretiyor |
| `reasons_consistent` | `true` | Sınıflar sebep karıştırmıyor |

İki yeni kabul kapısı bunu zorunlu kılar:
`unknown_uncertain_distinguishable` ve `uncertainty_reasons_consistent`.

Sebep alanı her değerlendirmenin başında sıfırlanır; aynı aday yeniden
değerlendirilirse eski sebep sızmaz ve bu ayrı bir testle sınanır.

Eski davranışı sabitleyen `test_unknown_uncertain_ayrimi_durustce_raporlanir`
testi, tasarlandığı gibi kırıldı ve
`test_unknown_uncertain_ayrimi_artik_yapilabiliyor` olarak güncellendi.

## Çalıştırma

```bash
python -m hga epistemik --seeds 1,2,3 \
  --out raporlar/epistemik_benchmark.json \
  --markdown raporlar/epistemik_benchmark.md
```

Protokol deterministiktir: tohum sonucu değiştirmez ve bu bir testle
(`test_determinizm`) sabitlenmiştir. Tohum yalnız manifest sözleşmesi
içindir. Çok tohumlu manifest/istatistik gerekirse
`run_epistemic_seed_sweep` kullanılır; sabit metrikler bootstrap yerine
`degenerate-constant` yöntemiyle işaretlenir.

## Bilimsel sınırlar

- Kontrollü ontoloji üzerinde epistemik karar testidir; gerçek dil anlama
  iddiası değildir.
- `UNKNOWN` ve `UNCERTAIN` tek duruma eşlendiği için ayrım ölçülür,
  varsayılmaz.
- Sonuç `ExperienceEvaluator` karar ağacının davranışıdır; neural üretim
  kalitesi değildir.
- `false_confidence_rate` yalnız bu fixture için geçerlidir; genel
  halüsinasyon oranı değildir.
- 28 vaka küçük bir kümedir; oranlar dar güven aralığı taşımaz.
