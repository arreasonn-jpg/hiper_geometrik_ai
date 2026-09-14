# Deneyim Verimi Ayrıştırması — EY'nin Ötesinde (P1-005)

`experience-yield-decomposition-v1`

## Neden tek EY yetmiyor

Depoda uzun süre tek bir verim ölçüsü vardı:

```
EY = doğrulanmış / üretilen
```

Bu ölçü üç farklı durumu **birbirinden ayıramaz** ve üçünde de yüksek çıkar:

1. Sistem zaten bildiği bir şeyi tekrar üretip tekrar doğrular — yeni bilgi
   yoktur.
2. Sistem bir olguyu doğrular ama olgu bellek çakışmasında kaybolur —
   bilgi kullanılamaz.
3. Sistem olguları ezberler ama hiçbir görülmemiş duruma genelleyemez.

Bu protokol EY'yi dört ayrı eksene ayırır. Eksenlerin anlamlı olabilmesi
için EY'den **ayrışmaları** gerekir; hepsi aynı sayıyı veriyorsa yeni bir
şey ölçmüyor, sadece yeniden adlandırıyorlardır.

## Dört eksen

| Metrik | Tanım | Neyi yakalar |
|---|---|---|
| **NY** — Novelty Yield | ayrık **yeni** doğrulanmış olgu / üretilen | Tekrar üretim paydayı şişirir, payı şişiremez |
| **UEY** — Useful Experience Yield | doğrulanmış **ve** doğru **ve** bellekten geri çağrılabilir / üretilen | Doğrulanıp kaybolan bilgiyi cezalandırır |
| **GY** — Generalization Yield | holdout kümesinde öğrenme sonrası doğruluk artışı | Ezberleme ile genellemeyi ayıran tek eksen |
| **VID** — Verified Information Density | doğrulanmış bit / üretilen deneyim | Bir olguyu `R` sonuç arasından sabitlemek `log2(R)` bittir |

## Ölçülen sonuçlar

5 tohum, 30 döngü × 32 batch, `operands_max=15`, %95 bootstrap güven
aralıklarıyla (`raporlar/verim_metrikleri.json`):

| Metrik | Ortalama | %95 GA |
|---|---:|---:|
| EY (klasik) | 0.2385 | [0.2327, 0.2444] |
| NY (yenilik) | 0.1904 | [0.1863, 0.1946] |
| UEY (kullanışlı) | 0.1606 | [0.1575, 0.1646] |
| GY (genelleme) | 0.6419 | [0.6116, 0.6721] |
| VID (bit/deneyim) | 0.9434 | [0.9227, 0.9640] |

Ayrışma gerçektir ve tek yönlüdür: **EY > NY > UEY**. Yani klasik EY hem
yeniliği hem kullanışlılığı sistematik olarak abartır. Tohum 1'in ham
sayımları bunu doğrular: 960 üretim, 220 doğrulama, ancak 178 ayrık yeni
olgu (224 tekrar üretim) ve bunların 150'si bellekten geri çağrılabilir
(70 çakışma). Bilgi tabanına giren yanlış olgu sayısı 0'dır.

> Not: Seyrek bellek adresleme düzeltmesi (`docs/SPARSE_ADDRESSING_FIX.md`)
> öncesinde bu sayılar 148 geri çağrılabilir / 72 çakışma ve UEY = 0.1565
> [0.1538, 0.1600] idi. Düzeltme çakışma kaybını azalttı; EY ve NY bellekten
> bağımsız olduğu için değişmedi. Ayrışma yönü (EY > NY > UEY) korunur.

## GY iki kez "ölü metrik" olarak yakalandı

Bir metriğin daima 0.0 dönmesi, ölçüm yapmadığı anlamına gelir. GY bu
tuzağa iki kez düştü ve ikisi de düzeltildi:

**1. Yanlış karar mekanizması.** Holdout kararı önce `ExperienceEvaluator`
ile veriliyordu. Fakat `R_EQUALS` ilişkisinin hiçbir tip/özellik kısıtı
yoktur, dolayısıyla evaluator holdout'un **tamamına** — öğrenme öncesi ve
sonrası aynı şekilde — `VALID` diyordu. GY yapısal olarak daima 0.0 çıkıyordu.
Bu, sistemin genelleyemediğinin değil, ölçüm aracının kör olduğunun
kanıtıydı.

Yerine **fonksiyonel teklik** çıkarımı kondu: `eşittir` ilişkisinde bir
ifadenin tek bir doğru sonucu vardır, o halde `(ifade, =, X)` doğrulanmışsa
`(ifade, =, Y≠X)` yanlıştır. Böylece öğrenilen pozitiften **görülmemiş
negatif** elenebilir.

**2. Yanlış eşik.** Çıkarım kuralı `score >= 1.0` arıyordu. Oysa doğrulama
hattı olguyu `1.0` ile değil, doğrulayıcı güvenine ölçeklenmiş bir skorla
(ölçülen: `0.6375`) yazar. Eşik, öğrenilen **her** olguyu sessizce eliyordu.
Artık skora değil kaynağa bakılır: `VERIFIED_RULE`, `EXTERNAL_VERIFIED`,
`HUMAN_CONFIRMED`, `REAL_DATA`. `MODEL_GENERATED` tek başına çıkarım
dayanağı sayılmaz.

Düzeltmeden sonra GY canlı bir metriktir ve öğrenmeyle monoton artar:

| Döngü | Holdout karar | Doğru | GY |
|---:|---:|---:|---:|
| 10 | 14/86 | 14 | 0.1628 |
| 20 | 31/86 | 31 | 0.3605 |
| 30 | 52/86 | 52 | 0.6047 |
| 40 | 56/86 | 56 | 0.6512 |
| 60 | 56/86 | 56 | 0.6512 |

Karar verilen her örnekte isabet **1.0000**'dır; sistem emin olmadığında
karar vermez (çekimser kalır). 40. döngüden sonra aday havuzu tükendiği
için GY doyuma ulaşır. Her iki regresyon da testlerle sabitlendi
(`test_genelleme_verimi_olculebilir`,
`test_cikarim_kurali_dogrulanmis_kaynak_arar`).

## Ayrışma raporu

Her koşu bir `divergence` bölümü üretir: `ey_vs_ny`, `ey_vs_uey`,
`ny_vs_uey` farkları ve `ey_overstates_novelty` /
`ey_overstates_usefulness` bayrakları. Farklar sıfırsa yeni metrikler
EY'nin takma adıdır; sıfırdan büyükse EY abartıyor demektir. Bu, iddianın
kendi kendini denetlemesidir ve testle sabitlenmiştir.

## Çalıştırma

```bash
python -m hga verim --cycles 30 --batch 32 --initial-facts 40 \
  --operands-max 15 --negatives-per-fact 3 --seeds 1,2,3,4,5 \
  --out raporlar/verim_metrikleri.json \
  --markdown raporlar/verim_metrikleri.md
```

Güven aralıkları P3 istatistik modülünden (`hga/evaluation/statistics.py`)
gelir. Çok tohumlu manifest için `run_yield_seed_sweep` kullanılır.

## Bilimsel sınırlar

- Sentetik aritmetik domain; sonuçlar genel dil görevlerine taşınmaz.
- GY tek bir çıkarım kuralıyla (fonksiyonel teklik) ölçülür: öğrenilen
  pozitiften görülmemiş negatifi elemek. Bu **gerçek ama dar** bir
  genellemedir; yeni bir toplama bağıntısı keşfetmek değildir.
- GY'nin paydası holdout kümesidir; kapsam düşükken GY de düşük görünür —
  bu öğrenme eksikliğidir, metrik kusuru değil.
- UEY bellek geri çağrımına bağlıdır; farklı slot sayısı farklı UEY verir.
- VID, her olgunun sonuç uzayında tekdüze dağıldığını varsayar.
- Tek koşu tek tohumdur; güven aralığı için çok tohumlu koşu gerekir.
- **5 tohumla `p<0.05` matematiksel olarak imkânsızdır** (en küçük iki
  yönlü sign-flip p değeri `2/2^n = 0.0625`). Yukarıdaki güven aralıkları
  betimleyicidir; anlamlılık iddiası için ≥6 tohum gerekir.
