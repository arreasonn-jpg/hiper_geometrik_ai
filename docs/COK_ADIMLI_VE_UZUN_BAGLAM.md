# Çok Adımlı Çıkarım ve Uzun Bağlam (P1-004 / P1-006)

`python -m hga cok-adimli`

Bu protokol tek bir soruyu ölçer: **sistem tek tek doğrulanmış olgulardan
zincirleme çıkarım yapabiliyor mu, ve araya alakasız olgular biriktikçe bu
yeteneği koruyor mu?**

## Protokol

`a --öncesi--> b --öncesi--> c` zinciri kurulur, sonra `a --öncesi--> c`
sorulur. Bu üçlü bilgi deposunda **yazılı değildir**; yalnızca geçişlilik
(transitivity) ile çıkarılabilir. İki eksen bağımsız taranır:

| eksen | anlamı |
|---|---|
| `hop` (derinlik) | zincirdeki kenar sayısı. 1 = doğrudan geri çağırma, 2+ = gerçek çıkarım |
| `distractor` (bağlam yükü) | zincir kenarlarının **arasına** serpiştirilen alakasız olgu sayısı |

Dolgu olgular zincirin ortasına serpiştirilir, sonuna eklenmez — böylece
araya giren yazmalar zincir takibinin tam ortasına denk gelir.

## Kritik tasarım kararı: zincir takibi bellekten geçer

İlk taslakta zincir takibi bir Python sözlüğü üzerinden yapılıyordu. Sonuç:
doğruluk **her koşulda 1.0** çıkıyordu; bellek slot sayısı 4096'dan 16'ya
indirilse bile hiçbir şey değişmiyordu. Bu bir **ölü metrikti** — bağlam
ekseni ölçüyormuş gibi görünüyor ama hiçbir şey ölçmüyordu.

Düzeltme: `zincir_takip` artık her kenarı ilerlemeden önce seyrek bellekten
(`DeneyimSlotlari.icerir`) doğrular. Bellek bir kenarı çakışma yüzünden
kaybettiyse zincir orada **kopar**. Böylece bellek baskısı doğruluğa yansır.

Bu davranış `tests/scientific/test_multi_hop.py` içindeki
`test_metrik_olu_degil_bellek_daralinca_dogruluk_dusuyor` ile kilitlenmiştir;
bellek kapısı kaldırılırsa test kırılır.

## Ölçülen sonuç (varsayılan: 4096 slot, tohum 1,2,3)

| hop \ dolgu | 0 | 16 | 64 | 256 |
|---|---:|---:|---:|---:|
| 1 adım (geri çağırma) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 3 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 5 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

* çok adımlı çıkarım (hop≥2): **1.0000**
* tek adımlı geri çağırma: **1.0000**
* en derin güvenilir zincir: **5 adım** *(ızgara tavanı — gerçek sınır çok
  daha derindedir, bkz. `docs/REASONING_DEPTH.md`: C_R = 16384)*
* bağlam bozulması: **0.0000**

> **Güncelleme (adresleme düzeltmesi).** Bu tablonun eski sürümünde 5
> adım/256 dolgu hücresi 0.5000'e düşüyordu ve `robust_to_long_context`
> kapısı KALIYORDU. O sınır gerçek bir kapasite sınırı değil, seyrek bellek
> adresleme kusurunun eseriydi: "Bloom tarzı" iki tablo sıfır bağımsızlık
> sağlıyordu ve `icerir()` AND semantiği kaybı büyütüyordu (ayrıntı:
> `docs/SPARSE_ADDRESSING_FIX.md`). Düzeltme sonrası bu ızgaradaki tüm
> hücreler temizdir; gerçek kırılma yüzlerce hop ötededir ve
> `cikarim-derinligi` protokolünde ölçülür. Eski sayılar tarihsel kayıt
> olarak bu blokta korunur: hop≥2 doğruluğu 0.9688, en derin güvenilir
> zincir 4 adım, 5 adım @256 dolgu = 0.5000.

### Bellek daraltılınca (256 slot) sınır öne çekiliyor

| hop \ dolgu | 0 | 16 | 64 | 256 |
|---|---:|---:|---:|---:|
| 3 adım | 1.0000 | 1.0000 | 1.0000 | 0.5000 |
| 4 adım | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 5 adım | 1.0000 | 1.0000 | 0.5000 | 0.5000 |

En derin güvenilir zincir 5 → **2 adıma** düşer (kesintisiz-derinlik kuralı:
3 adım 256 dolguda düştüğü için üstü sayılmaz). Metrik canlıdır: bellek
daraltılınca sınır hâlâ öne çekilir, adresleme düzeltmesi bunu değiştirmedi.

## Negatif kontrol

`always_yes` ve `always_no` kolları zinciri hiç takip etmeden sabit cevap
verir. Veri kümesi dengeli olduğu için ikisi de tam **0.5000** alır; motor
0.9750 ile bunları yener (`beats_degenerate` GEÇTİ). Motor bu kollardan
ayrışamazsa çok adımlı çıkarım iddiası düşer.

## Kabul kapıları

| kapı | anlamı | varsayılan sonuç |
|---|---|---|
| `multi_hop_inference_works` | hop≥2 doğruluğu ≥ 0.99 | GEÇTİ (1.0000) |
| `beats_degenerate` | sabit-cevap kollarını yener | GEÇTİ |
| `rejects_broken_chains` | kopuk zincire "evet" demez | GEÇTİ |
| `robust_to_long_context` | en yüksek bağlamda bozulma yok | GEÇTİ |
| `memory_preserved_chain` | bellek tüm kenarları korudu | GEÇTİ |

Eskiden beş kapıdan üçü KALIYORDU (0.9688 / bağlam bozulması / kenar kaybı).
Kapılar eşik gevşetilerek DEĞİL, kök nedendeki adresleme kusuru düzeltilerek
geçti; varsayılan ızgara artık sınırın çok altındadır. Sınır kaybolmadı,
binlerce hop ötesine taşındı. Deep profildeki eski C_RD=256 sınırının kök
nedeni de kovalandı: teşhis (`docs/DEPTH_DIAGNOSIS_RAW.md`) bunun çıkarım
değil bellek doygunluğu olduğunu gösterdi; slot bütçesi dolgu yüküne göre
boyutlandırılınca (2^20) deep profil 16384 dolguda **C_RD = 8192** ölçer
(retention 0.5) ve `retains_half_depth_under_max_distractors` kapısı GEÇER.
Kalan yarı kayıp gerçek girişim maliyetidir ve raporlanmaya devam eder.

## Sınırlar (dürüstlük)

* Bu bir dil modeli **"context window" testi değildir**. Token penceresi,
  attention span veya doğal metin anlama ölçülmez. Ölçülen şey sembolik depo
  + seyrek bellek üzerinde zincir takibidir.
* Zincir tek bir geçişli ilişki (`öncesi`) üzerinden kurulur. Gerçek ama
  **dar** bir çıkarımdır; çok ilişkili karma akıl yürütme kapsam dışıdır.
* `hop=1` çıkarım değil geri çağırmadır ve **negatif kontrol** olarak
  okunmalıdır. Çıkarım iddiası yalnız `hop≥2` için geçerlidir.
* Dolgu olgular sentetiktir ve zincirle aynı ilişkiyi kullanır; doğal
  metindeki anlamsal karışıklığı temsil etmez.
* Zincir takibi genişlik-öncelikli aramadır — öğrenilmiş bir yetenek değil,
  deterministik bir çıkarımdır. Buradaki başarı "model akıl yürütmeyi
  öğrendi" demek **değildir**.

## Üretim

```bash
python -m hga cok-adimli --hops 1,2,3,4,5 --distractors 0,16,64,256 \
    --seeds 1,2,3 --out raporlar/cok_adimli_benchmark.json \
    --markdown raporlar/cok_adimli_benchmark.md
```

Bellek baskısı denemek için `--memory-slots 256` ekleyin.
