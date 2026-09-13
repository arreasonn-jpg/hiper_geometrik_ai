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
| 5 adım | 1.0000 | 1.0000 | 1.0000 | 0.5000 |

* çok adımlı çıkarım (hop≥2): **0.9688**
* tek adımlı geri çağırma: **1.0000**
* en derin güvenilir zincir: **4 adım**
* bağlam bozulması: **-0.1000**

**Bulunan gerçek sınır:** 5 adımlık zincir, 256 dolgu olgu altında 0.5000'e
düşüyor — yani pozitif vakaları tamamen kaybediyor. Bu bir hata değil,
ölçülen bir kapasite sınırıdır ve `robust_to_long_context` kapısı bu yüzden
**KALDI** olarak raporlanır. Kapıyı geçirmek için eşiği gevşetmedik.

### Bellek daraltılınca (256 slot) sınır öne çekiliyor

| hop \ dolgu | 0 | 16 | 64 | 256 |
|---|---:|---:|---:|---:|
| 3 adım | 1.0000 | 1.0000 | 1.0000 | 0.5000 |
| 4 adım | 1.0000 | 0.5000 | 0.5000 | 0.5000 |
| 5 adım | 1.0000 | 0.5000 | 0.5000 | 0.5000 |

En derin güvenilir zincir 4 → **2 adıma** düşer. Metrik canlıdır.

## Negatif kontrol

`always_yes` ve `always_no` kolları zinciri hiç takip etmeden sabit cevap
verir. Veri kümesi dengeli olduğu için ikisi de tam **0.5000** alır; motor
0.9750 ile bunları yener (`beats_degenerate` GEÇTİ). Motor bu kollardan
ayrışamazsa çok adımlı çıkarım iddiası düşer.

## Kabul kapıları

| kapı | anlamı | varsayılan sonuç |
|---|---|---|
| `multi_hop_inference_works` | hop≥2 doğruluğu ≥ 0.99 | KALDI (0.9688) |
| `beats_degenerate` | sabit-cevap kollarını yener | GEÇTİ |
| `rejects_broken_chains` | kopuk zincire "evet" demez | GEÇTİ |
| `robust_to_long_context` | en yüksek bağlamda bozulma yok | KALDI |
| `memory_preserved_chain` | bellek tüm kenarları korudu | KALDI |

Beş kapıdan üçü varsayılan ayarda **kalıyor**. Bu kasıtlıdır: kapılar
geçsin diye eşik gevşetilmedi veya parametre seçilmedi.

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
