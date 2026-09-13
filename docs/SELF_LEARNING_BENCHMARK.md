# Kapalı Self-Learning ve Collapse Benchmarkı

## Amaç

Bu deney iki farklı sistemi aynı deterministik aritmetik ortamda ayırır:

```text
CLOSED_VERIFIED
K₀ → Generate → Evaluate → Independent Verify → Consolidate → Kₙ
```

ve kasıtlı failure injection:

```text
UNVERIFIED_SELF_TRAINING
Generate → Evaluate → VALID çıktıyı doğrulamadan tekrar bilgi/belleğe yaz
```

İlk protokolde yalnız `VERIFIED` olgular kalıcı bilgiye girer. İkinci protokol
bir öneri değildir; verifier'sız öz-beslemenin tekrar, contamination ve memory
interference etkisini yakalamak için sistemi kasıtlı olarak yanlış çalıştırır.

## Experience Yield

```text
Experience Yield = Verified New Knowledge / Generated Experiences
```

Bu oran tek başına kalite değildir. FAR, FRR, incorrect knowledge, diversity ve
memory collision ile birlikte raporlanır.

## Ölçümler

Her cycle için:

- generated ve novel experience,
- VALID-before-verifier,
- VERIFIED / INVALID / UNCERTAIN / CONFLICT,
- FAR ve FRR,
- knowledge size / correct / incorrect knowledge,
- cycle ve cumulative Experience Yield,
- cumulative diversity,
- normalized knowledge entropy,
- mean information gain,
- memory collision ve exact-ID retrieval accuracy.

Collapse probunda ayrıca evaluator novelty düşüşü, tekrar oranı, doğrulanmamış
yanlış model olguları ve memory interference ölçülür.

## Verifier robustness kırma testi

`verifier-epistemic-fault-injection-v1`, FAR ve FRR'nin yalnız sıfır değeri
raporlamakla kalmayıp bilinen hatalara tepki verdiğini sınar. Sekiz doğru ve
sekiz yanlış adayın %25'inde verifier kararı kasıtlı ters çevrilir. İki
kanıtı yetersiz ifade doğal biçimde `UNCERTAIN`, K₀'daki doğrulanmış eşitliğe
karşı iki iddia ise `CONFLICT` olur.

Beklenen confusion matrix: TA=6, TN=6, FA=2, FR=2; precision=recall=F1=0.75,
FAR=FRR=0.25. `UNCERTAIN` ve `CONFLICT` durable knowledge'a yazılmaz. Hatasız
kontrol koşusu FAR=FRR=0 ve precision=recall=F1=1 üretir. Bu kasıtlı hata
probu üretim verifier kalitesi tahmini değildir.

## Test izolasyonu

Aritmetik adaylar deterministik shuffle sonrasında generation pool ve %10 test
holdout olarak ayrılır. Holdout üçlüleri generation veya memory'ye verilmez;
her koşu `generation_test_overlap`, `memory_test_overlap` ve `isolation_clean`
alanlarını raporlar. Bu protokol neural eğitim yapmadığından ayrıca bir train
split'i yoktur; olmayan train aşaması varmış gibi sunulmaz.

## Memory kapasite eğrisi

Her seed için aynı benzersiz context akışı 1 ve 2 tabloluk memory'de sekiz slot
kapasitesinde tekrar çalıştırılır. Collision event rate, exact-ID retrieval,
interference ve %95 recall'a ulaşan minimum slot/yük faktörü rapora gömülür.
İki tablo sonucu mevcut `ALL-read` politikasıyla değerlendirilir; redundancy
kazancı varsayılmaz.

## 100-cycle kontrollü baseline

Aşağıdaki sonuç seed 42, `K₀=100`, 31'e kadar operand, cycle başına 16 aday,
7 negatif/doğru oranı ve 4096 doğrulanmış-memory slotu ile alınmıştır.
Tek seed olduğu için güven aralığı yoktur.

### CLOSED_VERIFIED

| Metrik | Sonuç |
|---|---:|
| Cycle | 100 |
| Generated experiences | 1,600 |
| K₀ | 100 |
| K₁₀₀ | 308 |
| Verified new knowledge | 208 |
| Experience Yield | 0.1300 |
| Invalid generated | 1,392 |
| Correct knowledge | 308 |
| Incorrect knowledge | 0 |
| FAR (Evaluator öncesi kabul) | 1.0 |
| FAR (Verifier sonrası) | 0.0 |
| FRR | 0.0 |
| Memory collisions | 3 |

### UNVERIFIED_SELF_TRAINING failure injection

| Metrik | Sonuç |
|---|---:|
| Generated experiences | 1,600 |
| Unique experiences | 16 |
| Repetition rate | 0.99 |
| Novelty rate | 1.0 → 0.0 |
| Evaluator novelty | 1.0 → 0.01 |
| Acceptance accuracy | 0.125 |
| FAR | 1.0 |
| Incorrect model facts | 14 / 16 |
| Memory collisions | 1,584 |
| Memory retrieval | 1.0 → 0.01 |
| Collapse detected | true |

Bu karşılaştırma bağımsız doğrulamanın kontrollü ortamda bilgi kirlenmesini
engellediğini ve aynı model çıktılarının tekrar beslenmesinin ölçülebilir
collapse sinyalleri ürettiğini gösterir. Genel dil modeli veya gerçek dünya
self-learning başarısı kanıtı değildir.

## Beş-seed sonucu

Aynı protokol seed 1–5 üzerinde çalıştırıldı. Aşağıdaki sapma population
standard deviation'dır; tam makine-okunur sonuç
`raporlar/self_learning_5seed_summary.json` dosyasındadır.

| Metrik | mean ± std |
|---|---:|
| K₁₀₀ | 295.8 ± 5.0359 |
| Verified new knowledge | 195.8 ± 5.0359 |
| Experience Yield | 0.122375 ± 0.003147 |
| Incorrect knowledge | 0.0 ± 0.0 |
| FAR, Evaluator öncesi | 1.0 ± 0.0 |
| FAR, Verifier sonrası | 0.0 ± 0.0 |
| FRR | 0.0 ± 0.0 |
| Closed-loop memory collisions | 5.2 ± 1.7205 |
| Collapse repetition rate | 0.99 ± 0.0 |
| Collapse acceptance accuracy | 0.125 ± 0.03953 |
| Collapse incorrect model facts | 14.0 ± 0.63246 |
| Collapse memory collisions | 1,584 ± 0 |
| Robustness precision / recall / F1 | 0.75 / 0.75 / 0.75 |
| Robustness FAR / FRR | 0.25 / 0.25 |
| Robustness UNCERTAIN / CONFLICT | 2 / 2 |
| Generation/test overlap | 0 ± 0 |
| Memory/test overlap | 0 ± 0 |

1,600 context ve %95 recall hedefli memory sweep'te tek tablo beş seed'in
dördünde 16,384, birinde 32,768 slot gerektirdi. Mevcut `ALL-read` çift tablo
politikası dört seed'de 32,768 slotta hedefi geçti; seed 4, taranan üst sınırda
hedefe ulaşamadı. Dolayısıyla tek bir evrensel kapasite eşiği iddia edilmez.

Koşular commit `63a82d1`, dataset hash
`a2605e3578e28d2100b39eb4702f902183ef0b2559ad69a3feb71213be3128ab` ve
config hash `2276f6cb32aa07a7e422555b65eb8cadc3a226dcba609bdec6ca72c68c06d373`
ile, temiz çalışma ağacında üretildi. Her koşunun tam manifesti, sonucu,
robustness confusion matrix'i, kapasite eğrisi ve 100-cycle trajectory'si özet
JSON'a gömülüdür.

## Kullanım

Hızlı tek-seed koşu:

```bash
python -m hga self-learning-benchmark \
  --cycles 100 --batch 16 --initial-facts 100 \
  --operands-max 31 --negatives-per-fact 7 --seeds 42
```

Beş seed:

```bash
python -m hga self-learning-benchmark \
  --cycles 100 --batch 16 --initial-facts 100 \
  --operands-max 31 --negatives-per-fact 7 \
  --seeds 1,2,3,4,5 \
  --experiment-root experiments \
  --out raporlar/self_learning_seed_summary.json
```

Her seed ayrı `EXP-NNNN` manifesti üretir.

## Bilimsel sınırlar

- Ground truth bağımsız ancak sentetik aritmetik environment'tır.
- Entity uzayı baştan sabittir; büyüyen öğe doğrulanmış relation fact sayısıdır.
- Collapse, sabit aday replay'iyle kasıtlı failure injection'dır.
- Sonuç gerçek corpus, neural generation veya açık dünya doğrulamasına
  genellenemez.
- Beş-seed raporu olmadan baseline yalnız deterministik regresyon bulgusudur.
