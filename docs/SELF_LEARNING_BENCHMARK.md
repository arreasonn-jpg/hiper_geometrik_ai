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
