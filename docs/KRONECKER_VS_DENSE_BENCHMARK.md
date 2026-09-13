# Kronecker vs Eşit Parametreli Baseline

## Önce terminoloji düzeltmesi

Eski deney karşılaştırdığı modeli “standart dense linear” olarak adlandırıyordu.
Bu doğru değildi. `n² → 1 → n²` bias'sız katman, tam dense operatör değil,
**rank-1 factorized bottleneck** modelidir.

| Model | Fiziksel parametre | Efektif operatör rank üst sınırı |
|---|---:|---:|
| Kronecker `A X B` | `2n²` | `rank(A) × rank(B) ≤ n²` |
| Rank-1 bottleneck | `2n²` | `1` |
| Kısıtsız full dense | `n⁴` | `n²` |

`n⁴`, Kronecker modelinin fiziksel parametre sayısı değildir. Flatten edilmiş
`Bᵀ ⊗ A` operatöründeki matris girdisi sayısıdır.

## Tek görev yanlılığını engelleme

Eski benchmark yalnız Kronecker öğretmen kullanıyordu:

```text
Y = A* X B*
```

Bu görev doğrudan Kronecker modelinin varsayımına uyduğu için evrensel üstünlük
kanıtı olamaz. V2 protokolü iki karşıt görev çalıştırır:

1. `kronecker_teacher`: Kronecker inductive bias lehine.
2. `rank1_teacher`: `vec(Y) = u(vᵀvec(X))`; rank-1 baseline lehine.

Amaç bir modelin her zaman üstün olduğunu göstermek değil, aynı fiziksel bütçe
altında her parametrizasyonun hangi yapıyı verimli öğrendiğini ölçmektir.

## Ölçümler

Her görev ve model için:

- fiziksel parametre ve parametre byte sayısı,
- yaklaşık Adam eğitim belleği,
- son train loss,
- held-out MSE ve normalized MSE,
- held-out R²,
- açık toleranslı regresyon accuracy,
- eğitim süresi ve step/s,
- maksimum gradient normu,
- son 10 loss standard sapması,
- non-finite adım sayısı ve optimization stability,
- gözlenen ve teorik efektif operatör rankı.

CPU tensor RSS değeri güvenilir biçimde ayrıştırılamadığı için uydurulmaz.
CUDA kullanıldığında `peak_cuda_bytes` ayrıca kaydedilir.

## Kullanım

```bash
python -m hga kronecker-benchmark \
  --n 16 --steps 100 --batch 16 \
  --seeds 1,2,3,4,5 --device cpu \
  --experiment-root experiments \
  --out raporlar/kronecker_seed_summary.json
```

Her seed ayrı `EXP-NNNN` manifestidir. Aynı task içindeki iki model aynı train
ve test girdilerini, aynı optimizer türünü ve tam aynı `2n²` parametre bütçesini
kullanır.

## Ölçülmüş beş-seed sonucu

CPU üzerinde PyTorch `2.3.1+cu121`, `n=16`, 300 step, batch=64, 128 held-out
örnek ve seed 1–5 ile çalıştırıldı. Her iki model tam 512 fiziksel parametreye
sahiptir.

| Teacher / model | Held-out NMSE mean ± std | R² mean ± std |
|---|---:|---:|
| Kronecker / Kronecker | `1.07112e-11 ± 5.3048e-12` | `0.999999999989 ± 5.305e-12` |
| Kronecker / rank-1 | `0.959713 ± 0.004953` | `0.040239 ± 0.004977` |
| Rank-1 / Kronecker | `0.977951 ± 0.011529` | `0.022042 ± 0.011529` |
| Rank-1 / rank-1 | `3.43394e-6 ± 2.96832e-6` | `0.999996566 ± 2.968e-6` |

Her seed'de yapı-matched model kendi teacher görevini kazandı. Sonuç evrensel
Kronecker üstünlüğü değil, iki parametrizasyonun farklı inductive bias'larını
counterbalanced protokolde doğrular.

Tam sonuçlar, per-seed train/test metrikleri ve temiz çalışma ağacı manifestleri
`raporlar/kronecker_5seed_summary.json` içindedir. Koşular commit `1aa4d15`,
dataset hash
`cd0e3d55d10179f78e14d7b128b1c7a5c7e545c73b3f2a1417df3d93ac32ad1`
ve config hash
`88026c8e9ed2c8f74b09cf6c433c8d00c5edb86e210c52c1353fe260ef9e523f`
ile üretildi.

## Yorumlama sınırları

- İki sentetik öğretmen gerçek dil modelleme başarısı değildir.
- Rank-1 baseline, kısıtsız `n⁴` dense katmanın yerine geçmez; yalnız eşit bütçe
  karşılaştırmasıdır.
- Hız cihaz, BLAS/CUDA ve PyTorch sürümüyle birlikte okunmalıdır.
- Seed ortalaması ve standard sapması raporlanmadan tek koşu iddia sayılmaz.
- Bir görevdeki üstünlük diğer fonksiyon ailelerine genellenemez.
