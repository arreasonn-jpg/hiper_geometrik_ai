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

## Yorumlama sınırları

- İki sentetik öğretmen gerçek dil modelleme başarısı değildir.
- Rank-1 baseline, kısıtsız `n⁴` dense katmanın yerine geçmez; yalnız eşit bütçe
  karşılaştırmasıdır.
- Hız cihaz, BLAS/CUDA ve PyTorch sürümüyle birlikte okunmalıdır.
- Seed ortalaması ve standard sapması raporlanmadan tek koşu iddia sayılmaz.
- Bir görevdeki üstünlük diğer fonksiyon ailelerine genellenemez.
