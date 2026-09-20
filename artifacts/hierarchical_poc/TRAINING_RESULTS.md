# Hierarchical Kronecker — Eğitim Sonuçları

## n_base=4 (final_size=32)

| Model | Params | Train | Test | Test/Param |
|---|---|---|---|---|
| Dense MLP | 1,075,208 | 1.000 | 0.828 | 0.00000077 |
| Flat Kron | 232 | 0.741 | 0.710 | 0.00306 |
| Hierarchical | 9,544 | 1.000 | **0.904** | 0.0000947 |

## n_base=8 (final_size=64)

| Model | Params | Train | Test | Test/Param |
|---|---|---|---|---|
| Dense MLP | 17,080,328 | 1.000 | 0.710 | 0.000000042 |
| Flat Kron | 904 | 0.710 | 0.434 | 0.00048 |
| Hierarchical | 38,152 | 1.000 | **0.768** | 0.0000201 |

## Bulgular

1. Hierarchical her iki ölçekte de kazanan
2. n=4'te 113× parametre verimliliği (dense vs hier)
3. Dense overfit ediyor (train=1.0, test=0.71-0.83)
4. Flat Kronecker ölçekle ÇÖKÜYOR (%39 düşüş)
5. Hierarchical ölçekle dayanıklı (%15 düşüş)

## Tek Cümlede

"Hiyerarşik Kronecker, parametre sayısının 113× azıyla, dense MLP'den
daha iyi genelleme yapıyor; flat Kronecker'in ölçekle çöküşünü engelliyor."
