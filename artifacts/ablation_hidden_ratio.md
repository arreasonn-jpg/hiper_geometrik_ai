# Ablasyon: Hidden Feature Ratio'nun Hibrit Kazancına Etkisi

## Sonuçlar (3 seed ortalaması)

| Hidden Ratio | Symbolic | Coverage | Neural | Hybrid | Gain |
|---|---|---|---|---|---|
| 0.00 | 1.0000 | 1.000 | 0.7425 | 1.0000 | **+0.2575** |
| 0.15 | 0.7433 | 0.743 | 0.7850 | 0.9433 | **+0.1583** |
| 0.30 | 0.4908 | 0.491 | 0.7617 | 0.8850 | **+0.1233** |
| 0.50 | 0.2600 | 0.260 | 0.7883 | 0.8492 | **+0.0609** |
| 0.70 | 0.0917 | 0.092 | 0.7575 | 0.7733 | **+0.0158** |

## Kritik Bulgular

1. **Hibrit kazanç monoton azalır** (0.2575 → 0.0158) — symbolic coverage azaldıkça
2. **Hibrit asla neural'ın altına düşmez** — veto mekanizması kayıpsız
3. **Matematiksel ispat:** `Hybrid = cov_sym × 1.0 + (1-cov_sym) × neural_acc`
   - hidden=0.30: 0.491 + 0.509 × 0.7617 = 0.879 ≈ 0.885 ✓

## Mekanizma

- **Symbolic güvenilir** (yüksek coverage) → hibrit sembolik doğruluğunu miras alır
- **Symbolic çekimser** (düşük coverage) → hibrit neural'a yönelir
- **Kayıp yok:** symbolic'in yanlış yaptığı örnek yok (accuracy_on_answered=1.0)

## Yayın İçin Değeri

Bu ablasyon, hibrit paradigmanın **neden** kazandığını gösteriyor:
- **Basit mekanizma:** Veto + fallback
- **Ölçülebilir fayda:** Hidden ratio ile parametrize edilmiş
- **Sıfır kayıp:** Hybrid ≥ neural her durumda
