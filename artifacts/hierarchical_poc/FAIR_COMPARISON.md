# Adil Parametre Bütçesi Karşılaştırması

## Sonuçlar

| Expansion | Dense (19.7K) | Hierarchical | Flat (232) |
|---|---|---|---|
| 2 | 0.926 | 0.904 (9.5K) | 0.710 |
| 3 | 0.918 | 0.892 (102K) | 0.694 |
| 4 | 0.920 | 0.870 (559K) | 0.626 |

## Kritik Bulgular

1. **Adil karşılaştırmada dense kazanıyor** (her üç senaryoda)
2. **Hierarchical > flat her yerde** (+19-24 puan) — sağlam
3. **Expansion arttıkça hierarchical KÖTÜLEŞİYOR** (0.904 → 0.870)
4. **Etkin boyut artışı, görev performansına yansımıyor**
5. **Önceki "113× verimlilik" iddiası GEÇERSİZ** — dense haksız büyüktü

## Dürüst Değerlendirme

Sentetik rastgele lineer sınıflandırma görevi, hiyerarşik yapıdan
fayda sağlamıyor çünkü verinin kendisi hiyerarşik değil.

Gerçek potansiyel, kompozisyonel/çok-ölçekli görevlerde test edilmeli.
