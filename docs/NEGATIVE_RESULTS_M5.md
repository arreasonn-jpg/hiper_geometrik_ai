# M5 Negatif Sonuç: SparseAdam Bellek Kazancı Sağlamıyor

**Tarih:** 2026-10-18
**Hedef:** M5 (audit) — "Seyrek tablo için SparseAdam veya dinamik KV"
**Durum:** Denendi, **başarısız oldu**. Dense AdamW'ye geri dönüldü.

## Hipotez

`mimari/seyrek_tablo.py` içindeki `nn.Embedding`'i `sparse=True` yapıp,
optimizer'ı `SparseAdam`'a geçirerek bellek kazancı sağlamak.

## Uygulama

1. `nn.Embedding(..., sparse=True)` eklendi.
2. `SparseAdam` (seyrek) + `AdamW` (yogun) birleştiren `BilesikOptimizer` yazıldı.
3. 1M × 32 tablo boyutunda (128 MB ağırlık) 5 adım koşuldu.

## Ölçüm Sonuçları

| Bileşen | Boyut |
|---|---:|
| `weight` | 134.2 MB |
| `grad` (sparse) | ✅ nnz=32 (sadece 32 satır) |
| `exp_avg` (1. moment) | ❌ **DENSE 134.2 MB** |
| `exp_avg_sq` (2. moment) | ❌ **DENSE 134.2 MB** |
| **Toplam bellek** | **~536 MB** |

## Kök Neden

PyTorch 2.14'ün `SparseAdam` implementasyonu **yalnızca gradyanı** sparse tutuyor.
Optimizer state'i (`exp_avg`, `exp_avg_sq`) `zeros_like(p)` ile **tam dense**
olarak tahsis ediliyor. Dolayısıyla:

- Sadece dokunulan satırlar için bellek tasarrufu **YOK**
- Ağırlık (134MB) + Dense state (268MB) + Dense gradyan (opsiyonel) = **~402+ MB**
- SparseAdam ile durum **kötüleşmedi ama iyileşmedi de**

## Neden LM görevinde işe yaramaz?

Hash tabanlı seyrek tabloda her adımda farklı satırlar adreslenir.
4000 adım × ~256 satır = ~1M benzersiz satır → **kısa sürede tam doluluk**.
Sparse bellek avantajı sadece ilk birkaç yüz adımda geçerli, sonra sıfırlanır.

## Sonuç

`SparseAdam` M5 hedefini **karşılamıyor**. Bu bir PyTorch kısıtıdır,
projenin hatası değildir. Gerçek kazanç için 2 alternatif kalır:

1. **Satır-bazlı özel optimizer** (`dict[int, (step, m, v)]`): sadece dokunulan
   satırlar için state tutar. Ancak `Optimizer` arayüzünü yeniden yazmayı,
   `LambdaLR` uyumunu ve checkpoint formatını değiştirmeyi gerektirir.
2. **Dinamik KV deposu** (audit'in diğer önerisi): seyrek tabloyu sabit
   boyutlu `nn.Embedding` yerine, talebe göre büyüyen bir bellek deposuna taşır.

Bu iki iş **Faz 3** kapsamına alınmıştır. Faz 2'de mevcut `AdamW` + dense
tablo ile devam edilecektir (doğruluk aynı, bellek beklentisi şeffaf).

## Referanslar
- PyTorch 2.14 `SparseAdam` kaynak kodu (dense state tahsisi)
- M5 (birleşik audit raporu)
- İlgili Faz 3 hedefleri: `MYPY_RUFF_ROADMAP.md` (gelecek)
