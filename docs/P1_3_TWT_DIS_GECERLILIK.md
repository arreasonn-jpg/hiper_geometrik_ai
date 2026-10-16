# P1-3: Semantik Çıkarım Dış Geçerlilik Testi

**Tarih:** 2026-10-16
**Script:** `artifacts/p1_3_twt_eval.py`
**Sonuç:** `artifacts/w7_runs/p1_3_twt_eval.json`
**Protokol:** `p1_3_twt_eval_v1`

## Amaç

`docs/SEMANTIC_EXTRACTION.md`'de raporlanan F1=1.0000 sonucunun
**genel Türkçe** performansı olup olmadığını test etmek. Audit raporu
bu sonucu "küçük, küratörlü fixture" olarak işaretlemişti (P1-3).

## Yöntem

- **Kaynak:** TWT v1 (Türkçe Web Treebank, 4.851 cümle, LREC 2020)
- **Altın:** UD anotasyonundan silver ground truth
  - Entity: `NOUN` / `PROPN` lemma'ları
  - Relation: `(nsubj, verb, obj)` üçlüsü
- **Tahmin:** `hga.experience.semantik_ayiklayici.cumle_coz`
- **Örneklem:** 2000 cümle (web + wiki karışık, deterministik)

## Sonuçlar

| Metrik | Fixture (P0-5) | TWT (P1-3) | Δ |
|---|---:|---:|---:|
| Entity Precision | 1.0000 | 0.1692 | −0.83 |
| Entity Recall | 1.0000 | 0.2833 | −0.72 |
| **Entity F1** | **1.0000** | **0.2119** | **−0.79** |
| Relation Precision | 1.0000 | 0.0000 | −1.00 |
| Relation Recall | 1.0000 | 0.0000 | −1.00 |
| **Relation F1** | **1.0000** | **0.0000** | **−1.00** |
| Boş çıkarım oranı | 0.0000 | 0.0010 | +0.001 |

## Neden bu kadar düşük?

**Kök neden 1 — Kapalı sözlük.**
`semantik_ayiklayici.py` varlık ve fiil tanımasını iki küçük
sözlüğe dayandırıyor:

- `VARSAYILAN_TIPLER`: ~30 isim (Ali, Ayşe, Ankara, araba, kitap...)
- `VARSAYILAN_FIILLER`: 7 fiil (almak, binmek, gelmek, gitmek,
  görmek, okumak, sevmek)

TWT'de **184 benzersiz fiil** var; motor yalnız 6'sını (normalize
sonrası) yakalıyor → **fiil recall = 0.033**.

**Kök neden 2 — Kök çıkarımı bozuk.**
`kok_ve_durum()` bazı ekleri yanlış soyuyor:

| Yüzey | Yanlış kök | Olması gereken |
|---|---|---|
| açıklanmayan | `aciklmek` | `açıkla` |
| adınızdır | `adinizdmek` | `ad` |
| bulunmaktadır | `bulunmaktadmak` | `bulun` |
| evrenselliktedir | `evrensellikted` | `evrensellik` |

Bu kökler hiçbir sözlükte bulunamaz → hem entity hem relation
kaybı.

**Kök neden 3 — Sentaktik yapı eksik.**
Motor özne/nesne rollerini **yüzey durum ekleriyle** tahmin ediyor
(`-i` → nesne, `-de` → konum). TWT'de tam UD bağımlılık ağacı var
ama motor bunu kullanmıyor. Bu yüzden ilişkilerde 7144 yanlış
pozitif üretip 0 doğru buluyor.

## Yorum

Fixture F1=1.0000 sonucu **dış geçerlilik taşımıyor**. Bu sonuç
audit raporunun (2026-10-13) şu tespitini sayısal olarak doğruluyor:

> "SEMANTIC_EXTRACTION.md F1=1.0 raporluyor ama aynı belge korpusun
> küratörlü/şablon tabanlı olduğunu ve genel Türkçe performansı
> göstermediğini söylüyor."

Yani mevcut F1=1.0 ne yanlış ne yanıltıcı — sadece **kendi
kapsamı içinde** doğru. Bu rapor kapsamı belgeliyor.

## Sonraki adımlar (kapsam dışı, ayrı PR)

Bu rapor **mevcut durumu** belgeliyor. Motoru iyileştirme işi
ayrı bir görevdir:

1. **Açık sözlük genişletme:**
   - `VARSAYILAN_TIPLER` yerine Türkçe WordNet / Wikidata
   - `VARSAYILAN_FIILLER` yerine morfolojik fiil çatısı
2. **Kök çıkarımı düzeltmesi:**
   - `kok_ve_durum()` için Zemberek veya `turkishnlp` kullan
   - Mevcut basit ek soyma yeterli değil
3. **UD ağacı kullanımı:**
   - TWT'de hazır UD var → supervised neural baseline
   - `cumle_coz` yerine, UD bağımlılık ağacı üzerinden
     relation extraction (klasik supervised)
4. **Bağımsız insan anotasyonu:**
   - 2 anotatör + adjudication
   - 500 cümle gold test seti
   - Krippendorff α raporu

## Dürüst değerlendirme

**Şu anki durum:**
- Fixture F1=1.0: doğru, ama **dar kapsamda** (7 fiil, 30 isim)
- TWT F1=0.21/0.00: motor **gerçek metinde çalışmıyor**
- "Türkçe semantik çıkarım" iddiası **desteklenmiyor**

**Bu, başarısızlık değil — ölçümdür.** Audit raporunun P1-3'te
istediği tam olarak buydu: fixture sonuçlarının genel Türkçe
performansı olmadığını **göstermek**. Rapor bunu yapıyor.

## Referans

- Fixture benchmark: `docs/SEMANTIC_EXTRACTION.md`
- TWT veri seti: `hga/evaluation/datasets/twt_v1/DATASET_CARD.md`
- Motor: `hga/experience/semantik_ayiklayici.py`
- P1-3 (birleşik audit raporu)
- SECURITY.md: SEC-2026-10-13-02
