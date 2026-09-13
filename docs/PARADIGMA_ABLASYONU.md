# Faz 21 — Neural-only vs Symbolic-only vs Hybrid

Üretim komutu:

```bash
python -m hga paradigma --seeds 1,2,3,4,5 --out raporlar/paradigma_ablation.json
```

Ham çıktı: `raporlar/paradigma_ablation.json` · Kod: `hga/evaluation/paradigma.py`
· Testler: `tests/test_paradigma_ablation.py` (9 test)

## Neden bu deney?

Proje baştan beri "hibrit nöro-sembolik" iddiası taşıyordu ama bu iddia hiç
**kontrollü** olarak test edilmemişti. Hibritin iyi olduğunu söylemek için,
saf sembolik ve saf nöral kolların aynı veri üzerinde nerede kırıldığını
göstermek gerekir. Bu, kullanıcının "en büyük 5 eksik" listesindeki maddelerden
biriydi.

## Protokol

Görev prosedüreldir: `(özne, ilişki, nesne)` üçlüsü, ilişkinin gerektirdiği
özellikleri hem özne hem nesne taşıyorsa doğrudur. Zorluk üç kaynaktan gelir:

1. **Gizli özellik (%30)** — varlıkların bir kısmının özellikleri bilgi
   tabanında yoktur. Sembolik kol burada çekimser kalmak zorundadır.
2. **Görülmemiş varlık (%20)** — test setinin bir bölümü eğitimde hiç geçmemiş
   varlıklar içerir (cold-start). Testler bu split'in gerçekten sızıntısız
   olduğunu doğrular.
3. **Dengeli etiket** — %50 doğru / %50 yanlış, yani şans seviyesi 0.500.

| Kol | Tanım |
|---|---|
| `symbolic` | Yalnız kural/bilgi tabanı. Öğrenme yok. Kanıt yoksa **çekimser** (`None`), tahmin uydurmaz. |
| `neural` | Yalnız varlık/ilişki gömmesi + MLP (BCE, Adam). Kural bilgisi yok. |
| `hybrid` | Sembolik kesin konuştuğunda veto hakkı sembolikte; çekimser kaldığında nöral doldurur. |

Görev boyutu: 120 varlık, 8 ilişki, 1200 eğitim / 400 test örneği.
Çekimser cevap **doğru sayılmaz**; `coverage` ayrı raporlanır.

## Sonuçlar (5 seed, mean ± std)

| Metrik | symbolic | neural | hybrid |
|---|---:|---:|---:|
| Accuracy | 0.507 ± 0.026 | 0.777 ± 0.028 | **0.893 ± 0.013** |
| Coverage | 0.507 ± 0.026 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Precision | 1.000 ± 0.000 | 0.787 ± 0.022 | 0.896 ± 0.015 |
| Recall | 1.000 ± 0.000 | 0.760 ± 0.039 | 0.889 ± 0.016 |
| F1 | 1.000 ± 0.000 | 0.773 ± 0.030 | 0.893 ± 0.013 |
| FAR | 0.000 ± 0.000 | 0.205 ± 0.019 | 0.103 ± 0.016 |
| FRR | 0.000 ± 0.000 | 0.240 ± 0.039 | 0.111 ± 0.016 |
| Acc (görülen varlık) | 0.510 ± 0.027 | 0.836 ± 0.031 | 0.920 ± 0.018 |
| Acc (görülmemiş varlık) | 0.495 ± 0.086 | 0.545 ± 0.032 | 0.785 ± 0.030 |

## Bulgular

**1. Sembolik kolun "mükemmel" skoru kapsam olmadan anlamsızdır.**
`precision = recall = F1 = 1.000` ve `FAR = FRR = 0.000` — kusursuz görünüyor.
Ama `coverage = 0.507`: örneklerin yarısında **cevap vermiyor**. Tüm set
üzerinden doğruluğu 0.507, yani yazı-tura seviyesinde. Bu, projenin başka
yerlerinde de geçerli bir uyarıdır: FAR=0 tek başına başarı değildir.

**2. Nöral kol ezberliyor, kural öğrenmiyor.**
Eğitim doğruluğu **1.000**, test doğruluğu 0.777 → genelleme açığı +0.223.
Kritik olan alt kırılım: görülen varlıklarda 0.836, **görülmemiş varlıklarda
0.545** — şans seviyesi 0.500'e neredeyse eşit. Gömmeler varlık kimliğini
ezberliyor; ilişkinin altındaki özellik kuralını çıkarmıyor.

**3. Hibrit her iki zaafı da kapatıyor ve bu ölçülebilir.**
Accuracy 0.893 ± 0.013: sembolik üzerine **+0.386**, nöral üzerine **+0.116**
puan. Kazanç sihir değil, mekanizması açık: sembolik kolun kesin olduğu %50'lik
dilimde hata sıfırdır, nöral kol yalnız sembolik kolun sustuğu yerde konuşur.

**4. Hibrit de sınırsız değil.**
Görülmemiş varlıklarda hibrit 0.785'e düşüyor (görülende 0.920). Sembolik kol
o örneklerin bir kısmında da çekimser kalıyor ve boşluğu dolduran nöral kol
cold-start'ta zayıf. Yani hibritlik cold-start problemini **azaltıyor, çözmüyor**.

**5. FAR tek başına raporlanmadı.** Hibrit FAR=0.103, FRR=0.111, F1=0.893 —
yanlış kabul ve yanlış ret dengeli, sistem bir tarafa kaymıyor.

## Sınırlar

- Görev sentetiktir ve prosedüreldir; gerçek Türkçe metin değildir. Sembolik
  kolun `accuracy_on_answered = 1.000` sonucu, oracle kuralın veri üretimiyle
  aynı olmasından kaynaklanır — gerçek veride bu düşecektir (Faz 27/28'in
  gerekçesi ile aynı nokta).
- Nöral kol kasıtlı olarak küçüktür (16-dim gömme, 32 gizli birim, 60 epoch).
  İddia "sinir ağları bu görevi öğrenemez" değil, "kimlik gömmesine dayalı bir
  model cold-start'ta genelleyemez"dir.
- Hibrit birleştirme kuralı basittir (sembolik veto). Güven ağırlıklı
  birleştirme denenmedi.
