# İnsan Değerlendirme Protokolü + Krippendorff α (P1)

Protokol: `human_evaluation_protocol_v1` · Modül: `hga/evaluation/human_evaluation.py`
CLI: `python -m hga insan-degerlendirme`

> **Bu belge insan değerlendirme SONUCU sunmaz.** Gerçek değerlendirici
> yoktur. Karnede `human_evaluation` bölümü `n/a` kalır ve bu kasıtlıdır.

## Neden `n/a`, neden "protokol hazır" diye puan verilmiyor?

Protokolü tanımlamak, aracı yazmak ve körlemeyi uygulamak ölçüm *altyapısı*
üretir — ölçüm *sonucu* üretmez. Araç kapılarına puan vermek "ölçmediğimi
ölçtüm" demek olurdu. Bu yüzden `build_scorecard` yalnız
`human_ratings_collected` kapısı geçtiğinde bu bölümü skorlar; aksi halde
`score: null` döner. Test `test_karne_bolumu_na_kaliyor` bunu zorunlu kılar.

Kapı kalıcı olarak kapalı da değildir: `collected_ratings` verildiği anda
bölüm skorlanır (`test_gercek_puan_verilince_skor_uretiliyor`).

## Teslim edilen üç şey

### 1. Protokol

| Alan | Değer |
|---|---|
| Prompt | 50 (şart: 50–100), Türkçe |
| Değerlendirici | 10 (şart: 10–20) |
| Kollar | `hga`, `dense`, `transformer`, `symbolic` |
| Tasarım | within-subject, tam çapraz |
| Toplam tekil yargı | 10 000 (10 × 200 öğe × 5 boyut) |

Puanlama boyutları: `dogruluk`, `tutarlilik`, `dil_kalitesi`,
`belirsizlik_durustlugu` (1–5 ordinal) ve `halusinasyon_var` (0/1 nominal).
`belirsizlik_durustlugu` kasıtlı olarak ayrı bir boyuttur: bilmediğini
söylemek ile yanlış cevap vermek aynı hata değildir.

### 2. Körleme — iddia değil, uygulanmış ve doğrulanmış

Körleme araç seviyesinde zorlanır:

- Kol adı değerlendirici paketine **hiç yazılmaz**; her öğe yalnız
  `item_id` (SHA-256 türevi) taşır.
- Kol ↔ öğe eşlemesi ayrı bir **kör açma anahtarında** tutulur; rapora
  yalnız 16 karakterlik özeti girer.
- Sunum sırası her değerlendiricide bağımsız karıştırılır (Latin-kare
  benzeri döndürme + tohumlu karıştırma), böylece sıra etkisi kolları
  dengeler.
- Değerlendirici başına dikkat kontrolü (attention check) öğeleri işaretlenir.

Rapor, üretilen paketlerde kol adı geçip geçmediğini **fiilen tarar** ve
`leaked_labels` olarak raporlar. Test, hiçbir kol adının ne paketlerde ne
de Markdown çıktısında görünmediğini doğrular.

### 3. Krippendorff's α — doğruluğu kanıtlanmış

Üç metrik desteklenir (nominal / ordinal / interval), eksik veri düşürülür,
ikiden fazla kodlayıcı desteklenir.

**Doğrulama:** Krippendorff'un kanonik örneğine (3 kodlayıcı × 15 birim,
eksik hücreler dahil) karşı test edilmiştir:

| Metrik | Referans | Ölçülen |
|---|---:|---:|
| nominal | 0.691 | **0.6914** |
| ordinal | 0.807 | **0.8067** |
| interval | 0.811 | **0.8108** |

Ayrıca davranışsal testler: tam uyumda α=1, rastgele etiketlemede α≈0
(|α|<0.12), sistematik uyuşmazlıkta α<0, ordinal metriğin komşu
uyuşmazlığı uzak uyuşmazlıktan ayırması.

**Varyans yoksa α `None` döner, 1.0 değil.** Herkes aynı etikete basarsa
yüzde uyum %100 çıkar ama hiçbir bilgi üretilmemiştir; α bunu "TANIMSIZ"
diye işaretler. Protokolde yüzde uyum kullanılmamasının sebebi budur.

Eşikler: α ≥ 0.800 kabul edilebilir, α ≥ 0.667 yalnız geçici sonuç.
Bir boyut bile eşiği geçemezse o boyuta dayanan hiçbir sonuç raporlanamaz.

## Kabul kapıları

8 kapı GEÇTİ (şartname sınırları, körleme, dikkat kontrolleri, α aracının
varlığı ve referans doğrulaması, çoklu boyut).
2 kapı **KASITLI olarak KALDI**: `human_ratings_collected` ve
`reliability_meets_threshold`.

## Sınırlar

- **Gerçek değerlendirici yok.** Bu bir protokol ve araçtır; insan
  değerlendirme sonucu değildir ve öyle sunulamaz.
- Örnek prompt'lar aracı göstermek içindir, temsili bir Türkçe
  değerlendirme korpusu değildir.
- **α tutarlılığı ölçer, doğruluğu değil.** Hepsi aynı şekilde yanılan
  değerlendiriciler yüksek α verir.
- Dikkat kontrolleri tanımlıdır ama doğru yanıt anahtarı gerçek model
  yanıtları üretilmeden doldurulamaz.
- Değerlendirici havuzunun demografisi, Türkçe yeterliği ve eğitim süreci
  bu modülün kapsamı dışındadır.
