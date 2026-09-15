# Hiper-Geometrik AI — Bilinear Kronecker Zinciri + Seyrek "Boş Küme" Belleği

Deneysel bir Türkçe dil modeli: klasik Transformer'daki saf doğrusal katman
yığınlarına alternatif olarak, **gerçek matris-sandviçi (bilinear) `A @ X @ B`
katmanlarının zinciri** ve kavramsal uzayı katrilyonların üzerinde olan
**hash'lenmiş seyrek "boş küme" belleği** üzerine kuruludur. Üretim tarafında
**3 katmanlı halüsinasyon kontrol mekanizması** (kayıtlı bilgi → beyaz listeli
üretim → işaretlenmiş serbest üretim) çalışır.

---

## ⚠️ Dürüst Kapasite Bildirimi (önce bunu okuyun)

Bu projenin eski README'si "1 Katrilyon sinaps" diyordu. Bu ifade **teknik
olarak bir üst sınırı tarif ediyordu ama gerçek eğitilebilir parametre sayısı
ile temsil kapasitesini birbirine karıştırıyordu**. Aşağıdaki tablo, varsayılan
yapılandırmanın (n=256, K=4, sözlük=8000, seyrek tablo 1.048.576 satır)
dürüst muhasebesidir:

| Ölçüm | Değer |
|---|---|
| **Yoğun (dense) gerçek parametre** | **6.970.433 (~6,97 milyon)** |
| Seyrek fiziksel depo (gerçekten RAM'de ayrılır) | 33.554.432 parametre (1.048.576 satır × 32 boyut ≈ 128 MB) |
| **TOPLAM gerçek eğitilebilir parametre** | **40.524.865 (~40,5 milyon)** |
| Seyrek depoda eğitim sonrası DOLU küme | yalnız veride görülen pencereler (demo: 2.171/1.048.576 ≈ %0,21) |
| 0→1 katmanı sanal köşe (n²) | 65.536 |
| Katman başına temsil edilen operatör (n⁴, Kronecker `Bᵀ⊗A`) | 4.294.967.296 (~4,3 milyar) |
| Zincir etkileşim üst sınırı (n^(2K) = 256⁸) | ~1,8 × 10¹⁹ |
| Seyrek bellek KAVRAMSAL anahtar uzayı (sözlük^pencere = 8000¹⁶) | ~2,8 × 10⁶² |
| Bir bilinear katmanı TAM matris olarak tutmak için gereken RAM | ~16 GB |
| Zincirin GERÇEK RAM tüketimi | ~2 MB |

**Bu sayılar ne demek, ne DEĞİL?**

Bu README üç ayrı büyüklüğü bilinçli olarak ayırır:

- **P (physical/trainable parameters):** RAM/VRAM'de gerçekten ayrılan ve
  optimizer tarafından güncellenen parametre sayısı.
- **C_I (interaction capacity):** Kronecker zincirinin temsil ettiği sanal
  etkileşim/operatör üst sınırı; gerçek parametre değildir.
- **C_M (memory address capacity):** seyrek belleğin adresleyebildiği kavramsal
  anahtar uzayı (`sözlük^pencere`); fiziksel depo boyutu değildir.

- **Bilinear/Kronecker tarafı:** Bir `A @ X @ B` katmanı, flatten uzayında
  `Y = (Bᵀ ⊗ A) X` dönüşümüdür: temsil ettiği tam operatör **n² × n² = n⁴
  boyutludur** ama bunu yalnızca **2n² gerçek parametre** ile taşır. 16 GB'lık
  operatörün ~0,5 MB'lık iki mercekle temsil edilmesi ("Kronecker İllüzyonu")
  gerçekten çalışır — ancak tam ranklı bir operatörün öğrenebileceği fonksiyon
  ailesinin yalnızca küçük bir alt kümesini süpürür (düşük-rank kısıtı).
- **Seyrek bellek tarafı:** Kavramsal anahtar uzayı (sözlük^pencere) gerçekten
  katrilyonların üzerindedir — 8000 parçalık sözlükte 4 kelimelik bir pencere
  bile 8000⁴ ≈ 4×10¹⁵ (katrilyon üzeri) farklı anahtar üretir. Fiziksel depo
  ise sabittir (128 MB) ve **başlangıçta tamamı boştur (sıfır vektör)**;
  yalnızca veride görülen pencerelere denk gelen satırlar eğitimle dolar,
  dokunulmayan satırlar asla değişmez. Bellek kullanımı dolu satırla değil,
  `tablo_boyutu` ile ölçeklenir (dürüst not: `nn.Embedding` tabloyu baştan
  tahsis eder; "sadece dokunulanlar bellekte" davranışı için dinamik bir
  anahtar-değer deposu gerekir — bu ölçekte gereksiz karmaşıklık).
- **Literal "1 katrilyon GERÇEK parametre" bu projede hedef olmamalıdır:**
  Katrilyon ölçeğinde fiziksel parametre; yüzlerce GPU'luk kümeler, petabayt
  veri ve çok yüksek bütçe gerektirir. Tek kişilik donanımda fiziksel olarak
  imkânsıza yakındır; bu repo böyle bir iddia taşımaz.
- Bu projenin gerçekçi ve dürüst hedefi: **"K katmanlı Kronecker zinciriyle
  katrilyon mertebesinin üzerinde sanal etkileşim kapasitesi + katrilyonların
  üzerinde adreslenebilir 'boş küme' uzayına sahip seyrek bellek; ~7M yoğun +
  ~34M seyrek fiziksel gerçek parametre."** Üstteki tablo bunu doğrular.

Sayıları ve araştırma protokollerini kendiniz doğrulayın:

```bash
python mimari/kuresel_model.py     # dürüst kapasite raporu
python test_mimari.py              # mimari duman testleri
python -m hga research-benchmark  # 5 seed, tek JSON/Markdown/HTML araştırma karnesi
```

Research suite ayrıca **C_G (Generalization Capacity)** ölçümünü raporlar.
Buradaki C_G teorik bir uzay büyüklüğü değildir: sürümlü held-out compositional
fixture'da doğru çözülen uygun örnek sayısı/oranıdır. `C_V ≤ C_E ≤ C_M`
eşitsizliğinin parçası değildir ve dataset hash olmadan yorumlanmaz.

---

## 📐 Mimari

```
token id'leri (B, S)
  │  nn.Embedding + öğrenilen pozisyon kodlaması
  ▼
Nedensel çok kafalı dikkat (Pre-LN + ReZero + SDPA/Flash)     ← hiper_attention.py
  │  düzleştir: (B, S·emb)
  │
  ├─→ [Seyrek bellek] pencere anahtarı → hash tablosu          ← seyrek_tablo.py
  │     "gen" vektörü (boş küme → sıfır; yalnız görülen
  │     pencereler dolar) → gen_kopru ile bağlama eklenir
  ▼
[0→1 Katman]  GeometrikVeriEncoder:  X = tanh(u ⊗ v)           ← encoder.py
  │  İki n boyutlu izdüşümün dış çarpımı → (B, n, n) matris
  │  ("n² sanal köşe"; tanh, rank-1 kısıtını kırar)
  ▼
[1→2+ Katmanlar]  KureselZincir: K adet  A @ X @ B             ← kuresel_bag.py
  │  Her katman: X ← X + SiLU(Aᵢ @ LayerNorm(X) @ Bᵢ)
  │  (Pre-Norm + artık bağlantı; opsiyonel gradient checkpointing)
  ▼
FraktalDecoder: odak = M₁ @ X @ M₂ → çift yönlü ortalama      ← decoder.py
  │  → tanh → Linear → HAM LOGİT (softmax yok — CrossEntropyLoss ile uyum)
  ▼
(B, sozluk_boyutu) logitler
```

Varsayılan yapılandırma (tek doğruluk kaynağı: `mimari/kuresel_model.py`):

| Ayar | Varsayılan | Açıklama |
|---|---|---|
| `n` | 256 | küresel bağ boyutu (128–256 önerilir) |
| `K` | 4 | bilinear katman sayısı (4–8 önerilir) |
| bağlam | 16 token | BPE alt-kelime penceresi |
| `emb_dim` | 128 | gömme boyutu |
| sözlük | ≤ 8000 | BPE alt-kelime |
| seyrek satır | 1.048.576 | "boş küme" tablosu (≈128 MB @ 32 boyut) |
| seyrek boyut | 32 | her kümenin vektör boyutu |

**Ölçekleme rehberi (rapor 8.3):** `n`'i büyütmek parametre ve işlem maliyetini
O(n²)/O(n³) büyütür; bunun yerine **K'yı artırmak** gerçek parametreyi doğrusal
tutarken sanal etkileşim üst sınırını n^(2K) ile katlar. Seyrek bellek
kapasitesi ise `tablo_boyutu` ile ölçeklenir (RAM = satır × boyut × 4 bayt):

| Yapı | Gerçek parametre (zincir) | Sanal etkileşim üst sınırı |
|---|---|---|
| n=128, K=4 | 131.072 | 128⁸ ≈ 7,2 × 10¹⁶ |
| **n=256, K=4 (varsayılan)** | **524.288** | **256⁸ ≈ 1,8 × 10¹⁹** |
| n=256, K=8 | 1.048.576 | 256¹⁶ ≈ 3,4 × 10³⁸ |

| Seyrek tablo boyutu | RAM (32 boyut, float32) |
|---|---|
| 262.144 satır | ~32 MB |
| 1.048.576 satır (varsayılan) | ~128 MB |
| 20.000.000 satır | ~2,4 GB |

---

## 🧠 Seyrek "Boş Küme" Belleği (Hashing Trick)

`mimari/seyrek_tablo.py` — `HashlenmisKureselTablo`. Kavramsal olarak
katrilyonlarca "boş küme" adreslenebilen, fiziksel olarak sabit boyutlu bir
hash'lenmiş gömme tablosu:

- **Anahtar = tam bağlam penceresi.** Pencere, taşmasız 31-bit polinomsal
  hash + splitmix sonlandırıcıyla karıştırılır; satır adresi
  `(anahtar × tuz) % tablo_boyutu` ile bulunur. Aynı pencere her zaman aynı
  satıra gider; sözlük^pencere farklı pencere adreslenebilir (8000¹⁶ ≈ 10⁶²).
- **Başlangıçta her küme boştur** (tamamı sıfır vektör). Bir pencere ilk kez
  görüldüğünde adreslenen satır gradyan alır ve eğitimle dolar; hiç
  görülmeyen kümeler sonsuza dek sıfır kalır (AdamW'da gradyanı 0 olan sıfır
  satır aynen sıfır kalır).
- **Doluluk izleme:** `doluluk_orani()` kaç kümenin dolduğunu sayar — eğitim
  loglarında çağ başına raporlanır ("1 katrilyonluk kapasitenin şu an X kümesi
  dolu" — pazarlama abartısı değil, ölçülebilir gerçek metrik).
- **Çakışma:** iki farklı pencere aynı satıra düşebilir. `carpisma_istatistigi()`
  benzersiz pencere → benzersiz adres imzası oranını ölçer ve %5 üstünde uyarı
  üretir; `tablo_sayisi=2` ile Bloom tarzı çift hash açılır (iki farklı tuzlu
  tablo, çıktılar toplanır).
- **Kronecker zinciriyle tamamlayıcıdır (rapor 9.5):** seyrek tablo HANGİ
  kümelerin dolu olduğunu taşır; `gen_kopru` dolu küme vektörünü bağlam
  vektörüne enjekte eder; bilinear zincir etkileşimi işler.

Kullanım: `egitim/egitici.py --seyrek-satir 1048576 --seyrek-boyut 32` veya
`--seyrek-yok` ile kapatın. Talimat fine-tuning'i aynı parametrelerle
kurulmalıdır (strict yükleme uyumsuzluğu açıkça hata verir).

### Collision / interference benchmarkı

Saf Python `DeneyimSlotlari` prototipi için streaming stres benchmarkı;
collision event rate, exact-ID retrieval accuracy, interference loss, false
positive rate, throughput ve yaklaşık fiziksel Python depolamasını ölçer.
Collision olaylarının tamamı sayılır; yalnız ilk 1000 teşhis örneği saklanır,
böylece 1M/10M koşuları collision log'u nedeniyle sınırsız RAM tüketmez.

```bash
# Hızlı varsayılan: 1K, 10K, 100K
python -m hga memory-benchmark

# Tam 1K→10M planı: fixed ve Engine'deki aynı bounded Dynamic KV
python -m hga memory-benchmark --active-dynamic \
  --scales 1000,10000,100000,1000000,10000000 \
  --slots 65536 --audit-samples 256 --seeds 1,2,3,4,5 \
  --out raporlar/memory_stress_1k_10m.json \
  --markdown raporlar/memory_stress_1k_10m.md
```

Aktif mod her seed için girdileri listelemeden **tek streaming geçişte** 10M'e
taşır; her checkpoint'i baştan koşturmaz. FIRST_WINS table-1 için dolu slot
sayısı, bounded unique-key Dynamic KV için aktif kayıt sayısı exact history
recall'ın fiziksel invariantıdır; ikinci bir 10M read turu yerine bu tam sayı
muhasebesi ve deterministik retained/evicted audit örnekleri birlikte raporlanır.
Journal ve lazy-LRU heap boyutları bounded kabul kapılarıdır. RSS ile CPython
storage tahmini ayrı alanlardır.

Her seed ayrı `EXP-NNNN` manifesti üretir. Fixed çift-tablo legacy modunda
**ALL-table** exact-ID okuma collision kaybını telafi etmez; aktif bounded
Dynamic KV collision'ı kaldırır fakat kapasite sonrasında LRU eviction nedeniyle
tüm tarih recall'ı düşer. Aktif kayıt exact recall'ı ile history recall aynı
metrik değildir. Bunların hiçbiri semantic/learned retrieval veya gerçek dil
kanıtı değildir. Ayrıntılar: `docs/MEMORY_COLLISION_BENCHMARK.md`.

Temiz `94fe3f6` commit'inde, 65.536 entry capacity ve 5 seed ile gerçek 1K→10M
koşusu tamamlandı (`raporlar/memory_stress_1k_10m.{json,md}`):

| Context | Fixed history recall | Dynamic history recall | Dynamic active recall | Eviction |
|---:|---:|---:|---:|---:|
| 1K | 0.991600±0.002154 | 1.000000±0 | 1.000000±0 | 0 |
| 10K | 0.926120±0.000773 | 1.000000±0 | 1.000000±0 | 0 |
| 100K | 0.512728±0.001186 | 0.655360±0 | 1.000000±0 | 34.464 |
| 1M | 0.065536±0 | 0.065536±0 | 1.000000±0 | 934.464 |
| 10M | 0.006554±0 | 0.006554±0 | 1.000000±0 | 9.934.464 |

Sonuç üstünlük ilanı değil, bounded kapasite sınırıdır: Dynamic KV hash
collision'ını giderdi, fakat 10M geçmişin yalnız son 65.536 kaydı aktiftir.
Aktif-küme audit recall'ı `1.0` iken tüm-geçmiş recall'ı `0.006554`'e düşer.

### Eşit-parametre Kronecker benchmarkı

```bash
python -m hga kronecker-benchmark \
  --n 16 --steps 300 --batch 64 --seeds 1,2,3,4,5
```

Karşılaştırma her iki modele tam `2n²` fiziksel parametre verir. Önceki
“standart dense” adı düzeltildi: baseline gerçekte `n²→1→n²` rank-1 factorized
bottleneck'tir; kısıtsız dense operatör `n⁴` parametre gerektirir. Tek görevli
Kronecker yanlılığını önlemek için hem `Y=A*XB*` hem rank-1 öğretmen görevi
çalıştırılır. Held-out normalized MSE/R²/tolerance accuracy, hız, parametre ve
optimizer belleği, gradient/loss kararlılığı ve efektif rank raporlanır.
`n⁴` her raporda açıkça **operatör girdisi, gerçek parametre değil** olarak
saklanır. Ölçülmüş beş-seed sonucunda yapı-matched modeller kendi görevlerini
kazandı: Kronecker-teacher NMSE'si Kronecker modelde `1.07e-11`, rank-1 modelde
`0.9597`; rank-1-teacher NMSE'si rank-1 modelde `3.43e-6`, Kronecker modelde
`0.9780` oldu. Tam rapor `raporlar/kronecker_5seed_summary.json`, ayrıntılar
`docs/KRONECKER_VS_DENSE_BENCHMARK.md` içindedir.

### 100-cycle self-learning ve collapse testi

```bash
python -m hga self-learning-benchmark \
  --cycles 100 --batch 16 --initial-facts 100 \
  --operands-max 31 --negatives-per-fact 7 --seeds 1,2,3,4,5
```

`CLOSED_VERIFIED` protokolünde yalnız bağımsız aritmetik environment tarafından
onaylanan deneyimler Kₙ'e yazılır. Yanında çalışan `UNVERIFIED_SELF_TRAINING`
probu ise aynı model çıktısını verifier olmadan tekrar besleyen kasıtlı failure
injection'dır. Experience Yield, FAR/FRR, correct/incorrect knowledge, novelty,
diversity, entropy ve memory collision/retrieval birlikte raporlanır.

Aynı CLI artık `arithmetic`, `logic` (modus ponens) ve `consistency`
(property constraints) environment'larını ayrı relation/entity namespace ve
bağımsız verifier router'larıyla, fakat ortak `KnowledgeStore` + aktif Dynamic
KV üzerinde interleave eder. Yanlış environment verifier'ı kabul etmek yerine
`None` dönmeli; per-environment FAR/FRR, durable contamination,
generation/memory/knowledge holdout overlap ve shared-memory recall ayrı
kapılardır. Holdout yalnız leakage kontrolüdür; neural task generalization
ölçümü değildir.

Beş-seed kontrollü baseline'ında (`K₀=100`, 100 cycle, batch=16) ortalama
K₁₀₀=`295.8±5.04`, Experience Yield=`0.1224±0.0031`, incorrect knowledge=0
ve Verifier sonrası FAR/FRR=0 ölçüldü (Evaluator tek başına FAR=1.0).
Verifier'sız failure injection'da repetition=0.99, FAR=1.0, yanlış model
olgusu=`14.0±0.63 / 16` ve memory collision=1,584 oldu. Tam config/dataset/Git
hash'leri ve beş koşu manifesti `raporlar/self_learning_5seed_summary.json`
içindedir. Ek fault-injection probu FAR/FRR=0.25, precision/recall/F1=0.75,
`UNCERTAIN=2` ve `CONFLICT=2` kontrolünü; memory kapasite sweep'i ise %95 recall
eşiğini sınar. Ayrılmış test holdout'un generation/memory overlap'i her koşuda
sıfır olmak zorundadır. Bunlar sentetik aritmetik laboratuvar sonuçlarıdır; gerçek dilde
otonom öğrenme iddiası değildir. Ayrıntılar: `docs/SELF_LEARNING_BENCHMARK.md`.

> **Güncelleme (P1, `deep` profil koşuldu).** `ogrenme-olcek` deep profili
> 100 → 1000 → 3000 cycle + alan eksenini (operands_max 31 → 63 → 127)
> gerçekten koştu, 9/9 kapı geçti: 3000 cycle'da yanlış bilgi **0**, FAR 0,
> izolasyon temiz (doğrulayıcı ÖNCESİ FAR 1.0 — doğrulayıcının iş yaptığının
> kanıtı). Ana bulgu ölçekleme fiziği: sabit alanda cycle 30× artınca bilgi
> yalnız 1.032× arttı (aday havuzu doyuyor; "daha çok döngü = daha çok
> bilgi" YANLIŞ), buna karşılık alan 4× büyüyünce bilgi **15.71×** arttı.
> Ölçekleme cycle sayısından değil ALANIN genişliğinden gelir.
> Ayrıntı: `docs/SELF_LEARNING_SCALING.md`.

### HGA Research Benchmark Suite

Tek komut Architecture, Kronecker, Memory, Verification, Compositional
Generalization, Neural/Symbolic/Hybrid, Self Learning, OOD, Turkish NLP ve
Reproducibility bölümlerini ortak sözleşmede çalıştırır:

```bash
python -m hga research-benchmark
```

Varsayılan `smoke` profil **5 seed** kullanır; her seed ayrı `EXP-NNNN`
manifestidir. Sonunda çalışma dizinine şu üç dosya yazılır:

```text
research_report.json
research_report.md
research_report.html
```

Yayın/uzun koşu ve seçili bölüm örnekleri:

```bash
python -m hga research-benchmark --profile full --seeds 1,2,3,4,5 \
  --out raporlar/research_report.json \
  --markdown raporlar/research_report.md \
  --html raporlar/research_report.html

python -m hga research-benchmark \
  --sections compositional-generalization,verification,ood
```

Turkish NLP ana skoru, Apache-2.0 lisanslı ve insan anotasyonlu **Turkish Web
Treebank** üzerinde hesaplanır. Upstream revision
`40838e5cbe3f2882d4e768a3d782e6219e50b52a` ve iki CoNLL-U dosyasının SHA-256
özetleri `hga/evaluation/datasets/twt_v1/PROVENANCE.json` içinde sabittir.
4.851 ham gerçek Türkçe cümleden duplicate/near-duplicate quarantine sonrası
3.881/484/484 train/dev/test kalır. Dependency-arc verification görevi
Accuracy/F1/FAR/FRR/coverage ile; entity-, relation-, composition-, wording- ve
sentence-disjoint dilimlerde raporlanır. Bu etiketler morphosyntactic'tir;
semantik relation extraction veya NER iddiası değildir.

Aynı bölüm, aynı model-visible TWT adaylarında gerçek PyTorch ile Dense,
Transformer, repository `KureselZincir` Kronecker kolu ve gerçek HGA
attention/outer-product/Kronecker/fraktal çekirdeğini karşılaştırır. Toplam
trainable parametre max/min oranı `≤1.01`, ortak embedding dışındaki body oranı
`≤1.05` olmak zorundadır; unused bütçe parametresi yasaktır. Veri, train-only
vocabulary, başlangıç embedding'i, batch schedule hash'i, AdamW, loss, adım ve
karar eşiği dört kolda aynıdır. Bu structured arc sınıflandırması tam language
model pretraining veya end-to-end dependency parser değildir.

Her neural kol için uncertainty calibration da aynı harness içinde ölçülür:
pozitif scalar temperature yalnız sabit **dev** splitinde NLL ile seçilir,
**test** ise yalnız değerlendirmedir. Testte kalibrasyon öncesi/sonrası NLL,
Brier, fixed/adaptive ECE; tüm disjoint dilimler ve önceden ilan edilmiş
%10/%25/%50/%75/%100 coverage noktalarında selective risk/AURC raporlanır.
Temperature argmax kararlarını değiştiremez ve bu invariant kapıdır. Engine
`weighted`/source-confidence alanları bu deneyle olasılık ilan edilmez.
Ayrıntılar: `docs/UNCERTAINTY_CALIBRATION.md`.

Verification bölümü ayrıca gerçek, hash-doğrulanmış TWT kaynak artifact'ları
üzerinde zaman/revizyon yaşam döngüsünü sınar. `ACTIVE`, `STALE`, `SUPERSEDED`
ve `RETRACTED` ayrıdır: süresi geçen veya erişilemeyen kayıt default sorgudan
çıkar ama **yanlış ilan edilmez**; aynı-hash revalidation geri açabilir, değişen
hash yeni revision gerektirir ve dependency staleness transitif yayılır. Tüm
geçişler hash-chain event defterindedir. Ağ/revizyon/withdrawal olayları
kontrollü müdahalelerdir; canlı TWT değişikliği iddiası değildir. Ayrıntılar:
`docs/KNOWLEDGE_LIFECYCLE.md`.

Neural compositional ablation ayrıca full HGA'yı `no_attention`, aynı
parametreli `additive_geometry` ve `no_kronecker_chain` kollarıyla aynı
başlangıç tensorları/batch schedule üzerinde karşılaştırır. `C_G_N`, yalnız
TWT `composition_disjoint` dengeli arc accuracy'sidir; mevcut kontrollü `C_G`
veya teorik kapasite değildir. Çıkarılan bileşenlerin parametre farkı unused
reserve ile kapatılmaz.

Türkçe compositional fixture çalışma anında üretilmez;
`hga/evaluation/datasets/compositional_tr_v1.json` içinde elle sabitlenmiştir.
Train'de `Ali ata bindi`, `Ayşe arabaya bindi`, `Mehmet otobüse bindi`
varken held-out `Ali arabaya bindi` ve `Ali otobüse bindi` aday/karar/metin
kanallarında ayrı ölçülür. Seen composition, unseen entity, unseen relation,
unseen combination, unseen wording, unseen sentence, negative ve OOD
boyutları ayrı raporlanır. Semantik ve exact-yüzey sızıntısı koşuyu durdurur.

Unseen entity'nin ontoloji kaydı ve unseen relation'ın ilişki şeması sisteme
verildiği için sonuç **entity discovery, relation induction veya neural dil
üretimi değildir**. Referans kurulumda PyTorch proje bağımlılığıdır ve tüm
bölümler `COMPLETED` olmalıdır; eksik kurulum `SKIPPED` olarak gizlenmez ve
`--strict` ile koşu başarısız kapatılabilir. Verification bölümü ayrıca false,
incomplete, contradictory, malformed, boundary ve adversarial proof fixture'ını
FAR/FRR/coverage/robustness ile ölçer. `overall_diagnostic_score` bir zekâ/SOTA
skoru değildir. Tam protokol: `docs/RESEARCH_BENCHMARK_SUITE.md`.

### Milestone tablosu: K₀ → E₀ → V₀ → K₁ → … → K₁₀₀

Yukarıdaki self-learning benchmarkının üstüne bilgi sürümleme (Faz 23),
değişmez deneyim defteri (Faz 24) ve ölçülen kapasite (Faz 13–14) bağlanmış
hâlidir. Çıktısı tek bir tablodur: **metrik × cycle**.

```bash
python -m hga milestone \
  --cycles 100 --batch 32 --initial-facts 100 \
  --operands-max 31 --negatives-per-fact 7 \
  --seeds 1,2,3,4,5 --checkpoints 0,10,50,100 \
  --out raporlar/milestone_100.json --markdown docs/MILESTONE_TABLOSU.md
```

Beş seed, 100 döngü, mean ± std (`docs/MILESTONE_TABLOSU.md`):

| Metrik | Cycle 0 | Cycle 10 | Cycle 50 | Cycle 100 |
| --- | ---: | ---: | ---: | ---: |
| Verified Knowledge | 100.0 ± 0.0 | 136.4 ± 4.2 | 295.8 ± 5.0 | 497.6 ± 12.5 |
| New Knowledge | 0.0 ± 0.0 | 36.4 ± 4.2 | 195.8 ± 5.0 | 397.6 ± 12.5 |
| Invalid | 0.0 ± 0.0 | 283.6 ± 4.2 | 1404.2 ± 5.0 | 2802.4 ± 12.5 |
| Uncertain | 0.0 ± 0.0 | 20.0 ± 0.0 | 100.0 ± 0.0 | 200.0 ± 0.0 |
| Conflict | 0.0 ± 0.0 | 10.0 ± 0.0 | 50.0 ± 0.0 | 100.0 ± 0.0 |
| FAR | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| FRR | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Experience Yield | 0.0000 | 0.1040 ± 0.0121 | 0.1119 ± 0.0029 | 0.1136 ± 0.0036 |
| Memory Collision | 0.0 ± 0.0 | 0.2 ± 0.4 | 5.2 ± 1.7 | 19.0 ± 4.1 |
| Memory Recall | 1.0000 | 0.9950 ± 0.0100 | 0.9735 ± 0.0086 | 0.9523 ± 0.0099 |
| Incorrect Knowledge | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |

Okunuşu — abartısız:

* Kapalı döngü 100 cycle boyunca **hiç yanlış olgu yazmadı** (incorrect=0) ve
  Experience Yield ~%11–12'de stabil kaldı; yani bilgi kalitesi çürümedi.
* **Bellek çürüdü**: recall 1.000 → 0.952, collision 0 → 19. Bu bir başarı
  değil, ölçülmüş bir sınırdır — sabit slot sayısında sparse memory doğrusal
  birikimi taşımıyor.
* `UNCERTAIN` ve `CONFLICT` sütunları epistemik problarla **gerçekten
  dolduruluyor**: "bilmiyorum" ile "yanlış" ayrı sayılır (Faz 4).
* Her koşuda rollback tatbikatı yapılır: bilerek yanlış bir olgu yazılıp
  (`K101`, yanlış=1) sağlam sürüme dönülür (`K102`, yanlış=0) ve **hatalı
  sürüm geçmişte korunur**.
* 100 döngüde seed başına 3.500 defter kaydı üretilir; hash zinciri her koşuda
  doğrulanır.

Sınır: ground truth bağımsız ama sentetik aritmetik environment'tan gelir.
Bu tablo genel dilde otonom bilgi keşfi kanıtı **değildir**.

### Kapasite çerçevesi: P, C_I, C_M, C_E, C_V

```bash
python -m hga kapasite --operands-max 9
```

`C_M` (adreslenebilir) ile `C_E` (üretilebilir) ve `C_V` (doğrulanabilir) ayrı
büyüklüklerdir; zorunlu sıralama `C_V ≤ C_E ≤ C_M`'dir. İlk ikisi teorik üst
sınır, son ikisi **ölçülen** değerdir:

| Kapasite | Sembol | Tip | Anlam |
|---|---|---|---|
| Fiziksel parametre | `P` | ölçülen | RAM/VRAM'de ayrılan, optimizer'ın güncellediği |
| Etkileşim kapasitesi | `C_I` | üst sınır | Kronecker operatör boyutu — parametre DEĞİL |
| Bellek adres kapasitesi | `C_M` | üst sınır | `sözlük^pencere` — fiziksel tablo DEĞİL |
| Deneyim kapasitesi | `C_E` | **ölçülen** | kısıtlar altında gerçekten üretilebilen deneyim |
| Doğrulanabilir kapasite | `C_V` | **ölçülen** | bağımsız verifier'ın karara bağlayabildiği alt küme |

Milestone koşusunda `C_M ≈ 2.8×10⁶²` iken `C_E = 1.184.832` ve
`C_V = 1.180.685` ölçüldü (`C_V/C_E = 0.9965`). Aradaki ~10⁵⁶'lık uçurum tam
olarak "adreslenebilir olmak ile üretip doğrulayabilmek arasındaki fark"tır.

### Bilgi sürümleme, rollback ve değişmez defter

```bash
python -m hga bilgi-surum   # K0 → K1 → K2(hatalı) → rollback → K3
python -m hga defter        # VERIFIED/INVALID/UNCERTAIN/CONFLICT kayıtları
```

* **Rollback `git revert` semantiğidir, `reset --hard` değil:** hedef sürümün
  içeriği yeni bir sürüm olarak geri yüklenir, hatalı sürüm zincirde kalır.
  Snapshot'lar içerik-adreslidir (SHA-256) ve `zincir_dogrula()` ile denetlenir.
* **Defter append-only ve hash-zincirlidir:** reddedilen deneyim de silinmez.
  Geçmiş bir kayıt değiştirilirse zincir doğrulaması bunu yakalar — "model
  geçmişte nerede hata yaptı?" sorusu ancak böyle cevaplanabilir.

### Neural vs Symbolic vs Hybrid (Faz 21)

```bash
python -m hga paradigma --seeds 1,2,3,4,5
```

"Hibrit mimari" iddiası ilk kez **kontrollü** olarak test edildi: aynı veri,
aynı split, aynı metrikler; %30 gizli özellik ve %20 görülmemiş varlık
(cold-start) ile. 5 seed, mean ± std:

| Metrik | symbolic | neural | hybrid |
|---|---:|---:|---:|
| Accuracy | 0.507 ± 0.026 | 0.777 ± 0.028 | **0.893 ± 0.013** |
| Coverage | 0.507 ± 0.026 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| F1 | 1.000 ± 0.000 | 0.773 ± 0.030 | 0.893 ± 0.013 |
| FAR / FRR | 0.000 / 0.000 | 0.205 / 0.240 | 0.103 / 0.111 |
| Acc (görülmemiş varlık) | 0.495 ± 0.086 | 0.545 ± 0.032 | 0.785 ± 0.030 |

Sembolik kolun `F1 = 1.000` skoru **kapsam olmadan anlamsızdır**: örneklerin
yarısında çekimser kalır, tüm set üzerinden doğruluğu 0.507 (yazı-tura).
Nöral kol eğitimde 1.000, testte 0.777, görülmemiş varlıkta **0.545 ≈ şans** —
kuralı genellemiyor, varlık kimliğini ezberliyor. Hibrit +0.386 / +0.116 puan
kazandırıyor ama cold-start'ta kendisi de 0.785'e düşüyor: hibritlik bu
problemi **azaltıyor, çözmüyor**. Ayrıntı: `docs/PARADIGMA_ABLASYONU.md`.

### Ölçeklendirilmiş golden benchmark (Faz 3/6)

```bash
python -m hga olcekli-golden --sizes 100,1000,10000 --seeds 1,2,3,4,5 --hard
```

`golden_dataset/` elle sabitlenmiş 8 kayıttır; 5 test örneğinde `FAR=0` görmek
gürültüden ayırt edilemez. Bu komut aynı epistemik sözleşmeyi koruyarak veri
setini prosedürel olarak ölçekler. Beklenen etiket evaluator'ın karar ağacından
değil **kanıt durumunun inşasından** türetilir, yani ölçüm tautolojik değildir.

| Ölçek | Accuracy | F1 | FAR | FRR | Örnek başına süre |
|---:|---:|---:|---:|---:|---:|
| 100 | 1.0000 ± 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.178 ms |
| 1.000 | 1.0000 ± 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.169 ms |
| 10.000 | 1.0000 ± 0.0000 | 1.0000 | 0.0000 | 0.0000 | **17.741 ms** |

Doğruluk 10.000 örnekte de bozulmuyor (zor modda, eşik/kısıt-önceliği/kaynak
çatışması sınır vakalarıyla birlikte). **Asıl bulgu maliyette**: örnek başına
süre ölçek 100× büyürken 99.7× arttı — `Scoring.information_gain` adayı tüm
varlıklarla karşılaştırdığı için değerlendirme O(N²). Ters indeksle sabit
çarpan düşürüldü; asimptotik sınıf metriğin tanımı gereği korunuyor.
Ayrıntı: `docs/OLCEKLI_GOLDEN_BENCHMARK.md`.

### Kronecker effective rank ve zincir çöküşü (Faz 19/20)

```bash
python -m hga kronecker-rank --n-values 4,8,16 --k-values 1,2,4
```

`n^(2K)` etkileşim uzayı sayısı hesaplanıyordu ama **hiç ölçülmüyordu**. Bu
komut onu kırmaya çalışır ve iki net sonuç verir:

* **Aktivasyonsuz K katmanlı zincir tek katmana ÇÖKÜYOR.** `A₂(A₁XB₁)B₂ =
  (A₂A₁)X(B₁B₂)` özdeşliği sayısal olarak doğrulandı (kalıntı ~3×10⁻⁷, 6/6
  konfigürasyon). Bu durumda `n^(2K)` **boş bir üst sınırdır**: K'nın ifade
  gücüne katkısı sıfırdır. Derinliğin katkısı tamamen **aktivasyondan** gelir
  (SiLU ile kalıntı 0.38–0.72'ye çıkıyor).
* **Rank tam ama yanıltıcı.** `rank(Bᵀ ⊗ A) = n²` yani operatör tam rank'tır
  (kullanım 1.000), fakat serbestlik derecesi yalnız `2n²`'dir ve spektral
  etkin boyut çok daha küçüktür (n=16'da 256 yerine **98.38**).

Ayrıntı: `docs/KRONECKER_RANK_VE_NK_TARAMASI.md`.

### Köken (provenance) ve Priority(E) ağırlıkları (Faz 27/28, 25)

```bash
python -m hga koken      # "nereden biliyorum?" denetimi
python -m hga oncelik    # Priority(E) terim ablasyonu
```

`RelationFact` artık opsiyonel `source_url` / `document_hash` / `sentence` /
`extractor` / `retrieved_at` taşır (geriye dönük uyumlu). `REAL_DATA` kaynaklı
bir olgu köken taşımıyorsa **yetim olgu** sayılır; belge sonradan değişirse
hash doğrulaması bunu yakalar. Ölçülen darboğaz: sözlük tabanlı ayıklayıcının
`extraction_yield`'i demo korpusta **0.5** — köken altyapısı hazır, ayıklama
kapsamı dar.

Priority(E) ağırlıkları tek kaynakta tanımlı, negatif değer reddediliyor,
`normalize=True` ile karşılaştırılabilir hâle geliyor ve `priority_dokumu()`
her terimin katkısını ayrı gösteriyor. Demo havuzundaki ablasyon önce bir
"zaaf" gösterdi (`w_novelty`/`w_uncertainty` seçimi değiştirmiyor); P0-1 kök
neden analizi bunun terimlerin değil **degenere ölçüm havuzunun** özelliği
olduğunu kanıtladı (kayıtsız adaylar `novelty=uncertainty=1.0` köşesinde
yığılıyor ve ilk-K tamamen köşeden seçiliyor). Havuz düzeltildikten sonra
**dört terimin dördü de** skor→sıralama→seçim→downstream zincirini uçtan uca
taşıyor. Ayrıntı: `docs/KOKEN_VE_ONCELIK.md`, `docs/PRIORITY_CAUSAL_CHAIN.md`.

### Memory interference: kasıtlı çakışma (Faz 15–17)

```bash
python -m hga memory-interference --slots 4096 --forced-collisions 50
```

Milestone tablosu 100 döngüde recall'ın `1.000 → 0.952`'ye düştüğünü gösterdi;
bu deney çakışmayı beklemek yerine **kurar** (A ve B zorla aynı slota):

| Politika | Kurban yaşar | Saldırgan yaşar | İkisi birden | Bozulma |
|---|---:|---:|---:|---:|
| `FIRST_WINS` (legacy) | 1.000 | 0.000 | 0.000 | 0.000 |
| `LAST_WINS` | 0.000 | 1.000 | 0.000 | 1.000 |
| `DYNAMIC_KV` | 1.000 | 1.000 | 1.000 | 0.000 |

Sabit tabloda `both_survival = 0` bir ayar meselesi değil, **yapısal**dır: tek
slot iki kimliği taşıyamaz. 4096 slotlu tabloda recall yük faktörüyle çöküyor
(100k context → **0.041**), dinamik KV recall'ı 1.0 tutuyor ama 27× bellek
istiyor. Ayrıntı ve dönüm noktası analizi: `docs/MEMORY_INTERFERENCE.md`.

`DYNAMIC_KV` artık yalnız kıyas prototipi değildir: `ExperienceEngine`'in
varsayılan aktif deneyim/replay deposudur ve kıyas da aynı
`DynamicKVMemory` sınıfını kullanır. Tam `(subject, relation, object)` anahtarı
kimliktir; 31-bit hash yalnız gözlem adresidir. Bounded Engine modunda LRU/FIFO
ve opsiyonel idle-TTL eviction, stale replay temizliği, atomik/hash-korumalı
JSON persistence, parent-hash snapshot zinciri, monoton store/record version,
journal compaction ve kayıp muhasebeli `FIRST_WINS → DYNAMIC_KV` migration
vardır. Collision kaybı kalkarken kapasite dolumu artık açık eviction kaybına
dönüşür; semantic/learned retrieval iddiası yoktur. Yaşam döngüsü Research
Suite Memory bölümünde her seed için kırılır.

### Epistemik benchmark: bilmediğini biliyor mu? (P0-007)

```bash
python -m hga epistemik --seeds 1,2,3
```

"Doğru cevap oranı" tek başına yanıltıcıdır; bir sistem her soruya emin
cevap verip yüksek skor alabilir. Asıl soru **kanıt yokken susmayı bilip
bilmediğidir**. Bu benchmark `ExperienceEvaluator`'ı beş epistemik sınıfa
karşı ölçer: `KNOWN` (kabul et), `FALSE` (reddet), `UNKNOWN` (kayıt yok),
`UNCERTAIN` (özellik yazılmamış), `CONFLICT` (kanıt kuralla zıt). Veri
kümesi elle sabitlenmiş, hash'lenmiş 28 vakalık depo fixture'ıdır.

Asıl metrikler **yanlış güven oranı (↓)** ve **bilinmeyen doğruluğu (↑)**
birlikte raporlanır, çünkü tek metrik kandırılabilir. Benchmark ilk koşuda
%100 aldığı için üç dejenere politika negatif kontrol olarak eklendi:

| Kol | Doğruluk | Bilinen | Bilinmeyen | Yanlış güven |
|---|---:|---:|---:|---:|
| `always_valid` | 0.286 | 1.000 | 0.000 | 1.000 |
| `always_abstain` | 0.321 | 0.000 | 1.000 | 0.000 |
| `always_invalid` | 0.286 | 0.000 | 0.000 | 1.000 |
| **Evaluator** | **1.000** | **1.000** | **1.000** | **0.000** |

`always_abstain` yalnız "bilinmeyen doğruluğu"na bakılsa mükemmel görünür
ama hiçbir bilineni kabul edemez. `beats_degenerate_baselines` kapısı
gerçek değerlendiricinin her kolu dört eksende birden domine etmesini
zorunlu kılar.

**Önce ölçülen, sonra kapatılan sınır:** `UNKNOWN` (kayıt hiç yok) ile
`UNCERTAIN` (özellik yazılmamış) epistemik olarak farklıdır — *bilmiyorum*
ile *emin değilim*. Evaluator ikisini de tek bir `UNCERTAIN` durumuna
indiriyordu; bu önce gizlenmeden `distinguishable = false` diye raporlandı,
sonra kapatıldı. Çözüm `DeneyimDurumu`'na yeni üye eklemek değil (durum
makinesi ve 30+ karşılaştırma noktası kırılırdı), `ExperienceCandidate`'e
makine-okunur `belirsizlik_sebebi` alanı eklemek oldu: `KAYIT_YOK` /
`OZELLIK_YOK`. Rapor artık `distinguishable_by_state = false` (durum kodu
hâlâ tek) ile `distinguishable_by_reason = true` ayrımını birlikte verir ve
iki yeni kabul kapısı bunu zorunlu kılar.
Ayrıntı: `docs/EPISTEMIK_BENCHMARK.md`.

### Çok adımlı çıkarım ve uzun bağlam (P1-004 / P1-006)

```bash
python -m hga cok-adimli --hops 1,2,3,4,5 --distractors 0,16,64,256 --seeds 1,2,3
```

`a→b→c` zinciri kurulup depoda **yazılı olmayan** `a→c` sorulur; ayrıca zincir
kenarlarının arasına alakasız dolgu olgular serpiştirilerek uzun bağlam
baskısı uygulanır. Adresleme düzeltmesi sonrası çok adımlı çıkarım (hop≥2)
**1.0000**, tek adımlı geri çağırma 1.0000; varsayılan 1–5 hop / 0–256 dolgu
ızgarası tamamen temiz ve beş kabul kapısının **beşi de geçiyor**. (Düzeltme
öncesi: hop≥2 0.9688, en derin güvenilir zincir 4 adım, 5 adım @256 dolgu
0.5000 ve üç kapı KALIYORDU — o sınır kapasite değil, adresleme kusurunun
eseriydi; bkz. `docs/SPARSE_ADDRESSING_FIX.md`. Gerçek sınır yüzlerce hop
ötededir ve `cikarim-derinligi` protokolünde ölçülür.)

İlk taslakta zincir takibi Python sözlüğünden yapılıyordu ve doğruluk her
koşulda 1.0 çıkıyordu — **ölü metrik**. Takip artık her kenarı seyrek
bellekten doğrular; bellek 4096→256 slota indirilince en derin güvenilir
zincir 5→2 adıma düşer. Bu davranış testle kilitlidir.

> **Güncelleme (P0-7).** Buradaki "4 adım", `cok-adimli` protokolünün
> 1–5 hop / 0–256 dolgu ızgarasına aittir ve o ızgaranın tavanına yakındır.
> Daha geniş taramalar iki aşamada ilerledi: önce ızgara genişletilince
> C_R=32 ölçüldü; sonra bu sınırın kaynağı kovalanınca seyrek bellek
> adreslemesinde gerçek bir KUSUR bulundu ("Bloom tarzı" iki tablo sıfır
> bağımsızlık sağlıyordu — bkz. `docs/SPARSE_ADDRESSING_FIX.md`). Düzeltme
> sonrası 2^18 slotla ölçülen C_R=2048 / C_RD=256 sınırının da kök nedeni
> kovalandı: teşhis protokolü (`docs/DEPTH_DIAGNOSIS_RAW.md`) düşüşün
> çıkarım değil **bellek doygunluğu** olduğunu gösterdi (derinlik slotla
> log-log eğim ~1.33 ile ölçekleniyor). Deep profil dolgu yüküne göre
> boyutlandırılınca (2^20 slot) ölçüm: **C_R = 16384** (ızgara-içi, tavan
> değil); 16384 dolguda C_RD **8192** (retention 0.5,
> `retains_half_depth_under_max_distractors` GEÇTİ). Dar taramanın "yetenek
> sınırı" gibi görünmesi ve o sınırın iki kez de mühendislik/yapılandırma
> çıkması, bu deponun "sınırı gizleme, kaynağını ölç" ilkesinin somut örneğidir.
> Ayrıntı: `docs/REASONING_DEPTH.md`, `docs/REASONING_DEPTH_ROOT_CAUSE.md`.

Ayrıntı: `docs/COK_ADIMLI_VE_UZUN_BAGLAM.md`.

### Deneyim verimi: EY'nin ötesinde (P1-005)

```bash
python -m hga verim --cycles 30 --batch 32 --initial-facts 40 \
  --operands-max 15 --negatives-per-fact 3 --seeds 1,2,3,4,5
```

`EY = doğrulanmış/üretilen` üç farklı durumu ayıramaz ve üçünde de yüksek
çıkar: zaten bilineni tekrar doğrulamak, doğrulayıp bellekte kaybetmek ve
ezberleyip hiç genelleyememek. EY dört eksene ayrıldı (5 tohum, %95
bootstrap GA):

| Metrik | Ortalama | %95 GA |
|---|---:|---:|
| EY (klasik) | 0.2385 | [0.2327, 0.2444] |
| **NY** yenilik | 0.1904 | [0.1863, 0.1946] |
| **UEY** kullanışlı | 0.1606 | [0.1575, 0.1646] |
| **GY** genelleme | 0.6419 | [0.6116, 0.6721] |
| **VID** bit/deneyim | 0.9434 | [0.9227, 0.9640] |

Ayrışma tek yönlüdür: **EY > NY > UEY**. Yani klasik EY hem yeniliği hem
kullanışlılığı sistematik olarak abartıyor (960 üretim → 220 doğrulama →
178 ayrık yeni → yalnız 148 geri çağrılabilir; 224 tekrar üretim, 72
bellek çakışması, 0 yanlış olgu).

**GY iki kez "ölü metrik" olarak yakalandı.** Önce holdout kararı
`ExperienceEvaluator` ile veriliyordu; `R_EQUALS`'ın hiç kısıtı olmadığı
için evaluator holdout'un tamamına — öğrenme öncesi *ve* sonrası — `VALID`
diyordu ve GY yapısal olarak daima 0.0'dı. Sonra çıkarım kuralı
`score >= 1.0` arıyordu; oysa doğrulama hattı olguları `0.6375` ile yazıyor,
yani eşik öğrenilen her olguyu sessizce eliyordu. Düzeltmeden sonra GY
öğrenmeyle monoton artıyor (10/20/30 döngü → 0.163 / 0.361 / 0.605) ve
karar verilen her örnekte isabet 1.0000'dır. Daima sıfır dönen bir metrik
ölçüm yapmıyor demektir; ikisi de testle sabitlendi.
Ayrıntı: `docs/VERIM_METRIKLERI.md`.

---

## 🛡️ 3 Katmanlı Halüsinasyon Kontrol Mekanizması

`legacy/bilgi_katmani.py` — `BilgiKatmani` (LEGACY). Rapor 10'daki temel ödünleşim:
**"kelimeyi bilmek" ≠ "cümleyi/ilişkiyi bilmek"**. Saf ezbere kilitlenmiş bir
sistemde halüsinasyon ~0'a iner ama hiç görmediği cümleyi de kuramaz; saf
genellemede (LLM'ler) esneklik yüksek ama uydurma riski de vardır. Bu proje
özdünleşimi **görünür** kılar — her cevap hangi katmandan geldiğini açıkça
söyler:

| Katman | Koşul | Davranış | Etiket |
|---|---|---|---|
| 1 — Tam eşleşme | Normalize edilmiş soru kayıt/kural ile birebir | Kayıtlı cevap doğrudan verilir; hiçbir şey üretilmez | 🔎 [KAYITLI BİLGİ ✓] |
| 2 — Kısmi eşleşme | Kelime örtüşmesi (Jaccard) ≥ 0,40 | Eşleşen kaydın kelimeleriyle BEYAZ LİSTE; model yalnız bilinen parçalarla üretir | 🤔 [KISMİ EŞLEŞME] + skor |
| 3 — Açık genelleme | Eşleşme yok | Sinir ağı serbest üretir; açıkça işaretlenir | ⚠️ [DOĞRULANMAMIŞ] |

Bu, endüstrideki RAG (Retrieval-Augmented Generation) yaklaşımının
basitleştirilmiş hâlidir. Güven skoru kullanıcıdan gizlenmez. Terminal ve
Gradio arayüzleri aynı mekanizmayı ve aynı tokenizer/model üretim runtime'ını
paylaşır (eski `legacy/arayuz.py`'nin `intent_cevap`'ı bu katmanın ilkel bir örneğiydi;
artık bilgi kararı `legacy/bilgi_katmani.py`, üretim/yükleme ortaklığı
`hga/ui_runtime.py` üzerinden gelir).

Örnek oturum (eğitilmiş demo modeliyle):

```
👤 Sen: merhaba nasılsın
🤖 AI 🔎 [KAYITLI BİLGİ ✓] (güven 1.00): Merhaba, iyiyim teşekkür ederim. ...

👤 Sen: türkiyenin başkenti neresidir
🤖 AI 🔎 [KAYITLI BİLGİ ✓] (güven 1.00): Türkiye Cumhuriyeti'nin başkenti Ankara'dır.

👤 Sen: su kaç derecede kayar        ← kayıt: "su kaç derecede kaynar"
🤖 AI 🤔 [KISMİ EŞLEŞME] skor=0.60: deniz seviyesinde yüz ... (yalnız kayıttaki kelimelerle)

👤 Sen: kuantum bilgisayar nedir     ← kayıtlarda yok
🤖 AI ⚠️ [DOĞRULANMAMIŞ — sinir ağı üretimi]: ...
```

---

## 🧠 Experience Engine (Knowledge + Experience + Memory)

`hga/` paketi, geometrik çekirdeğin ÜZERİNE eklenen **kendi kendini genişleten
deneyim mimarisidir** ("Hiper Geometrik AI: Experience Engine / Self-Expanding
Knowledge Architecture" yol haritasının fiziksel karşılığı). Saf Python'dur
(torch gerektirmez) ve çekirdeği değiştirmez:

- **Knowledge/Index** (`hga/knowledge/`) — EntityIndex / PropertyIndex /
  RelationIndex / KnowledgeStore. Her kavramın kaynağı (`source`) ve güveni
  (`confidence`) saklanır; entity_id, tokenizer token ID'sinden AYRIDIR.
- **Experience** (`hga/experience/`) — kontrollü kombinasyon üreten Generator,
  7+1 bağımsız sinyalle puanlama, CANDIDATE→VALID/UNCERTAIN/CONFLICT/INVALID
  durum makinesi; kanıt eksikliği (`UNCERTAIN`) ile gerçek kanıt çelişkisini
  (`CONFLICT`) ayıran araştırma yolu ve INVALID→REJECT terminal yolu,
  Conflict→Exploration çözücüsü, konsolidasyon, metin/olay üretimi (v0.2,
  Türkçe ek uyumu: yönelme/belirtme/bulunma/ayrılma + ünsüz yumuşaması +
  ünlü düşmesi + iyelik (6 kişi) + iyelik+durum zinciri + fiil çekimi
  (6 kişi × geçmiş/şimdiki/gelecek/geniş zaman),
  cümle/dosya→üçlü ayıklayıcı + `REAL_DATA` aktarımı, information-gain (v0.3),
  deterministik aritmetik mini-environment (v0.5), çelişki araştırma kuyruğu,
  sürekli öğrenme döngüsü (v1.0), ground-truth benchmark ve dış korpus borusu
  (`veri_toplayici.py` çıktısını sözlük büyütüp `REAL_DATA` olarak akıtır;
  ilişki asla uydurulmaz; elle küratörlü 7 ilişkili sözlük + yumuşama geri
  çevirmeli kök çıkarma) ile çevrimdışı belirleyici korpus üretici
  (`korpus_uretici.py` — ağsız sentetik ölçek provası).
- **Kalıcılık** (`hga/knowledge/persistence.py`) — bilgi tabanını atomik JSON
  olarak kaydet/yükle (VERIFIED bilgi gerçekten kalıcı).
- **Bağımsız kapalı doğrulama** (`hga/experience/dogrulama.py`) — Evaluator
  hiçbir adayı doğrudan VERIFIED yapmaz. Önce `VALID`, sonra bağımsız Verifier
  ile `VERIFYING → VERIFIED/INVALID/UNCERTAIN` geçişi uygulanır. Kontrollü
  aritmetik laboratuvarında N=30 adayın 6'sı VERIFIED, 24'ü INVALID olmuş ve
  bu sabit veri içindeki false acceptance 24→0 ölçülmüştür; bu sonuç genel
  sistem için FAR≈0 iddiası değildir.
- **Tek yüz** (`hga/engine.py` + `python -m hga`) — tüm katmanı yapılandırılabilir
  tek motor + komut satırı arayüzü.
- **Memory** (`hga/memory/`) — Engine'de aktif tam-anahtarlı Dynamic KV,
  bounded eviction + stale-safe replay, atomik persistence/snapshot/versioning,
  compaction ve legacy migration; ayrıca torch seyrek tabloya köprü (v0.6),
  doğrulanmış deneyimleri MODELİN kendi seyrek belleğine bağlayan `NeuralKopru`,
  bilgi yazmanın aşağı-akış etkisini ölçen `AblasyonDeneyi` (boş bellek ~%50,
  bilgi yazılı ~%100) ve bunu modelin KENDİ tamamlama görevine taşıyan
  `GorevAblasyonu` (yoğun gövde donukken ~şans → ~%100, ölü-yol → canlı-yol;
  TABLO-SIFIR: tablo boşaltılınca %100 → %0; HELD-OUT: yazılmamış olgu
  genellemez — "katrilyon" tezinin görev boyutlu kanıtı) ve `GenellemeAblasyonu` (belleğin
  KANONİK kodu eğitimde görülmeyen öznelere de genelliyor: held-out ~%25 → %100,
  boş bellek şansta kalır — genelleme ezberden değil bellekten gelir).

**En kritik güvenlik kuralı:** `MODEL_GENERATED` kaynaklı bir deneyim hiçbir
zaman otomatik `VERIFIED` kabul edilmez — en fazla `VALID` (bellek adayı) olur.
Ayrıca harici kaynak etiketi de tek başına yeterli değildir: `VERIFIED` geçişi
bağımsız doğrulayıcı kimliği (`verified_by`) ister. Konsolidasyon bu kuralı
ikinci kez denetler ve ihlali çelişki günlüğüne yazar.

### Golden benchmark ve sızıntı denetimi

`golden_dataset/` altındaki yedi JSON dosyası elle sabitlenmiştir; benchmark
çalışırken üretilmez. Train/test üçlüleri kanonik SHA-256 izleriyle ayrılır ve
test üçlülerinin train/memory/knowledge/candidate bölümlerine sızması koşuyu
başarısız yapar. Rapor `accuracy`, `precision`, `recall`, `F1`, `FAR`, `FRR`,
confusion matrisi ve veri kümesi hash'ini içerir.

```bash
python -m hga golden-benchmark
python -m hga golden-benchmark --out raporlar/golden.json
python -m hga golden-benchmark --seeds 1,2,3,4,5
```

Çoklu-seed modu her koşuyu atomik `EXP-NNNN` kimliğiyle `experiments/`
altında saklar. Her koşuda `config.yaml`, `manifest.json`, `results.json`,
`stdout.log` ve `model_hash.txt` bulunur. Manifest; Git commit/dirty durumu,
seed, dataset/config/model hash'leri, Python/Torch sürümü, device, parametreler
ve koşu sonucunu taşır. Üretilen `experiments/EXP-*` dizinleri Git'e alınmaz;
yayımlanacak sonuçların ayrıca `raporlar/` altında küratörlenmesi gerekir.

Golden v1 küçük ve deterministik bir semantik sözleşme/regresyon setidir
(N=5 test örneği). Beş seed'de `std=0`, yalnız koşunun tekrarlanabilir olduğunu
gösterir; genel dil başarısı veya istatistiksel model kalitesi kanıtı değildir.

```bash
python experiments/experience_loop/run_full.py        # v0.1 → v1.0 tam demosu
python experiments/experience_loop/run_gercek_veri.py # gerçek veri → temsil → deneyim → doğrulama
python experiments/experience_loop/run_benchmark.py   # kontrollü benchmark + kapalı doğrulama
python experiments/experience_loop/run_ablation.py    # bilgi yazmanın öğrenmeye etkisi (torch)
python experiments/experience_loop/run_gorev_ablasyonu.py # bilgi → modelin tamamlama görevi (torch)
python experiments/experience_loop/run_genelleme_ablasyonu.py # bilgi → görülmeyen olguya genelleme (torch)
python experiments/experience_loop/run_genelleme_olcegi.py  # genelleme × ölçek + gürültü (torch)
python experiments/experience_loop/run_morfoloji.py   # ünlü düşmesi + iyelik + fiil çekimi (6 kişi)
python experiments/experience_loop/run_korpus_boru.py # veri toplayıcı → sözlük büyütme → REAL_DATA
python experiments/experience_loop/run_korpus_olcegi.py # çevrimdışı korpus ölçeği provası
python -m hga bilgi                                   # tek yüz (CLI) demosu
python -m hga dogrulama                               # false accept 24→0
python -m hga graf                                    # Experience Graph & lineage demosu (Faz 23)
python -m hga kesif                                   # Exploration Map & Active Learning seçimi (Faz 24-25)
python -m hga halusinasyon                            # factual consistency metriği
python -m hga sweep                                   # n/K/context kapasite taraması
python -m hga tokenizer                               # mini Türkçe tokenizer benchmark
python -m hga perplexity --tiny                       # küçük modelle perplexity smoke (torch)
python -m hga turkce-lm                               # GERÇEK Türkçe LM: tr_corpus_v1 (1.11M kelime) held-out PPL (torch)
python -m hga checkpoint-rapor checkpoints/temel/latest.pt # checkpoint/model uyumluluğu
python -m hga benchmark-rapor --out raporlar/benchmark_report.json --markdown raporlar/benchmark_report.md
python -m hga research-benchmark                    # birleşik 5-seed araştırma karnesi
python -m hga veri-kalite                             # veri kalite filtresi demo raporu
python -m hga veri-canli-smoke --kontrollu --out raporlar/controlled_data_smoke.json
python -m hga manifest turkce_metin.txt               # veri SHA-256 manifesti
python -m hga observability --out raporlar/observability_panel.json --html raporlar/observability_panel.html
python -m egitim.mini_smoke --out raporlar/mini_training_report.json --markdown raporlar/mini_training_report.md
python tests/test_milestone_v01.py                    # §14'ün 12 maddesi + §15 senaryosu
python tests/test_tokenizer_guvenligi.py              # Türkçe BPE + special token + byte fallback
python tests/test_core_integration.py                 # tokenizer→model uçtan uca zincir (torch varsa)
python tests/test_training_saglamlik.py               # KV cache + padding mask + checkpoint (torch varsa)
python tests/test_egitim_saglamlik.py                 # AMP/checkpoint smoke helper'ları (torch varsa)
```

Ayrıntılı Mimari ve Kod Sınıflandırması:
- `docs/KOD_TABANI_VE_MIMARI_DUZENI.md` — Modül statüleri (Active/Legacy/Experimental) ve tek gerçek kaynak rehberi.
- `docs/EXPERIENCE_ENGINE.md` — Experience Engine mimari notu.
- `docs/EPISTEMIK_BENCHMARK.md` — KNOWN/UNKNOWN/UNCERTAIN/CONFLICT/FALSE protokolü, negatif kontrol kolları ve ölçülmüş UNKNOWN↔UNCERTAIN sınırı.
- `docs/VERIM_METRIKLERI.md` — NY/UEY/GY/VID ayrıştırması ve GY'nin iki kez ölü metrik olarak yakalanıp düzeltilmesi.
- `docs/COK_ADIMLI_VE_UZUN_BAGLAM.md` — zincirleme çıkarım × bağlam yükü ızgarası, bellekten geçen zincir takibi ve ölçülen derinlik sınırı.

---

## 🗂️ Klasör Yapısı

```text
hiper_geometrik_ai/
├── README.md                    # Bu belge
├── pyproject.toml               # TEK bağımlılık/paketleme kaynağı (extras: test, data, ui)
├── requirements.txt             # pyproject'e ince köprü (`--editable .`)
├── requirements-lock.txt        # Referans pin'li ortam
├── test_mimari.py               # 22 duman testi (pytest ile de çalışır)
├── legacy/                      # DONDURULMUŞ eski prototipler (aktif mimari kullanmaz)
│   ├── bilgi_katmani.py         # 3 katmanlı halüsinasyon kontrol prototipi
│   ├── calistir.py              # Terminal sohbet (chatbot)
│   └── arayuz.py                # Gradio sohbet arayüzü (opsiyonel)
├── hga/ui_runtime.py            # Terminal/Gradio ortak tokenizer-model-üretim runtime'ı (KV-cache üretim yolu)
├── raporlar/                    # Küçük smoke/benchmark/observability JSON-MD-HTML çıktıları
├── mimari/
│   ├── kuresel_model.py         # Bütünleşik model + model_olustur (TEK KAYNAK) + kapasite raporu
│   ├── kuresel_bag.py           # Bilinear A@X@B katmanı + K katmanlı zincir
│   ├── encoder.py               # 0→1 katmanı: dış çarpım köprüsü (u ⊗ v)
│   ├── decoder.py               # Bilinear odak merceği okuma katmanı
│   ├── hiper_attention.py       # Nedensel dikkat (SDPA/Flash, Pre-LN, ReZero)
│   ├── seyrek_tablo.py          # Hash'lenmiş seyrek 'boş küme' belleği
│   ├── bpe_tokenizer.py         # BPE alt-kelime tokenizer (varsayılan)
│   ├── tokenizer.py             # Eski kelime-bazlı tokenizer (uyumluluk için duruyor)
│   └── kuresel_loss.py          # CrossEntropy tabanlı loss
├── hga/                         # Experience Engine (saf Python, çekirdeğin üstünde)
│   ├── knowledge/               # Entity/Property/Relation indexleri + KnowledgeStore + versioning/rollback
│   ├── experience/              # Generator, Evaluator, Conflict, Consolidation, Loop, Ledger, Milestone
│   ├── memory/                  # Aktif Dynamic KV + lifecycle + legacy slot + replay
│   ├── evaluation/              # Halüsinasyon, perplexity, golden/kapasite (C_E/C_V), epistemik benchmark, istatistik
│   ├── observability/           # Attention/geometri/bellek/deneyim akışı + panel çıktısı
│   ├── data/                    # Veri kalite filtresi, canlı/kontrollü smoke + SHA-256 manifest
│   └── config/                  # experience_config.yaml + model_config.yaml
├── tests/                       # Knowledge/Experience katmanı testleri (torch'suz çalışır)
├── experiments/experience_loop/ # v0.1 ve v0.1→v1.0 uçtan uca demoları
├── docs/EXPERIENCE_ENGINE.md    # Experience Engine mimari notu
└── egitim/
    ├── egitici.py               # Temel eğitim (AdamW + AMP + grad izleme + checkpoint + early stopping)
    ├── mini_smoke.py            # Gerçek küçük eğitim koşusu + JSON/Markdown raporu
    ├── degerlendirme.py         # Perplexity/loss ölçümü
    ├── determinizm.py           # Seed/deterministik çalışma yardımcıları
    ├── talimat_egitici.py       # Instruction fine-tuning
    ├── talimat_toplayici.py     # Yerleşik talimat seti
    └── veri_toplayici.py        # Korpus toplama (Wikipedia + HF; güvenilir kaynak rehberi)
```

---

## 🚀 Kurulum ve Kullanım

Önerilen ortam: Python 3.11, PyTorch 2.0+ (CUDA kullanacaksanız kurulu CUDA
sürümünüze uygun PyTorch tekerini seçin). Saf Python `hga/` testleri torch
olmadan da çalışır; mimari/eğitim testleri torch varsa gerçeklenir.

Bağımlılıkların **tek gerçek kaynağı `pyproject.toml`**'dur; `requirements.txt`
yalnız ona işaret eden ince bir köprüdür (eski `gereksinimler.txt` kaldırıldı).

```bash
pip install -e .                        # çekirdek (torch, numpy, pyyaml)
pip install -e ".[test]"                # + pytest, ruff, mypy
pip install -e ".[data]"                # + requests, pyarrow, pandas
pip install -r requirements-lock.txt    # tekrarlanabilir referans ortam
```

### 5 Dakikalık Hızlı Başlangıç

```bash
python -m hga bilgi
python -m hga dogrulama
python tests/test_tokenizer_guvenligi.py
```

Bu üç komut; Knowledge/Experience durum makinesini, MODEL_GENERATED→VERIFIED
korumasını ve Türkçe tokenizer doğrulamasını eğitim gerektirmeden gösterir.

**1) Korpus topla** (internet gerekir; opsiyonel `requests` bağımlılığı):

```bash
python -c "import sys; sys.path.insert(0, 'egitim'); \
from veri_toplayici import OtomatikVeriToplayici; \
OtomatikVeriToplayici('.').genis_korpus_cek(hedef_kelime=50000)"
```

**2) Temel eğitim** (BPE sözlüğünü kurar, kilitler; seyrek doluluğu izler):

```bash
python egitim/egitici.py --cag 5
python egitim/egitici.py --config hga/config/model_config.yaml
python egitim/egitici.py --n 128 --katman 2 --batch 32   # küçük/deneysel koşu
python egitim/egitici.py --validation-split 0.1 --early-stopping-patience 3 \
  --warmup-cag 1 --lr-min-factor 0.05 \
  --log-dizini logs/egitim --checkpoint-dizini checkpoints/temel
python egitim/egitici.py --seyrek-satir 2097152          # 2M satırlık seyrek bellek
python egitim/egitici.py --seyrek-yok                    # seyrek bellek kapalı
```

**3) Talimat (instruction) fine-tuning** (temel eğitimle aynı seyrek parametreler):

```bash
python egitim/talimat_egitici.py --config hga/config/model_config.yaml
python egitim/talimat_egitici.py --validation-split 0.1 --early-stopping-patience 3 \
  --warmup-cag 1 --lr-min-factor 0.05 \
  --log-dizini logs/talimat --checkpoint-dizini checkpoints/talimat
```

**4) Sohbet:**

```bash
python legacy/calistir.py   # LEGACY terminal chatbot (3 katmanlı kontrol + etiketler)
python legacy/arayuz.py     # LEGACY Gradio arayüzü (pip install -e ".[ui]")
```

**5) Doğrulama, benchmark ve gözlem raporları:**

```bash
python test_mimari.py
python mimari/kuresel_model.py
python -m hga perplexity --tiny
python -m hga benchmark-rapor --out raporlar/benchmark_report.json --markdown raporlar/benchmark_report.md
python -m hga veri-canli-smoke --kontrollu --konular "Türkçe,İstanbul" \
  --cikis logs/controlled_data_smoke.txt --out raporlar/controlled_data_smoke.json
python -m hga observability --out raporlar/observability_panel.json \
  --markdown raporlar/observability_panel.md --html raporlar/observability_panel.html
python -m egitim.mini_smoke --out raporlar/mini_training_report.json \
  --markdown raporlar/mini_training_report.md
```

Bu branch'te güncel örnek artefaktlar `raporlar/` altında tutulur:

- `mini_training_report.{json,md}` — 1 çağlık gerçek mini eğitim: finite loss,
  finite gradient, CSV+JSONL log, checkpoint uyumluluk ve CPU/GPU bilgisi.
- `benchmark_report.{json,md}` — tokenizer kapsamı, held-out mini perplexity,
  hallucination/factual consistency, seyrek bellek ve cihaz/VRAM fallback.
- `controlled_data_smoke.json` — ağsız/deterministik Türkçe veri hattı smoke;
  kalite filtresi + SHA-256 manifest zincirini kanıtlar.
- `live_data_smoke.json` — canlı Wikipedia smoke denemesi; ağ/kaynak boşsa bunu
  açıkça `insufficient_data` olarak raporlar.
- `observability_panel.{json,md,html}` — bellek haritası, deneyim akışı ve
  Kronecker katman benzerliği paneli.

Çalışma zamanı ağırlıkları/verileri (`bpe_sozluk.json`, `hiper_model_*.pt`,
`turkce_metin.txt`, `talimat_verisi.json`, `logs/`, `checkpoints/`)
`.gitignore`'dadır; repoya girmez.

---

## ⚙️ Donanım Gerçekçiliği

- Varsayılan yapı (n=256, K=4 + 128 MB seyrek tablo) **CPU'da eğitilebilir**
  (2 çekirdekte bile; bu repodaki demo ağırlıklar öyle eğitildi), ama ciddi
  koşular için GPU şart: her bilinear katman başına ~2n³ çarpma yapılır.
- **Karışık hassasiyet (AMP):** CUDA + bf16 destekliyse eğitim motorunda
  otomatik açılır (`--amp hayir` ile kapatılabilir). Eğitim döngüsü loss/logit
  NaN/Inf kontrolünü her adımda yapar.
- **LR scheduler:** temel eğitim ve talimat fine-tuning aynı warmup + cosine
  decay matematiğini kullanır (`--warmup-cag`, `--lr-min-factor`).
- **Logging:** `--log-dizini` CSV'ye ek olarak W&B tarzı satır-satır JSONL
  metrikleri (`metrics.jsonl` / `talimat_metrics.jsonl`) üretir; TensorBoard
  kuruluysa aynı dizine event log da yazılır.
- **Gradient clipping + monitoring:** `--grad-clip` ile toplam norm kırpılır;
  çağ loglarında toplam norm ve en yüksek katman normu raporlanır.
- **Gradient checkpointing:** derin zincirlerde (büyük K) `--checkpoint`.
- **Checkpoint uyumluluğu/resume:** eğitim checkpoint'ları `checkpoint_version`
  ve `model_meta` taşır; `--resume` kayıtlı çağ/epoch'tan devam eder. Şekil/
  anahtar denetimi için `egitim.saglamlik.checkpoint_uyumluluk_raporu` veya
  `python -m hga checkpoint-rapor <ckpt>` kullanılabilir.
- **KV-cache üretim yolu:** attention katmanındaki `forward_cacheli` artık model
  seviyesinde `forward_cacheli_pencere` ile bağlanır; prefix aynı kaldığında
  K/V ve attention çıktıları yeniden kullanılabilir. Sliding-window kayınca
  cache güvenli biçimde yeniden kurulur; çıktı `forward(...)` ile eşdeğer kalır.
  Ortak UI runtime (`metin_uret`) varsayılan olarak bu yolu dener.
- Seyrek tabloyu büyütmek RAM'i doğrusal artırır (20M satır ≈ 2,4 GB); doluluk
  için `doluluk_orani()`, hash çakışması için `carpisma_istatistigi()` kullanın;
  gerekirse `tablo_sayisi=2` (Bloom).
- Çok büyük n/K denemeleri için katmanları farklı cihazlara dağıtmak
  (`torch.distributed` veya manuel `device_map`) yol haritasındadır.

---

## 📜 Bu Sürümde Değişenler (Refactor Günlüğü)

Bu sürüm, karşılaştırmalı inceleme raporundaki **8.1–8.6** ve **9–10**
yol haritalarını uygular:

1. **Tek gerçek kaynak (8.1.1):** `model_olustur` üç dosyadaki kopyasından
   `mimari/kuresel_model.py`'ye indirgendi. Eski kopyalarda `n` parametresi,
   anahtar adları eşleşmediği için **hiçbir zaman modele iletilmiyordu** —
   artık iletiliyor ve `test_fabrika_n_gercekten_gecer` bunu regression
   olarak koruyor.
2. **Ölü/eksik parametreler temizlendi (8.1.2):** eski `kuresel_bag` iki
   bağımsız `Linear`'dı ve `v_yeni` çıktısı hesaplanıp çöpe atılıyordu
   (`mercek_B` hiç eğitilmiyordu). Yeni zincirde her parametre gradyan alır.
3. **Bağımlılık kaynağı `pyproject.toml`'a taşındı (8.1.4):** `.gitignore`'daki `*.txt` /
   `*.json` genel yasakları kaldırıldı. PyTorch başlatma uyarısını önlemek için
   `numpy>=1.26,<2` açık bağımlılık olarak tutulur.
4. **`strict=True` ağırlık yükleme (8.1.5):** uyumsuzluk sessizce yutulmıyor.
5. **Gerçek bilinear/Kronecker mimarisi geri döndü (8.2):** `A @ X @ B`
   sandviçi `torch.einsum` ile; temsil edilen operatörün Kronecker boyutu
   (n⁴) kapasite raporunda açıkça raporlanıyor.
6. **K katmanlı zincir, küçük n (8.3):** `n=256, K=4` varsayılan.
7. **AMP + gradient checkpointing (8.4.1–8.4.2).**
8. **Nedensel dikkat + BPE devreye alındı (8.4.5–8.4.6):** `is_causal=True`;
   BPE varsayılan ve `bpe_sozluk.json`'a kilitli.
9. **Veri kaynakları (8.4.7):** `genis_korpus_cek()` (Wikipedia API) +
   OSCAR/CC-100/mC4 önerileri; `veri_toplayici.py`'deki erişilemez ölü kod
   bloğu temizlendi; `hazirla_veya_yukle` artık gerçekten diskten yüklüyor.
10. **README dürüstleştirildi (8.6.7).**
11. **Entegrasyon onarımları:** `encoder.py` modelin 0→1 katmanı oldu; istem
    biçimi talimat eğitimiyle aynı; bağlam penceresi tüm bileşenlerde
    modelden okunuyor.
12. **Seyrek "boş küme" belleği (9):** `HashlenmisKureselTablo` — kavramsal
    uzayı katrilyonların üzerinde (sözlük^pencere), fiziksel depo sabit;
    başlangıçta tamamı boş, yalnız görülen pencerelerin satırları doluyor;
    `doluluk_orani()` ile çağ başı izleme; Bloom çift hash seçeneği; modelde
    `gen_kopru` ile entegrasyon (köprü sıfır-başlatma tuzağı bilinçli olarak
    önlandı — aksi halde yol ölü kalırdı; `test_model_seyrek_yol_ogreniyor`
    iki adımda canlanmayı doğrular); collision metriği, erişim izleme, LRU
    temizliği ve decay mekanizması eklendi.
13. **3 katmanlı halüsinasyon kontrolü (10):** `BilgiKatmani` + beyaz listeli
    kısıtlı üretim + [DOĞRULANMAMIŞ] etiketleme; güven skoru kullanıcıya
    gösterilir; `norm()` artık kesme işaretini siler ('Türkiye'nin' →
    'turkiyenin' tam eşleşmesi düzeltildi); terminal ve Gradio aynı mekânizmayı
    paylaşır.
14. **Eğitim sağlamlaştırma:** temel ve talimat eğitiminde her adımda
    loss/logit/gradient sonluluk kontrolü, `--grad-clip`, per-layer gradient
    norm özeti, validation split, perplexity, early stopping, `best.pt/latest.pt`
    checkpoint ve CSV/TensorBoard logging.
15. **Config tek kaynağı:** `hga/config/model_config.yaml` + `mimari/model_config.py`
    ile `n`, `K`, bağlam, aktivasyon, seyrek tablo ve eğitim ayarları merkezileşti.
16. **Tokenizer doğrulaması:** Türkçe `İ/I` küçültme düzeltildi, special token
    ve vocab/embedding tutarlılık kontrolleri eklendi, Unicode için byte-level
    fallback açıldı.
17. **Doğrulama ortamları + durum makinesi:** aritmetik doğrulayıcıya ek olarak
    temel modus ponens (`MantikOrtam`) ve property/ilişki kısıtı tutarlılığı
    (`TutarlilikOrtam`) sağlandı; `DeneyimDurumMakinesi` MODEL_GENERATED→VERIFIED
    yükseltmesini engeller ve CONFLICT→EXPLORE / INVALID→REJECT yollarını tanımlar.
18. **Gözlemlenebilirlik + değerlendirme:** attention heatmap/head diversity,
    attention-level KV cache, model-level `forward_cacheli_pencere`, Kronecker
    katman benzerliği, bellek doluluk haritası, deneyim akışı, mini Türkçe
    benchmark harness'i, n/K/context tarama tahminleyicisi ve halüsinasyon/
    factual consistency metrikleri eklendi.
19. **Veri kalite + versiyonlama:** duplicate/spam/bozuk encoding filtreleri
    `hga/data/quality.py`; SHA-256 manifest tabanlı hafif veri sürüm izi
    `hga/data/versioning.py`; canlı/kontrollü veri hattı smoke'u
    `hga/data/live_smoke.py` içinde sağlandı.
20. **Rapor artefaktları:** gerçek mini eğitim smoke'u (`egitim/mini_smoke.py`),
    birleşik benchmark raporu (`python -m hga benchmark-rapor`) ve statik
    gözlemlenebilirlik paneli (`python -m hga observability --html ...`) eklendi;
    örnek çıktılar `raporlar/` altındadır.

**Bilinen kırılma:** mimari değiştiği için önceki sürümlerin checkpoint'ları
(seyrek tablosuz `hiper_model_*.pt`) yeni modele YÜKLENEMEZ (strict yükleyici
bunu açıkça söyler) — modeli yeniden eğitmek gerekir.

---

## 🗺️ Yol Haritası Durumu

Bu branch için önceki “eksik kalan işler” smoke/rapor düzeyinde tamamlandı.
Büyük-ölçek hedefleri hâlâ araştırma ve donanım meselesidir; README bunları
gerçekleşmiş kalite iddiası gibi sunmaz.

| Alan | Durum | Kanıt/komut |
|---|---:|---|
| Eğitim döngüsü stabilitesi | ✅ %100 smoke | `tests/test_training_saglamlik.py`, `egitim/mini_smoke.py`, `metrics.csv/jsonl`, finite loss/grad |
| Checkpoint uyumluluğu/resume | ✅ %100 smoke | `checkpoint_version`, `model_meta`, `python -m hga checkpoint-rapor`, resume testleri |
| Türkçe tokenizer güvenliği | ✅ %100 smoke | özel token, `İ/I`, byte fallback, vocab/embedding tutarlılığı testleri |
| Seyrek bellek metrikleri | ✅ %100 smoke | doluluk, collision, determinism, LRU/aging, kapasite raporları |
| Veri kalite + manifest | ✅ %100 smoke | `veri-kalite`, `manifest`, `veri-canli-smoke --kontrollu` |
| Benchmark/değerlendirme | ✅ %100 smoke | `perplexity --tiny`, `benchmark-rapor`, hallucination/factual consistency |
| Gerçek Türkçe benchmark | ✅ TWT v1 | 4.851 ham insan-anotasyonlu cümle, sabit hash/split, parameter-matched 4 mimari, neural compositional HGA ablasyonu |
| Gerçek Türkçe LM (P1) | ✅ 1.11M kelime held-out | `python -m hga turkce-lm`, `docs/TURKISH_LM.md`: tr_corpus_v1 (UD+Bible+TWT, hash doğrulamalı), belge-ayrık split, train-only BPE, unigram/bigram kontrolleri, parametre-eşli dense/transformer/HGA; `corpus_at_least_1m_words` kapısı full profilde gerçek veriyle PASS |
| Uncertainty calibration | ✅ dev-only T scaling | 4 neural kol × 5 seed; ECE/adaptive ECE, Brier, NLL, AURC, disjoint slices, selective risk |
| Knowledge lifecycle | ✅ gerçek artifact + kontrollü olaylar | ACTIVE/STALE/SUPERSEDED/RETRACTED, same-hash revalidation, dependency propagation, event chain |
| Research Benchmark Suite | ✅ manifestli protokol | `research-benchmark`, 5 seed, JSON/MD/HTML, gerçek TWT + compositional C_G + bölüm bazlı skip/error |
| Observability | ✅ %100 smoke | JSON/Markdown/HTML panel, attention/geometri/bellek/deneyim metrikleri |
| KV-cache entegrasyonu | ✅ %100 smoke | attention cache + model-level `forward_cacheli_pencere` + UI runtime yolu |
| Knowledge/Experience/state machine | ✅ %100 smoke | `MODEL_GENERATED ≠ VERIFIED`, doğrulama ortamları, kapalı validation |
| Aktif Dynamic KV lifecycle | ✅ Research Suite | Engine default, LRU/FIFO/TTL, replay sync, atomik persistence, snapshot/version, compaction, migration |
| 1K→10M memory stress | ✅ 5 seed | Tek streaming geçiş; fixed/Dynamic history recall, active recall, eviction, RSS/throughput |
| GPU/VRAM raporlama | ✅ CPU fallback | raporlar `cuda_available` ve VRAM bilgisini/eksikliğini açık yazar |
| Bilgi sürümleme + rollback | ✅ ölçüldü | `python -m hga bilgi-surum`, `tests/test_knowledge_versioning.py` (K₀→Kₙ, içerik-adresli, geçmiş silinmez) |
| Immutable experience ledger | ✅ ölçüldü | `python -m hga defter`, `tests/test_experience_ledger.py` (append-only, hash-zincirli, tamper-evident) |
| Kapasite çerçevesi (C_E/C_V) | ✅ ölçüldü | `python -m hga kapasite`, `tests/test_capacity_framework.py` (`C_V ≤ C_E ≤ C_M`) |
| 100-cycle milestone tablosu | ✅ 5 seed | `python -m hga milestone`, `docs/MILESTONE_TABLOSU.md` (incorrect=0, EY≈0.11, recall 1.000→0.952) |
| Epistemik benchmark (P0-007) | ✅ ölçüldü + negatif kontrol | `python -m hga epistemik`, `docs/EPISTEMIK_BENCHMARK.md` (yanlış güven 0.000, 3 dejenere kol domine edildi, UNKNOWN↔UNCERTAIN ayrımı `false` olarak raporlanıyor) |
| Verim metrikleri (P1-005) | ✅ 5 seed + %95 GA | `python -m hga verim`, `docs/VERIM_METRIKLERI.md` (EY 0.239 > NY 0.190 > UEY 0.161; GY 0.642 monoton artıyor) |
| İstatistiksel çıkarım (P3) | ✅ bootstrap + etki büyüklüğü | `hga/evaluation/statistics.py` + `tohum-istatistik` core profili: **20 tohum**, 76 eşleşmiş karşılaştırma, min p=2e-6, 8/8 kapı (`docs/SEED_STATISTICS.md`) |
| Legacy izolasyonu | ✅ denetleniyor | `legacy/` paketi + `tests/test_legacy_isolation.py` (aktif katmanda sıfır legacy import) |
| Tek bağımlılık kaynağı | ✅ tamam | `pyproject.toml` (+ `requirements-lock.txt`); `gereksinimler.txt` kaldırıldı |

**Sonraki ölçek işleri (tamamlandı iddiası değildir):** ~~milyon-kelime Türkçe
korpus~~ TAMAMLANDI: `tr_corpus_v1` (1.11M kelime; UD r2.14 ×8 + Bible CC0 +
TWT, hash doğrulamalı, `hga/evaluation/datasets/tr_corpus_v1/`) `turkce-lm`
full profilinde koşar ve `corpus_at_least_1m_words` kapısını gerçek insan
metniyle açar; smoke profil TWT üzerinde kalır ve kapı orada bilinçli FAIL'dir.
Kalanlar: 10M+ kelime ölçeği, gerçek instruction set büyütme, 64+ token uzun
bağlam, katman-bazlı çoklu GPU/model paralelliği ve sohbet kalitesi için insan
değerlendirmesi.

**Milestone tablosunun açığa çıkardığı ve artık aktif yola taşınan iş:** 100
döngüde bilgi kalitesi korunurken fixed-slot recall `1.000 → 0.952`'ye düştü.
Bu nedenle Dynamic KV yalnız karşılaştırılmadı; aktif Engine deneyim/replay
deposu yapıldı. Sorun gizlenmedi: fixed sonuçları raporda korunur, bounded
Dynamic KV'nin kapasite kaybı ise collision yerine eviction olarak ölçülür.

---

## 🔬 Matematiksel Arka Plan (kısaca)

- **Bilinear sandviç:** `Y = A @ X @ B`. Flatten uzayında
  `vec(Y) = (Bᵀ ⊗ A) vec(X)` — katmanın tam operatörü iki merceğin
  **Kronecker çarpımıdır**: n⁴ girdilik operatör, 2n² parametreyle.
- **0→1 katmanı:** `X = tanh(u ⊗ v)` — iki vektörün dış çarpımı n² "sanal
  köşe" üretir; eleman bazlı tanh rank-1 kısıtını kırar.
- **Seyrek bellek:** pencere → h(pencere) → `tablo[h % M]`; kavramsal uzay
  sözlük^pencere, fiziksel depo M satır. Boş küme = sıfır vektör; gradyanı 0
  olan satır optimize edicide değişmez.
- **Okuma:** `M₁ @ X @ M₂` odak merceklerinden satır/sütun ortalamalarıyla
  çift yönlü okuma; softmax bilinçli olarak YOKTUR.
- **Dürüst sınır:** Zincir bileşkesi yine bir (n² × n²) doğrusal operatördür;
  K katman ona 2Kn² gerçek serbestlik katar. n^(2K) "etkileşim uzayı" ve
  sözlük^pencere "anahtar uzayı" geometrik büyüme anlatısının üst
  sınırlarıdır — gerçek parametre sayısı değildir.
