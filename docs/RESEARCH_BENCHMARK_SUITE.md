# HGA Research Benchmark Suite

`hga-research-benchmark-v1`, dağınık deneyleri tek bir manifest ve rapor
sözleşmesinde toplar. Amaç yeni bir model özelliği eklemek değil, mevcut
iddiaları aynı komutla ölçmek ve eksik/atlanmış ölçümleri görünür kılmaktır.

```bash
python -m hga research-benchmark
```

Varsayılan davranış:

- `smoke` profilini çalıştırır,
- `1,2,3,4,5` seed'lerini kullanır,
- her seed için ayrı `experiments/EXP-NNNN/` manifesti üretir,
- çalışma dizinine `research_report.json`, `research_report.md` ve
  `research_report.html` yazar.

Yollar ve profil açıkça değiştirilebilir:

```bash
python -m hga research-benchmark \
  --profile full \
  --seeds 1,2,3,4,5 \
  --experiment-root experiments \
  --out raporlar/research_report.json \
  --markdown raporlar/research_report.md \
  --html raporlar/research_report.html
```

Yalnız belirli protokolleri çalıştırmak için:

```bash
python -m hga research-benchmark \
  --sections compositional-generalization,verification,ood
```

`--strict`, optional bir bölüm atlanırsa veya bir bölüm hata verirse komutu
non-zero durumla kapatır. Strict verilmediğinde atlama **başarı sayılmaz**;
raporda `SKIPPED` ve suite sonucunda `COMPLETED_WITH_SKIPS` olarak kalır.

## Bölümler

| Bölüm | Ölçüm | Temel sınır |
|---|---|---|
| Architecture | P / C_I / C_M ayrımı ve config tabanlı parametre muhasebesi | Parametre sayısı formül tahminidir |
| Kronecker | aktivasyonsuz çöküş, SiLU ile kırılma, eşit-parametre iki öğretmen | Sentetik öğretmen; Transformer değildir |
| Memory | fixed exact-ID + forced collision + Dynamic KV + ölçek maliyeti | Semantic/learned retrieval değildir |
| Verification | golden + fault injection + proof attack + gerçek-artifact STALE lifecycle | Theorem prover veya canlı-web freshness ölçümü değildir |
| Compositional Generalization | seen/unseen entity/relation/combination/wording/sentence | Küçük, elle sabit fixture |
| Neural / Symbolic / Hybrid | aynı split üzerinde coverage ve hata metrikleri | Prosedürel sentetik görev |
| Self Learning | kapalı aritmetik kontrol + shared-store arithmetic/logic/consistency multi-environment izolasyonu | Neural/genel dil self-learning değildir |
| OOD | kanıt yokluğunda `UNCERTAIN ≠ FALSE` | Tek mini OOD vaka |
| Turkish NLP | 4.849 etkin gerçek TWT cümlesi + parameter-matched dört neural kol + dev-only uncertainty calibration + tokenizer smoke | Structured morphosyntactic görevdir; end-to-end parser/semantik extraction değildir |
| Reproducibility | seed sayısı, manifest/hash tutarlılığı ve skor dağılımı | Byte-identical koşu zorunlu değildir |

PyTorch proje bağımlılığıdır ve kurulu referans ortamda Kronecker ile
neural/symbolic/hybrid bölümleri de `COMPLETED` olmalıdır. Eksik/bozuk bir
kurulumda suite bunu başarı gibi göstermemek için açıkça `SKIPPED` raporlar;
`--strict` böyle bir koşuyu non-zero kapatır.

## Verifier adversarial fixture

`hga/evaluation/datasets/verifier_adversarial_v1.json`, çalışma anında
üretilmeyen şu proof sınıflarını içerir:

- geçerli kontrol ve negatif/boundary vakaları,
- false proof,
- incomplete proof,
- contradictory proof,
- malformed proof,
- adversarial bool-as-int, sınır aşımı ve fazla alan girdileri,
- verifier'ın desteklemediği kural için `UNCERTAIN` kontrolü.

Rapor `FAR`, `FRR`, precision, recall, F1, coverage ve attack robustness
metriklerini birlikte verir. Desteklenmeyen iyi biçimli kuralı yanlış diye
uydurmaz; `UNCERTAIN` döndürür ve coverage düşüşü görünür kalır. Bu katı tam
sayı toplama doğrulayıcısı genel theorem prover veya formal verification
iddiası taşımaz.

Aynı Verification bölümü `real-artifact-knowledge-lifecycle-v1` ile iki gerçek,
hash-doğrulanmış TWT kaynağında expiry, same-hash revalidation, changed-revision
supersession, outage, explicit retraction ve transitive dependency staleness
kontrollerini yürütür. `STALE` hiçbir zaman `FALSE` veya `RETRACTED` olarak
sayılmaz; default kullanım dışına alınırken audit görünürlüğü korunur. Olaylar
SHA-256 parent-chain defterindedir. Kaynak dosyaları gerçektir, yaşam döngüsü
olayları kontrollüdür; canlı upstream değişiklik iddiası yoktur. Ayrıntılar:
`docs/KNOWLEDGE_LIFECYCLE.md`.

## Memory: fixed tablo ve Dynamic KV

Memory bölümü artık dört kontrolü aynı seed altında birlikte çalıştırır:

1. fixed-table streaming exact-ID recall/collision,
2. kasıtlı aynı-slot saldırısında `FIRST_WINS`, `LAST_WINS`, `DYNAMIC_KV`,
3. context ölçeği büyürken fixed recall ile dinamik depolama maliyeti,
4. aktif `ExperienceEngine` belleğinde bounded eviction, replay tutarlılığı,
   persistence, snapshot/restore, sürüm monotonluğu, journal compaction,
   tamper rejection ve legacy migration (`active-dynamic-kv-lifecycle-v1`).

`hga.memory.DynamicKVMemory` tam üçlü anahtarı fiziksel kimlik olarak kullanır;
31-bit hash yalnız gözlem adresidir. Böylece benchmarktaki Dynamic KV ile aktif
Engine aynı implementasyondur. `ExperienceEngine` varsayılan olarak
`DYNAMIC_KV` kullanır; `FIRST_WINS` yalnız açık legacy/migration seçeneğidir.
Kapasite dolunca deterministik LRU (veya açıkça seçilirse FIFO), opsiyonel idle
TTL eviction uygulanır. Evicted/overwrite edilmiş anahtarın stale replay kopyası
aynı işlemde çıkarılır.

Snapshot şema v2, store/record version, parent snapshot hash, kayıtlar, sayaçlar
ve sıkıştırılabilir mutation journal'ını taşır. JSON persistence geçici dosya +
`fsync` + `os.replace` ile atomiktir ve içerik SHA-256 değişikliği yüklemede
reddedilir. Compaction logical kayıtları değiştirmeden eski upsert/tombstone
olaylarını etkin kayıt başına teke indirir. Legacy FIRST_WINS migration yalnız
replay payload'ı mevcut ve kaynak slotta gerçekten korunmuş kayıtları taşır;
collision ile daha önce kaybolan bilgiyi uydurmaz, `missing_from_source` ve
fidelity olarak raporlar. Eski minimal persistence schema v1 yüklemede v2'ye
migrate edilir.

Dynamic KV iki hash-çakışan tam anahtarı birlikte korumalı ve unbounded ölçek
kıyaslarında exact recall `1.0` vermelidir. Bunun karşılığında fiziksel maliyet
girdi sayısıyla büyür; bounded Engine modunda ise collision yerine açık eviction
kaybı vardır. Suite'in Memory diagnostic skoru bilinçli olarak fixed-table
recall olarak kalır; aktif Dynamic KV kontrolünün başarısı mevcut sabit belleğin
zayıflığını maskelemez. Bu exact lookup'tır; semantic/learned retrieval değildir.

Uzun 1K→10M koşusu, normal smoke suite süresini yaklaşık 30 dakika uzatmamak
için aynı üst-katman CLI'nin `memory-benchmark --active-dynamic` modundadır.
Temiz `94fe3f6` üzerinde 5 seed tamamlanmış manifestli sonuç
`raporlar/memory_stress_1k_10m.{json,md}` dosyalarındadır. 65.536 bounded
capacity ile 10M'de aktif-küme audit recall `1.0`, fakat tüm-geçmiş recall
`0.0065536` ve eviction `9.934.464` olmuştur; active recall history recall gibi
sunulmaz.

## Multi-environment self-learning

Self Learning bölümü tek aritmetik kontrolü korur ve ek olarak
`multi-environment-closed-verified-self-learning-v1` protokolünü her seed'de
çalıştırır. Arithmetic, modus-ponens logic ve property-consistency adayları
ayrı namespace/holdout/verifier ile dengeli interleave edilir; shared
KnowledgeStore ve aktif Dynamic KV kullanılır. Cross-verifier acceptance,
cross-namespace durable fact, yanlış durable knowledge ve üç ayrı holdout
leakage kanalı sıfır; her environment growth ve shared-memory exact recall
pozitif olmak zorundadır. Bu epistemik yaşam döngüsü testidir; neural model
öğrenmesi veya gerçek dünya self-learning değildir.

## Gerçek Türkçe benchmark: Turkish Web Treebank v1

Turkish NLP bölümü artık bir mini corpus sonucu yerine, sabitlenmiş **Turkish
Web Treebank (TWT)** üzerinde ana skorunu üretir. Vendored ham dosyalar ve
provenance paketi `hga/evaluation/datasets/twt_v1/` altındadır.

- Kaynak: Turkish forum/blog/how-to/review/guide sayfaları ve Türkçe Wikipedia.
- Upstream revision: `40838e5cbe3f2882d4e768a3d782e6219e50b52a`.
- Lisans: `Apache-2.0`; lisansın SHA-256 doğrulanan kopyası pakettedir.
- İnsan anotasyonu: segmentation, morphology, POS, basic dependency head ve
  44 dependency relation.
- Ölçek: 4.851 ham cümle, 66.466 sözcük, 81.370 inflectional-group token.
- Etkin split: 3.881 train / 484 dev / 484 test.

Split, upstream TWT API kuralını her kaynak dosyada ayrı uygular: `%10 < 8`
train, `%10 == 8` dev, `%10 == 9` test. Ham kayıtlar değiştirilmez. Train/dev
exact duplicate olan bir cümle ile train/test lemma-set Jaccard değeri `0.8333`
olan bir yakın cümle sabit kimliklerle quarantine edilir. Etkin splitlerde
sentence ID, normalize yüzey ve sabit `0.80` eşiğindeki lexical-semantic
Jaccard proxy denetimleri temiz olmak zorundadır. Proxy, semantik eşdeğerlik
kanıtı olarak sunulmaz.

Görev **basic morphosyntactic dependency-arc candidate verification**'dır.
İnsan anotasyonlu `(dependent, relation, head)` arc'ı `VALID`; aynı token için
başka bir legal head seçen deterministik bozulma, basic tree'nin single-head
sözleşmesi nedeniyle `INVALID`'dır. Accuracy, precision, recall, F1, FAR, FRR,
selective accuracy ve coverage birlikte raporlanır. Test ayrıca şu saf dilimleri
taşır:

- `entity_disjoint`: en az bir gold lemma/form token kimliği train dışında,
  relation train'de;
- `relation_disjoint`: iki endpoint train'de, `csubj`/`parataxis` relation'ı
  model-visible train dışında;
- `composition_disjoint`: entity ve relation bileşenleri görülmüş, exact gold
  bileşim görülmemiş;
- `wording_disjoint`: gold bileşim görülmüş, normalize cümlenin tamamı train'de
  görülmemiş;
- `sentence_disjoint`: sabit test sentence ID'si train/dev dışında.

Her çalışmada kaynak, dataset, config, split ve candidate SHA-256 değerleri
rapora girer. Selective POS/relation/direction frequency verifier yalnız veri ve
metrik hattının sağlık baseline'ıdır. Özellikle relation-disjoint diliminde
abstention coverage'ı `0.0` olur ve bu başarı sayılmaz. TWT relation'ları
**morphosyntactic dependency relation**'dır; semantik knowledge triple değildir.
“Entity” de NER etiketi değil, split için kullanılan token kimliğidir.

Mini tokenizer round-trip ve kontrollü compositional parser sonuçları geriye
dönük smoke ölçümü olarak tutulur, ancak gerçek TWT F1 ana Turkish NLP skorunu
şişirmez. Dataset kartı, citation, değişiklik bildirimi ve tüm hashler için:
`hga/evaluation/datasets/twt_v1/DATASET_CARD.md` ve `PROVENANCE.json`.

### Parameter-matched Dense / Transformer / Kronecker / HGA

Aynı Turkish NLP bölümü `twt-parameter-matched-architectures-v1` harness'ını
her seed'de gerçek PyTorch ile çalıştırır. Ortak structured girdi altı alandan
oluşur: dependent lemma/form kimliği, dependent UPOS, relation, head lemma/form
kimliği, head UPOS ve arc direction/distance. Vocabulary yalnız model-visible
train splitinden kurulur; relation holdout etiketleri train ve dev'den çıkarılır.
Bu encoder repository BPE tokenizer'ından ayrıdır ve tokenizer kodunu değiştirmez.

Adillik kapıları:

- dört kol aynı dataset/split/candidate SHA-256 değerlerini kullanır;
- aynı train-only vocabulary ve aynı başlangıç embedding matrisi kullanılır;
- aynı seed'e ait dengeli batch indekslerinin SHA-256 değeri dört kolda aynıdır;
- optimizer `AdamW`, loss `CrossEntropyLoss`, adım, batch, learning rate,
  weight decay, gradient clipping ve karar eşiği aynıdır;
- gerçek `requires_grad` parameter `numel` sayıları toplamda max/min `≤1.01`,
  ortak embedding çıkarıldıktan sonra architecture body'de `≤1.05` olmalıdır;
- bütçe doldurmak için unused/reserve parameter eklenmez ve her trainable
  parameter'ın gradient aldığı doğrulanır;
- Accuracy, precision, recall, F1, FAR, FRR ve coverage, TWT'nin tüm disjoint
  dilimlerinde her model için ayrı raporlanır.

Kolların gerçek implementasyonları:

| Kol | Gövde |
|---|---|
| Dense | `Embedding → Linear → GELU → Linear` |
| Transformer | `torch.nn.TransformerEncoder` |
| Kronecker | `mimari.kuresel_bag.KureselZincir` |
| HGA | `HiperGeometrikAttention → GeometrikVeriEncoder → KureselZincir → FraktalDecoder` |

HGA kolu repository'nin aktif attention/outer-product/bilinear/fraktal
çekirdeğini kullanır; yalnız ismi HGA olan bağımsız bir MLP değildir. Buna
karşın bu harness tam autoregressive `HiperGeometrikAI` language-model
pretraining'i veya ham cümleden end-to-end dependency parsing değildir. Fiziksel
parametre adilliği FLOP, aktivasyon belleği ve duvar süresi eşitliği anlamına
gelmez; bunlar ayrıca raporlanır. Smoke profil 32 ortak adımla protokol/regresyon,
full profil 256 ortak adımla daha uzun optimizasyon içindir; ikisinden de SOTA
veya genel dil üstünlüğü sonucu çıkarılmaz.

### Dev-only uncertainty calibration

Aynı dört neural kolun scalar temperature değeri yalnız sabit dev logits ve
NLL ile seçilir; test labels sıcaklık veya abstention threshold seçiminde
kullanılmaz. NLL, Brier, fixed/adaptive ECE, AURC ve önceden tanımlı
risk-coverage noktaları overall ile tüm disjoint dilimlerde raporlanır.
Temperature argmax kararını korumak zorundadır. Test metriklerinin iyileşmesi
kabul kapısı değildir; aksi test seçimine yol açardı. Ayrıntılı sözleşme:
`docs/UNCERTAINTY_CALIBRATION.md`.

### Neural compositional HGA ablasyonu

`twt-neural-compositional-ablation-v1`, aynı train-only vocabulary, başlangıç
state'i, dengeli batch schedule hash'i, AdamW/loss/adım ve gerçek TWT
composition-disjoint test diliminde şu dört kolu ölçer:

| Kol | Müdahale |
|---|---|
| `full` | Aktif HGA attention + çarpımsal dış-çarpım + Kronecker zincir |
| `no_attention` | `HiperGeometrikAttention` modülü fiziksel olarak çıkarılır |
| `additive_geometry` | aynı `proj_u/proj_v` parametreleri korunur; `u⊗v` yerine `u+v` matrisi kullanılır |
| `no_kronecker_chain` | `KureselZincir` fiziksel olarak çıkarılır |

Ortak kalan tensorlar full kolun başlangıç state'inden bire bir kopyalanır ve
byte-eşitliği eğitimden önce doğrulanır. Gerçek component removal parametre
sayısını doğal olarak azaltır; fark unused/reserve parameter ile doldurulmaz.
`full` ile `additive_geometry` trainable `numel` olarak tam eşittir. Her kalan
trainable parameter gradient almak zorundadır.

Ayrı ölçüm:

```text
C_G_N = gerçek TWT testindeki dengeli composition_disjoint arc adaylarında accuracy
```

`C_G_N`, kontrollü fixture'ın mevcut `C_G` ölçümünün yerine geçmez ve teorik
kapasite değildir. JSON raporu her kol için Accuracy/F1/FAR/FRR/coverage,
parametre, süre, seed sonucu ve `full - ablation` farklarını; Markdown raporu
5-seed mean ± std tablosunu saklar. Bu, structured morphosyntactic arc
genellemesidir; semantik compositional reasoning veya genel dil kanıtı değildir.

## Türkçe compositional fixture

Veri `hga/evaluation/datasets/compositional_tr_v1.json` içindedir ve çalışma
anında üretilmez. Entity, relation, train ve test kayıtları ile beklenen
epistemik durumlar elle sabitlenmiştir. Dataset'in kanonik SHA-256 özeti hem
suite raporuna hem deney manifestlerine girer.

Merkez örnek:

```text
Train:
  Ali ata bindi.
  Ayşe arabaya bindi.
  Mehmet otobüse bindi.

Held-out:
  Ali arabaya bindi.
  Ali otobüse bindi.
```

Protokol üç kanalı ayrı ölçer:

1. **Aday + karar:** Generator görülmeyen üçlüyü üretir mi, Evaluator doğru
   `VALID/INVALID/UNCERTAIN` durumuna koyar mı?
2. **Yüzey üretimi:** ilişkiye özel morfolojik üretici beklenen Türkçe cümleyi
   exact-match ile kurar mı?
3. **Yüzey ayıklama:** train'de olmayan wording, küratürlü parser tarafından
   aynı kanonik üçlüye bağlanır mı?

Semantik olarak yeni test kayıtlarının train üçlüleriyle SHA-256 çakışması
koşuyu durdurur. `seen_composition` ve `unseen_wording` kayıtlarındaki kasıtlı
semantik tekrar bu denetimin dışında, ayrı kontrol grubu olarak tutulur. Exact
normalize cümle sızıntısı ayrıca denetlenir.

### `C_G`

Bu suite'te:

```text
C_G = doğru çözülen uygun held-out VALID örnek / uygun held-out VALID örnek
```

Uygun örnek en az bir `unseen_entity`, `unseen_relation`,
`unseen_combination` veya `unseen_sentence` boyutu taşır. JSON raporunda hem
sayı (`correctly_generalized`) hem oran (`score`) saklanır.

`C_G` burada:

- teorik üst sınır değildir,
- `C_M`, `C_E` veya `C_V` ile büyüklük eşitsizliğine sokulmaz,
- genel Türkçe performansı değildir,
- sabit benchmark protokolünde ölçülmüş görev skorudur.

Unseen entity'nin ontoloji kaydı/özellikleri, unseen relation'ın ilişki şeması
sisteme verilir. Dolayısıyla bunlar sıfırdan entity discovery veya bilinmeyen
ilişki indüksiyonu iddiası değildir.

## Rapor ve skor sözleşmesi

Her bölüm:

```json
{
  "status": "COMPLETED",
  "score": 1.0,
  "checks": {},
  "metrics": {},
  "limitations": [],
  "elapsed_seconds": 0.01
}
```

üretir. Çoklu-seed üst raporu `score_mean`, `score_std`, minimum ve maksimumu
saklar. `overall_diagnostic_score`, yalnız tamamlanan bölüm skorlarının basit
ortalamasıdır. **Zekâ, dil kalitesi veya SOTA skoru değildir.** Heterojen
metrikleri tek panelde hızlı karşılaştırmak için diagnostic bir göstergedir.

## Manifest sözleşmesi

Her `EXP-NNNN/manifest.json` en az şunları taşır:

- Git commit ve dirty durumu,
- dataset/config/model hash,
- seed,
- Python ve Torch sürümü,
- CPU, CPU sayısı, toplam RAM,
- GPU, CUDA ve cuDNN bilgisi (yoksa `null`),
- parametre sayısı ve sayım türü,
- toplam/eğitim/inference süre alanları,
- `RUNNING → COMPLETED | FAILED` sonucu.

Alt benchmark için süre uygulanmıyorsa alan silinmez, `null` kalır. Böylece
manifest şeması her ortamda sabittir.
