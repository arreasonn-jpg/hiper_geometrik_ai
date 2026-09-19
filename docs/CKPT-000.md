# CKPT-000 — v0.1.0-prototype baseline

**Durum:** current · **Amaç:** sonraki her değişimin karşılaştırma noktası ·
**Makine-okunur manifest:** [`checkpoints/CKPT-000.json`](checkpoints/CKPT-000.json)

Bu checkpoint bir modelin genel zekâ, ölçeklenebilirlik veya SOTA iddiası
olduğunu söylemez. Küçük ölçekli HGA prototipinin kod, veri kökeni, deney
sözleşmesi ve dürüst kapasite ayrımlarını sabitler. `v0.1.0-prototype` Git
etiketi bir **araştırma checkpoint** etiketidir; depodaki mevcut Python dağıtım
sürümünü geriye doğru yeniden numaralandırmaz.

## Sabit sözleşme

| Alan | CKPT-000 değeri |
|---|---:|
| Yoğun gerçek parametre | 6.970.433 (~6,97M) |
| Seyrek fiziksel depo | 33.554.432 parametre (1.048.576 × 32) |
| Toplam gerçek eğitilebilir parametre | 40.524.865 (~40,5M) |
| Varsayılan seed seti | 1, 2, 3, 4, 5 |
| İstatistik raporu | ortalama ± popülasyon standart sapması |
| Repo lisansı | Apache-2.0 |

Kapasite raporlaması üç farklı büyüklüğü ayırmak zorundadır:

- **P:** RAM/VRAM'de bulunan gerçek, eğitilebilir parametreler;
- **C_I^UB:** Kronecker zincirinin etkileşim/operatör üst sınırı;
- **C_M^UB:** seyrek belleğin kavramsal adres üst sınırı.

`C_I^UB` ve `C_M^UB` parametre sayısı değildir. Bu nedenle bu baseline, “1
katrilyon gerçek sinaps/parametre” iddiası taşımaz.

## Kanıt artefaktları

- Bilinear zincir: [`mimari/kuresel_bag.py`](../mimari/kuresel_bag.py) içindeki
  `A @ X @ B` uygulaması; mimari duman testi bunu doğrudan hesaplanan sonuçla
  karşılaştırır.
- Hash'lenmiş seyrek “boş küme” belleği:
  [`mimari/seyrek_tablo.py`](../mimari/seyrek_tablo.py). Fiziksel tablo baştan
  ayrılır ve sıfırla başlar; yalnız erişilen slotlar eğitimle değişir.
- Tek komutlu Research Benchmark Suite:
  `python -m hga research-benchmark`. Her seed için atomik `EXP-NNNN` dizini,
  `config.yaml`, `manifest.json`, `results.json`, `stdout.log` ve model hash'i
  üretir. Ayrıntılı sözleşme:
  [`RESEARCH_BENCHMARK_SUITE.md`](RESEARCH_BENCHMARK_SUITE.md).
- TWT v1: upstream revision
  `40838e5cbe3f2882d4e768a3d782e6219e50b52a` sabittir. `web.conllu` ve
  `wiki.conllu` SHA-256 değerleri,
  [`PROVENANCE.json`](../hga/evaluation/datasets/twt_v1/PROVENANCE.json)
  içinde kayıtlı ve yükleyici tarafından doğrulanır.

## Kapsayıcı ile yeniden üretim

Referans CPU ortamı Python 3.11.9 ve `requirements-lock.txt` ile kurulur.
Araştırma çıktısını image katmanına değil, bind-mounted bir dizine yazın:

```bash
docker build -t hga:v0.1.0-prototype .
mkdir -p artifacts
docker run --rm \
  -v "$PWD/artifacts:/artifacts" \
  hga:v0.1.0-prototype research-benchmark \
  --experiment-root /artifacts/experiments \
  --out /artifacts/research_report.json \
  --markdown /artifacts/research_report.md \
  --html /artifacts/research_report.html
```

Bu komut varsayılan olarak beş seed çalıştırır. Ortam bilgisi (Python, Torch,
CPU/RAM, CUDA aygıtı varsa CUDA sürümü), veri ve konfigürasyon hash'leri her
`manifest.json` dosyasında korunur. CPU/GPU ve işletim sistemi farkları nedeniyle
sayısal eğitim çıktılarının bit-düzeyinde özdeşliği iddia edilmez; tohum,
girdi, sürüm ve sonuç sözleşmesi denetlenebilir durumdadır.

## Tamamlanan genişletmeler ve dürüst kalan sınırlar

CKPT-000 kapsamı aşağıdaki yeni, doğrulanabilir artefaktlarla genişletildi:

- **Teori/rank:** [`THEORETICAL_FRAMEWORK.md`](THEORETICAL_FRAMEWORK.md),
  Kronecker rank özdeşliğini, aktivasyonsuz zincir çöküşünü, manifold boyutunu
  ve VC dil sınırını ayırır. Tam zincir için VC/pseudo-dimension sonucu yoktur;
  bu alan `NOT_ESTABLISHED` olarak işaretlenir, uydurma sayı verilmez.
- **İngilizce + iki dilli kapsama:** hash/doğrulamalı UD English EWT v1,
  Türkçe TWT v1 ile birlikte
  [`BILINGUAL_BENCHMARK.md`](BILINGUAL_BENCHMARK.md)'de tanımlıdır. EWT'nin
  CC-BY-SA-4.0 lisansı, upstream attribution'ı ve SHA-256 kaynak sözleşmesi
  [`ewt_v1/PROVENANCE.json`](../hga/evaluation/datasets/ewt_v1/PROVENANCE.json)
  içinde pakete dahil edilir.
- **Transformer/BERT/GPT-style kontroller:**
  [`ENGLISH_EWT_BASELINES.md`](ENGLISH_EWT_BASELINES.md) beş seed üzerinde
  dense, Transformer, BERT-style ve GPT-style küçük mimarileri raporlar.
  BERT/GPT-style modeller pretrained checkpoint değildir. Bu bölüm artık
  varsayılan `research-benchmark` çağrısında her suite seed'i için yer alır;
  beş-seed gereksinimi üst-suite manifest agregasyonunda denetlenir.
- **Ölçek deneyi:** gerçek HGA çekirdeği için
  [`HGA_ENGLISH_SCALING_PROBE.md`](HGA_ENGLISH_SCALING_PROBE.md) üç boyut ve
  beş seed içerir. 3,28× aralık bir scaling law tanımlamak için yetersizdir;
  rapor bunu özellikle `EXPLORATORY_NOT_A_SCALING_LAW` olarak işaretler.
- **Preprint taslağı:** [`paper/main.tex`](../paper/main.tex) ve
  [`paper/README.md`](../paper/README.md), hiçbir uydurma yazar ya da yayın
  durumu içermeyen inceleme taslağıdır.
- **İnceleme/submission kapısı:** [`PUBLICATION_CHECKLIST.md`](PUBLICATION_CHECKLIST.md)
  ve [`CONTRIBUTING.md`](../CONTRIBUTING.md), dış yeniden üretim, gerçek author
  onayı ve bağımsız teknik inceleme için yapılması gerekenleri ayırır.

Kod ile dürüstçe kapatılamayan dış sınırlar şunlardır: geniş çok dilli
pretraining/cross-lingual transfer, pretrained BERT/GPT checkpoint kıyası,
geniş compute-data-model grid'iyle scaling-law analizi, gerçek yazar onayıyla
arXiv yükleme ve bağımsız hakem değerlendirmesi. Tek geliştiricili risk CI,
manifest ve test sözleşmeleriyle azaltılabilir; kod tarafından yok edilemez.

Bu sınırları aşan çalışma CKPT-000'u yeniden yazmak yerine yeni manifest ve
karşılaştırmalı sonuçla kaydedilmelidir.
