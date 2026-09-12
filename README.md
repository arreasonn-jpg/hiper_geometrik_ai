# hiper_geometrik_ai

Türkçe için **küçük, kelime düzeyli, pencere tabanlı bir sonraki-token dil modeli** ve
bunun üzerine kurulu **kural tabanlı + sinir ağı hibrit** bir sohbet arayüzü.

> **Dürüst boyut notu:** Bu proje ~13,5 milyon parametrelik küçük bir modeldir;
> "katrilyon sinaps", Tesseract/120-hücreli projeksiyonu veya 4B hiper-küp hesabı
> **içermez**. "Hiper-geometrik" adı, katman adlandırmasından (`kuresel_bag`,
> `mercek`, izdüşüm) kalan deneysel bir isimdir. Bu README, kodda gerçekten olanı
> anlatır: iyi tanımlı küçük bir dil modeli + Türkçe sohbet denemesi.

## Mimari (kodda gerçekten olan)

`HiperGeometrikAI` (`mimari/kuresel_model.py`), sabit 8 kelimelik bir bağlam
penceresinden bir sonraki kelimeyi tahmin eder:

1. **Gömme + konum:** `nn.Embedding(sözlük, 64)` + öğrenilen konum parametresi
2. **Dikkat:** `HiperGeometrikAttention` (`mimari/hiper_attention.py`) — fused-QKV,
   Pre-LN, ReZero (`alpha=0` başlangıçlı) ve `F.scaled_dot_product_attention`
   (donanım destekliyorsa Flash Attention çekirdeği; `is_causal` parametrelidir)
3. **İki izdüşüm:** pencere düzleştirilir; `u_kure` ve `v_kure` ile `n` boyutuna
   taşınır, L2-normalize edilir ve çapraz kapılarla (sigmoid) ölçeklenir
4. **Küresel bağ:** `OptimizeEdilmisKureselBag` — iki `(n×n)` lineer "mercek"in
   SiLU çıktıları **toplanarak birleştirilir** (toplam `2·n²` parametre)
5. **Decoder:** LayerNorm + sözlük boyutuna lineer katman (logits)

Toplam ~13,5M parametre (n=1000, 8000 kelimelik sözlükle). Bağlam penceresi 8 kelime
olduğu için model klasik n-gram modellere yakındır; uzun ve tutarlı metin üretimi
beklenmemelidir (bkz. [Sınırlamalar](#sınırlamalar)).

Sohbet tarafı hibrittir: bilinen soru kalıplarına kural tabanlı (intent) ve
Jaccard benzerlikli eşleşme (`arayuz.py`), eşleşme yoksa sinir ağı üretimi
(top-k örnekleme + tekrar cezası, `ortak.metin_uret`).

## Kurulum

Python 3.10+ önerilir.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r gereksinimler.txt
```

Bağımlılıklar: `torch`, `gradio`, `requests`, `pyarrow`, `pandas`
(`numpy` doğrudan kullanılmaz; gerekiyorsa torch/pandas tarafından kurulur).
Geliştirici/test bağımlılıkları için: `pip install -r gereksinimler_dev.txt`.

## Eğitim Akışı (sırasıyla)

Model ağırlıkları (`*.pt`), sözlük (`sozluk.json`), korpus (`turkce_metin.txt`) ve
talimat verisi (`talimat_verisi.json`) repoya dahil **değildir**; hepsi aşağıdaki
akışla üretilir. Akışı atlayıp doğrudan `python calistir.py` çalıştırırsanız
eğitimsiz (rastgele) bir modelle karşılaşırsınız.

### 1) Veri toplama → `turkce_metin.txt`

```bash
python -m egitim.veri_toplayici                       # sonraki yama, ~40.000 kelime hedefi
python -m egitim.veri_toplayici --kelime 60000        # hedef kelime sayısı
python -m egitim.veri_toplayici --patch all           # tüm dosyalar
python -m egitim.veri_toplayici --patch 3             # belirli yama numarası
python -m egitim.veri_toplayici --konu "Türkiye" --konu "Ankara"   # Wikipedia makaleleri
```

> **Üçüncü taraf veri kaynağı uyarısı:** Varsayılan kaynak, kişisel bir Hugging Face
> veri seti olan `ShigeoKageyama/NLP_SUITE`'tir. Bu repo silinir/gizlenir/yapısı
> değişirse veri toplama akışı kırılabilir. Farklı bir kaynak için `--repo`
> argümanını, harici bağımlılık istemiyorsanız `--konu` (Wikipedia) seçeneğini
> kullanın. Toplayıcı; HTTP Range Request ile parquet'i sanal okuma, kalite süzgeci
> (Türkçe karakter oranı, rakam oranı, minimum uzunluk) ve ilerleme durumu
> (`yama_durumu.json`) içerir.

### 2) Ön-eğitim → `hiper_model_<n>.pt` (+ `sozluk.json` ilk kez burada kurulur)

```bash
python -m egitim.egitici --n 1000 --cag 15 --batch 256
```

Sözlük ilk çalıştırmada korpusdan kurulur ve `sozluk.json` olarak **kilitlenir**;
sonraki çalıştırmalarda yüklendiği için kelime kimlikleri kaymaz. Var olan
ağırlıktan devam etmek için `--devam`.

### 3) Instruction fine-tuning → `hiper_model_<n>_talimat.pt`

```bash
python -m egitim.talimat_egitici --n 1000
```

- `talimat_verisi.json` **varsa yüklenir, asla ezilmez**: kendi soru–cevap
  örneklerinizi bu dosyaya ekleyebilirsiniz (yoksa gömülü 30 örnekle oluşturulur;
  fabrika ayarlarına dönmek için `--talimat-sifirla`).
- Bu adım yalnızca `hiper_model_<n>_talimat.pt` yazar; temel model
  (`hiper_model_<n>.pt`) **üzerine yazılmaz**.

### 4) Kullanım

```bash
python calistir.py              # terminal sohbeti (n=1000)
python calistir.py --n 500      # farklı boyut (ağırlık varsa)
python arayuz.py                # Gradio arayüzü (http://127.0.0.1:7860)
```

Arayüzdeki **N_GEN** dropdown'ı model boyutunu gerçekten değiştirir: seçim
değişince model o boyutta yeniden kurulur ve `hiper_model_<n>_talimat.pt` /
`hiper_model_<n>.pt` (varsa) yüklenir. Boyut için ağırlık yoksa durum panelinde
görünür. Sunucu adresini `GRADIO_SERVER_NAME` / `GRADIO_SERVER_PORT` ortam
değişkenleriyle değiştirebilirsiniz (konteynerlerde `0.0.0.0` gerekir).

## Klasör Yapısı

```text
hiper_geometrik_ai/
├── README.md                  # Bu belge
├── LICENSE                    # MIT
├── gereksinimler.txt          # Çalışma zamanı bağımlılıkları
├── gereksinimler_dev.txt      # Test bağımlılıkları (pytest)
├── ortak.py                   # Ortak fabrika/ağırlık yükleme/üretim/loglama
├── calistir.py                # Terminal sohbet
├── arayuz.py                  # Gradio sohbet arayüzü
├── mimari/                    # Model katmanları
│   ├── __init__.py
│   ├── kuresel_model.py       # HiperGeometrikAI (pencere tabanlı LM)
│   ├── hiper_attention.py     # Fused-QKV + Pre-LN + ReZero + SDPA dikkat
│   ├── kuresel_bag.py         # İki mercek izdüşümünü birleştiren katman
│   ├── kuresel_loss.py        # CrossEntropyLoss sarmalayıcısı
│   ├── decoder.py             # n → sözlük logits lineer katmanı
│   └── tokenizer.py           # Kelime düzeyli GeometrikTokenizer
├── egitim/
│   ├── __init__.py
│   ├── veri_toplayici.py      # Korpus toplama (HF + Wikipedia) — adım 1
│   ├── egitici.py             # Ön-eğitim — adım 2
│   ├── talimat_toplayici.py   # Soru–cevap verisi yönetimi
│   └── talimat_egitici.py     # Instruction fine-tuning — adım 3
├── experimental/              # Akışa bağlı OLMAYAN kodlar (bkz. experimental/README.md)
│   ├── README.md
│   ├── bpe_tokenizer.py       # Alternatif alt-kelime tokenizer (entegre değil)
│   └── encoder.py             # Eski tasarım: 512 boyutlu ham veri encoder'ı
├── tests/                     # pytest birim testleri
│   ├── conftest.py
│   ├── test_tokenizer.py
│   ├── test_kuresel_bag.py
│   ├── test_hiper_attention.py
│   ├── test_model.py
│   ├── test_ortak.py
│   └── test_talimat_toplayici.py
└── .github/workflows/ci.yml   # GitHub Actions: pytest
```

Üretilen (git harici) dosyalar: `turkce_metin.txt`, `sozluk.json`,
`talimat_verisi.json`, `yama_durumu.json`, `hiper_model_*.pt`.

## Testler

```bash
pip install -r gereksinimler_dev.txt
pytest -v
```

Kapsam: tokenizer gidiş-dönüşü, model forward şekli, **tüm parametrelerin gradyan
alması** (ölü parametre regresyonu), `n` parametresinin modele iletilmesi,
ağırlık uyuşmazlığı raporlama, talimat verisinin ezilmemesi.

## Sınırlamalar (dürüst liste)

- **8 kelimelik bağlam penceresi:** model her adımda tüm pencereyi düzleştirip tek
  sonraki-token tahmini üretir; n-gram modellere yakın davranır, uzun bağlam
  tutarlılığı yoktur.
- **Kelime düzeyli tokenizer:** nadir kelimeler `<UNK>` olur; alt-kelime (BPE)
  sürümü `experimental/bpe_tokenizer.py`'de vardır ama akışa bağlı değildir.
- **Küçük kapasite (~13,5M parametre):** genel amaçlı bir LLM değildir; eğitimsiz
  açılırsa sinir ağı kısmı anlamsız üretir (kural tabanlı yanıtlar yine çalışır).
- **Hibrit sohbet:** `arayuz.py`'deki intent kuralları elle yazılmıştır ve her yeni
  "yetenek" için yeni kural/talimat örneği gerekir. Talimat eşleştirme eşiği
  `BENZERLIK_ESIGI` sabitinden ayarlanabilir.
- **Uyumluluk notu:** Bu sürümde dikkat katmanı `nn.MultiheadAttention` yerine
  `HiperGeometrikAttention`'a taşındı ve `kuresel_bag` tek tensör döndürür; eski
  sürümle eğitilmiş `*.pt` dosyaları kısmen yüklenir ve `ortak.agirlik_yukle`
  uyuşmazlıkları uyarıyla raporlar. Temiz sonuç için yeniden eğitin.

## Katkı

PR'lerde `pytest -v` geçmiş olmalıdır. Kod adlandırması Türkçe'dir; yeni kodda da
Türkçe adlar kullanın. Loglama `logging` modülü üzerinden yapılır
(`ortak.log_kur` ile yapılandırılır).

## Lisans

[MIT](LICENSE)
