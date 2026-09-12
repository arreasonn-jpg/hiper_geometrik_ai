# Hiper-Geometrik AI — Bilinear Kronecker Zincirli Küresel Bağ Mimarisi

Deneysel bir Türkçe dil modeli: klasik Transformer'daki saf doğrusal katman
yığınlarına alternatif olarak, **gerçek matris-sandviçi (bilinear) `A @ X @ B`
katmanlarının zinciri** üzerine kuruludur. Projenin "hiper-geometrik" anlatısı
(tesseract, fraktal katlanma, küresel bağlar) bu zincirde somutlaşır: her
katman, (n, n) boyutlu bir ara temsili iki taraftan matris çarpımıyla
dönüştürür.

---

## ⚠️ Dürüst Kapasite Bildirimi (önce bunu okuyun)

Bu projenin eski README'si "1 Katrilyon sinaps" diyordu. Bu ifade **teknik
olarak bir üst sınırı tarif ediyordu ama gerçek eğitilebilir parametre sayısı
ile temsil kapasitesini birbirine karıştırıyordu**. Aşağıdaki tablo, varsayılan
yapılandırmanın (n=256, K=4) dürüst muhasebesidir:

| Ölçüm | Değer |
|---|---|
| **GERÇEK eğitilebilir parametre** | **6.902.849 (~6,9 milyon)** |
| 0→1 katmanı sanal köşe (n²) | 65.536 |
| Katman başına temsil edilen operatör (n⁴, Kronecker `Bᵀ⊗A`) | 4.294.967.296 (~4,3 milyar) |
| Zincir etkileşim uzayı üst sınırı (n^(2K) = 256⁸) | ~1,8 × 10¹⁹ |
| Bir katmanın operatörünü TAM matris olarak tutmak için gereken RAM | ~16 GB |
| Zincirin GERÇEK RAM tüketimi | ~2 MB |

**Bu sayılar ne demek, ne DEĞİL?**

- Bir `A @ X @ B` katmanı, flatten uzayında `Y = (Bᵀ ⊗ A) X` dönüşümüdür:
  temsil ettiği tam operatör **n² × n² = n⁴ boyutludur** ama bunu yalnızca
  **2n² gerçek parametre** ile taşır. 16 GB'lık operatörün ~0,5 MB'lık iki
  mercekle temsil edilmesi — README'nin "Kronecker İllüzyonu" dediği şey —
  tam olarak budur ve gerçekten çalışır.
- Ancak bu, **katrilyonlarca bağımsız/öğrenilebilir ağırlığa sahip olmakla
  aynı ifade gücü DEĞİLDİR**. Kronecker yapılı operatör, tam ranklı bir
  operatörün öğrenebileceği fonksiyon ailesinin yalnızca çok küçük bir
  alt kümesini süpürür (düşük-rank kısıtı). Öğrenebilir serbestlik her zaman
  gerçek parametre sayısıyla sınırlıdır.
- **Literal "1 katrilyon gerçek parametre" bu projede hedef olmamalıdır:**
  Bilinen en büyük ticari modeller (GPT-4, Gemini sınıfı) dahi ~1-2 trilyon
  parametre civarında tahmin edilmektedir; katrilyon bunun ~1000 katıdır ve
  yüzlerce GPU'luk kümeler + petabayt veri + milyonlarca dolarlık bütçe
  gerektirir. Tek kişilik donanımda fiziksel olarak imkânsızdır.
- Bu projenin gerçekçi ve dürüst hedefi: **"K katmanlı Kronecker zinciriyle
  katrilyon mertebesinde (10¹⁵) sanal etkileşim kapasitesi; ~7 milyon gerçek
  parametre."** Üstteki tablo, varsayılan ayarların bu eşiği aştığını gösterir
  (256⁸ ≈ 1,8 × 10¹⁹ ≈ 18.000 katrilyon).

Sayıları kendiniz doğrulayın:

```bash
python mimari/kuresel_model.py     # dürüst kapasite raporu
python test_mimari.py              # mimari duman testleri
```

---

## 📐 Mimari

```
token id'leri (B, S)
  │  nn.Embedding + öğrenilen pozisyon kodlaması
  ▼
Nedensel çok kafalı dikkat (Pre-LN + ReZero + SDPA/Flash)     ← hiper_attention.py
  │  düzleştir: (B, S·emb)
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

**Ölçekleme rehberi (rapor 8.3):** `n`'i büyütmek parametre ve işlem maliyetini
O(n²)/O(n³) büyütür; bunun yerine **K'yı artırmak** gerçek parametreyi doğrusal
tutarken sanal etkileşim üst sınırını n^(2K) ile katlar:

| Yapı | Gerçek parametre (zincir) | Sanal etkileşim üst sınırı |
|---|---|---|
| n=128, K=4 | 131.072 | 128⁸ ≈ 7,2 × 10¹⁶ |
| **n=256, K=4 (varsayılan)** | **524.288** | **256⁸ ≈ 1,8 × 10¹⁹** |
| n=256, K=8 | 1.048.576 | 256¹⁶ ≈ 3,4 × 10³⁸ |

---

## 🗂️ Klasör Yapısı

```text
hiper_geometrik_ai/
├── README.md                    # Bu belge
├── gereksinimler.txt            # Bağımlılıklar (torch; opsiyonel: gradio, requests, pyarrow)
├── test_mimari.py               # Mimari duman testleri (pytest ile de çalışır)
├── calistir.py                  # Terminal sohbet (chatbot)
├── arayuz.py                    # Gradio sohbet arayüzü (opsiyonel)
├── mimari/
│   ├── kuresel_model.py         # Bütünleşik model + model_olustur (TEK KAYNAK) + kapasite raporu
│   ├── kuresel_bag.py           # Bilinear A@X@B katmanı + K katmanlı zincir
│   ├── encoder.py               # 0→1 katmanı: dış çarpım köprüsü (u ⊗ v)
│   ├── decoder.py               # Bilinear odak merceği okuma katmanı
│   ├── hiper_attention.py       # Nedensel dikkat (SDPA/Flash, Pre-LN, ReZero)
│   ├── bpe_tokenizer.py         # BPE alt-kelime tokenizer (varsayılan)
│   ├── tokenizer.py             # Eski kelime-bazlı tokenizer (uyumluluk için duruyor)
│   └── kuresel_loss.py          # CrossEntropy tabanlı loss
└── egitim/
    ├── egitici.py               # Temel eğitim motoru (batch + AdamW + AMP + kayan pencere)
    ├── talimat_egitici.py       # Instruction fine-tuning
    ├── talimat_toplayici.py     # Yerleşik talimat seti
    └── veri_toplayici.py        # Korpus toplama (Wikipedia + HF; güvenilir kaynak rehberi)
```

---

## 🚀 Kurulum ve Kullanım

```bash
pip install -r gereksinimler.txt
```

**1) Korpus topla** (internet gerekir; opsiyonel `requests` bağımlılığı):

```bash
python -c "import sys; sys.path.insert(0, 'egitim'); \
from veri_toplayici import OtomatikVeriToplayici; \
OtomatikVeriToplayici('.').genis_korpus_cek(hedef_kelime=50000)"
```

**2) Temel eğitim** (BPE sözlüğünü kurar, kilitler; modeli `hiper_model_256.pt`'ye yazar):

```bash
python egitim/egitici.py --cag 5
python egitim/egitici.py --n 128 --katman 2 --batch 32   # küçük/deneysel koşu
```

**3) Talimat (instruction) fine-tuning:**

```bash
python egitim/talimat_egitici.py
```

**4) Sohbet:**

```bash
python calistir.py      # terminal chatbot
python arayuz.py        # Gradio arayüzü (pip install gradio)
```

**5) Doğrulama ve raporlar:**

```bash
python test_mimari.py          # tüm mimari testleri
python mimari/kuresel_model.py # dürüst kapasite raporu
```

Çalışma zamanı artefaktları (`bpe_sozluk.json`, `hiper_model_*.pt`,
`turkce_metin.txt`, `talimat_verisi.json`) `.gitignore`'dadır; repoya girmez.

---

## ⚙️ Donanım Gerçekçiliği

- Varsayılan yapı (n=256, K=4) **CPU'da eğitilebilir** (2 çekirdekte bile),
  ama ciddi koşular için GPU şart: her bilinear katman başına ~2n³ çarpma
  yapılır ve K katman bunu K kez tekrarlar.
- **Karışık hassasiyet (AMP):** CUDA + bf16 destekliyorsa eğitim motorunda
  otomatik açılır (`--amp hayir` ile kapatılabilir).
- **Gradient checkpointing:** derin zincirlerde (büyük K) `--checkpoint`
  ile ara aktivasyon belleği aktivasyon-başına yeniden hesaplamaya
  dönüştürülür.
- Çok büyük n/K denemeleri için katmanları farklı cihazlara dağıtmak
  (`torch.distributed` veya manuel `device_map`) yol haritasındadır; bkz. aşağıda.

---

## 📜 Bu Sürümde Değişenler (Refactor Günlüğü)

Bu sürüm, karşılaştırmalı inceleme raporundaki 8.1–8.6 yol haritasını uygular:

1. **Tek gerçek kaynak (8.1.1):** `model_olustur` üç dosyadaki kopyasından
   `mimari/kuresel_model.py`'ye indirgendi. Eski kopyalarda `n` parametresi,
   anahtar adları eşleşmediği için (`n_gen`/`gen_sayisi`/`boyut` ararken model
   `n` bekliyordu) **hiçbir zaman modele iletilmiyordu** — artık iletiliyor ve
   `test_fabrika_n_gercekten_gecer` bunu regression olarak koruyor.
2. **Ölü/eksik parametreler temizlendi (8.1.2):** eski `kuresel_bag` iki
   bağımsız `Linear`'dı ve `v_yeni` çıktısı hesaplanıp çöpe atılıyordu
   (`mercek_B` hiç eğitilmiyordu). Yeni zincirde her parametre gradyan alır;
   `test_model_ileri_geri_olu_parametre_yok` bunu garantiler.
3. **`gereksinimler.txt` repoya girdi (8.1.4):** `.gitignore`'daki `*.txt` /
   `*.json` genel yasakları kaldırıldı; yalnızca üretilen artefaktlar yok
   sayılıyor. Kullanılmayan `numpy` bağımlılığı da düşürüldü.
4. **`strict=True` ağırlık yükleme (8.1.5):** `agirlik_yukle()` checkpoint
   uyumsuzluklarını sessizce yutmaz; uyumsuzluk açıkça raporlanır.
5. **Gerçek bilinear/Kronecker mimarisi geri döndü (8.2):** `A @ X @ B`
   sandviçi `torch.einsum` ile uygulandı; temsil edilen operatörün Kronecker
   boyutu (n⁴) kapasite raporunda açıkça raporlanıyor.
6. **K katmanlı zincir, küçük n (8.3):** `n=256, K=4` varsayılan; gerçek
   parametre ~7M kalırken sanal etkileşim üst sınırı katrilyonun üzerine çıkar.
7. **AMP + gradient checkpointing (8.4.1–8.4.2):** eğitim motorunda.
8. **Nedensel dikkat + BPE devreye alındı (8.4.5–8.4.6):** `hiper_attention.py`
   artık ölü kod değil (`is_causal=True`); BPE tokenizer varsayılan ve
   `bpe_sozluk.json`'a kilitleniyor.
9. **Veri kaynakları (8.4.7):** kırılgan yama repo'suna alternatif olarak
   `genis_korpus_cek()` (Wikipedia API) eklendi; OSCAR/CC-100/mC4 önerileri
   belgelendi. Ayrıca `veri_toplayici.py`'deki erişilemez (return sonrası)
   ölü kod bloğu temizlendi ve `hazirla_veya_yukle` artık gerçekten
   diskten yüklüyor.
10. **README dürüstleştirildi (8.6.7):** "1 katrilyon parametre" değil,
    "katrilyon mertebesinde sanal etkileşim kapasitesi + ~7M gerçek parametre".
11. **Entegrasyon onarımları:** `encoder.py` (eskiden 512 boyutlu ham vektör
    bekleyen ölü kod) artık modelin 0→1 katmanı; `calistir.py`'nin istem
    biçimi talimat eğitimiyle aynı (`soru ... cevap ... son`); bağlam penceresi
    tüm bileşenlerde modelden okunuyor (sabit 8 değil).

**Bilinen kırılma:** mimari değiştiği için eski `hiper_model_1000.pt`
checkpoint'ları yeni modele YÜKLENEMEZ (strict yükleyici bunu açıkça söyler).
Modeli yeniden eğitmek gerekir.

---

## 🗺️ Yol Haritası

1. **Korpus ölçeği:** Wikipedia dökümü, OSCAR/CC-100/mC4 "tr" alt kümeleriyle
   milyon-kelime seviyesine çıkmak (sanal kapasiteyi besleyecek gerçek veri).
2. **Uzun bağlam:** 16 token → 64+ token pencere (dikkat maliyeti S² ile büyür).
3. **GPU/dağıtık:** AMP hazırdır; katman-bazlı cihaz yerleştirme ve
   `torch.distributed` ile model paralelliği belgelenecek.
4. **Talimat seti büyütme:** 30 yerleşik örnek → gerçek Türkçe instruction
   veri setleri.
5. **Değerlendirme:** perplexity + uçtan uca sohbet karşılaştırmaları
   (rastgele başlangıç vs eğitilmiş model).

---

## 🔬 Matematiksel Arka Plan (kısaca)

- **Bilinear sandviç:** `Y = A @ X @ B`. Flatten uzayında
  `vec(Y) = (Bᵀ ⊗ A) vec(X)` — yani katmanın tam operatörü iki merceğin
  **Kronecker çarpımıdır**: n⁴ girdilik bir operatör, 2n² parametreyle.
- **0→1 katmanı:** `X = tanh(u ⊗ v)` — n boyutlu iki vektörün dış çarpımı
  n² "sanal köşe" üretir; eleman bazlı tanh bu rank-1 matrisi tam ranklı
  hâle getirir.
- **Okuma:** `M₁ @ X @ M₂` odak merceklerinden satır/sütun ortalamalarıyla
  çift yönlü okuma yapılır; softmax bilinçli olarak YOKTUR
  (`nn.CrossEntropyLoss` kendi `log_softmax`'unu uygular).
- **Dürüst sınır:** Zincir bileşkesi yine bir (n² × n²) doğrusal operatördür;
  K katman ona 2Kn² gerçek serbestlik katar. n^(2K) "etkileşim uzayı" bu
  yapının geometrik büyüme anlatısının üst sınırıdır — gerçek parametre sayısı
  değildir.
