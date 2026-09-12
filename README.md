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
  Bilinen en büyük ticari modeller (GPT-4, Gemini sınıfı) dahi ~1-2 trilyon
  parametre civarında tahmin edilmektedir; katrilyon bunun ~1000 katıdır ve
  yüzlerce GPU'luk kümeler + petabayt veri + milyonlarca dolarlık bütçe
  gerektirir. Tek kişilik donanımda fiziksel olarak imkânsızdır.
- Bu projenin gerçekçi ve dürüst hedefi: **"K katmanlı Kronecker zinciriyle
  katrilyon mertebesinin üzerinde sanal etkileşim kapasitesi + katrilyonların
  üzerinde adreslenebilir 'boş küme' uzayına sahip seyrek bellek; ~7M yoğun +
  ~34M seyrek fiziksel gerçek parametre."** Üstteki tablo bunu doğrular.

Sayıları kendiniz doğrulayın:

```bash
python mimari/kuresel_model.py     # dürüst kapasite raporu
python test_mimari.py              # 22 mimari duman testi
```

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
- **Çakışma:** iki farklı pencere aynı satıra düşebilir. `tablo_sayisi=2` ile
  Bloom tarzı çift hash açılır (iki farklı tuzlu tablo, çıktılar toplanır).
- **Kronecker zinciriyle tamamlayıcıdır (rapor 9.5):** seyrek tablo HANGİ
  kümelerin dolu olduğunu taşır; `gen_kopru` dolu küme vektörünü bağlam
  vektörüne enjekte eder; bilinear zincir etkileşimi işler.

Kullanım: `egitim/egitici.py --seyrek-satir 1048576 --seyrek-boyut 32` veya
`--seyrek-yok` ile kapatın. Talimat fine-tuning'i aynı parametrelerle
kurulmalıdır (strict yükleme uyumsuzluğu açıkça hata verir).

---

## 🛡️ 3 Katmanlı Halüsinasyon Kontrol Mekanizması

`bilgi_katmani.py` — `BilgiKatmani`. Rapor 10'daki temel ödünleşim:
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
Gradio arayüzleri aynı mekanizmayı paylaşır (eski `arayuz.py`'nin
`intent_cevap`'ı bu katmanın ilkel bir örneğiydi; artık tek kaynak
`bilgi_katmani.py`'dir).

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
  7+1 bağımsız sinyalle puanlama, VALID/CONFLICT/INVALID durum makinesi,
  Conflict→Exploration çözücüsü, konsolidasyon, metin/olay üretimi (v0.2,
  Türkçe ek uyumu: yönelme/belirtme/bulunma/ayrılma + ünsüz yumuşaması +
  ünlü düşmesi + iyelik (6 kişi) + iyelik+durum zinciri + fiil çekimi
  (6 kişi × geçmiş/şimdiki/gelecek/geniş zaman),
  cümle/dosya→üçlü ayıklayıcı + `REAL_DATA` aktarımı, information-gain (v0.3),
  deterministik aritmetik mini-environment (v0.5), çelişki araştırma kuyruğu,
  sürekli öğrenme döngüsü (v1.0), ground-truth benchmark ve dış korpus borusu
  (`veri_toplayici.py` çıktısını sözlük büyütüp `REAL_DATA` olarak akıtır;
  ilişki asla uydurulmaz).
- **Kalıcılık** (`hga/knowledge/persistence.py`) — bilgi tabanını atomik JSON
  olarak kaydet/yükle (VERIFIED bilgi gerçekten kalıcı).
- **Kapalı doğrulama** (`hga/experience/dogrulama.py`) — deterministik
  doğrulayıcıyla MODEL_GENERATED deneyimleri doğrular; false acceptance'ı
  sıfırlar (benchmark: 30 adaydan 6 VERIFIED, 24 INVALID).
- **Tek yüz** (`hga/engine.py` + `python -m hga`) — tüm katmanı yapılandırılabilir
  tek motor + komut satırı arayüzü.
- **Memory** (`hga/memory/`) — seyrek deneyim slotları, experience replay ve
  bunları birleştiren entegrasyon + torch seyrek tabloya köprü (v0.6),
  doğrulanmış deneyimleri MODELİN kendi seyrek belleğine bağlayan `NeuralKopru`,
  bilgi yazmanın aşağı-akış etkisini ölçen `AblasyonDeneyi` (boş bellek ~%50,
  bilgi yazılı ~%100) ve bunu modelin KENDİ tamamlama görevine taşıyan
  `GorevAblasyonu` (yoğun gövde donukken ~şans → ~%100, ölü-yol → canlı-yol —
  "katrilyon" tezinin görev boyutlu kanıtı) ve `GenellemeAblasyonu` (belleğin
  KANONİK kodu eğitimde görülmeyen öznelere de genelliyor: held-out ~%25 → %100,
  boş bellek şansta kalır — genelleme ezberden değil bellekten gelir).

**En kritik güvenlik kuralı:** `MODEL_GENERATED` kaynaklı bir deneyim hiçbir
zaman otomatik `VERIFIED` kabul edilmez — en fazla `VALID` (bellek adayı) olur.
Konsolidasyon bu kuralı ikinci kez denetler ve ihlali çelişki günlüğüne yazar.

```bash
python experiments/experience_loop/run_full.py        # v0.1 → v1.0 tam demosu
python experiments/experience_loop/run_gercek_veri.py # gerçek veri → temsil → deneyim → doğrulama
python experiments/experience_loop/run_benchmark.py   # kontrollü benchmark + kapalı doğrulama
python experiments/experience_loop/run_ablation.py    # bilgi yazmanın öğrenmeye etkisi (torch)
python experiments/experience_loop/run_gorev_ablasyonu.py # bilgi → modelin tamamlama görevi (torch)
python experiments/experience_loop/run_genelleme_ablasyonu.py # bilgi → görülmeyen olguya genelleme (torch)
python experiments/experience_loop/run_morfoloji.py   # ünlü düşmesi + iyelik + fiil çekimi (6 kişi)
python experiments/experience_loop/run_korpus_boru.py # veri toplayıcı → sözlük büyütme → REAL_DATA
python -m hga bilgi                                   # tek yüz (CLI) demosu
python -m hga dogrulama                               # false accept 24→0
python tests/test_milestone_v01.py                    # §14'ün 12 maddesi + §15 senaryosu
```

Ayrıntı: `docs/EXPERIENCE_ENGINE.md`.

---

## 🗂️ Klasör Yapısı

```text
hiper_geometrik_ai/
├── README.md                    # Bu belge
├── gereksinimler.txt            # Bağımlılıklar (torch; opsiyonel: gradio, requests, pyarrow)
├── test_mimari.py               # 22 duman testi (pytest ile de çalışır)
├── bilgi_katmani.py             # 3 katmanlı halüsinasyon kontrol mekanizması
├── calistir.py                  # Terminal sohbet (chatbot)
├── arayuz.py                    # Gradio sohbet arayüzü (opsiyonel)
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
│   ├── knowledge/               # Entity/Property/Relation indexleri + KnowledgeStore
│   ├── experience/              # Generator, Evaluator, Conflict, Consolidation, Loop
│   ├── memory/                  # Seyrek deneyim slotları + replay + torch köprüsü
│   └── config/                  # experience_config.yaml + bağımlılıksız yükleyici
├── tests/                       # Knowledge/Experience katmanı testleri (torch'suz çalışır)
├── experiments/experience_loop/ # v0.1 ve v0.1→v1.0 uçtan uca demoları
├── docs/EXPERIENCE_ENGINE.md    # Experience Engine mimari notu
└── egitim/
    ├── egitici.py               # Temel eğitim (batch + AdamW + AMP + seyrek doluluk izleme)
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

**2) Temel eğitim** (BPE sözlüğünü kurar, kilitler; seyrek doluluğu izler):

```bash
python egitim/egitici.py --cag 5
python egitim/egitici.py --n 128 --katman 2 --batch 32   # küçük/deneysel koşu
python egitim/egitici.py --seyrek-satir 2097152          # 2M satırlık seyrek bellek
python egitim/egitici.py --seyrek-yok                    # seyrek bellek kapalı
```

**3) Talimat (instruction) fine-tuning** (temel eğitimle aynı seyrek parametreler):

```bash
python egitim/talimat_egitici.py
```

**4) Sohbet:**

```bash
python calistir.py      # terminal chatbot (3 katmanlı kontrol + etiketler)
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

- Varsayılan yapı (n=256, K=4 + 128 MB seyrek tablo) **CPU'da eğitilebilir**
  (2 çekirdekte bile; bu repodaki demo ağırlıklar öyle eğitildi), ama ciddi
  koşular için GPU şart: her bilinear katman başına ~2n³ çarpma yapılır.
- **Karışık hassasiyet (AMP):** CUDA + bf16 destekliyorsa eğitim motorunda
  otomatik açılır (`--amp hayir` ile kapatılabilir).
- **Gradient checkpointing:** derin zincirlerde (büyük K) `--checkpoint`.
- Seyrek tabloyu büyütmek RAM'i doğrusal artırır (20M satır ≈ 2,4 GB); çakışma
  oranını `doluluk_orani()` ile izleyin, gerekirse `tablo_sayisi=2` (Bloom).
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
3. **`gereksinimler.txt` repoya girdi (8.1.4):** `.gitignore`'daki `*.txt` /
   `*.json` genel yasakları kaldırıldı. Kullanılmayan `numpy` düşürüldü.
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
    iki adımda canlanmayı doğrular).
13. **3 katmanlı halüsinasyon kontrolü (10):** `BilgiKatmani` + beyaz listeli
    kısıtlı üretim + [DOĞRULANMAMIŞ] etiketleme; güven skoru kullanıcıya
    gösterilir; `norm()` artık kesme işaretini siler ('Türkiye'nin' →
    'turkiyenin' tam eşleşmesi düzeltildi); terminal ve Gradio aynı mekânizmayı
    paylaşır.

**Bilinen kırılma:** mimari değiştiği için önceki sürümlerin checkpoint'ları
(seyrek tablosuz `hiper_model_*.pt`) yeni modele YÜKLENEMEZ (strict yükleyici
bunu açıkça söyler) — modeli yeniden eğitmek gerekir.

---

## 🗺️ Yol Haritası

1. **Experience Engine ölçeklemesi:** `hga/` katmanını korpus ve talimat
   eğitimiyle beslemek — gerçek üçlüleri `REAL_DATA` kaynağıyla KnowledgeStore'a
   akıtmak, `memory/kopru.py` köprüsünü `mimari/seyrek_tablo.py` eğitim
   döngüsüne bağlamak (v0.6), `text_generator.py` şablonlarını Türkçe morfoloji
   üreteciyle değiştirmek.
2. **Korpus ölçeği:** Wikipedia dökümü, OSCAR/CC-100/mC4 "tr" alt kümeleriyle
   milyon-kelime seviyesine çıkmak — seyrek tablonun dolmasını ve gerçek
   genelleme yeteneğini besleyecek gerçek veri.
2. **Uzun bağlam:** 16 token → 64+ token pencere (dikkat maliyeti S² ile
   büyür; kavramsal anahtar uzayı sözlük^pencere ile katlanarak büyür).
3. **Anahtar tasarımını zenginleştir (9.6.3):** pencere anahtarına katman/
   konum bileşenleri ekleyerek katman derinleştikçe adres uzayının katlanması.
4. **GPU/dağıtık:** AMP hazırdır; katman-bazlı cihaz yerleştirme ve
   `torch.distributed` ile model paralelliği.
5. **Talimat seti büyütme** → gerçek Türkçe instruction veri setleri.
6. **Değerlendirme:** perplexity + uçtan uca sohbet karşılaştırmaları;
   katman-2 beyaz liste üretiminin kalitesinin (kelime sırası) ölçülmesi.

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
