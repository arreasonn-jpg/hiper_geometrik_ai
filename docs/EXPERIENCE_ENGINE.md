# HGA Experience Engine — Uygulama Notu (v0.1 → v1.0)

> Hiper Geometrik AI: Experience Engine / Self-Expanding Knowledge Architecture
> yol haritasının (12 Eylül 2026) fiziksel karşılığı.

Bu belge, `hga/` paketi altında kurulan **Knowledge + Experience + Evaluation +
Memory** katmanını açıklar: hangi dosya hangi rapor bölümünü gerçekler, nasıl
çalıştırılır ve hangi güvenlik kuralları korunur. Yol haritasındaki v0.1–v1.0
aşamalarının tamamı bu katmanda gerçekleştirilmiştir (v0.6 köprüsü torch
kurulu ortamda aktifleşir).

---

## 1. Ne kuruldu, ne değişmedi

Mevcut geometrik çekirdek (`mimari/` — Kronecker zinciri, seyrek "boş küme"
belleği, dikkat, decoder) ve `bilgi_katmani.py` (3 katmanlı halüsinasyon
kontrolü) **hiç değiştirilmedi**. Yeni katman, çekirdeğin ÜZERİNE eklenen saf
Python modüllerinden oluşur (torch bağımlılığı yoktur — v0.1 bu şekilde tek
başına test edilebilir ve doğrulanabilir).

Rapordaki `hga/model/` önerisi mevcut `mimari/` paketine karşılık gelir; geriye
dönük uyumluluk için çekirdek yerinden oynatılmadı.

## 2. Gerçekleşen klasör yapısı

```text
hiper_geometrik_ai/
├── hga/
│   ├── __init__.py                # paket kökü (şemaları dışa açar)
│   ├── knowledge/
│   │   ├── __init__.py
│   │   ├── schemas.py             # Entity, PropertyValue, Relation, RelationFact,
│   │   │                          #   ExperienceCandidate + KaynakTuru + DeneyimDurumu
│   │   ├── entity_index.py        # EntityIndex   (§4)
│   │   ├── property_index.py      # PropertyIndex (§5)
│   │   ├── relation_index.py      # RelationIndex (§6)
│   │   ├── knowledge_store.py     # KnowledgeStore (§3) + bellek desteği + çelişki günlüğü
│   │   └── persistence.py         # atomik JSON kaydet/yükle (kalıcı bilgi) (§12, EK-C)
│   ├── experience/
│   │   ├── __init__.py
│   │   ├── generator.py           # ExperienceGenerator — kontrollü kombinasyon (§7)
│   │   ├── text_generator.py      # TextGenerator — üçlüden metin/olay üretimi (§19 v0.2)
│   │   ├── turkce.py              # Türkçe ek uyumu: yönelme/belirtme/bulunma/ayrılma/çoğul
│   │   │                          #   + ünsüz yumuşaması + ünlü düşmesi + iyelik (6 kişi)
│   │   │                          #   + iyelik+durum zinciri + fiil çekimi (6 kişi ×
│   │   │                          #   geçmiş/şimdiki/gelecek/geniş zaman)
│   │   ├── cumle_ayiklayici.py    # Cümle → üçlü + REAL_DATA aktarımı (döngünün "gerçek veri" aşaması)
│   │   ├── corpus.py              # metin dosyasından cümle → üçlü → REAL_DATA
│   │   ├── sozluk_buyutme.py      # gerçek korpustan yeni VARLIK desenleri (ilişki uydurmaz)
│   │   ├── korpus_boru.py         # veri_toplayici çıktısı → sözlük büyütme → REAL_DATA
│   │   ├── korpus_uretici.py      # çevrimdışı belirleyici SOV cümle üretici (sentetik stres testi)
│   │   ├── scoring.py             # bağımsız sinyaller + ağırlıklı puan + information_gain (§8, §16, v0.3)
│   │   ├── evaluator.py           # VALID/UNCERTAIN/CONFLICT/INVALID aday değerlendirmesi
│   │   ├── conflict.py            # Conflict → Exploration (§11)
│   │   ├── arastirma.py           # ArastirmaKuyrugu — çelişkiyi deterministik kanıtla toplu çözme
│   │   ├── consolidation.py       # Consolidator — belleğe/araştırmaya/redde yönlendirme (§12)
│   │   ├── mini_env.py            # AritmetikOrtam — deterministik doğrulayıcı (§18, §19 v0.5)
│   │   ├── loop.py                # DeneyimDongusu — sürekli öğrenme + metrikler (§19 v1.0, §20)
│   │   ├── dogrulama.py           # DogrulamaHatti — deterministik kanıtla VERIFIED/INVALID
│   │   └── benchmark.py           # ground-truth'a karşı ölçüm + özet rapor (§18, §20)
│   ├── engine.py                  # ExperienceEngine — tüm katmanın tek yüzden orkestrasyonu
│   └── __main__.py                # CLI: python -m hga bilgi|gercek-veri|benchmark|dogrulama|ozet
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── sparse_memory.py       # DeneyimSlotlari — seyrek deneyim slotları (§22 commit 6)
│   │   ├── replay.py              # DeneyimTekrari — experience replay (§19 v0.4)
│   │   ├── entegrasyon.py         # BellekEntegrasyonu — replay + consolidation ↔ seyrek bellek
│   │   ├── kopru.py               # TorchKoprusu — deneyim ↔ torch seyrek tablo köprüsü (§19 v0.6)
│   │   ├── neural_kopru.py        # NeuralKopru — deneyim ↔ MODELİN seyrek belleği + gen_kopru
│   │   ├── ablation.py            # AblasyonDeneyi — belleğe yazılan bilginin etkisini ölçer (§17/§20)
│   │   ├── gorev_ablasyonu.py     # GorevAblasyonu — bilgi, modelin KENDİ görevini çözüyor mu?
│   │   └── genelleme_ablasyonu.py # GenellemeAblasyonu — bilgi, GÖRÜLMEYEN olgulara genelliyor mu?
│   └── config/
│       ├── __init__.py
│       ├── config.py              # bağımlılıksız YAML yükleyici (PyYAML varsa onu kullanır)
│       └── experience_config.yaml # ağırlıklar + eşikler + üretici ayarları
├── tests/
│   ├── test_entity_index.py
│   ├── test_relation_index.py
│   ├── test_experience_eval.py
│   ├── test_conflict.py
│   ├── test_milestone_v01.py      # §14'ün 12 maddesi + §15 senaryosu
│   ├── test_v02_v03.py            # v0.2 metin üretimi + v0.3 information gain
│   ├── test_mini_env.py           # v0.5 aritmetik doğrulayıcı
│   ├── test_loop.py               # v1.0 sürekli döngü + metrikler
│   ├── test_kopru.py              # v0.6 torch köprüsü (torch yoksa güvenli atlama)
│   ├── test_turkce.py             # ek uyumu + yumuşama + ünlü düşmesi + iyelik + fiil
│   ├── test_cumle_ayiklayici.py   # cümle → üçlü + REAL_DATA aktarımı
│   ├── test_corpus.py             # dosyadan cümle → üçlü → REAL_DATA
│   ├── test_sozluk_buyutme.py     # gerçek korpustan yeni varlık desenleri
│   ├── test_korpus_boru.py        # veri toplayıcı çıktısı → sözlük büyütme → REAL_DATA
│   ├── test_korpus_uretici.py     # çevrimdışı korpus üretici (determinizm/atlama/ölçek)
│   ├── test_persistence.py        # atomik JSON kaydet/yükle
│   ├── test_benchmark.py          # ground-truth'a karşı ölçüm
│   ├── test_dogrulama.py          # kapalı doğrulama hattı (false accept 24→0)
│   ├── test_engine.py             # ExperienceEngine entegrasyonu
│   ├── test_neural_kopru.py       # deneyim ↔ model seyrek belleği (torch gerekir)
│   ├── test_ablation.py           # bilgi yazmanın aşağı-akış etkisi (torch gerekir)
│   ├── test_gorev_ablasyonu.py    # bilgi → modelin tamamlama görevi (torch gerekir)
│   ├── test_genelleme_ablasyonu.py # bilgi → GÖRÜLMEYEN olguya genelleme (torch gerekir)
│   └── test_arastirma.py          # araştırma kuyruğu (CONFLICT → kanıt → kesin durum)
├── experiments/
│   └── experience_loop/
│       ├── run_demo.py            # v0.1 milestone demosu
│       ├── run_full.py            # v0.1 → v1.0 tam yol haritası demosu
│       ├── run_gercek_veri.py     # gerçek veri → temsil → deneyim → doğrulama
│       ├── run_benchmark.py       # kontrollü benchmark + kapalı doğrulama
│       ├── run_kopru.py           # deneyim ↔ torch seyrek bellek köprüsü (v0.6)
│       ├── run_neural_kopru.py    # deneyim ↔ MODELİN seyrek belleği + gen_kopru (torch)
│       ├── run_ablation.py        # bilgi yazmanın öğrenmeye etkisi (kontrol/deney)
│       ├── run_gorev_ablasyonu.py # bilgi → modelin KENDİ tamamlama görevi (torch)
│       ├── run_genelleme_ablasyonu.py # bilgi → GÖRÜLMEYEN olguya genelleme (torch)
│       ├── run_genelleme_olcegi.py  # genelleme × ölçek + gürültü bozulma eğrisi (torch)
│       ├── run_morfoloji.py       # ünlü düşmesi + iyelik + fiil çekimi (6 kişi) demosu
│       ├── run_korpus_boru.py     # veri toplayıcı çıktısı → sözlük büyütme → REAL_DATA
│       └── run_korpus_olcegi.py   # çevrimdışı korpus ölçeği provası (sentetik, ağsız)
└── docs/
    └── EXPERIENCE_ENGINE.md       # bu belge
```

## 3. Çalıştırma

```bash
# Tüm yeni katman testleri (her dosya tek başına da çalışır; pytest de kabul eder)
python tests/test_entity_index.py
python tests/test_relation_index.py
python tests/test_experience_eval.py
python tests/test_conflict.py
python tests/test_milestone_v01.py
python tests/test_v02_v03.py
python tests/test_mini_env.py
python tests/test_loop.py
python tests/test_kopru.py
python tests/test_turkce.py
python tests/test_cumle_ayiklayici.py
python tests/test_corpus.py
python tests/test_sozluk_buyutme.py
python tests/test_korpus_boru.py
python tests/test_korpus_uretici.py
python tests/test_persistence.py
python tests/test_benchmark.py
python tests/test_dogrulama.py
python tests/test_engine.py
python tests/test_neural_kopru.py   # torch gerekir
python tests/test_ablation.py       # torch gerekir
python tests/test_gorev_ablasyonu.py # torch gerekir
python tests/test_genelleme_ablasyonu.py # torch gerekir
python tests/test_arastirma.py

# Uçtan uca döngü demoları
python experiments/experience_loop/run_demo.py        # v0.1 milestone
python experiments/experience_loop/run_full.py        # v0.1 → v1.0 tam yol haritası
python experiments/experience_loop/run_gercek_veri.py # gerçek veri → temsil → deneyim → doğrulama
python experiments/experience_loop/run_benchmark.py   # kontrollü benchmark + kapalı doğrulama
python experiments/experience_loop/run_kopru.py       # deneyim ↔ torch seyrek bellek (torch gerekir)
python experiments/experience_loop/run_neural_kopru.py # deneyim ↔ MODELİN seyrek belleği (torch)
python experiments/experience_loop/run_ablation.py     # bilgi yazmanın öğrenmeye etkisi (torch)
python experiments/experience_loop/run_gorev_ablasyonu.py # bilgi → modelin tamamlama görevi (torch)
python experiments/experience_loop/run_genelleme_ablasyonu.py # bilgi → GÖRÜLMEYEN olguya genelleme (torch)
python experiments/experience_loop/run_genelleme_olcegi.py  # genelleme × ölçek + gürültü (torch)
python experiments/experience_loop/run_morfoloji.py    # ünlü düşmesi + iyelik + fiil çekimi (6 kişi)
python experiments/experience_loop/run_korpus_boru.py  # veri toplayıcı → sözlük büyütme → REAL_DATA
python experiments/experience_loop/run_korpus_olcegi.py  # çevrimdışı korpus ölçeği provası

# Komut satırı (tek yüz)
python -m hga bilgi            # bilgi tabanı + durum makinesi demosu
python -m hga dogrulama        # kapalı doğrulama hattı (false accept 24→0)
python -m hga benchmark        # metrik tablosu
python -m hga ozet bilgi.json  # bilgi tabanı özeti (dosyadan yükleme)

# Tüm testler (torch kuruluysa çekirdek + Experience Engine birlikte)
pip install -r gereksinimler.txt pytest
python -m pytest -q            # 165 test: 23 çekirdek + 142 Experience Engine
```

## 5b. Doğrulama durumu

Saf-Python katmanı (Knowledge/Experience/Memory) torch'suz tek başına çalışır.
torch kurulu bir ortamda (ör. `python -m venv .venv && .venv/bin/pip install
torch pytest`) aşağıdakiler birlikte doğrulandı:

- **Mevcut geometrik çekirdek bozulmadı:** `test_mimari.py`'deki 23 duman testi
  (bilinear sandviç, zincir gradyanı, nedensel dikkat, seyrek "boş küme", kapasite
  raporu, strict yükleme, mini eğitim) geçti.
- **v0.6 köprüsü gerçek tabloya yazıyor:** `run_kopru.py` doğrulanmış 6 üçlüyü
  `HashlenmisKureselTablo`'ya adresler; gradyan adımı sonrası doluluk 0 → 6
  (boş küme → dolu küme). Aynı üçlü her zaman aynı satıra düşer.
- **NeuralKopru (v0.6+):** doğrulanmış deneyimler MODELİN kendi seyrek
  belleğine yazılır; `gen_kopru` köprüsü "ölü-yol"dan (yazmadan önce gradyan 0)
  "canlı-yol"a (yazdıktan sonra gradyan > 0) geçer; token eğitimiyle birlikte
  paylaşımlı bellekte loss düşer.
- **Ablasyon (v1.0+):** aynı dengeli kümede boş bellekten salt okuma ~%50
  (şans), bilgi yazılı bellekten salt okuma ~%100 — doğrulanmış bilgi,
  aşağı-akış öğrenme için ölçülebilir bir sinyale dönüşür (etki +%50).
- **Görev ablasyonu (v1.0+):** ölçüm modelin KENDİ ileri geçiş yoluna taşındı
  (yoğun gövde DONUK). Dizisel tamamlama görevinde (özne+ilişki → nesne)
  bellek yolu boş+donukken doğruluk ~şans (model çözemez), bellek yolu
  eğitildiğinde ~%100 (etki +%100) ve `gen_kopru` gradyanı 0 → >0
  (ölü-yol → canlı-yol). Ters yönlü iki kanıtla pekiştirildi: TABLO-SIFIR
  (`tablo_sifir`) — eğitilmiş köprü DONUK kalırken tablo boşaltılınca doğruluk
  %100 → %0 düşer (bilgi köprüde değil, tablo SATIRLARINDADIR); HELD-OUT
  SINIRI (`heldout_siniri`) — hiç yazılmamış [özne, ilişki] pencereleri
  (yeni olgular) ~şans kalır (bellekte satırı olmayan olgu yoktan var olmaz;
  3 tohumda kararlı). Bellek, göreve yönelik gerçek ve dürüst bir bilgi kanalıdır.
- **Tam morfoloji (v1.0+):** `turkce.py`'ye ünlü düşmesi (burun→burna,
  şehir→şehri; ünsüzle başlayan eklerde YOK: burunda), 6 kişilik iyelik ekleri
  (kitabım…kitapları), iyelik+durum zinciri (evi→evine 'n' ara harfi) ve
  3. tekil fiil çekimleri — geçmiş (bin→bindi), şimdiki (bin→biniyor,
  git→gidiyor, oku→okuyor), gelecek (bin→binecek, ye→yiyecek) ve geniş
  zaman/aorist (bak→bakar, gel→gelir, gör→görür, git→gider, ye→yer) — ile
  tam kişi ekli çekim (`fiil_cekimi`): bindim…bindiler, biniyorum…
  biniyorlar, bineceğim (k→ğ yumuşaması)…binecekler, bakarım…bakarlar
  eklendi; ünsüz yumuşaması/ünlü düşmesi/aorist düzensizlikleri küratörlü
  listelerle sınırlı.
- **Dış korpus borusu (v1.0+):** `veri_toplayici.py`'nin `metni_kaydet` çıktısı
  (`turkce_metin.txt`) `korpus_boru.py` üzerinden okunur: cümlelere bölünür,
  sözlük bilinen fiil desenleriyle büyütülür (yeni özne=insan, yeni nesne=
  varlik; ilişki ASLA uydurulmaz) ve üçlüler `REAL_DATA` olarak KnowledgeStore'a
  yazılır. 8 cümlelik örnekte 7 üçlü aktarıldı, 4 yeni özne + 3 yeni nesne
  eklendi, kuşkulu cümle atlandı. (Ağ/requests/pyarrow gerekmez: yalnız dosya
  sözleşmesi; canlı korpus çekimi `egitim/veri_toplayici.py`'nin işidir.)
- **Genişletilmiş sözlük (v1.0+):** `VARSAYILAN_SOZLUK` elle küratörlü 7
  ilişkiye genişletildi — binmek/bakmak (yönelme) + gitmek/gelmek (yönelme)
  + okumak/yazmak/sevmek (belirtme); fiil yüzey biçimleri `fiil_cekimi` (4
  zaman) ile birebir uyumlu. Kök çıkarma (`_kok_bul`) artık ünsüz yumuşamasını
  GERİ çevirir (kitabı→Kitap, ağacı→Ağaç; ğ ambigua → dokunulmaz, tek heceli
  gövde yalnızca bilinen istisnalarda çevrilir). İlişki yine ASLA çalışma
  anında uydurulmaz.
- **Çevrimdışı korpus ölçeği (v1.0+):** `korpus_uretici.py` belirleyici
  (tohumlu) ve dilbilgisel olarak doğru SOV cümleleri üretir (7 ilişki ×
  yönelme/belirtme); aynı boru ~4.000 cümleyi <0,1 saniyede (~40.000
  cümle/sn) REAL_DATA olarak akıtır, kasıtlı gürültüyü atlar, ilişki icat
  etmez. Bu bir SENTETİK stres testidir (belgeli); gerçek Wikipedia/OSCAR
  ölçeği ağ gerektirir.
- **Genelleme ablasyonu (v1.0+):** bellek, KANONİK (paylaşılan) nesne
  kodları taşıyorsa; bellek donukken yalnızca `gen_kopru` köprüsü + küçük
  kafa eğitilen okuyucu, EĞİTİMDE HİÇ GÖRMEDİĞİ öznelerde de doğru nesneyi
  tamamlar: kontrol (boş bellek) eğitim ~%25 / held-out ~%25 (şans) → deney
  eğitim %100 / held-out %100 (etki +%75; 3 tohumda kararlı). Genelleme
  ezberden değil BELLEKTEN gelir. (Tam yoğun gövde eğitilebilir bırakılırsa
  model ezberler: train ~%100 ama held-out ~şans — dürüstlük notu olarak
  belgeli; bu yüzden gövde donuk tutulur.)
- **Genelleme × ölçek + gürültü (v1.0+):** `gurultulu_kos` aynı protokolü
  BÜYÜK bilgi tabanında (6 kategori × 4 özne = 24 olgu) tekrarlar: gürültüsüz
  held-out %100 (3 tohumda kararlı). Eğitim olgularına YANLIŞ nesne kodu
  yazılırsa held-out dürüstçe bozulur: her kategori eğitimde TEK temiz
  örnekle temsil edildiği için held-out ≈ 1 − gürültü_oranı düşer (%17 → %83,
  %50 → %50, %100 → %0). Bu, seyrek belleğin değil OKUYUCU eğitiminin veri
  bağımlılığıdır: tek örnek kırılgandır; yedekli temiz veri genellemeyi
  sağlamlaştırır.
- **Toplam:** `pytest` ile 165 test tek seferde geçti.

Testlerin hiçbiri torch gerektirmez; yalnızca standart kütüphane kullanılır.

## 4. Temel döngü ve durum makinesi

```text
Bilgi tabanı ──> Generator (kontrollü kombinasyon) ──> CANDIDATE
      │                                                    │
      │                                                    ▼
      │                                    Evaluator (kural + puan)
      │                                    ├─ kural ihlali   → INVALID   → reddet
      │                                    ├─ kanıt eksikliği→ UNCERTAIN → araştır
      │                                    ├─ kanıt çelişkisi→ CONFLICT  → araştır
      │                                    └─ kısıtlarla uyum→ VALID     (kaynak fark etmez)
      │                                                    │
      │                                                    ▼
      │                                    Bağımsız Verifier
      │                                    └─ VERIFYING → VERIFIED/INVALID/UNCERTAIN
      ▼
Consolidator ──> VALID → belleğe aday; VERIFIED → kalıcı bilgi; UNCERTAIN/CONFLICT → kuyruk
      │
      ▼
(Yeni bilgi) ──> Generator yeniden üretir   [self-expanding loop, §12]
```

**En kritik güvenlik kuralı (§9, §21):** `source=MODEL_GENERATED` bir deneyim
hiçbir koşulda otomatik `VERIFIED` kabul edilmez — en fazla `VALID` (bellek
adayı) olur. `Consolidator` bunu ikinci kez savunur: elle `VERIFIED` işaretlense
bile MODEL_GENERATED adayı kalıcı bilgiye yazmaz ve çelişki günlüğüne ihlal
kaydı düşer. Bu davranış `test_model_generated_kalici_olamaz` ile kilitlenir.

## 5. Rapor bölümü → kod eşlemesi

| Rapor | Gerçekleşen yer |
|---|---|
| §3 temel kayıt | `schemas.py` (`Entity`, `PropertyValue`, `Relation`, `RelationFact`, `ExperienceCandidate`) |
| §4 Entity Index | `entity_index.py` — entity_id ≠ tokenizer token ID |
| §5 Property Index | `property_index.py` — 1/0 boolean, `[0,1]` güven değerine genişletilebilir |
| §6 Relation Index | `relation_index.py` — seyrek sözlük + kanıt listesi (tensör değil) |
| §7 Generator | `generator.py` — kontrollü kombinasyon, `tip_filtresi` |
| §8 Evaluator | `evaluator.py` + `scoring.py` — 7 bağımsız sinyal |
| §9 Durum makinesi | `evaluator.py` karar ağacı + `schemas.DeneyimDurumu` |
| §10 Confidence/Source | `KaynakTuru` + `KAYNAK_GUVENIRLIGI` + her kayıtta `confidence` |
| §11 Conflict→Exploration | `conflict.py` — alternatif üret → deterministik test → güven güncelle → yeniden değerlendir |
| §12 Self-expanding loop | `consolidation.py` + `loop.py` |
| §16 1+1+1 yorumu | `scoring.py` — toplam puan kanıt değildir; kısıtlar ayrıca karar verir |
| §18 AlphaGo benzetmesi | `mini_env.py` — domain-specific deterministik doğrulayıcı |
| §19 v0.2 | `text_generator.py` + `turkce.py` — üçlüden ek uyumlu metin/olay üretimi |
| §19 v0.3 | `scoring.py` — `information_gain` sinyali |
| §19 v0.4 | `memory/replay.py` + `memory/entegrasyon.py` — replay + konsolidasyon ↔ seyrek bellek |
| §19 v0.5 | `mini_env.py` — `AritmetikOrtam` (güvenli `ast` ile, eval yok) |
| §19 v0.6 | `memory/kopru.py` — `TorchKoprusu` (torch kurulu ortamda aktif) |
| §19 v0.6+ / EK-B | `memory/neural_kopru.py` — deneyim ↔ modelin seyrek belleği + `gen_kopru` |
| §17/§20 "katrilyon tezi" | `memory/ablation.py` — belleğe yazılan bilginin öğrenmeye etkisi (ölçülebilir) |
| §17/§20 + gerçek görev | `memory/gorev_ablasyonu.py` — bilgi, modelin KENDİ görevini çözüyor (tamamlama) |
| §17/§20 + genelleme | `memory/genelleme_ablasyonu.py` — bilgi, GÖRÜLMEYEN olgulara genelliyor (held-out) |
| §19 v1.0 | `loop.py` — `DeneyimDongusu` + kontrollü metrikler |
| §2 "Gerçek veri → Temsil" | `cumle_ayiklayici.py` + `corpus.py` — cümle/dosya → üçlü + REAL_DATA aktarımı |
| §11 ölçekleme | `arastirma.py` — CONFLICT kuyruğunu deterministik kanıtla toplu çözme |
| §11/§18 kapalı döngü | `dogrulama.py` — deterministik kanıtla VERIFIED/INVALID (false accept 24→0) |
| §12/EK-C "kalıcı bilgi" | `persistence.py` — atomik JSON kaydet/yükle |
| §18/§20 "benchmarklarla ölç" | `benchmark.py` — ground-truth'a karşı false accept/reject ölçümü |
| Orkestrasyon | `engine.py` — ExperienceEngine (config + bileşim) + `__main__.py` CLI |
| §20 Metrikler | `loop.AdimRaporu` (acceptance/conflict/false-accept/false-reject/knowledge growth/replay) |
| §21 Riskler | MODEL_GENERATED asla VERIFIED değil; çelişki günlüğü; versiyon; çakışma ölçümü |
| §22 commit 6 | `memory/sparse_memory.py` + `memory/kopru.py` — seyrek bellek bağlantısı |

## 6. Puanlama formülü (EK-A.8)

```text
Score(e) = w1*property_compatibility + w2*relation_compatibility
         + w3*context_consistency    + w4*memory_support
         + w5*novelty                + w_info_gain*information_gain
         + w6*source_confidence
         - w7*contradiction
```

Ağırlıklar ve eşikler `hga/config/experience_config.yaml` içindedir; `config.py`
PyYAML kuruluysa onu, değilse gömülü düz ayrıştırıcıyı kullanır.

## 7. Durum makinesi özeti (EK-C)

| Durum | Anlam | İşlem |
|---|---|---|
| CANDIDATE | Henüz değerlendirilmedi | Evaluator'a gönder |
| VALID | Mevcut bilgiyle uyumlu | Belleğe aday olarak ekle |
| CONFLICT | Mevcut bilgiyle çelişiyor / yetersiz kanıt | Araştırma kuyruğuna gönder |
| INVALID | Kural/ilişki/özellik açısından uyumsuz | Reddet |
| VERIFIED | Harici/deterministik doğrulama aldı | Kalıcı bilgiye yükselt |

**Güvenlik:** `MODEL_GENERATED` kaynaklı deneyim asla otomatik `VERIFIED`
edilmez (en fazla `VALID`); `Consolidator` bu kuralı ikinci kez denetler.

## 8. Sonraki aşamalar (yol haritasının ötesi)

v0.1–v1.0 çekirdeği tamamlandı; ek olarak Türkçe ek uyumu (`turkce.py`,
ünsüz yumuşaması + 4 durum eki), cümle/dosya → üçlü ayıklayıcı
(`cumle_ayiklayici.py` + `corpus.py`), çelişki kuyruğu otomasyonu
(`arastirma.py`), kalıcılık (`persistence.py`) ve ground-truth benchmark
(`benchmark.py`) eklendi. Kalan gerçekçi adımlar:

- **Tam morfoloji (tamamlandı):** `turkce.py`'ye ünlü düşmesi, 6 kişilik
  iyelik ekleri + iyelik+durum zinciri, 3. tekil fiil çekimleri
  (geçmiş/şimdiki/gelecek/geniş zaman) ve tam kişi ekli çekim
  (`fiil_cekimi`, 6 kişi) eklendi. Kalan (bilinçli kapsam dışı): isim
  tamlamaları ve emir/istek/şart kipleri, daha geniş istisna listeleri
  (docstring'te belgeli).
- **Gerçek görev sinyali + ablasyon ölçekleme:** `GorevAblasyonu` aynı
  protokolü modelin `gen_kopru` çıktı yolu üzerinde, GERÇEK bir görevin
  (dizisel tamamlama) çapraz-entropi sinyaliyle birleştirdi: yoğun gövde
  donukken kontrol ~şans → deney ~%100 (+%100), ölü-yol → canlı-yol.
  Ters yönlü kanıtlar: TABLO-SIFIR (tablo boşaltılınca %100 → %0 — bilgi
  tabloda yaşar) ve HELD-OUT SINIRI (yazılmamış olgular genellemez, ~şans;
  3 tohumda kararlı). Belleğin genellemeye katkısı `genelleme_ablasyonu.py`
  ile ayrıca ölçüldü:
  donuk gövde + eğitilen köprü/kafa, held-out olgularda %100 tamamlama
  (boş bellek ~şans). Bu protokol BÜYÜK + GÜRÜLTÜLÜ korpusta da tekrarlandı
  (`gurultulu_kos`): ölçekte genelleme korunur (%100), gürültü ise dürüstçe
  bozar (held-out ≈ 1 − gürültü_oranı; tam gürültüde ~şans). Sıradaki adım:
  canlı korpus çekimi.
- **Dış korpus ölçeği (tamamlandı):** `egitim/veri_toplayici.py` çıktısı
  (`turkce_metin.txt`) `korpus_boru.py` üzerinden `REAL_DATA` olarak akıtılıyor;
  sözlük bilinen fiil desenleriyle büyütülüyor (yeni özne/nesne varlıkları;
  ilişki asla uydurulmaz). Elle küratörlü sözlük 7 ilişkiye genişletildi
  (yönelme + belirtme); kök çıkarma ünsüz yumuşamasını geri çeviriyor. Ağsız
  ölçek provası `korpus_uretici.py` + `run_korpus_olcegi.py` ile eklendi
  (~40k cümle/sn, belirleyici sentetik korpus). Kalan ölçek adımı: canlı
  korpusu (Wikipedia dökümü, OSCAR/CC-100/mC4 "tr") gerçekten çekip
  milyon-kelime seviyesinde bu borudan geçirmek — bu ağ gerektirir ve
  `egitim/veri_toplayici.py`'nin görevidir.
- **Deneyim döngüsünü sinir ağına bağlamak:** doğrulanmış bilginin gömme
  temsillerini geometrik çekirdeğe enjekte etmek (kapalı doğrulama döngüsü
  `dogrulama.py` + sinirsel köprü `neural_kopru.py` + ablasyon `ablation.py`
  + görev ablasyonu `gorev_ablasyonu.py` kuruldu).

## 9. Dürüst kapasite notu (rapor §17)

`P` (gerçek öğrenilebilir parametre), `C_I` (temsil/etkileşim kapasitesi) ve
`C_M` (adreslenebilir bellek/deneyim uzayı) ayrı ölçeklerdir. Bu katman `P`'yi
artırmaz; `C_M`'yi (deneyim/kanıt uzayı) seyrek ve kaynak-güvenli biçimde
genişletir. "Katrilyon" hedefi, katrilyon bağımsız ağırlık depolamak değil,
küçük bir fiziksel modelin çok büyük bir potansiyel deneyim uzayını aktif olarak
keşfetmesi olarak sınanır.
