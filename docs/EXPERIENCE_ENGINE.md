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
│   │   └── knowledge_store.py     # KnowledgeStore (§3) + bellek desteği + çelişki günlüğü
│   ├── experience/
│   │   ├── __init__.py
│   │   ├── generator.py           # ExperienceGenerator — kontrollü kombinasyon (§7)
│   │   ├── text_generator.py      # TextGenerator — üçlüden metin/olay üretimi (§19 v0.2)
│   │   ├── turkce.py              # Türkçe ek uyumu (yönelme -a/-e, özel isim kesme işareti)
│   │   ├── cumle_ayiklayici.py    # Cümle → üçlü + REAL_DATA aktarımı (döngünün "gerçek veri" aşaması)
│   │   ├── scoring.py             # bağımsız sinyaller + ağırlıklı puan + information_gain (§8, §16, v0.3)
│   │   ├── evaluator.py           # ExperienceEvaluator + VALID/CONFLICT/INVALID (§9)
│   │   ├── conflict.py            # Conflict → Exploration (§11)
│   │   ├── arastirma.py           # ArastirmaKuyrugu — çelişkiyi deterministik kanıtla toplu çözme
│   │   ├── consolidation.py       # Consolidator — belleğe/araştırmaya/redde yönlendirme (§12)
│   │   ├── mini_env.py            # AritmetikOrtam — deterministik doğrulayıcı (§18, §19 v0.5)
│   │   └── loop.py                # DeneyimDongusu — sürekli öğrenme + metrikler (§19 v1.0, §20)
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── sparse_memory.py       # DeneyimSlotlari — seyrek deneyim slotları (§22 commit 6)
│   │   ├── replay.py              # DeneyimTekrari — experience replay (§19 v0.4)
│   │   ├── entegrasyon.py         # BellekEntegrasyonu — replay + consolidation ↔ seyrek bellek
│   │   └── kopru.py               # TorchKoprusu — deneyim ↔ torch seyrek tablo köprüsü (§19 v0.6)
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
│   ├── test_turkce.py             # Türkçe ek uyumu (yönelme + kesme işareti)
│   ├── test_cumle_ayiklayici.py   # cümle → üçlü + REAL_DATA aktarımı
│   └── test_arastirma.py          # araştırma kuyruğu (CONFLICT → kanıt → kesin durum)
├── experiments/
│   └── experience_loop/
│       ├── run_demo.py            # v0.1 milestone demosu
│       ├── run_full.py            # v0.1 → v1.0 tam yol haritası demosu
│       └── run_gercek_veri.py     # gerçek veri → temsil → deneyim → doğrulama
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
python tests/test_arastirma.py

# Uçtan uca döngü demoları
python experiments/experience_loop/run_demo.py        # v0.1 milestone
python experiments/experience_loop/run_full.py        # v0.1 → v1.0 tam yol haritası
python experiments/experience_loop/run_gercek_veri.py # gerçek veri → temsil → deneyim → doğrulama
```

Testlerin hiçbiri torch gerektirmez; yalnızca standart kütüphane kullanılır.

## 4. Temel döngü ve durum makinesi

```text
Bilgi tabanı ──> Generator (kontrollü kombinasyon) ──> CANDIDATE
      │                                                    │
      │                                                    ▼
      │                                    Evaluator (kural + puan)
      │                                    ├─ kural ihlali  → INVALID  → reddet
      │                                    ├─ kanıt çelişkisi→ CONFLICT → araştır
      │                                    ├─ MODEL_GENERATED→ VALID    (asla VERIFIED değil)
      │                                    └─ harici+ yüksek puan → VERIFIED
      ▼
Consolidator ──> VALID → belleğe aday yaz; VERIFIED → kalıcı bilgi; CONFLICT → kuyruk
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
| §19 v1.0 | `loop.py` — `DeneyimDongusu` + kontrollü metrikler |
| §2 "Gerçek veri → Temsil" | `cumle_ayiklayici.py` — cümle → üçlü + REAL_DATA aktarımı |
| §11 ölçekleme | `arastirma.py` — CONFLICT kuyruğunu deterministik kanıtla toplu çözme |
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

v0.1–v1.0 çekirdeği tamamlandı; ek olarak Türkçe ek uyumu (`turkce.py`),
cümle → üçlü ayıklayıcı (`cumle_ayiklayici.py`) ve çelişki kuyruğu otomasyonu
(`arastirma.py`) eklendi. Kalan gerçekçi adımlar:

- **Tam morfoloji:** `turkce.py`'deki yönelme ekini ünsüz yumuşaması
  (kitap→kitaba), diğer durum ekleri ve çekimli fiil üretimiyle genişletmek
  (bilinçli olarak kapsam dışı bırakıldı; docstring'te belgeli).
- **Gerçek torch köprüsü:** `memory/kopru.py`'yi `mimari/seyrek_tablo.py`
  eğitim döngüsüne bağlayıp deneyim vektörlerinin gradyanla dolmasını ölç.
- **Dış korpus ölçeği:** `egitim/veri_toplayici.py` korpusunu
  `cumle_ayiklayici` üzerinden `REAL_DATA` olarak KnowledgeStore'a akıtmak —
  sözlük, gerçek korpustan çıkarılan desenlerle büyütülmeli.
- **Deneyim döngüsünü sinir ağına bağlamak:** doğrulanmış bilginin gömme
  temsillerini geometrik çekirdeğe enjekte etmek.

## 9. Dürüst kapasite notu (rapor §17)

`P` (gerçek öğrenilebilir parametre), `C_I` (temsil/etkileşim kapasitesi) ve
`C_M` (adreslenebilir bellek/deneyim uzayı) ayrı ölçeklerdir. Bu katman `P`'yi
artırmaz; `C_M`'yi (deneyim/kanıt uzayı) seyrek ve kaynak-güvenli biçimde
genişletir. "Katrilyon" hedefi, katrilyon bağımsız ağırlık depolamak değil,
küçük bir fiziksel modelin çok büyük bir potansiyel deneyim uzayını aktif olarak
keşfetmesi olarak sınanır.
