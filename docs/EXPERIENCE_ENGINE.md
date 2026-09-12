# HGA Experience Engine — v0.1 Uygulama Notu

> Hiper Geometrik AI: Experience Engine / Self-Expanding Knowledge Architecture
> yol haritasının (12 Eylül 2026, v1.0) ilk fiziksel karşılığı.

Bu belge, `hga/` paketi altında kurulan **Knowledge + Experience + Evaluation +
Memory** katmanını açıklar: hangi dosya hangi rapor bölümünü gerçekler, nasıl
çalıştırılır ve hangi güvenlik kuralları korunur.

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
│   │   ├── scoring.py             # bağımsız sinyaller + ağırlıklı puan (§8, §16)
│   │   ├── evaluator.py           # ExperienceEvaluator + VALID/CONFLICT/INVALID (§9)
│   │   ├── conflict.py            # Conflict → Exploration (§11)
│   │   └── consolidation.py       # Consolidator — belleğe/araştırmaya/redde yönlendirme (§12)
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── sparse_memory.py       # DeneyimSlotlari — seyrek deneyim slotları (v0.6 köprüsü)
│   │   └── replay.py              # DeneyimTekrari — experience replay (v0.4 hazırlığı)
│   └── config/
│       ├── __init__.py
│       ├── config.py              # bağımlılıksız YAML yükleyici (PyYAML varsa onu kullanır)
│       └── experience_config.yaml # ağırlıklar + eşikler + üretici ayarları
├── tests/
│   ├── test_entity_index.py
│   ├── test_relation_index.py
│   ├── test_experience_eval.py
│   ├── test_conflict.py
│   └── test_milestone_v01.py      # §14'ün 12 maddesi + §15 senaryosu
├── experiments/
│   └── experience_loop/
│       └── run_demo.py            # uçtan uca döngü demosu
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

# Uçtan uca döngü demosu
python experiments/experience_loop/run_demo.py
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
| §12 Self-expanding loop | `consolidation.py` + `run_demo.py` |
| §16 1+1+1 yorumu | `scoring.py` — toplam puan kanıt değildir; kısıtlar ayrıca karar verir |
| §21 Riskler | MODEL_GENERATED asla VERIFIED değil; çelişki günlüğü; versiyon; çakışma ölçümü |
| §22 commit 6 | `memory/sparse_memory.py` — saf Python slot deposu (torch'a v0.6 köprüsü) |

## 6. Puanlama formülü (EK-A.8)

```text
Score(e) = w1*property_compatibility + w2*relation_compatibility
         + w3*context_consistency    + w4*memory_support
         + w5*novelty                + w6*source_confidence
         - w7*contradiction
```

Ağırlıklar ve eşikler `hga/config/experience_config.yaml` içindedir; `config.py`
PyYAML kuruluysa onu, değilse gömülü düz ayrıştırıcıyı kullanır.

## 7. Sonraki aşamalar (rapor §19)

- **v0.2** — Generator'ı kontrollü kombinasyondan gerçek metin/olay üretimine genişlet.
- **v0.3** — `novelty` / `information-gain` skorlarını zenginleştir.
- **v0.4** — `memory/replay.py`'yi mevcut seyrek bellek + eğitim döngüsüyle birleştir.
- **v0.5** — deterministik kurallarla doğrulanabilir mini-environment (matematik/mantık).
- **v0.6** — `memory/sparse_memory.py` köprüsünü `mimari/seyrek_tablo.py` (torch) ile birleştir:
  deneyim vektörlerini gerçek `HashlenmisKureselTablo`'ya yaz.
- **v1.0** — gerçek veri + üretilen deneyim + doğrulama + continual learning döngüsünü
  kontrollü benchmarklarla ölç.

## 8. Dürüst kapasite notu (rapor §17)

`P` (gerçek öğrenilebilir parametre), `C_I` (temsil/etkileşim kapasitesi) ve
`C_M` (adreslenebilir bellek/deneyim uzayı) ayrı ölçeklerdir. Bu katman `P`'yi
artırmaz; `C_M`'yi (deneyim/kanıt uzayı) seyrek ve kaynak-güvenli biçimde
genişletir. "Katrilyon" hedefi, katrilyon bağımsız ağırlık depolamak değil,
küçük bir fiziksel modelin çok büyük bir potansiyel deneyim uzayını aktif olarak
keşfetmesi olarak sınanır.
