# Bilgi Sürümleme, Rollback ve Immutable Experience Ledger

Roadmap Faz 23 (Knowledge Rollback) ve Faz 24 (Immutable Experience Log).

Kendi ürettiği deneyimlerle büyüyen bir sistemde iki soru er ya da geç sorulur:

1. **"K₃ yanlışsa K₂'ye dönebiliyor muyuz?"**
2. **"Model geçmişte tam olarak nerede hata yaptı?"**

Bu iki soruyu cevaplamak self-learning için lüks değil sigortadır. Aşağıdaki
iki katman tam olarak bunun içindir.

---

## 1. Knowledge Versioning (`hga/knowledge/versioning.py`)

```text
K0 ──commit──▶ K1 ──commit──▶ K2 ──commit──▶ K3
                ▲
                └── rollback("K1") → K4 (içerik = K1)
```

### Sözleşme

| Kural | Gerekçe |
|---|---|
| Snapshot **içerik-adreslidir** (`state_hash`, `content_hash`) | "Bu iki sürüm gerçekten aynı mı?" sorusu deterministik cevaplanır |
| Snapshot **derin kopyadır** | `KnowledgeStore.to_dict()` bazı alanlarda canlı referans döndürür; snapshot sonradan değişmemelidir |
| `rollback` **geçmişi silmez** | git `revert` semantiği; hatalı sürüm denetim için zincirde kalır |
| Sürüm sayacı **geriye gitmez** | Bilgi tabanı zamanda geri sarmaz; yalnız içerik geri gelir |
| `content_hash`, `versiyon` alanını **dışlar** | Rollback sonrası içerik eşitliği sayaç farkına takılmasın |
| `checkout` **yan etkisizdir** | Eski bir sürümü incelemek aktif store'u bozamaz |

### Kullanım

```python
from hga.knowledge import KnowledgeVersionStore

kv = KnowledgeVersionStore()                 # K0
kv.store.olgu_kaydet("E_001", "R_001", "E_002", score=1.0)
kv.commit("K1: doğrulanmış olgu")            # K1
kv.store.olgu_kaydet("E_001", "R_001", "E_003", score=1.0)   # hatalı
kv.commit("K2: hatalı olgu")                 # K2

kv.rollback("K1")                            # K3 = K1 içeriği
kv.diff("K2", "K3").silinen_olgular          # [['E_001','R_001','E_003']]
kv.zincir_dogrula()                          # True
kv.kaydet("logs/surumler.json")
```

CLI: `python -m hga bilgi-surum`

### Bütünlük denetimi

`zincir_dogrula()` şunları kontrol eder: indeks sırası, `parent_id` bağları,
her snapshot'ın hash'i, ve ROLLBACK kayıtlarının hedefle birebir aynı içeriğe
sahip olması. Snapshot gövdesine sonradan dokunulursa `False` döner
(`tests/test_knowledge_versioning.py::test_bozulan_snapshot_zincir_dogrulamasinda_yakalanir`).

---

## 2. Immutable Experience Ledger (`hga/experience/ledger.py`)

Reddedilen deneyim de bilgidir. Defter **append-only** ve **hash-zincirlidir**:

```text
entry_hash = SHA256( prev_hash || kanonik_json(kayıt gövdesi) )
```

Her kayıt:

```json
{
  "seq": 12,
  "experience": "E_001|R_001|E_004",
  "experience_id": "MS-1-0003-0007",
  "status": "INVALID",
  "reason": "bağımsız doğrulayıcı çürüttü",
  "evaluator": "ExperienceEvaluator",
  "verifier": "arithmetic-env-v1",
  "source": "MODEL_GENERATED",
  "knowledge_version": "K12",
  "timestamp": "2026-09-13T10:00:00+00:00",
  "scores": {"novelty": 0.5},
  "cycle": 3,
  "prev_hash": "…",
  "entry_hash": "…"
}
```

### Sözleşme

* `sil` / `guncelle` API'si **bilinçli olarak yoktur**.
* `entries` property'si kopya döndürür; dıştan mutasyon defteri bozamaz.
* Geçmiş bir kaydı "düzeltmek" (ör. INVALID → VERIFIED) zinciri kırar ve
  `zincir_dogrula()` `False` döner.
* Tek deneyimin yaşam öyküsü `gecmis(experience_id)` ile izlenir:
  `VALID → VERIFYING → INVALID` gibi tüm geçişler ayrı kayıtlardır.
* JSONL biçimi append-only dosya yazımını destekler (`ekle_jsonl`).

### Kullanım

```python
from hga.experience import ExperienceLedger

defter = ExperienceLedger()
defter.kaydet(aday, knowledge_version="K12", reason="kural ihlali", cycle=3)
defter.ozet()          # durum dağılımı + head_hash + zincir_gecerli
defter.kaydet_jsonl("logs/defter.jsonl")
```

CLI: `python -m hga defter`

Milestone koşusunda defter otomatik doldurulur:

```bash
python -m hga milestone --cycles 100 --batch 32 --ledger logs/defter.jsonl
```

100 döngü × (32 aday + 3 epistemik prob) = seed başına **3.500 kayıt**, hepsi
hash-zincirli.

---

## 3. Neden bu ikisi birlikte anlamlı?

Milestone deneyi her döngü sonunda Kₙ'i mühürler ve her adayı deftere yazar.
Sonuç: **herhangi bir cycle'daki bilgi durumu ile o duruma yol açan tüm
deneyim kararları birlikte yeniden kurulabilir.**

Koşunun sonunda kasıtlı bir rollback tatbikatı yapılır:

| Adım | Sürüm | Yanlış olgu |
|---|---|---:|
| Sağlam durum | `K100` | 0 |
| Kasıtlı bozma | `K101` | 1 |
| Rollback | `K102` | 0 |

`content_restored=True`, `history_preserved=True`. Yani hata geri alındı ama
**kanıt kaybolmadı** — bilimsel denetim için gereken tam olarak budur.
