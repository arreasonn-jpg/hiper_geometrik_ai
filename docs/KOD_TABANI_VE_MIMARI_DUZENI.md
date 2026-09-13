# HGA Kod Tabanı Mimarisi ve Modül Sınıflandırması (v1.0)

Bu belge, **HGA Project Stabilization & Research Roadmap v1.0 (P0-002, P0-003, P0-004)** gereğince kod tabanının tek gerçek kaynağını (Single Source of Truth), mimari katmanlarını ve modül statülerini resmileştirir.

---

## 🏛️ 1. MİMARİ HİYERARŞİ VE TEK GERÇEK KAYNAK

Proje 3 temel sütun üzerinde yapılandırılmıştır:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                      HGA EXPERIENCE & KNOWLEDGE ENGINE                  │
│                                  (hga/)                                 │
│                                                                         │
│  ├── hga.knowledge       : Entity, Property, Relation, KnowledgeStore   │
│  ├── hga.experience      : Generator, Evaluator, StateMachine, Scoring  │
│  ├── hga.memory          : SparseMemory, Replay, Bridge & Integration  │
│  ├── hga.evaluation      : Benchmark, Hallucination Metrics, Reporting  │
│  ├── hga.observability   : Attention, Geometric, Memory & Flow Panels   │
│  └── hga.data            : Quality Filter, Manifest & Versioning        │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 │                                       │
                 ▼                                       ▼
┌──────────────────────────────────┐   ┌──────────────────────────────────┐
│        NEURAL CORE ENGINE        │   │    EXPERIMENT & BENCHMARK        │
│             (mimari/)            │   │         (experiments/)           │
│                                  │   │                                  │
│  • KureselZincir (Bilinear A@X@B)│   │  • run_full.py                   │
│  • HiperGeometrikAttention       │   │  • run_benchmark.py              │
│  • HashlenmisKureselTablo        │   │  • run_ablation.py               │
│  • BPETokenizer (Tek Standart)   │   │  • run_genelleme_ablasyonu.py    │
│  • GeometrikVeriEncoder/Decoder  │   │  • run_korpus_boru.py            │
└──────────────────────────────────┘   └──────────────────────────────────┘
```

1. **Orkestrasyon ve Bilgi/Deneyim Motoru (`hga/`):** Tek kanonik karar, deneyim üretimi, değerlendirme ve bilgi tabanı katmanıdır.
2. **Sinirsel Çekirdek (`mimari/`):** Kronecker bilinear zinciri, dikkat ve seyrek bellek tensörlerini işleten PyTorch hesaplama çekirdeğidir.
3. **Köprü Katmanı (`hga/memory/kopru.py` & `neural_kopru.py`):** Semantik/epizodik deneyimleri sinirsel çekirdeğin seyrek tablosuna ve latent uzayına taşır.

---

## 🏷️ 2. DOSYA VE MODÜL STATÜ MATRİSİ

| Modül / Dosya | Statü | Açıklama |
| :--- | :---: | :--- |
| `hga/engine.py` | `ACTIVE` | Tüm motorun tek yüzlü orkestratörü (`ExperienceEngine`). |
| `hga/knowledge/*` | `ACTIVE` | Kanonik Varlık, Özellik ve İlişki dizinleri (`KnowledgeStore`). |
| `hga/experience/*` | `ACTIVE` | Deneyim durum makinesi, puanlama, üretici, değerlendirici ve döngü. |
| `hga/memory/*` | `ACTIVE` | Epizodik bellek, replay buffer ve sinirsel köprü (`NeuralKopru`). |
| `hga/evaluation/*` | `ACTIVE` | Standart metrikler, halüsinasyon kontrolü ve raporlama. |
| `hga/observability/*` | `ACTIVE` | Bellek doluluk, dikkat ve deneyim akışı paneli. |
| `hga/data/*` | `ACTIVE` | Veri kalitesi, SHA-256 manifestosu ve versiyonlama. |
| `hga/config/*` | `ACTIVE` | Merkezi YAML konfigürasyon yöneticisi. |
| `mimari/kuresel_model.py` | `ACTIVE` | Ana PyTorch modeli (`HiperGeometrikAI`). |
| `mimari/kuresel_bag.py` | `ACTIVE` | Bilinear Kronecker matris sandviç katmanı (`A @ X @ B`). |
| `mimari/hiper_attention.py` | `ACTIVE` | Nedensel geometrik dikkat mekanizması. |
| `mimari/seyrek_tablo.py` | `ACTIVE` | Sabit boyutlu hash tabanlı seyrek bellek (`HashlenmisKureselTablo`). |
| `mimari/bpe_tokenizer.py` | `ACTIVE` | **Tek standart tokenizer**. |
| `egitim/*` | `ACTIVE` | Eğitim döngüleri, determinizm ve scheduler yardımcıları. |
| `experiments/*` | `EXPERIMENTAL` | Ablasyon ve ölçekleme deney koşucuları. |
| `bilgi_katmani.py` | `LEGACY` | Eski 3 katmanlı string eşleme prototipi. Yerini `hga.knowledge` ve `hga.evaluation.hallucination` almıştır. |
| `mimari/tokenizer.py` | `LEGACY` | Karakter tabanlı eski tokenizer. Yerini `mimari/bpe_tokenizer.py` almıştır. |
| `calistir.py` / `arayuz.py` | `LEGACY / WRAPPER` | Eski prototip arayüzleri. CLI için standart: `python -m hga`. |

---

## 🔒 3. DEDUPLİKASYON VE ÇİFT SİSTEMLERİN KALDIRILMASI

* **Tokenizer:** `mimari/tokenizer.py` yerine tamamen `mimari/bpe_tokenizer.py` (`BPETokenizer`) kullanılır. Eski tokenizer sadece geriye dönük test uyumluluğu için tutulur.
* **Knowledge Store:** `bilgi_katmani.py` yerine `hga.knowledge.KnowledgeStore` tek yetkilidir.
* **Config:** `mimari/model_config.py` ile `hga/config/` arasındaki parametreler `pyproject.toml` ve YAML configleri ile tam uyumlu hale getirilmiştir.
* **Memory:** Saf Python `DeneyimSlotlari` (prototip/bağımsız test) ile PyTorch `HashlenmisKureselTablo` arasındaki bağ `hga.memory.kopru` üzerinden açık protokol ile yürütülür.
