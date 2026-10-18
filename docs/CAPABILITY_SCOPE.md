# Capability Vector — Kapsam ve Sınırlar (P1-4)

Her capability metriği belirli bir protokolün ürünüdür. Bu belge,
`docs/SCORECARD.md`'deki skorların hangi dar kapsamda geçerli olduğunu
ve hangi geniş iddiaları DESTEKLEMEDİĞİNİ açıkça listeler.

Kaynak: `hga/experiments/capability_vector.py` içindeki `TERMINOLOGY`.

## Daraltılmış 5 Metrik (P1-4, 2026-10-15)

| Sembol | Yeni Ad | Protokol | Kapsam | Dışında Kalan |
|---|---|---|---|---|
| `C_R` | Verified Relational Chain Depth | `reliable_reasoning_depth_v1` | Sembolik BFS/ilişki motoru, tek ilişki tipi | Doğal metin üzerinde öğrenilmiş reasoning |
| `C_RD` | Distractor-Resistant Chain Depth | `reliable_reasoning_depth_v1` | Aynı motor, sentetik dolgu | Genel dayanıklılık |
| `C_G` | Compositional Fixture v2 Score | `compositional_generalization_v2_raw_text` | Küçük, kural-tabanlı fixture (~14 cümle) | Genel kompozisyonel genelleme, dış geçerlilik |
| `C_MR` | Exact Record Lookup Rate | `hierarchical_memory_v1` | Exact-key lookup; çakışma ve eviction sınırlı | Semantik retrieval, RAG |
| `C_H` | Signature Benchmark Resistance | `hga_signature_benchmark_v1` | Kontrollü imza fixture'ı | Açık dünya halüsinasyon oranı |

## Değişmeyen Semboller

`P`, `C_I^UB`, `C_M^UB`, `C_E`, `C_V`, `C_U` — isimleri zaten ölçtükleri
şeyi dürüstçe yansıtıyordu.

## Neden Daraltma?

Audit raporu (2026-10-13): "C_R = genel reasoning kapasitesi" gibi
okunuşlar, ölçülen protokolün dar kapsamını gizliyordu. Bu belge ve
`TERMINOLOGY` içindeki `scope`/`limitations` alanları, her metriğin
yanında kapsamını taşır.

## JSON Şema Etkisi

`CapabilityEntry` veri sınıfına 3 alan eklendi (default boş string,
geriye uyumlu):
- `protocol: str`
- `scope: str`
- `limitations: str`

Eski JSON çıktıları okunabilir; yeni çıktılarda bu alanlar dolu gelir.

## Regresyon

`tests/test_capability_naming.py` (9 test):
- Yeni isimler doğru mu
- Kapsam alanları dolu mu (5 metrik için)
- Eski geniş isimler kaldırıldı mı
- `CapabilityEntry` default değerleri geriye uyumlu mu
