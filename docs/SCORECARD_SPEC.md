# SCORECARD_SPEC — Kanonik Skor Kartı Sözleşmesi

## Kanonik Dosyalar
- `docs/scorecard.json` — makine okunur
- `docs/SCORECARD.md` — insan okunur

## Tek Üretim Yolu
Elle düzenleme YASAKTIR. Kart yalnız şu komutla üretilir:

    python -m hga championship-benchmark \
        --signature-profile smoke --seeds 1 \
        --out docs/scorecard.json --markdown docs/SCORECARD.md

## Zorunlu Alanlar (JSON)
- `generated_at_utc`: ISO 8601 (T ve +00:00 içerir)
- `overall`: float veya null (skorlu bölümlerin ortalaması)
- `scored_sections`: int (0..13)
- `unscored_sections`: string listesi
- `sections`: dict, 13 kanonik bölüm
  Her bölüm: `section`, `score`, `inputs`, `evidence`, `rationale`
  `score`: null veya [0.0, 10.0]

## Kanonik Bölümler (13)
architecture, memory, verification, generalization, reasoning,
self_learning, statistical_rigor, human_evaluation, turkish_nlp,
language_modeling, reproducibility, scientific_evidence, engineering

## human_evaluation Kuralı (SEC-2026-10-13-02)
Sadece üçü birlikte doğruysa skorlanır:
1. `checks.human_ratings_collected == true`
2. `checks.rater_attestation_present == true`
3. `source_kind == "real"`

Aksi halde `score=null` ve `unscored_sections` içinde yer alır.

## CI Kontrolü
`tests/test_scorecard_canonical.py` şunları doğrular:
- Zorunlu alanlar mevcut
- 13 kanonik bölüm
- Skorlar [0,10] veya null
- overall = skorlu bölümlerin ortalaması (yaklaşık)
- unscored_sections doğru
- human_evaluation kuralı uygulanmış

## Referanslar
- `SECURITY.md`: SEC-2026-10-13-01, SEC-2026-10-13-02
- `artifacts/historical/scorecards/README.md`
- `MYPY_RUFF_ROADMAP.md`
