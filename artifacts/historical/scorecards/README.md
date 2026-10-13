# Tarihsel Skor Kartları

Bu dizin, **P0-4 fix öncesi** (2026-09-21/22) üretilen skor kartlarını
içerir. Bu kartlar artık **kanonik değildir**.

## Neden Arşivlendi?

Bu kartlar `human_evaluation` bölümünü **10.0** olarak gösteriyordu.
Bu puan, gerçek değerlendirici beyanı olmadan (yalnız doldurulmuş
sentetik CSV ile) üretilmişti. P0-4 sonrası guard üç koşulu birlikte
arar: `human_ratings_collected` + `rater_attestation_present` +
`source_kind == "real"`. Üçü sağlanmadığı için yeni kanonik kartta
`human_evaluation: n/a`.

## Kanonik Kart

- `docs/scorecard.json` (JSON)
- `docs/SCORECARD.md`   (Markdown)

Üretim komutu:

    python -m hga championship-benchmark \
        --signature-profile smoke --seeds 1 \
        --out docs/scorecard.json --markdown docs/SCORECARD.md

## Referans

- `SECURITY.md` → SEC-2026-10-13-02
- `docs/SCORECARD_SPEC.md` (yazılacak)
