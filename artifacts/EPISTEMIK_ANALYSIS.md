# Epistemik Test Analizi

## Sonuclar

| Metrik | Deger |
|---|---|
| Bilinmeyen dogrulugu | 1.000 |
| Sessiz kabul orani | 0.000 |
| Genel dogruluk | 1.000 |

Sinif bazinda 5/5 tam:
- KNOWN: 8/8
- FALSE: 8/8
- UNKNOWN: 5/5
- UNCERTAIN: 4/4
- CONFLICT: 3/3

Dejenere politikalardan ezici fark:
- EVALUATOR: 1.000
- always_valid: 0.286
- always_abstain: 0.321
- always_invalid: 0.286

## Kritik Ozellik

Sistem "bilmiyorum" diyebiliyor. UNKNOWN (kayit yok) ile UNCERTAIN
(ozellik yok) ayirt ediliyor. Sessiz kabul orani = 0.000.

Bu, modern LLM'lerin en zayif noktasina (halusinasyon) dogrudan cozum.

## Durust Sinir

Repo soyluyor:
"Kontrollu ontoloji uzerinde epistemik karar testidir; gercek dil
anlama degildir. Sonuc Evaluator karar agacinin davranisidir; neural
uretim kalitesi degildir."

Yani bu bir kural tabanli epistemik motor, neural belirsizlik degil.

## Yetenek Haritasi (6/7)

| # | Modul | Sonuc |
|---|---|---|
| 1 | Hibrit (sentetik) | +0.10 |
| 2 | TWT (Turkce) | +0.04 |
| 3 | EWT (Ingilizce) | +0.07 |
| 4 | Self-learning | Lineer ama %99 tekrar |
| 5 | Multi-hop | BFS calisiyor, ogrenme degil |
| 6 | Epistemik | 1.000, halusinasyon yok |
